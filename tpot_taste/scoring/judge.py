"""Taste judge (Phase 6c, D29) — separate tpot-taste from generic-competent-virality WITHOUT engagement.

The convergence artifact from D28: directives 1 (curate the goods) and 3 (self-play / "same model in
scoring") both reduce to one tool — a model that, given a post (or a pair), judges *tpot taste* rather than
*engagement*. It then (a) relabels/mines preference pairs to retrain the BT scorer sans the engagement
proxy, and (b) curates goods (keep tpot, drop generic-viral/promo/platitude).

Risk (the crux): a base 3B judge may share the mainstream prior that LIKES platitudes — the very failure.
So the judge runs on a hand-tuned RUBRIC (not its raw prior) and is graded on the dipstick panel before we
trust it (scripts/judge_eval.py). Pairwise (matches BT + less miscalibration) is primary; pointwise scoring
is for curation thresholds. Order is randomized + double-checked (agree_winner) to kill position bias.

Pure helpers here are unit-tested (tests/test_judge.py); model calls are smoke-tested via the eval script.
"""
from __future__ import annotations

import re

# First-draft rubric. CALIBRATED against the dipstick panel (tweak until judge ranks tpot>platitude there).
RUBRIC = (
    "You judge whether a social-media post has *tpot* taste — the voice of the introspective, "
    "intellectually playful corner of tech Twitter.\n\n"
    "MORE tpot (good):\n"
    "- specific and concrete: a real detail, a particular experience, a named thing\n"
    "- a surprising or earned turn of thought; shows rather than tells\n"
    "- first-person observation; wry, earnest, or a little strange\n"
    "- compressed insight — a sharp aphorism that had to be NOTICED, not looked up\n\n"
    "LESS tpot (bad) — these often do well on BROADER Twitter but are NOT tpot:\n"
    "- generic motivational platitudes / self-help advice ('you should...', 'discipline is...')\n"
    "- engagement-bait ('RT if', 'agree?', 'tag someone'), hashtag/emoji spam\n"
    "- corporate / LinkedIn voice; promotional announcements ('excited to announce', launches)\n"
    "- vague, abstract, or assistant-like ('Absolutely! Here's a post...')\n\n"
    "The hard call: a sharp APHORISM is tpot; a motivational PLATITUDE is not — even when they rhyme. "
    "Judge the specificity and the earned-ness, not the surface uplift."
)


def build_pairwise_prompt(a: str, b: str) -> list[dict]:
    user = (
        "Two posts. Which has more tpot taste, per the rubric?\n\n"
        f"Post A:\n{a}\n\nPost B:\n{b}\n\n"
        "Reason in ONE sentence, then end with exactly one line: `VERDICT: A` or `VERDICT: B`."
    )
    return [{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}]


def build_score_prompt(text: str) -> list[dict]:
    user = (
        "Rate this post's tpot taste from 0 (generic / bait / corporate / platitude) to "
        "10 (sharp, specific, unmistakably tpot).\n\n"
        f"Post:\n{text}\n\n"
        "Reason in ONE sentence, then end with exactly one line: `SCORE: <0-10>`."
    )
    return [{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}]


def parse_verdict(text: str) -> str | None:
    """'A' or 'B' from a judge reply; None if undecidable. Prefers the explicit VERDICT line."""
    m = re.search(r"VERDICT:\s*([AB])", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([AB])\b", text)  # loose fallback: first standalone A/B
    return m.group(1).upper() if m else None


def parse_score(text: str) -> float | None:
    """Float in [0,10] from a judge reply; None if no number. Clamps out-of-range."""
    for pat in (r"SCORE:\s*(-?\d+\.?\d*)", r"(-?\d+\.?\d*)\s*/\s*10", r"(-?\d+\.?\d*)"):
        m = re.search(pat, text, re.I)
        if m:
            return max(0.0, min(10.0, float(m.group(1))))
    return None


def agree_winner(v1: str | None, v2: str | None) -> str | None:
    """Resolve two swapped-order verdicts to a winning POST ('a'/'b'), or None on disagreement.

    v1 = verdict for prompt order (a, b): 'A'->a, 'B'->b.
    v2 = verdict for swapped order (b, a): 'A'->b, 'B'->a.
    Returns the post only if both rounds agree (kills position bias); else None.
    """
    w1 = {"A": "a", "B": "b"}.get(v1 or "")
    w2 = {"A": "b", "B": "a"}.get(v2 or "")
    return w1 if (w1 is not None and w1 == w2) else None


# ---- model-using paths (smoke-tested via scripts/judge_eval.py, not unit-tested) ----
def _generate(model, tok, messages: list[dict], *, max_new_tokens: int = 160, temperature: float = 0.0) -> str:
    import torch

    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    with torch.no_grad():
        do_sample = temperature > 0
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                            temperature=(temperature if do_sample else None),
                            pad_token_id=tok.pad_token_id)
    return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def judge_pairwise(model, tok, a: str, b: str, **kw) -> str | None:
    return parse_verdict(_generate(model, tok, build_pairwise_prompt(a, b), **kw))


def judge_pairwise_2x(model, tok, a: str, b: str, **kw) -> str | None:
    """Robust winner: judge both orders, return 'a'/'b' only if they agree (else None)."""
    return agree_winner(judge_pairwise(model, tok, a, b, **kw),
                        judge_pairwise(model, tok, b, a, **kw))


def judge_score(model, tok, text: str, **kw) -> float | None:
    return parse_score(_generate(model, tok, build_score_prompt(text), **kw))


def judge_score_batch(model, tok, texts: list[str], *, batch_size: int = 16,
                      max_new_tokens: int = 80) -> list[float | None]:
    """Pointwise taste scores for many texts via LEFT-padded batched greedy generation (D31).

    ~batch_size× faster than looping judge_score — makes corpus-scale judge-labeling feasible (minutes,
    not hours). Restores the tokenizer's padding side afterwards.
    """
    import torch

    old_side = tok.padding_side
    tok.padding_side = "left"
    out: list[float | None] = []
    try:
        for i in range(0, len(texts), batch_size):
            bt = texts[i : i + batch_size]
            prompts = [tok.apply_chat_template(build_score_prompt(t), tokenize=False,
                                               add_generation_prompt=True) for t in bt]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for j in range(len(bt)):
                out.append(parse_score(tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)))
    finally:
        tok.padding_side = old_side
    return out
