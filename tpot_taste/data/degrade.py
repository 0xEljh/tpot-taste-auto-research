"""Reusable taste-degradation (the deopt machinery, factored out — Phase 6c, D32).

Take a good post and rewrite it as a LENGTH/REGISTER-matched worse version (platitude/corporate/bland/...),
so a chosen≻rejected pair isolates TASTE, not length or register. The v5 length-matched styles (D15/D16):
every degradation is held to ~the source length, so "short=bad" never becomes the signal.

Used by `build_deopt_pairs.py` (legacy, engagement-anchored) and `build_taste_deopt_pairs.py` (judge-anchored).
"""
from __future__ import annotations

import random

SYS = ("You rewrite social-media posts exactly as instructed. Output ONLY the rewritten "
       "post text — no preamble, no quotes, no notes, no explanation.")

_LEN = " Keep it roughly the same length as the original — do NOT make it noticeably longer or shorter."
STYLES = [
    "Rewrite this post as low-effort and lazy: drop the substance, specificity and insight." + _LEN,
    "Rewrite this post as engagement-bait that fishes for replies/RTs ('does anyone else...', 'agree?', 'RT if...')." + _LEN,
    "Rewrite this post as vague and contentless — it hints at something but says nothing concrete." + _LEN,
    "Rewrite this post as a generic hot-take with the nuance and the specifics stripped out." + _LEN,
    "Rewrite this post to be bland, generic and forgettable: strip the specific detail and the punch." + _LEN,
    "Rewrite this post to be dry and humourless, stripping out any wit, voice or personality." + _LEN,
    "Rewrite this post in corporate / LinkedIn phrasing with a tidy little takeaway." + _LEN,
    "Rewrite this post as a generic motivational platitude on the same theme." + _LEN,
    "Rewrite this post to be try-hard and cringe, with forced enthusiasm." + _LEN,
    "Rewrite this post so it buries the point and loses the crisp phrasing." + _LEN,
]
# Subset that pushes specifically toward the generic-viral / platitude register v6/v7 over-reward.
PLATITUDE_STYLES = [STYLES[3], STYLES[4], STYLES[6], STYLES[7], STYLES[2]]

_PREFIXES = ("sure,", "here'", "here is", "rewritten", "okay", "ok,", "certainly")


def clean_gen(s: str) -> str:
    s = s.strip().strip('"').strip("'").strip()
    low = s.lower()
    if any(low.startswith(p) for p in _PREFIXES) and ":" in s[:40]:
        s = s.split(":", 1)[1].strip().strip('"').strip()
    return " ".join(s.split())


def degrade_texts(model, tok, texts: list[str], *, styles: list[str] | None = None,
                  batch_size: int = 64, max_new_tokens: int = 90, temperature: float = 0.9,
                  seed: int = 0) -> list[tuple[str, str]]:
    """Return [(degraded_text, style), ...] aligned with `texts` (one random style each)."""
    import torch

    rng = random.Random(seed)
    pool = styles or STYLES
    chosen_styles = [rng.choice(pool) for _ in texts]
    old_side = tok.padding_side
    tok.padding_side = "left"
    out: list[str] = []
    try:
        for i in range(0, len(texts), batch_size):
            bt, bs = texts[i : i + batch_size], chosen_styles[i : i + batch_size]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": SYS}, {"role": "user", "content": f"{s}\n\nPost:\n{t}"}],
                tokenize=False, add_generation_prompt=True) for t, s in zip(bt, bs)]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                     temperature=temperature, top_p=0.95, pad_token_id=tok.pad_token_id)
            for j in range(len(bt)):
                out.append(clean_gen(tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)))
            if i % (batch_size * 10) == 0:
                print(f"  ...degrade {i + len(bt)}/{len(texts)}")
    finally:
        tok.padding_side = old_side
    return list(zip(out, chosen_styles))
