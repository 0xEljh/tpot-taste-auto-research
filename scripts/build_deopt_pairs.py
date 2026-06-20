"""Build good-vs-deoptimized taste pairs (Scorer v3, D13).

Curate real tpot hits (high within-author z + high-fav liked), then use the local
Qwen to DEGRADE each into a bland/wordy/corporate/cringe version on the same topic.
chosen = real, rejected = degraded → the Scorer learns crafted-tpot ≻ degraded.
Train/test split is preserved (test goods come from the held-out time window).

  uv run python scripts/build_deopt_pairs.py
"""
from __future__ import annotations

import random
import re
from datetime import datetime
from pathlib import Path

import polars as pl
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

SYS = ("You rewrite social-media posts exactly as instructed. Output ONLY the rewritten "
       "post text — no preamble, no quotes, no notes, no explanation.")
# Length-BALANCED degradation styles (v4, D14): ~half make the post SHORTER/terser,
# ~half LONGER/wordier, so length stops predicting quality and bait/low-effort become
# explicit negatives.
STYLES = [
    # shorter / terser-bad
    "Rewrite this post as a low-effort, lazy one-liner that drops the substance and specificity.",
    "Rewrite this post as engagement-bait: a 'RT if you agree' / 'like if you...' / poll-style post that fishes for interaction.",
    "Rewrite this post as a vague, contentless subtweet that hints at something but says nothing.",
    "Rewrite this post as a generic hot-take that strips out the nuance and the specifics.",
    # length-preserving-bad
    "Rewrite this post to be bland, generic and forgettable: strip the specific detail and the punch, keep only the rough topic.",
    "Rewrite this post to be dry and humourless, stripping out any wit, voice or personality.",
    # longer / wordier-bad
    "Rewrite this post as a wordy, over-explained LinkedIn-style post with corporate phrasing and a tidy takeaway.",
    "Rewrite this post so it rambles and buries the point, and make it about twice as long.",
    "Rewrite this post as a generic motivational platitude on the same theme.",
    "Rewrite this post to be try-hard and cringe, with forced enthusiasm, emojis and hashtags.",
]
_PREFIXES = ("sure,", "here'", "here is", "rewritten", "okay", "ok,", "certainly")
_NONLATIN = re.compile(r"[぀-ヿ㐀-鿿가-힯؀-ۿ฀-๿]")
_SPAM = re.compile(
    r"(?i)(giveaway|airdrop|\bnft\b|presale|whitelist|mint(ing)?\b|link in bio|claim your|"
    r"retweet to win|follow (us|me) (and|to)|use code|promo ?code|\$[A-Z]{2,6}\b)"
)


def _quality_english(t: str) -> bool:
    if _NONLATIN.search(t) or _SPAM.search(t):
        return False
    ascii_letters = sum(1 for c in t if c.isalpha() and ord(c) < 128)
    return ascii_letters >= 20  # enough real English text


def _clean_gen(s: str) -> str:
    s = s.strip().strip('"').strip("'").strip()
    low = s.lower()
    if any(low.startswith(p) for p in _PREFIXES) and ":" in s[:40]:
        s = s.split(":", 1)[1].strip().strip('"').strip()
    return " ".join(s.split())


def _good(df: pl.DataFrame, by: str, min_val, n: int, cap: int, lo: int, hi: int) -> list[str]:
    d = df.filter((pl.col("text_clean_len") >= lo) & (pl.col("text_clean_len") <= hi))
    if "is_reply" in d.columns:
        d = d.filter(~pl.col("is_reply"))
    d = d.filter(pl.col(by) >= min_val).sort(by, descending=True)
    d = d.group_by("account_id", maintain_order=True).head(cap).sort(by, descending=True)
    texts = [t for t in d["text_clean"].to_list() if _quality_english(t)]
    return texts[:n]


@app.command()
def main(
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    splits_dir: Path = Path("data/splits"),
    liked_path: Path = Path("data/processed/liked_taste.parquet"),
    cutoff: str = "2025-06-01",
    maturity: str = "2026-01-17",
    n_up_train: int = 7000,
    n_liked_train: int = 3000,
    n_up_test: int = 1000,
    n_liked_test: int = 500,
    z_min: float = 1.0,
    liked_min_fav: int = 50,
    cap: int = 60,
    lo: int = 40,
    hi: int = 240,
    batch_size: int = 64,
    max_new_tokens: int = 90,
    tag: str = "",
    seed: int = 0,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rng = random.Random(seed)
    cut, mat = datetime.fromisoformat(cutoff), datetime.fromisoformat(maturity)
    liked = pl.read_parquet(liked_path)
    liked_tr = liked.filter(pl.col("created_dt") < cut)
    liked_te = liked.filter((pl.col("created_dt") >= cut) & (pl.col("created_dt") < mat))

    goods = {
        "train": _good(pl.read_parquet(splits_dir / "train_tweets.parquet"), "z", z_min, n_up_train, cap, lo, hi)
        + _good(liked_tr, "favorite_count", liked_min_fav, n_liked_train, cap, lo, hi),
        "test": _good(pl.read_parquet(splits_dir / "test_temporal_tweets.parquet"), "z", z_min, n_up_test, cap, lo, hi)
        + _good(liked_te, "favorite_count", liked_min_fav, n_liked_test, cap, lo, hi),
    }
    for k in goods:
        rng.shuffle(goods[k])
    print(f"[curate] train goods={len(goods['train']):,} test goods={len(goods['test']):,}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16).eval()

    def degrade(texts: list[str]) -> list[tuple[str, str]]:
        styles = [rng.choice(STYLES) for _ in texts]
        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            bt, bs = texts[i : i + batch_size], styles[i : i + batch_size]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYS}, {"role": "user", "content": f"{s}\n\nPost:\n{t}"}],
                    tokenize=False, add_generation_prompt=True,
                )
                for t, s in zip(bt, bs)
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                     temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
            for j in range(len(bt)):
                out.append(_clean_gen(tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)))
            if i % (batch_size * 10) == 0:
                print(f"  ...{i + len(bt)}/{len(texts)}")
        return list(zip(out, styles))

    for split, texts in goods.items():
        print(f"[degrade] {split}: {len(texts):,}")
        degs = degrade(texts)
        rows = []
        for good, (bad, style) in zip(texts, degs):
            if len(bad) < 15 or bad.lower() == good.lower():
                continue
            rows.append({"text_clean_w": good, "text_clean_l": bad, "style": style})
        out = splits_dir / f"deopt{tag}_{split}_pairs.parquet"
        pl.DataFrame(rows).write_parquet(out)
        print(f"[write] {out} : {len(rows):,} pairs (dropped {len(texts) - len(rows)})")
        for r in rows[:3]:
            print(f"   GOOD: {r['text_clean_w'][:90]}")
            print(f"   BAD : {r['text_clean_l'][:90]}\n")
    print("[done]")


if __name__ == "__main__":
    app()
