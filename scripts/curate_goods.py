"""Re-curate the Writer's SFT goods with the LOCKED scorer (Phase 6d D34 → Phase 7, re-run on v8).

The Writer was SFT'd on ENGAGEMENT-selected goods (platitude/promo pollution, D28). The LOCKED scorer is a
fast scalar taste scorer, so use it to re-curate at scale: score a broad pool (uploader across ALL z — to
recover the deadpan tpot engagement buried — + liked), keep the top by TASTE. Per-author cap keeps the ~334
voices diverse. Phase 7: LOCKED is now v8 (human-aligned, D43) — re-running re-curates on the human's taste,
the signal D35 lacked. Engagement-BAIT is dropped outright (is_baity): v8 has a 👇-bait blindspot (it scored a
generated bait post +0.71), so ranking-by-score alone would let bait survive into the voice-cloning corpus.

  uv run python scripts/curate_goods.py            # ~40k pool -> top 12k curated goods (LOCKED=v8)
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
_NONLATIN = re.compile(r"[぀-ヿ㐀-鿿가-힯؀-ۿ฀-๿]")
# function words that English tweets (≥40 chars) almost always carry; ASCII-Latin non-English (Slovenian,
# Indonesian, ...) and hash/garbage strings have none — a cheap language proxy without a langdetect dep.
_STOPWORDS = set(
    "the a an is are was were be been of to in into that this it its i you we they he she me my your our "
    "and but or for on with as at by not no if so do does did have has had just like what when why how "
    "about from all one out can will would there your i'm it's don't you're".split()
)
_WORD = re.compile(r"[a-z']+")


def _english(t: str) -> bool:
    # reject non-Latin scripts, diacritic-heavy Latin (Turkish/Romanian), AND stopword-less text
    # (ASCII-only non-English / hash garbage): the v8 scorer over-scores OOD text, so a script check
    # alone lets "çikolata yerine bok..." and "To je potencalno zanimiv..." top the kept set.
    alpha = [c for c in t if c.isalpha()]
    if len(alpha) < 20 or _NONLATIN.search(t):
        return False
    if sum(ord(c) > 127 for c in alpha) / len(alpha) > 0.08:
        return False
    return any(w in _STOPWORDS for w in _WORD.findall(t.lower()))


@app.command()
def main(
    out: Path = Path("data/splits/curated_goods.parquet"),
    splits_dir: Path = Path("data/splits"),
    liked_path: Path = Path("data/processed/liked_taste.parquet"),
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),  # = v8 (Qwen3-8B)
    base: str = typer.Option(None, help="scorer base id; default: read from the scorer's adapter_config"),
    n_up: int = 30000,
    n_liked: int = 12000,
    keep: int = 12000,
    cap: int = 60,
    batch_size: int = 48,
    seed: int = 0,
) -> None:
    import numpy as np
    import polars as pl

    from tpot_taste.engine import resolve_adapter_base
    from tpot_taste.scoring.model import load_trained_scorer, score_texts
    from tpot_taste.writer.grpo_reward import is_baity

    base = base or resolve_adapter_base(scorer, "Qwen/Qwen2.5-3B-Instruct")  # the v8 scorer is on Qwen3-8B

    def sample_rows(df, n):  # sample in polars BEFORE the python/regex pass (filtered sets are large)
        df = df.sample(min(int(n * 1.3), df.height), seed=seed)
        return [(r["text_clean"], r["account_id"]) for r in df.iter_rows(named=True)
                if _english(r["text_clean"]) and not is_baity(r["text_clean"])][:n]

    up = pl.read_parquet(splits_dir / "train_tweets.parquet")
    up = up.filter((pl.col("text_clean_len") >= 40) & (pl.col("text_clean_len") <= 240) & (~pl.col("is_reply")))
    up_rows = sample_rows(up, n_up)

    liked = pl.read_parquet(liked_path).filter(pl.col("created_dt") < datetime.fromisoformat("2025-06-01"))
    liked = liked.filter((pl.col("text_clean_len") >= 40) & (pl.col("text_clean_len") <= 240)
                         & (~pl.col("is_reply")) & (pl.col("favorite_count") >= 20))
    lk_rows = sample_rows(liked, n_liked)

    rows = up_rows + lk_rows
    # dedup
    seen, pool = set(), []
    for t, a in rows:
        k = " ".join(t.split()).lower()
        if k not in seen:
            seen.add(k)
            pool.append((t, a))
    print(f"[pool] {len(pool):,} unique (uploader {len(up_rows):,} + liked {len(lk_rows):,}) — scoring with {scorer.name} ...")

    sm, stok = load_trained_scorer(str(scorer), base)
    scores = score_texts(sm, stok, [t for t, _ in pool], batch_size=batch_size)

    order = np.argsort(-scores)
    kept, used = [], {}
    for i in order:
        t, a = pool[i]
        if used.get(a, 0) >= cap:
            continue
        used[a] = used.get(a, 0) + 1
        kept.append({"text_clean": t, "score": float(scores[i])})
        if len(kept) >= keep:
            break
    df = pl.DataFrame(kept)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    s = np.array([r["score"] for r in kept])
    print(f"[write] {out}: {len(kept):,} curated goods  taste score p10={np.percentile(s,10):+.2f} "
          f"p50={np.percentile(s,50):+.2f} p90={np.percentile(s,90):+.2f}  (threshold {s.min():+.2f})")
    print("[top curated]")
    for r in kept[:6]:
        print(f"  {r['score']:+6.2f}  {r['text_clean'][:96]}")
    print("[done]")


if __name__ == "__main__":
    app()
