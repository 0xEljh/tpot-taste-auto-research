"""Re-curate the Writer's SFT goods with the v7.1 scorer (Phase 6d, D34) — distillation pays off.

The Writer was SFT'd on ENGAGEMENT-selected goods (platitude/promo pollution, D28). Now that v7.1 is a fast
scalar taste scorer, use it to re-curate at scale: score a broad pool (uploader across ALL z — to recover the
deadpan tpot engagement buried — + liked), keep the top by TASTE. Per-author cap keeps the ~334 voices diverse.

  uv run python scripts/curate_goods.py            # ~40k pool -> top 12k curated goods
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
_NONLATIN = re.compile(r"[぀-ヿ㐀-鿿가-힯؀-ۿ฀-๿]")


def _english(t: str) -> bool:
    return not _NONLATIN.search(t) and sum(c.isalpha() and ord(c) < 128 for c in t) >= 20


@app.command()
def main(
    out: Path = Path("data/splits/curated_goods.parquet"),
    splits_dir: Path = Path("data/splits"),
    liked_path: Path = Path("data/processed/liked_taste.parquet"),
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),  # = v7.1
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    n_up: int = 30000,
    n_liked: int = 12000,
    keep: int = 12000,
    cap: int = 60,
    batch_size: int = 48,
    seed: int = 0,
) -> None:
    import numpy as np
    import polars as pl

    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    def sample_rows(df, n):  # sample in polars BEFORE the python/regex pass (filtered sets are large)
        df = df.sample(min(int(n * 1.3), df.height), seed=seed)
        return [(r["text_clean"], r["account_id"]) for r in df.iter_rows(named=True) if _english(r["text_clean"])][:n]

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
    print(f"[write] {out}: {len(kept):,} curated goods  v7.1 score p10={np.percentile(s,10):+.2f} "
          f"p50={np.percentile(s,50):+.2f} p90={np.percentile(s,90):+.2f}  (threshold {s.min():+.2f})")
    print("[top curated]")
    for r in kept[:6]:
        print(f"  {r['score']:+6.2f}  {r['text_clean'][:96]}")
    print("[done]")


if __name__ == "__main__":
    app()
