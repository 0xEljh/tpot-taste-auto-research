"""Audit the curated 'good' set for platitude / generic-virality pollution (Phase 6, D28).

The goods are selected purely by ENGAGEMENT (uploader within-author z>=z_min, liked fav>=min_fav).
But taste ⟂ engagement (~0.53 pairwise, D13), so engagement-selection systematically over-includes
generic-viral content (platitudes, motivational, relatable hot-takes) that works on broad Twitter but
is NOT tpot taste. Those goods become the scorer's `chosen` AND (via build_writer_sft) the Writer's
ideate targets — so the rot is in the soil, and GRPO just surfaced it (D27).

This reconstructs the goods exactly as build_deopt_pairs._good does, scores them with the locked v6
scorer, and tests the smoking-gun: do platitude-marked goods score HIGHER on v6?

  uv run python scripts/audit_goods.py            # full train goods
  uv run python scripts/audit_goods.py --n-up 2000 --n-liked 1000   # faster sample
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# --- pollution markers (crude first-pass heuristics; the eyeball of top-scoring goods is the real test) ---
# Generic-motivational / advice register: imperative 2nd-person advice + abstract self-help nouns.
_PLATITUDE = re.compile(
    r"(?i)\b("
    r"you (should|must|need to|have to|gotta|deserve|owe it)|"
    r"the (key|secret|trick|answer) to|the most important thing|at the end of the day|"
    r"surround yourself|never give up|don'?t give up|trust the process|comfort zone|"
    r"believe in yourself|best version of (your|you)|your (full )?potential|"
    r"work hard|hard work pays|stay (focused|consistent|hungry|humble)|"
    r"the only way to|if you want to (succeed|win|grow)|"
    r"discipline (beats|over)|consistency is|growth mindset|level up your"
    r")\b"
)
# Abstract self-help noun density (a softer signal than the phrase list)
_SELFHELP_NOUN = re.compile(
    r"(?i)\b(success|discipline|motivation|mindset|hustle|grind|greatness|potential|"
    r"journey|purpose|passion|consistency|habits?|growth|productivity|optimi[sz]e)\b"
)


def selfhelp_density(t: str) -> int:
    return len(_SELFHELP_NOUN.findall(t))


def is_platitude(t: str) -> bool:
    return bool(_PLATITUDE.search(t)) or selfhelp_density(t) >= 2


def _quality_english(t: str) -> bool:
    _NONLATIN = re.compile(r"[぀-ヿ㐀-鿿가-힯؀-ۿ฀-๿]")
    if _NONLATIN.search(t):
        return False
    return sum(1 for c in t if c.isalpha() and ord(c) < 128) >= 20


def _good(df, by: str, min_val, n: int, cap: int, lo: int, hi: int) -> list[str]:
    import polars as pl

    d = df.filter((pl.col("text_clean_len") >= lo) & (pl.col("text_clean_len") <= hi))
    if "is_reply" in d.columns:
        d = d.filter(~pl.col("is_reply"))
    d = d.filter(pl.col(by) >= min_val).sort(by, descending=True)
    d = d.group_by("account_id", maintain_order=True).head(cap).sort(by, descending=True)
    texts = [t for t in d["text_clean"].to_list() if _quality_english(t)]
    return texts[:n]


@app.command()
def main(
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    splits_dir: Path = Path("data/splits"),
    liked_path: Path = Path("data/processed/liked_taste.parquet"),
    cutoff: str = "2025-06-01",
    n_up: int = 7000,
    n_liked: int = 3000,
    z_min: float = 1.0,
    liked_min_fav: int = 50,
    cap: int = 60,
    lo: int = 40,
    hi: int = 240,
    top_k: int = 25,
) -> None:
    import numpy as np
    import polars as pl

    from tpot_taste.scoring.model import load_trained_scorer, score_texts
    from tpot_taste.writer.grpo_reward import is_baity

    cut = datetime.fromisoformat(cutoff)
    up = _good(pl.read_parquet(splits_dir / "train_tweets.parquet"), "z", z_min, n_up, cap, lo, hi)
    liked = pl.read_parquet(liked_path).filter(pl.col("created_dt") < cut)
    lk = _good(liked, "favorite_count", liked_min_fav, n_liked, cap, lo, hi)
    texts = up + lk
    src = ["uploader"] * len(up) + ["liked"] * len(lk)
    print(f"[load] goods: uploader={len(up):,} liked={len(lk):,} total={len(texts):,}")

    sm, stok = load_trained_scorer(str(scorer), base)
    print("[score] v6 scoring goods ...")
    scores = score_texts(sm, stok, texts, batch_size=16)

    plat = np.array([is_platitude(t) for t in texts])
    bait = np.array([is_baity(t) for t in texts])
    src = np.array(src)

    def q(a):
        return f"p10={np.percentile(a,10):+.2f} p50={np.percentile(a,50):+.2f} p90={np.percentile(a,90):+.2f} mean={a.mean():+.2f}"

    print("\n=== v6 score distribution (goods) ===")
    for s in ["uploader", "liked"]:
        m = src == s
        print(f"  {s:9s} n={m.sum():5d}  {q(scores[m])}")
    print(f"  {'ALL':9s} n={len(scores):5d}  {q(scores)}")

    print("\n=== marker prevalence among goods ===")
    print(f"  platitude-marked: {plat.mean()*100:4.1f}%  ({plat.sum():,})")
    print(f"  baity:            {bait.mean()*100:4.1f}%  ({bait.sum():,})")

    print("\n=== SMOKING GUN: mean v6 score by platitude marker ===")
    if plat.any() and (~plat).any():
        a, b = scores[plat].mean(), scores[~plat].mean()
        print(f"  platitude-marked goods: mean v6 = {a:+.3f}")
        print(f"  unmarked goods:         mean v6 = {b:+.3f}")
        print(f"  delta (marked - unmarked) = {a-b:+.3f}   "
              f"({'POLLUTION: platitudes score HIGHER' if a > b else 'ok: platitudes score lower'})")
        # share of the top quartile that is platitude-marked
        thr = np.percentile(scores, 75)
        topq = scores >= thr
        print(f"  platitude share of top-quartile goods (v6>={thr:+.2f}): "
              f"{plat[topq].mean()*100:.1f}%  (vs {plat.mean()*100:.1f}% overall)")

    order = np.argsort(-scores)
    print(f"\n=== top-{top_k} highest-v6 goods (eyeball for platitudes) ===")
    for i in order[:top_k]:
        flags = ("P" if plat[i] else " ") + ("B" if bait[i] else " ")
        print(f"  {scores[i]:+6.2f} [{flags}] ({src[i][:3]}) {texts[i][:110]}")

    pol = [i for i in order if plat[i]][:15]
    print(f"\n=== highest-scoring PLATITUDE-marked goods (the pollution to curate out) ===")
    for i in pol:
        print(f"  {scores[i]:+6.2f} ({src[i][:3]}) {texts[i][:120]}")
    print("\n[done]")


if __name__ == "__main__":
    app()
