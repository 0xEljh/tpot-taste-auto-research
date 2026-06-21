"""Mine judge-labeled TASTE pairs for Scorer v7 (Phase 6c, D31) — no engagement in the label.

Distillation: the generative judge (validated D31) labels a diverse pool by tpot-taste; we then form
preference pairs (chosen=judge-high, rejected=judge-low) to train the FAST BT scorer v7. v7 inherits the
judge's taste at scalar-head speed (for corpus-scale scoring + the RL reward).

Confound guards (the hard-won lessons from v3–v6):
  - LENGTH-STRATIFIED pairing — chosen & rejected drawn from the SAME length bucket, so length carries no
    signal (v4's bias was short=bad). Verified post-build (mean len chosen ≈ rejected).
  - per-author caps on BOTH sides (~334 voices; don't let eigenrobot dominate).
  - margin gate (judge_hi - judge_lo >= margin); the ambiguous middle (4–6) is dropped.
  - diverse pool: high-z goods (de-pollute the current distribution) + broad uploader sample (recover the
    deadpan tpot engagement buried) + liked. English-only (drop OOD the judge mis-scores).

  uv run python scripts/build_taste_pairs.py            # ~4k pool, judged + paired
  uv run python scripts/build_taste_pairs.py --reuse    # re-pair from the saved scored pool (no re-judge)
"""
from __future__ import annotations

import random
import re
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

_NONLATIN = re.compile(r"[぀-ヿ㐀-鿿가-힯؀-ۿ฀-๿]")
_BUCKETS = [(40, 80), (80, 130), (130, 190), (190, 240)]


def _english(t: str) -> bool:
    return not _NONLATIN.search(t) and sum(c.isalpha() and ord(c) < 128 for c in t) >= 20


def _sample_pool(splits_dir: Path, liked_path: Path, tweets_file: str, cut_before: bool,
                 n_hi: int, n_rand: int, n_liked: int, seed: int):
    """Return list of (text, length, account_id). Mix high-z goods + broad uploader + liked."""
    import polars as pl

    rng = random.Random(seed)
    up = pl.read_parquet(splits_dir / tweets_file)
    up = up.filter((pl.col("text_clean_len") >= 40) & (pl.col("text_clean_len") <= 240) & (~pl.col("is_reply")))
    rows = [(r["text_clean"], r["text_clean_len"], r["account_id"], r["z"])
            for r in up.iter_rows(named=True) if _english(r["text_clean"])]
    rows.sort(key=lambda x: -x[3])  # by z desc
    hi = rows[:n_hi]                                   # engagement-goods (judge sorts tpot from pollution)
    rest = rows[n_hi:]
    rng.shuffle(rest)
    rand = rest[:n_rand]                               # broad sample (recover buried tpot + clear negatives)

    liked = pl.read_parquet(liked_path)
    liked = liked.filter(pl.col("created_dt") < datetime.fromisoformat("2025-06-01")) if cut_before \
        else liked.filter(pl.col("created_dt") >= datetime.fromisoformat("2025-06-01"))
    liked = liked.filter((pl.col("text_clean_len") >= 40) & (pl.col("text_clean_len") <= 240)
                         & (~pl.col("is_reply")) & (pl.col("favorite_count") >= 20))
    lrows = [(r["text_clean"], r["text_clean_len"], r["account_id"])
             for r in liked.iter_rows(named=True) if _english(r["text_clean"])]
    rng.shuffle(lrows)
    pool = [(t, l, a) for t, l, a, _ in hi + rand] + lrows[:n_liked]
    # dedup by normalized text
    seen, out = set(), []
    for t, l, a in pool:
        k = " ".join(t.split()).lower()
        if k not in seen:
            seen.add(k)
            out.append((t, l, a))
    return out


def _pairs(scored, hi: float, lo: float, cap: int, max_pairs: int, seed: int):
    """Length-stratified, author-capped, margin-gated pairing. scored: list of dict(text,len,acct,score)."""
    rng = random.Random(seed)
    used: dict = {}
    pairs = []
    for blo, bhi in _BUCKETS:
        hs = [r for r in scored if r["score"] is not None and r["score"] >= hi and blo <= r["len"] < bhi]
        ls = [r for r in scored if r["score"] is not None and r["score"] <= lo and blo <= r["len"] < bhi]
        rng.shuffle(hs)
        rng.shuffle(ls)
        li = 0
        for c in hs:
            if used.get(c["acct"], 0) >= cap:
                continue
            # next rejected whose author isn't over cap
            while li < len(ls) and used.get(ls[li]["acct"], 0) >= cap:
                li += 1
            if li >= len(ls):
                break
            r = ls[li]
            li += 1
            used[c["acct"]] = used.get(c["acct"], 0) + 1
            used[r["acct"]] = used.get(r["acct"], 0) + 1
            pairs.append({"text_clean_w": c["text"], "text_clean_l": r["text"],
                          "score_w": c["score"], "score_l": r["score"], "len_w": c["len"], "len_l": r["len"]})
    rng.shuffle(pairs)
    return pairs[:max_pairs]


@app.command()
def main(
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    splits_dir: Path = Path("data/splits"),
    liked_path: Path = Path("data/processed/liked_taste.parquet"),
    n_hi: int = 1800,
    n_rand: int = 1400,
    n_liked: int = 800,
    n_hi_test: int = 300,
    n_rand_test: int = 200,
    n_liked_test: int = 100,
    hi: float = 7.0,
    lo: float = 3.0,
    cap: int = 18,
    max_pairs: int = 4000,
    max_pairs_test: int = 600,
    batch_size: int = 16,
    seed: int = 0,
    reuse: bool = False,
) -> None:
    import polars as pl

    pool_path = splits_dir / "taste_v7_pool_scored.parquet"
    pool_test_path = splits_dir / "taste_v7_pool_test_scored.parquet"

    if reuse and pool_path.exists():
        print("[reuse] loading saved scored pools")
        scored = pl.read_parquet(pool_path).to_dicts()
        scored_te = pl.read_parquet(pool_test_path).to_dicts()
    else:
        train_pool = _sample_pool(splits_dir, liked_path, "train_tweets.parquet", True,
                                  n_hi, n_rand, n_liked, seed)
        test_pool = _sample_pool(splits_dir, liked_path, "test_temporal_tweets.parquet", False,
                                 n_hi_test, n_rand_test, n_liked_test, seed + 1)
        # dedup test against train
        tr_keys = {" ".join(t.split()).lower() for t, _, _ in train_pool}
        test_pool = [(t, l, a) for t, l, a in test_pool if " ".join(t.split()).lower() not in tr_keys]
        print(f"[pool] train={len(train_pool):,} test={len(test_pool):,} — judging (batched) ...")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from tpot_taste.scoring.judge import judge_score_batch

        tok = AutoTokenizer.from_pretrained(base)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        jm = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16).eval()

        def judge_pool(pool, label):
            texts = [t for t, _, _ in pool]
            scores = judge_score_batch(jm, tok, texts, batch_size=batch_size)
            print(f"  [{label}] judged {len(texts)}  parse-miss={sum(s is None for s in scores)}")
            return [{"text": t, "len": l, "acct": a, "score": s}
                    for (t, l, a), s in zip(pool, scores)]

        scored = judge_pool(train_pool, "train")
        scored_te = judge_pool(test_pool, "test")
        pl.DataFrame(scored).write_parquet(pool_path)
        pl.DataFrame(scored_te).write_parquet(pool_test_path)
        print(f"[save] scored pools -> {pool_path.name}, {pool_test_path.name}")

    # score distribution
    import numpy as np
    sc = np.array([r["score"] for r in scored if r["score"] is not None])
    print(f"[judge dist] train pool n={len(sc)}  "
          f"hi(>={hi:.0f})={(sc>=hi).mean()*100:.0f}%  lo(<={lo:.0f})={(sc<=lo).mean()*100:.0f}%  "
          f"mid={((sc>lo)&(sc<hi)).mean()*100:.0f}%  mean={sc.mean():.2f}")

    tr_pairs = _pairs(scored, hi, lo, cap, max_pairs, seed)
    te_pairs = _pairs(scored_te, hi, lo, cap, max_pairs_test, seed + 1)
    for name, pairs in [("taste_v7_train_pairs", tr_pairs), ("taste_v7_test_pairs", te_pairs)]:
        df = pl.DataFrame(pairs)
        df.write_parquet(splits_dir / f"{name}.parquet")
        lw, ll = np.mean([p["len_w"] for p in pairs]), np.mean([p["len_l"] for p in pairs])
        print(f"[write] {name}: {len(pairs):,} pairs  len chosen={lw:.0f} rejected={ll:.0f} (Δ={lw-ll:+.0f})")
    print("\n[sample pairs]")
    for p in tr_pairs[:4]:
        print(f"  CHOSEN ({p['score_w']:.0f}): {p['text_clean_w'][:88]}")
        print(f"  REJECT ({p['score_l']:.0f}): {p['text_clean_l'][:88]}\n")
    print("[done]")


if __name__ == "__main__":
    app()
