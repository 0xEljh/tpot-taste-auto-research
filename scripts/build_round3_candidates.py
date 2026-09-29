"""Round-3 active-labeling candidate selection (doc 05 §8).

Picks unlabeled tweets for human verification using the label-limited finding's own tools:
  - boundary  : the fast embed+Ridge probe (trained on all 137 labels) predicts a continuous
                taste score; items landing in the ambiguous mid-band [0.25,0.75] (between not /
                borderline / tpot) are where a fresh label resolves the most.
  - disagree  : items where embed+Ridge and the shipped v7.2 (3B-BT) scorer most disagree in
                rank — query-by-committee.
  - random    : uniform control arm ("Random Is Hard to Beat" — only keep active if it wins).
Dedups vs the 140 already-labeled texts and drops near-duplicate candidates.

  uv run python scripts/build_round3_candidates.py
Writes data/splits/round3_candidates.parquet (id, text, ridge, v72, bucket) and prints the batch.
"""
from __future__ import annotations

import re
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

_NONLATIN = re.compile(r"[぀-ヿ㐀-鿿가-힯؀-ۿ฀-๿Ѐ-ӿ]")


def _english(t: str) -> bool:
    return not _NONLATIN.search(t) and sum(c.isalpha() and ord(c) < 128 for c in t) >= 20


def _norm(t: str) -> str:
    return " ".join(t.split()).lower()


@app.command()
def main(
    pool_path: Path = Path("data/splits/train_tweets.parquet"),
    calib: Path = Path("data/splits/calibration_set.parquet"),
    labels: Path = Path("outputs/calibration_labels.json"),
    out: Path = Path("data/splits/round3_candidates.parquet"),
    scorer: str = "outputs/scorer/qwen3b-bt-taste-LOCKED",
    pool_size: int = 3000,
    n_boundary: int = 20,
    n_disagree: int = 10,
    n_random: int = 10,
    id_start: int = 141,
    seed: int = 0,
) -> None:
    import random

    import numpy as np
    import polars as pl
    from sklearn.linear_model import Ridge

    from tpot_taste.data.embed import embed_texts, load_embedder
    from tpot_taste.scoring.bakeoff import load_labels
    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    lab = load_labels(labels)
    crows = {r["id"]: r["text"] for r in pl.read_parquet(calib).to_dicts()}
    lab_texts = [crows[i] for i in lab if i in crows]
    lab_y = np.array([lab[i] for i in lab if i in crows])
    seen = {_norm(t) for t in crows.values()}
    print(f"[labels] {len(lab_y)} labeled texts for the probe; {len(seen)} normalized seen-texts to exclude")

    df = pl.read_parquet(pool_path)
    df = df.filter((pl.col("text_clean_len") >= 40) & (pl.col("text_clean_len") <= 240)
                   & (~pl.col("is_reply")) & (pl.col("z") >= 1.0))
    pool = [r["text_clean"] for r in df.iter_rows(named=True)
            if _english(r["text_clean"]) and _norm(r["text_clean"]) not in seen]
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[:pool_size]
    print(f"[pool] {len(pool)} fresh English z>=1.0 goods (deduped vs 1-140)")

    model, tok = load_embedder()
    X_lab = embed_texts(model, tok, lab_texts)
    X_pool = embed_texts(model, tok, pool)
    ridge = Ridge(alpha=10.0).fit(X_lab, lab_y)
    r_score = ridge.predict(X_pool)

    from tpot_taste.engine import resolve_adapter_base

    scorer_base = resolve_adapter_base(scorer, "Qwen/Qwen2.5-3B-Instruct")
    sm, stok = load_trained_scorer(scorer, scorer_base)
    v_score = np.asarray(score_texts(sm, stok, pool, batch_size=16))

    def rank01(a):
        return a.argsort().argsort() / (len(a) - 1)

    disagree = np.abs(rank01(r_score) - rank01(v_score))

    picked: list[int] = []
    taken: set[int] = set()

    # boundary: ambiguous mid-band, spread evenly across it for diversity
    band = [i for i in range(len(pool)) if 0.25 <= r_score[i] <= 0.75]
    band.sort(key=lambda i: r_score[i])
    if band:
        step = max(1, len(band) // n_boundary)
        for i in band[::step][:n_boundary]:
            if i not in taken:
                picked.append(i); taken.add(i)

    # disagree: largest committee disagreement, not already taken
    for i in np.argsort(-disagree):
        if len(picked) >= n_boundary + n_disagree:
            break
        if i not in taken:
            picked.append(int(i)); taken.add(int(i))

    # random control
    pool_idx = [i for i in range(len(pool)) if i not in taken]
    rng.shuffle(pool_idx)
    rand_sel = pool_idx[:n_random]

    bucket = {}
    for i in picked[:n_boundary]:
        bucket[i] = "boundary"
    for i in picked[n_boundary:]:
        bucket[i] = "disagree"
    for i in rand_sel:
        bucket[i] = "random"

    sel = picked + rand_sel
    # near-duplicate removal among the selected (cosine on L2-normalized embeds)
    keep: list[int] = []
    for i in sel:
        if all(float(X_pool[i] @ X_pool[j]) < 0.92 for j in keep):
            keep.append(i)
    rows = [{"id": id_start + k, "text": pool[i], "ridge": round(float(r_score[i]), 3),
             "v72": round(float(v_score[i]), 2), "bucket": bucket[i]} for k, i in enumerate(keep)]
    pl.DataFrame(rows).write_parquet(out)
    print(f"[write] {out}: {len(rows)} candidates "
          f"({sum(b=='boundary' for b in (r['bucket'] for r in rows))} boundary / "
          f"{sum(r['bucket']=='disagree' for r in rows)} disagree / "
          f"{sum(r['bucket']=='random' for r in rows)} random)\n")
    for r in rows:
        print(f"#{r['id']} [{r['bucket']:8}] ridge={r['ridge']:+.2f} v72={r['v72']:+.2f} | {r['text'][:120]}")


if __name__ == "__main__":
    app()
