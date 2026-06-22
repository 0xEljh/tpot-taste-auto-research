"""Build direct human-pair BT training data + the shared A-vs-B held-out split (Phase 6f, Approach B).

Turns the human's 137 calibration labels into Bradley-Terry pairs (no judge in the loop). Also writes the
FIXED, STRATIFIED held-out id list that BOTH Approach B (this) and Approach A (fewshot_distill_test.py
--heldout) are evaluated on — so the head-to-head (which scorer aligns with the human better?) is fair: neither
trains on the held-out.

  uv run python scripts/build_human_pairs.py                      # all 3 contrasts, no length restriction
  uv run python scripts/build_human_pairs.py --pair-types hi_lo   # only the strong (tpot vs not) contrast
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
_MAP = {"👍 tpot": 1.0, "🤔 borderline": 0.5, "👎 not tpot": 0.0}


@app.command()
def main(
    calib: Path = Path("data/splits/calibration_set.parquet"),
    labels: Path = Path("outputs/calibration_labels.json"),
    out_pairs: Path = Path("data/splits/taste_human_train_pairs.parquet"),
    out_heldout: Path = Path("outputs/align_heldout_ids.json"),
    n_heldout: int = 46,
    pair_types: str = "hi_lo,hi_mid,mid_lo",
    length_tol: int = 0,        # 0 => no length restriction (labels are already length-balanced)
    reuse_cap: int = 0,         # 0 => unbounded reuse (all-pairs)
    seed: int = 0,
) -> None:
    import numpy as np
    import polars as pl

    from tpot_taste.data.human_pairs import build_pairs, stratified_heldout

    lab = {int(k): _MAP[v] for k, v in json.loads(labels.read_text()).items() if v in _MAP}
    rows = [r for r in pl.read_parquet(calib).to_dicts() if r["id"] in lab]
    items = [(r["id"], lab[r["id"]]) for r in rows]

    held, train = stratified_heldout(items, n_heldout=n_heldout, seed=seed)
    out_heldout.parent.mkdir(parents=True, exist_ok=True)
    out_heldout.write_text(json.dumps(sorted(held)))
    hc = {c: sum(1 for i in held if lab[i] == c) for c in (1.0, 0.5, 0.0)}
    tc = {c: sum(1 for i in train if lab[i] == c) for c in (1.0, 0.5, 0.0)}
    print(f"[split] held-out {len(held)} (tpot {hc[1.0]} / bord {hc[0.5]} / not {hc[0.0]})  "
          f"train {len(train)} (tpot {tc[1.0]} / bord {tc[0.5]} / not {tc[0.0]})")
    print(f"[split] -> {out_heldout}")

    train_items = [{"id": r["id"], "text": r["text"], "label": lab[r["id"]]} for r in rows if r["id"] in train]
    pairs = build_pairs(
        train_items,
        pair_types=tuple(pair_types.split(",")),
        length_tol=length_tol or None,
        reuse_cap=reuse_cap or None,
        seed=seed,
    )
    # rename to the scorer trainer's expected schema (text_clean_w / text_clean_l already match)
    df = pl.DataFrame(pairs)
    df.write_parquet(out_pairs)
    lw = np.mean([p["len_w"] for p in pairs])
    ll = np.mean([p["len_l"] for p in pairs])
    by_type: dict = {}
    for p in pairs:
        by_type[(p["label_w"], p["label_l"])] = by_type.get((p["label_w"], p["label_l"]), 0) + 1
    print(f"[write] {out_pairs}: {len(pairs):,} pairs  "
          f"len chosen={lw:.0f} rejected={ll:.0f} (Δ={lw - ll:+.0f}, want ~0)")
    print(f"[contrasts] " + "  ".join(f"{k[0]:.1f}>{k[1]:.1f}: {v}" for k, v in sorted(by_type.items())))
    print(f"\nNext: train_scorer.py --train-pairs {out_pairs} --out outputs/scorer/qwen3b-bt-v72human --epochs 2")


if __name__ == "__main__":
    app()
