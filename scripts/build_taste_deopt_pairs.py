"""Judge-ANCHORED deopt pairs for Scorer v7.1 (Phase 6c, D32) — the fix for v7's register confound.

v7 failed: distilling UNRELATED judge-high vs judge-low real tweets taught register, not taste (D32).
v6 generalized because chosen/rejected were the SAME tweet degraded. So: combine both strengths —
  anchor = a JUDGE-HIGH real tweet (clean tpot; fixes D28's polluted engagement anchor), and
  rejected = that anchor DEGRADED to generic/platitude at matched length/register (same-content contrast
             = learnable like v6; register-matched = kills the v7 confound).
Reads the cached judge-scored pool (no re-judging). Combines with the judge-taste pairs for semantic breadth.

  uv run python scripts/build_taste_deopt_pairs.py
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    splits_dir: Path = Path("data/splits"),
    hi: float = 7.0,
    n_styles: int = 3,
    batch_size: int = 64,
    max_new_tokens: int = 90,
    seed: int = 0,
) -> None:
    import polars as pl
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from tpot_taste.data.degrade import degrade_texts

    def anchors(pool_file: str) -> list[str]:
        df = pl.read_parquet(splits_dir / pool_file)
        return [r["text"] for r in df.iter_rows(named=True)
                if r["score"] is not None and r["score"] >= hi]

    a_tr = anchors("taste_v7_pool_scored.parquet")
    a_te = anchors("taste_v7_pool_test_scored.parquet")
    print(f"[anchors] train={len(a_tr):,} test={len(a_te):,} (judge>={hi:.0f})")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16).eval()

    def build(anchor_list, tag, n_styles_):
        # repeat anchors n_styles times -> each copy gets a random degradation style
        texts = anchor_list * n_styles_
        degs = degrade_texts(model, tok, texts, batch_size=batch_size,
                             max_new_tokens=max_new_tokens, seed=seed)
        rows = []
        for good, (bad, style) in zip(texts, degs):
            if len(bad) < 15 or bad.lower() == good.lower():
                continue
            rows.append({"text_clean_w": good, "text_clean_l": bad, "style": style})
        df = pl.DataFrame(rows)
        df.write_parquet(splits_dir / f"taste_deopt_{tag}_pairs.parquet")
        import numpy as np
        lw = np.mean([len(r["text_clean_w"]) for r in rows])
        ll = np.mean([len(r["text_clean_l"]) for r in rows])
        print(f"[write] taste_deopt_{tag}_pairs: {len(rows):,} pairs  "
              f"len chosen={lw:.0f} rejected={ll:.0f} (Δ={lw-ll:+.0f})")
        return df

    tr = build(a_tr, "train", n_styles)
    build(a_te, "test", max(1, n_styles - 1))

    # combine deopt (register-matched, learnable) + judge-taste (semantic breadth) -> v7.1 train set
    jt = splits_dir / "taste_v7_train_pairs.parquet"
    if jt.exists():
        jtdf = pl.read_parquet(jt).select(["text_clean_w", "text_clean_l"])
        combined = pl.concat([tr.select(["text_clean_w", "text_clean_l"]), jtdf]).sample(fraction=1.0, shuffle=True, seed=seed)
        combined.write_parquet(splits_dir / "taste_v71_train_pairs.parquet")
        print(f"[write] taste_v71_train_pairs (deopt {tr.height:,} + judge-taste {jtdf.height:,}) = {combined.height:,}")
    print("\n[sample]")
    for r in tr.head(4).iter_rows(named=True):
        print(f"  CHOSEN: {r['text_clean_w'][:84]}")
        print(f"  DEGRAD: {r['text_clean_l'][:84]}  [{r['style'][:30]}...]\n")
    print("[done]")


if __name__ == "__main__":
    app()
