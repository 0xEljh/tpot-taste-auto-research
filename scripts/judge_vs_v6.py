"""Validate the taste judge on REAL data by its disagreements with v6 (Phase 6c, D30).

The judge aced the hand-authored panel (D30) — but that's archetypes. The decisive test: on REAL goods,
where do the judge and v6 DISAGREE, and is the judge right? Those disagreements ARE the curation decisions:
  - v6-HIGH / judge-LOW  -> suspected pollution the judge would DROP (platitude/promo/generic-viral)
  - v6-LOW  / judge-HIGH -> suspected tpot the judge would RESCUE (the deadpan/oblique false-negatives)
Ranks are percentile-normalized (v6 is unbounded, judge is 0-10) so disagreement = |pct_v6 - pct_judge|.

  uv run python scripts/judge_vs_v6.py --n 240
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _sample_goods(splits_dir: Path, liked_path: Path, n: int, seed: int) -> list[str]:
    import random
    import re

    import polars as pl

    nonlatin = re.compile(r"[぀-ヿ㐀-鿿가-힯؀-ۿ฀-๿]")

    def good(df, by, mn):
        d = df.filter((pl.col("text_clean_len") >= 40) & (pl.col("text_clean_len") <= 240))
        if "is_reply" in d.columns:
            d = d.filter(~pl.col("is_reply"))
        d = d.filter(pl.col(by) >= mn).sort(by, descending=True)
        d = d.group_by("account_id", maintain_order=True).head(60)
        return [t for t in d["text_clean"].to_list()
                if not nonlatin.search(t) and sum(c.isalpha() and ord(c) < 128 for c in t) >= 20]

    up = good(pl.read_parquet(splits_dir / "train_tweets.parquet"), "z", 1.0)
    liked = pl.read_parquet(liked_path).filter(pl.col("created_dt") < datetime.fromisoformat("2025-06-01"))
    lk = good(liked, "favorite_count", 50)
    rng = random.Random(seed)
    rng.shuffle(up)
    rng.shuffle(lk)
    return up[: n // 2] + lk[: n // 2]


@app.command()
def main(
    out: Path = Path("outputs/judge_vs_v6.md"),
    splits_dir: Path = Path("data/splits"),
    liked_path: Path = Path("data/processed/liked_taste.parquet"),
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    n: int = 240,
    seed: int = 0,
    show: int = 12,
) -> None:
    import numpy as np

    from tpot_taste.scoring.judge import judge_score
    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    texts = _sample_goods(splits_dir, liked_path, n, seed)
    print(f"[sample] {len(texts)} real goods")

    # v6 (fast, batched)
    sm, stok = load_trained_scorer(str(scorer), base)
    v6 = np.asarray(score_texts(sm, stok, texts, batch_size=16))
    del sm
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()

    # judge (slow, sequential) — reuse the SAME base as a generative judge
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    jm = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16).eval()
    jr = []
    for i, t in enumerate(texts):
        jr.append(judge_score(jm, tok, t, max_new_tokens=80, temperature=0.0))
        if i % 40 == 0:
            print(f"  judge {i}/{len(texts)}")
    miss = sum(v is None for v in jr)
    judge = np.array([v if v is not None else np.nan for v in jr])

    ok = ~np.isnan(judge)

    def _spearman(a, b):  # rank-correlation = Pearson of ranks (no scipy dep)
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    rho = _spearman(v6[ok], judge[ok])
    # percentile ranks for scale-free disagreement
    def pct(a):
        order = a.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.linspace(0, 1, len(a))
        return r
    pv, pj = pct(v6[ok]), pct(judge[ok])
    diff = pv - pj  # >0: v6 ranks higher than judge (suspected pollution); <0: judge rescues
    tx = [t for t, o in zip(texts, ok) if o]
    v6o, jo = v6[ok], judge[ok]

    L = [f"# judge vs v6 on {len(tx)} real goods ({datetime.now():%Y-%m-%d %H:%M})",
         f"Spearman(v6, judge) = {rho:+.3f}  (low corr => they disagree a lot; judge parse-miss={miss})",
         "Disagreements ARE the curation decisions. Eyeball: is the judge right?", ""]
    order_drop = np.argsort(-diff)  # v6 high, judge low
    L += [f"## v6-HIGH / judge-LOW — suspected POLLUTION to drop (top {show})", "```"]
    for i in order_drop[:show]:
        L.append(f"  v6={v6o[i]:+5.2f} judge={jo[i]:4.1f}  {tx[i][:96]}")
    L += ["```", "", f"## v6-LOW / judge-HIGH — suspected tpot the judge RESCUES (top {show})", "```"]
    for i in np.argsort(diff)[:show]:
        L.append(f"  v6={v6o[i]:+5.2f} judge={jo[i]:4.1f}  {tx[i][:96]}")
    L += ["```", "", "## both agree HIGH (sanity)", "```"]
    agree_hi = np.argsort(-(pv + pj))
    for i in agree_hi[:6]:
        L.append(f"  v6={v6o[i]:+5.2f} judge={jo[i]:4.1f}  {tx[i][:96]}")
    L += ["```"]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"[write] {out}")
    print(f"[corr] Spearman(v6,judge)={rho:+.3f}  parse-miss={miss}/{len(texts)}")


if __name__ == "__main__":
    app()
