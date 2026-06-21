# tpot-taste — benchmark report

Final scorecard for the two-model system. Numbers are from this repo's eval scripts (cited per row);
full rationale in `docs/design/00-decision-log.md` (D1–D23). Both models are Qwen2.5-3B-Instruct + 4-bit
QLoRA adapters on one 12 GB RTX 3080 Ti.

| Capability | Deliverable | Model |
|---|---|---|
| Judge if a post is worth posting | #2 | **Scorer** = Qwen2.5-3B + BT reward head (**v7.1**) |
| Recommend post ideas | #1 | **Writer** = Qwen2.5-3B SFT→DPO (v2) |
| Suggest improvements | #3 | Writer (best-of-N) + Scorer (ranks) |

## Phase 6 (2026-06-21): the platitude fix — Scorer v6 → v7.1

The first pass shipped Scorer v6, but it had a systematic blind spot: it **over-rewarded generic-viral
platitudes** (the root of GRPO's drift, D27). Phase 6 traced this to the label source — goods were selected by
*engagement*, and **taste ⟂ engagement** (D13), so engagement-selection is *adversarial* to taste (it imports
platitudes/promo/news virality). The fix, validated qualitatively at every step (decision log D28–D35):

1. **Demonstrative dipstick** (`scripts/demo_eval.py`, a fixed archetype panel + writer panel, pushed to Notion)
   measured the flaw: on clean archetypes v6 ranked **generic_viral its TOP category** (+2.66 > tpot +1.68).
2. **A taste judge** (`tpot_taste/scoring/judge.py`) — base Qwen + a tpot-taste rubric, scoring taste not
   engagement — **inverts** that (tpot 7.8 ≫ platitude 2.8) and, on 236 *real* goods, drops v6's pollution
   (crypto/promo/news/a hex hash) and rescues v6's buried tpot (Factorio/DeepSeek, cursor-for-excel). The base
   model already *had* taste; it just had to be elicited, not trained (D30/D31).
3. **Distillation:** judge-label → train a fast BT scorer. v7 (unrelated judge-taste pairs) **FAILED** — learned
   register not taste (held-out chance, D32). **v7.1 = judge-anchored deopt** (judge picks clean tpot anchors →
   degrade to generic at matched register) **WORKS** (D33):

| panel category | v6 | **v7.1** | held-out taste acc | v6→v7.1 |
|---|---|---|---|---|
| tpot_canon | +1.68 | **+3.08** | (unrelated real judge-pairs) | |
| aphorism | +1.59 | +2.22 | v6/v7: chance | |
| **generic_viral** | **+2.66 (top)** | **+1.27** | **v7.1: 0.616** | platitudes now below tpot |
| promo | +0.91 | **−2.51** | | leak fixed |

HIGH-vs-LOW separation **+0.65 → +2.59 (~4×)**; OOD-non-English over-scoring fixed. Residual: corporate-listicle
still mildly over-rated (v7.2 lever). **Scorer LOCKED v6 → v7.1** (`qwen3b-bt-v71-judge-deopt`).

4. **Writer propagation (D34/D35)** — re-curated goods via v7.1 → re-SFT → re-DPO vs v7.1. The independent
   judge-arbitrated A/B (new vs old writer, *not* scored by v7.1) was a **TIE (0.43 of 7 decided)**. The lesson:
   **the platitude problem was a *scorer* problem, not a *writer* one** — fixing the scorer fixed the system via
   best-of-N; the writer was already adequate. **Writer stays DPO-v2.**

> The v6-era scorecard below is retained as history; the locked scorer is now v7.1.

## Taste Scorer (v6, historical) — superseded by v7.1

`scripts/eval_scorer.py`, `scripts/inspect_scorer.py`.

- **Synthetic good-vs-deoptimized pairwise acc ≈ 0.72** (held-out). Trained on real tpot hits vs LLM-degraded
  versions of the same post (the "taste" reframe, D13–D17), because…
- **Within-author engagement-lift prediction has a ~0.61 ceiling** — on topic-matched same-author pairs it is
  ~chance. *Key finding:* which of two same-author posts gets more engagement is mostly luck/timing, not
  text-extractable. So we model **taste**, not virality; real-engagement pairwise stays ~0.53 (taste ⟂ engagement).
- **Direction probes (the qualitative anchor):** engagement-bait **lowest**, corporate-cringe & platitude negative,
  funny-punchy on top. **Length-confound** pearson(score, chars) = **+0.13** (mild; v4 +0.23 and v5 −0.04 were
  rejected for trading bait-penalization against length — D16).
- **Known limits:** can't finely separate a sharp aphorism from a bland platitude (near the noise ceiling); can be
  fooled by substantive content carrying an assistant preamble; OOD on non-English text.

## Taste Writer — base → SFT → DPO v2 (`qwen3b-writer-LOCKED`)

`scripts/eval_writer.py` (A/B, judged by the locked v6 scorer, n=60, with a length-drift audit).

| Comparison | win-rate (v6) | mean score | length (chars) |
|---|---|---|---|
| SFT v1 vs **base** | **0.67** | +0.33 vs −0.64 | — |
| DPO v1 vs SFT | 0.67 | +1.27 vs +0.74 | **175 vs 121 — length-hack** |
| **DPO v2 vs SFT** ← shipped | **0.75** | +1.51 vs +0.48 | **112 vs 123 — clean** |
| GRPO v1 vs DPO v2 | 0.52 (tie) | +1.44 vs +1.31 | 113 vs 118 |
| GRPO v2 vs DPO v2 | 0.65* | +1.78 vs +1.11 | 118 vs 126 |

\* GRPO v2's v6-win is a **platitude-drift** — pushing RL hard on v6 exploits its aphorism≈platitude blind spot
(elaborated motivational advice scores high). A human read prefers DPO v2, so **GRPO is not adopted**; it marks the
practical **v6 ceiling**. The length/bait/repetition guardrails held throughout (no easy hacks).

- **SFT** (full-text on 19,970 ideate+improve records; deopt pairs reused as free bad→good supervision) produced a
  decisive register shift: base outputs are assistant-slop ("Absolutely! Here's a post for TPOT: ---", emojis,
  hashtags) the scorer scores −5..−6; the SFT Writer writes lowercase, terse, tpot-voiced (D19).
- **DPO** (best-of-N judged by v6): **v1 reward-hacked length** (+45% chars by gaming the scorer's mild length
  lean) — *caught by the length audit, not the win-rate.* **v2** neutralized length in best-of-N selection
  (`adj = score − λ·len`, λ=0.005) → hack gone **and** win-rate rose 0.67→0.75 (removing the shortcut forced
  genuine-quality learning). D21/D22.

## Reward-hacking audit

The recurring lesson (the whole project): **the reward number lies; ground it qualitatively.**
- Scorer v5 got a *perfect* length number but ranked corporate-cringe/bait highest → rejected (D16).
- DPO v1's 0.67 win-rate hid a 45% length inflation → caught by the audit → fixed in v2 (D21/D22).
- Residual: DPO faithfully optimizes v6, so it inherits v6's blind spots (occasional platitude-with-emoji). The
  RL reward's explicit length/bait penalties (design §3) are the next guard, for GRPO (Phase 4b).

## Use it

```
nix develop
uv run python scripts/tpot.py score   "is this worth posting?"
uv run python scripts/tpot.py ideate  --topic "recursion"
uv run python scripts/tpot.py improve "lit a fake cig"     # best-of-N, scorer-ranked
uv run python scripts/tpot.py repl                          # loads once, interactive
```
Smoke example — `improve "lit a fake cig"` (draft −0.17) → ranked improvements +1.88 / +1.53 / +1.10.

## Reproduce

`uv sync --extra train`, then: `eval_scorer.py --adapter <scorer> --pairs …`; `inspect_scorer.py`;
`eval_writer.py --a <writer> --b <ref>`. Frozen splits live in `data/splits/`; W&B project `tpot-taste`.

## Next

The v6 ceiling (D27) was broken by the better Scorer it called for: **v7.1** (judge-anchored, D33) fixes the
platitude blind spot. The frontier is now elsewhere:

1. **Human-calibrate the judge** (the one thing we can't bootstrap) — the judge's taste = base-model prior ×
   *our* rubric, graded on *our* archetypes. A few hundred human labels (on the judge/v6 disagreements) would
   turn "eyeballed correct" into a *measured* judge accuracy and tune the rubric to the real target.
2. **v7.2 — the corporate-listicle residual** (the one panel category v7.1 still mildly over-rates): up-weight
   corporate/listicle degradations.
3. **Taste pluralism** — is "tpot taste" one axis or several (rationalist / post-rat / e-acc / builder /
   shitposter)? Possibly a conditional scorer.
4. **GRPO redux vs v7.1** — re-running the RL push against a scorer that no longer rewards platitudes (the D27
   drift was v6's fault) may now yield a genuine gain, not metric-gaming.
5. **Adopt `dpo-v2sft`?** — the clean-data-foundation writer (tie on evidence; cleaner on principle).
