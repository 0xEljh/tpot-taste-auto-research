# tpot-taste — benchmark report

Final scorecard for the two-model system. Numbers are from this repo's eval scripts (cited per row);
full rationale in `docs/design/00-decision-log.md` (D1–D23). Both models are Qwen2.5-3B-Instruct + 4-bit
QLoRA adapters on one 12 GB RTX 3080 Ti.

| Capability | Deliverable | Model |
|---|---|---|
| Judge if a post is worth posting | #2 | **Scorer** = Qwen2.5-3B + BT reward head (v6) |
| Recommend post ideas | #1 | **Writer** = Qwen2.5-3B SFT→DPO (v2) |
| Suggest improvements | #3 | Writer (best-of-N) + Scorer (ranks) |

## Taste Scorer (v6) — `qwen3b-bt-taste-LOCKED`

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

GRPO (Phase 4b) is **done** — it confirmed the v6 ceiling (D27): a stronger push raised the v6 number (0.65) but
by drifting to platitudes v6 can't distinguish from insight, so DPO v2 ships. The single highest-leverage next
step is therefore **a better Scorer** — human-labeled taste pairs and/or a calibrated LLM-judge ensemble — which
would lift the ceiling on the scorer (deliverable #2), the DPO/GRPO reward, *and* best-of-N at inference. Only then
is more RL worthwhile. (Unsloth + vLLM would also make a 7B Writer or many-step GRPO feasible, if desired — D26.)
