# Taste alignment — Phase 6 synthesis (the judge, v7.1, and the human-alignment ceiling)

A standalone reference for the Phase-6 arc (decision log D28–D39). The one-line story: **we fixed the *coarse*
taste problem (platitudes) and discovered the *fine* taste problem (matching a specific human) is label-gated.**

## 1. The root: engagement is adversarial to taste

The corpus "good" examples were selected by **engagement** (within-author z, favourites). But the project's own
finding is **taste ⟂ engagement** (~0.53 pairwise, near chance — D13). So engagement-selection doesn't just add
noise, it **imports the anti-pattern**: platitudes, promo, news-virality, crypto — content that genuinely racks up
engagement on broad Twitter but is *not* tpot. This polluted both the Scorer's targets and the Writer's SFT goods
(`build_writer_sft.py`), and GRPO's platitude-drift (D27) was that rot surfacing — not an RL artifact.

## 2. The fix that worked: a judge → distilled scorer v7.1

- **The taste judge** (`tpot_taste/scoring/judge.py`): the base Qwen prompted with a tpot rubric, scoring *taste*
  not *engagement*. On a fixed dipstick panel + 236 real goods it cleanly drops the pollution (crypto/promo/news,
  even a hex hash) and rescues buried tpot (Factorio/DeepSeek, "cursor for excel"). **The base model already had a
  coarse sense of taste — it had to be elicited, not trained.** (D30/D31)
- **Distillation** into a fast scalar scorer. v7 (unrelated judge-high-vs-low real-tweet pairs) **FAILED** —
  learned a register confound (real tpot is terser than real generic), not taste (D32). The fix — **judge-anchored
  deopt** (judge picks clean tpot anchors → degrade each into generic at *matched* length/register) — gives a
  learnable, register-controlled signal. **v7.1 works**: platitudes fell from v6's top panel category to below
  tpot; held-out taste acc 0.54→0.62. **Scorer LOCKED = v7.1.** (D33)
- **Writer propagation was a wash** (D34/D35): re-curating goods + re-SFT + re-DPO vs v7.1 tied the old DPO-v2
  writer (judge-arbitrated A/B 0.43). The lesson: **the platitude bug was a *scorer* problem, not a *writer* one** —
  fixing the scorer fixed the system through best-of-N. *Where you put the taste signal matters more than retraining
  the generator.* Writer stays DPO-v2.

## 3. The ceiling: the judge is only weakly aligned with the *human*

Calibration (D36): 60 human-labeled tweets, judge score hidden, in a Notion DB. Measured:
**Spearman(judge, human) = 0.13.** The "judge looks great" eyeball was measuring the wrong thing — v7.1 fixes the
*coarse* platitude distinction (which the base model agrees on) but not the human's *fine* taste.

What aligns it? (D37/D38)

| lever | held-out Spearman | verdict |
|---|---|---|
| rubric edits (v1→v2) | 0.06 | **no leverage** — the base prior is rigid to abstract rules |
| bigger judge (7B vs 3B) | 0.07 / 0.27 ≈ 3B | **size is not the ceiling** |
| **few-shot human exemplars** (12) | 0.06 → **0.23** | **the only lever — and it scales with labels** |

**The human's taste isn't in any base model's prior; only the human's examples carry it.** It is *coherent*, not
noise: **tpot = wit / deadpan / absurdist / genuine-delight** (meta-jokes "many people do not realize this but
this is actually true"; escalating-absurd lists; "warp" in GPU = "warp and weft"); **NOT-tpot = self-serious
"smart" takes, promo, niche trivia, humblebrags**. The human is *indifferent to specificity itself* — it's the joke,
not the detail.

## 4. The path forward (label-gated)

**labels → few-shot (or fine-tune) the judge → distill → fast aligned scorer.** Status:

- **Linchpin dry-run** (D39): the few-shot judge learned the human's *strictness* (9% pass its bar vs the rubric
  judge's 37%); and notably **both distilled scorers (~0.25) align with the human better than the raw judges
  (~0.1)** — deopt-distillation adds alignment *beyond* the teacher. But on 60 provisional labels + 262 pairs the
  few-shot-vs-rubric distill comparison is inconclusive (both ~0.25, weak). **Both training and eval are
  label-limited.**
- **Ready now:** round-2 labeling set (80 fresh items, ids 61–140) in the Notion DB; `build_score_prompt_fewshot` /
  `judge_score_batch(exemplars=…)` primitives; the full few-shot-distill pipeline (`fewshot_distill_test.py`,
  `eval_scorer_vs_human.py`).
- **Next (when ~140 labels land):** few-shot-judge a large pool → judge-anchored deopt → train v7.2 → re-measure
  vs human on a held-out slice. If the few-shot-distill clearly beats v7.1, scale labels further; if not, train the
  scorer **directly** on human pairs (the labels become the supervision, not the judge).

## 5. Honest limits

- 60 labels are **provisional** (the human flagged uncertainty + context-dependence) — every alignment number here
  is noisy and likely an underestimate (label noise attenuates correlation).
- The few-shot judge is **strict + slow** (long prompts), so labeling a large pool for deopt anchors is a real
  compute cost — fine-tuning the judge on labels (fast at inference) may beat few-shot at scale.
- **Taste pluralism** (rationalist / post-rat / e-acc / builder / shitposter) is unaddressed — one scorer assumes
  one taste; feasibility-gated on having enough per-style labels.

## Reproduce

`uv run python scripts/{build_calibration_set,pull_calibration_labels,score_calibration,judge_fewshot_test,
fewshot_distill_test,eval_scorer_vs_human}.py`. Scorer LOCKED = `qwen3b-bt-v71-judge-deopt`; judge rubric in
`tpot_taste/scoring/judge.py`. W&B project `tpot-taste`.
