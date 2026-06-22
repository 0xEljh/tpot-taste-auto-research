# Taste alignment — Phase 6 synthesis (the judge, v7.1, and the human-alignment ceiling)

A standalone reference for the Phase-6 arc (decision log D28–D41). The one-line story: **we fixed the *coarse*
taste problem (platitudes) with a base-model judge, then cracked the *fine* taste problem (matching a specific
human) by training the scorer DIRECTLY on the human's labels — judge-distillation provably cannot reach fine
alignment, because the judge's own taste is partially *opposed* to the human's.**

> **Update (D40/D41, 2026-06-22):** the "label-gated ceiling" below was the state at 60 labels. With 137 labels the
> few-shot judge hit Spearman **0.42**, and direct human-pair training (`v7.2-human`) reached held-out pairwise
> **0.72** (v7.1 was at chance, 0.50). **Scorer LOCKED = v7.2-human.** Sections 4–5 carry the resolved story.

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

## 4. The fix — direct human pairs (v7.2), and why the judge route is a dead end (D40/D41)

With **137 labels** (round-1 60 + round-2 77), the alignment thread resolved — but not the way the plan assumed.

**4a. The lever scaled (as predicted).** Few-shot judge ↔ human: rubric-only **0.09** (chance) → few-shot (20
exemplars) **0.42**. The human's taste is capturable, only in-context.

**4b. But judge-distillation is a dead end — two independent failures:**
- **Approach A (few-shot-distill) is self-defeating.** The aligned few-shot judge is so calibrated to the human's
  bar that it endorsed **0 of 2000** pool items ≥7 → 0 deopt anchors → junk scorer (0.005). *The better the judge
  aligns, the fewer items it endorses* — judge-anchored distillation cannot scale to fine taste.
- **Approach C (hybrid: human pairs + v7.1 deopt pairs) goes anti-aligned** (Spearman **−0.19**, panel inverted).
  v7.1's "coarse taste" is partially *opposed* to the human's — the rubric lineage rewards substance/vividness/
  profundity, exactly what the human rejects as performative. The two signals fight; there is **no free breadth**
  from the judge lineage.

**4c. Approach B — skip the judge, supervise directly — WINS.** The human's labels are length-balanced
(len↔label r=−0.036), so the v7 register confound that *forced* deopt does not apply: cross-class human pairs
(tpot>borderline>not) carry pure taste. `human_pairs.py` → 2,684 pairs from 91 train labels → `v7.2-human`.

| scorer (shared 46 held-out) | Spearman | pairwise-acc(tpot>not) | precision |
|---|---|---|---|
| v7.1 (rubric-distilled) | +0.03 | 0.50 (chance) | 0.44 |
| **v7.2-human (direct pairs)** | **+0.39** | **0.72** | **0.71** |

And it ranks *generated* drafts correctly (writer best-of-N de-platitudes the traps; the synthetic-panel
compression is an OOD artifact, not a real-use failure). **Scorer LOCKED = v7.2-human.**

**The principle:** for an idiosyncratic taste absent from any base-model prior, *the human's labels are the only
faithful supervision*. A judge built on the same prior approximates a **different** taste; distilling it — at any
size, with any rubric — converges to that other taste, not the human's. Put the human's examples in the loss, not
in a teacher's prompt.

## 5. Honest limits

- **137 labels is still small.** v7.2's 0.72 pairwise is on a 46-item held-out (12 tpot / 20 not); the *direction*
  (B ≫ v7.1, chance) is robust, the *magnitude* is noisy. Round-1 labels were flagged provisional. More labels
  (round-3) is the clear lever to push alignment past ~0.4 Spearman.
- **Residuals (non-blocking):** mild platitude residual on *synthetic* generic-viral probes (not seen on real
  writer output); over-rates non-English in raw goods (OOD — the curation English filter handles it). The one
  *aligned* breadth source if needed: **human-tpot-anchored deopt** (degrade the human's own confirmed-tpot —
  agrees with the signal, unlike v71). Deferred (YAGNI).
- **Taste pluralism** (rationalist / post-rat / e-acc / builder / shitposter) is unaddressed — one scorer assumes
  one taste. Now *more* feasible: direct-pair training per-style needs only per-style labels (no judge).

## Reproduce

`uv run python scripts/{pull_calibration_labels,build_human_pairs,train_scorer,eval_scorer_vs_human,demo_eval}.py`.
Scorer LOCKED = `qwen3b-bt-v72human` (`qwen3b-bt-taste-LOCKED`); v7.1 = `qwen3b-bt-v71-judge-deopt` preserved.
Pairing logic: `tpot_taste/data/human_pairs.py` (+`tests/test_human_pairs.py`). W&B project `tpot-taste`.
