# 05 — The model frontier: have we pushed hard enough? (research synthesis + plan)

**Date:** 2026-06-22. **Status:** research done, plan proposed. Companion to `04-taste-alignment.md`.

**One-line:** The biggest unrealized lever is **not a bigger base model** — it's the *scorer's objective and
representation*. We are on the weakest combination (3B decoder + Bradley-Terry pairs) for a ~140-label taste task,
where the literature says *regression on a small/encoder/embedding representation* wins. Growing labels via an
active loop is the second lever. The Writer base upgrade (Qwen3-4B-Instruct-2507) is real but **lower ROI** until
the scorer improves — D35 already showed the Writer was adequate and the system is scorer-bound through best-of-N.

> This doc was produced from four parallel research briefs (base models, reward modeling, 12GB engineering, infra
> map). Every external claim is cited. **Caveat:** post-Jan-2026 model specifics (Qwen3.5, Gemma 4, Ministral 3)
> come from agent web-fetches that should be re-verified before any download. The *core* recommendation (scorer
> architecture) rests on well-established 2024–2025 results and does **not** depend on those post-cutoff details.

---

## 0. The question

User: *"I'm not sure we've pushed hard enough on the model front. Surely there are better base models given our
hardware, and more ML engineering to push on."* Correct instinct — Qwen2.5-3B-Instruct is ~21 months old and 12GB
QLoRA can train much more. But the research says the highest-value move is **not** "swap in a bigger base."

---

## 1. Finding A — Writer base models (the obvious upgrade exists, lower priority)

The clear in-family upgrade is **Qwen3-4B-Instruct-2507** (dense, Apache-2.0, *non-thinking*): IFEval 83.4,
MMLU-Pro 69.6, EQ-Bench Creative Writing v3 **83.5** — large gains over Qwen2.5-3B on every axis, fits 12GB QLoRA
trivially, and the SFT/DPO recipe + chat template port cleanly ([HF card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)).
Alternatives: **SmolLM3-3B** (fully-open, lighter, slightly weaker — [blog](https://huggingface.co/blog/smollm3));
**Qwen3-8B** as a scale-up ([Qwen3 report](https://arxiv.org/pdf/2505.09388)).

**Traps to avoid:** (1) *Thinking/reasoning* variants (Qwen3-4B-**Thinking**, Phi-4-reasoning, Gemma-4's
reasoning-first design) emit chain-of-thought that wrecks a terse deadpan tweet voice. (2) **Qwen3.5-4B** despite top
benchmarks is MoE + multimodal + thinking-default ([card](https://huggingface.co/Qwen/Qwen3.5-4B)) — doesn't QLoRA
cleanly on bitsandbytes. (3) **Gemma 3** license forbids using it to train other models ([analysis](https://markaicode.com/gemma-3-apache-license-commercial-use-guide/),
[terms](https://ai.google.dev/gemma/terms)) — disqualifying if it also seeds the scorer; Gemma 4 fixed the license
but is reasoning-first.

**Why lower priority:** D34/D35 found re-curating + re-SFT + re-DPO the Writer was a **tie** vs the old DPO-v2 — *the
platitude bug was a scorer problem, not a writer one.* A better Writer only helps once the **scorer** ranks
best-of-N well. So: queue the Qwen3-4B-2507 Writer upgrade, but **after** the scorer work proves the bottleneck.

---

## 2. Finding B — Scorer architecture & objective  **[THE HEADLINE]**

Our scorer is **Qwen2.5-3B + Bradley-Terry scalar reward head, LoRA, cross-class pairs**. The 2025–2026 reward-
modeling literature says this is the **least sample-efficient** option for ~140 idiosyncratic labels:

- **Regression > Bradley-Terry at small/weak bases — "by a large margin"**; the gap only closes for strong 7–9B
  bases ([base-model analysis, EMNLP'25 / arXiv 2505.10775](https://arxiv.org/pdf/2505.10775)). We force
  `borderline=0.5` into *pairs* and train BT on a 3B — exactly the regime where BT is weakest.
- **Small reward models decisively compete.** On RewardBench 2, Skywork-Reward-V2-**Qwen3-0.6B** nearly matches the
  prior-gen **27B** SOTA, and the **1.7B surpasses a 70B** RM ([Skywork-V2](https://huggingface.co/papers/2507.01352);
  [RewardBench 2](https://arxiv.org/html/2506.01937v2)). Data quality + base *choice* dominate scale.
- **Encoders beat decoders at classification scale-for-scale:** ~400M encoders > 1B decoders ([Ettin](https://arxiv.org/html/2507.11412v1);
  [ModernBERT](https://arxiv.org/pdf/2412.13663)).
- **Frozen-embedding + small-MLP head is the strongest *small-data* play** — beats small LM-RMs across the board and
  beats *larger* LM-RMs under restricted annotation, trains in minutes ([Reusing Embeddings, arXiv 2502.04357](https://arxiv.org/html/2502.04357v1)).
- RMs **inherit value biases from their base's pretraining** even with identical data ([arXiv 2601.20838](https://arxiv.org/abs/2601.20838))
  — an argument for a *neutral* representation + our own head rather than baking taste into an opinionated 3B decoder.

**Project-specific caveat (the orchestrator's drill-down — this is NOT in the generic research):** our D40/D41
finding is that this taste is **absent from any base-model prior**. That directly threatens the *frozen*-embedding
approach: if the taste isn't linearly decodable from a general embedder's space, a frozen embedder + MLP will
underperform no matter how good the embedder. **Therefore the bake-off must include a fine-tuned-representation arm**
(a fine-tuned encoder, and the current fine-tuned decoder), not only frozen-embedding+MLP. We settle this
empirically on *our* data — general RM benchmarks measure broad correctness, not one person's taste.

**Tie handling for `borderline`:** Davidson / Rao-Kupper BT-with-ties extensions exist and fold into DPO without
degradation ([arXiv 2409.17431](https://arxiv.org/html/2409.17431)), but the extra ν parameter is hard to fit at
N=140 — **not worth it**. The cleaner win is to stop forcing 0.5 and model the three classes **ordinally** with a
rank-consistent **CORN/CORAL** head ([coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch);
[arXiv 2111.08851](https://arxiv.org/pdf/2111.08851)) — which pairs naturally with the regression recommendation and
gives `borderline` first-class status.

---

## 3. Finding C — 12GB engineering (modern stack, mostly for the Writer)

- **Trainable sizes on 12GB QLoRA:** 4B comfortable; **7–8B tight-but-reliable** at seq≤512, batch=1, grad-ckpt
  (7B QLoRA fit in ~10.9GB — [Birch-san gist](https://gist.github.com/Birch-san/57878c4a27cf34f57d3e861865a7d0a2));
  **14B not advisable** (weights alone ~7–8GB, no activation headroom). **Biggest risk for 8B: activation spikes at
  long seq** (14–15GB at seq 2048 — [12GB guide](https://www.buildmvpfast.com/blog/fine-tune-llm-laptop-qlora-local-gpu-2026)) → cap seq 512, keep paged optimizer in reserve. DPO is the tightest stage.
- **Adopt Unsloth:** ~2× faster, >70% less VRAM, supports Qwen3/Gemma3 and DPO+RM (not just SFT)
  ([benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks)). This is what makes 8B comfortable.
- **PEFT levers worth turning on:** `rsLoRA` (near-free quality, [arXiv 2312.03732](https://arxiv.org/abs/2312.03732));
  target **all-linear** modules (QLoRA-paper standard); rank 16 / alpha 32. **Skip** DoRA (task-dependent, adds cost)
  and **PiSSA** (can *collapse* in preference/RL training — [arXiv 2512.23165](https://arxiv.org/html/2512.23165)),
  given our DPO-heavy pipeline. `liger-kernel` optional (40–60% VRAM, [arXiv 2410.10989](https://arxiv.org/pdf/2410.10989)).

These mostly matter once we move the **Writer** to 8B. The scorer redesign (Finding B) makes the scorer *smaller*,
not bigger, so 12GB stops being the binding constraint for it.

---

## 4. Finding D — labels & active learning (the verify-loop)

The user offered: *"You make a first pass on labels, I verify, and label what you're unsure about."* Best practice:

- Per cycle (~20–40 items): score the unlabeled pool with **v7.2**, take the most **uncertain** items near the
  decision boundaries (borderline↔tpot, borderline↔not — "small-margin pairs are most informative",
  [arXiv 2602.01581](https://arxiv.org/pdf/2602.01581)), then **diversify** within that pool (embedding-cluster /
  coreset) so the batch isn't 30 near-duplicates. Optional query-by-committee via an MLP ensemble (ActPRM hit SOTA at
  **20% of the label budget** — [arXiv 2504.10559](https://arxiv.org/html/2504.10559v1)).
- **Always run a random control arm.** "Random Is Hard to Beat" found random ≈ active for modern-LLM preference
  learning ([arXiv 2604.02766](https://arxiv.org/pdf/2604.02766)); only keep the active strategy if it beats random on
  held-out. The verify-loop's *second* virtue is independent of accuracy: uncertainty sampling routes the human to
  exactly the cases the model is worst at, maximizing label value per minute of verification — which is precisely the
  "label what you're unsure about" division of labor the user proposed.

---

## 5. The key insight

> For a 140-label idiosyncratic-taste scorer, **objective + representation beat base size.** We have been running the
> weakest combination (3B decoder + BT pairs). The cheapest, highest-information thing we can do is a **bake-off on the
> data we already have** — it answers "have we pushed hard enough on modeling?" empirically, and it de-risks every
> bigger bet (8B Writer, more labels) that would otherwise be a guess.

---

## 6. Design approaches

```
APPROACH A: "Bake-off on current data, then commit"  (minimal viable)
  Summary:    Fix the 137 labels; bake off scorer variants on the SAME 91-train / 46-held-out split —
              (1) current 3B-BT [baseline], (2) 3B + regression head, (3) frozen-embedding + MLP regressor,
              (4) ModernBERT-large fine-tune, (5) ordinal CORN head on the winner. Adopt the best. Touch nothing else.
  Complexity: Low      Risk: Low
  Pros:       Answers the user's question with evidence on OUR data; arms 2–3 train in minutes (k-fold-able);
              encoder/embedding winners free VRAM + 10× faster iteration; cannot be "wrong" — pure information.
  Cons:       Doesn't grow labels (the other lever); doesn't touch the Writer; magnitude bounded by 137-label noise.
  Reuses:     human_pairs.py, eval_scorer_vs_human.py (pairwise-acc + Spearman), align_heldout_ids.json (46-item),
              calibration_labels.json, scoring/model.py.

APPROACH B: "Both levers in parallel"  (ideal architecture)  ← RECOMMENDED
  Summary:    Run A's bake-off AND stand up the v7.2-driven active-labeling loop (uncertainty×diversity + random
              control); I pre-label a round-3 batch, user verifies async while the bake-off runs; retrain the
              bake-off winner on the grown label set. Fold in Unsloth/rsLoRA/all-linear opportunistically on retrain.
  Complexity: Med      Risk: Low–Med
  Pros:       Attacks both scaling levers; they compound (more labels × better-fit model); the loop is the only
              proven path past ~0.4 Spearman; good use of the user's availability (they label while GPU computes);
              matches the explicit collaboration offer.
  Cons:       Needs the user's verification time; more moving parts; active selection may not beat random (control arm
              hedges this).
  Reuses:     build_calibration_set.py (sampler), pull_calibration_labels.py, notion-cat + chunk_for_notion.py,
              the scorer-as-sampler, plus A's bake-off harness.

APPROACH C: "Full v8 re-base"  (creative / aggressive — defer)
  Summary:    Take every top research rec end-to-end now: Writer → Qwen3-4B-Instruct-2507 (or 8B via Unsloth),
              scorer → best bake-off architecture, stack → Unsloth+rsLoRA+all-linear+liger. A clean modern v8.
  Complexity: High     Risk: Med
  Pros:       Captures the full ~21-month jump; modern stack makes 8B feasible + iteration faster; best long-term base.
  Cons:       Large surface area; rebuilds working pieces; D35 says Writer ROI is speculative until the scorer is the
              proven bottleneck-breaker; most of the realizable gain likely comes from B's scorer+labels alone.
  Reuses:     Existing SFT/DPO scripts (ported to Unsloth), data splits, eval harness.
```

**Recommendation: B, sequenced — A's bake-off is literally step 1 of B.**
Rationale: research *and* D35 both point at the scorer, not the base; the bake-off is the highest-information cheap
step and settles the project-specific frozen-embedder caveat; labels are the only proven path past 0.4 Spearman; the
Writer/8B upgrade (C) is real but speculative ROI now, so we earn it after the scorer improves. We get C's *cheap*
wins (rsLoRA, all-linear, Unsloth) for free whenever we retrain, without committing to its big surface area.

---

## 7. Bake-off spec (TDD — tests first, must fail initially)

**Arms** (all on the frozen 91-train / 46-held-out split from `align_heldout_ids.json`; same 137 labels):
| # | representation | objective | cost |
|---|---|---|---|
| 1 | Qwen2.5-3B (LoRA) | Bradley-Terry pairs | baseline (done: 0.72 pw / 0.39 ρ) |
| 2 | Qwen2.5-3B (LoRA) | pointwise regression (0/0.5/1) | minutes |
| 3 | frozen embedder (gte/bge/Qwen3-Embedding) + MLP | regression | minutes (CPU-ok) |
| 4 | ModernBERT-large (fine-tuned) | regression / 3-class | minutes |
| 5 | winner-of-{2,3,4} | CORN ordinal head | minutes |

**Metrics (per arm, on the 46-held-out):** pairwise-acc(tpot>not) [primary], Spearman ↔ human, 3-class macro-F1.
Report alongside chance (0.50 pw). **N=46 is noisy** → for the cheap arms (3, and 2/4 where feasible) also report
**k-fold CV over all 137 labels** to de-noise the comparison; only trust differences that survive CV.

**New, testable units (write `tests/` first):**
- `ordinal_targets(labels)` → {0.0, 0.5, 1.0} mapping + class weights (test: mapping, balance).
- `embedding_features(texts, model)` → (N, d) matrix (test: shape, determinism, cache hit).
- a thin `train_regressor` / `eval_arm` returning the 3 metrics (test: monotonic sanity on a toy separable set).

**Decision rule:** adopt the arm with the best CV-stable pairwise-acc; if a fine-tuned encoder/embedding arm matches
or beats the 3B decoder (likely per Finding B, *unless* the frozen-embedder caveat bites), switch — it frees VRAM and
speeds iteration ~10×, compounding with the labeling loop.

### Results (2026-06-22) — full bake-off (all 5 arms)

`scripts/bakeoff_scorer.py` (`tests/test_bakeoff.py` green). All arms on the **same 46-item held-out** (12 tpot / 20
not → 240 tpot×not pairs, so a ~0.06 pw swing ≈ 14 pairs flipping — read inter-arm gaps as noise):

| arm | representation × objective | held-out pw | held-out ρ | de-noised | cost |
|---|---|---|---|---|---|
| 3B + Bradley-Terry (v7.2, current) | Qwen2.5-3B LoRA, 2684 pairs | 0.72 | +0.39 | — | ~4 min GPU |
| frozen **MiniLM-L6** (22M) + Ridge | frozen 384-d, 91 pts regression | **0.78** | **+0.44** | CV(137) 0.67 / +0.28 | **~2 s CPU** |
| frozen **mpnet-base** (110M) + Ridge | frozen 768-d, 91 pts regression | 0.74 | +0.36 | CV(137) 0.68 / +0.27 | ~3 s CPU |
| **regress-3b** | Qwen2.5-3B LoRA, 91 pts regression | 0.66 | +0.26 | overfit (train loss→0.002) | ~4 min GPU |
| **ModernBERT-large** (395M) + LoRA | fine-tuned 1024-d, 91 pts regression | 0.76 | +0.34 | — | **~33 s GPU** |
| (MiniLM/mpnet + MLP head) | frozen + nonlinear | 0.45–0.72 | — | CV 0.55–0.62 | overfit at n=91, discarded |

**Verdict — label-limited, not model-limited (settled).** Five very different representation×objective combinations
— a 22M frozen embedder, a 110M frozen embedder, a fine-tuned 400M encoder, a fine-tuned 3B decoder, and the 3B BT —
all land in the **same pw 0.66–0.78 / ρ 0.26–0.44 band**, indistinguishable at n=46. **Fine-tuning the representation
did not break the ceiling** (ModernBERT 0.76 ≈ frozen 0.74–0.78; regress-3b *overfit* the 91 points down to 0.66).
Bigger base, fine-tuned encoder, better objective — none moves the number. **The ceiling is the 137 labels.** This
answers the original question: we have pushed hard enough on the *model*; the model is not the bottleneck.

**Chosen recipe:**
- **Shipped scorer stays v7.2 (3B-BT)** — within noise of the best and already wired into `engine.py`'s best-of-N;
  no accuracy reason to swap, and swapping touches the Writer pipeline.
- **Fast inner-loop scorer = frozen MiniLM + Ridge** — matches the 3B at ~zero cost and **retrains in seconds on
  CPU**, exactly what active selection needs (re-score uncertainty every round). The real engineering win the
  bake-off surfaced: *scoring no longer needs the GPU at all.*
- **Re-bake-off after round-3 labels** — with more labels the cheap arm may pull decisively ahead (or the 3B may);
  reconsider what ships then.
- **Base upgrade (Qwen3-4B-2507) is now clearly secondary** — it can't help a label-limited scorer; its value is
  Writer-side, earned only after labels raise the scorer ceiling.

---

## 8. Labeling-loop spec (round-3, active)

1. **Score the unlabeled pool** (English z≥1.0 goods, deduped vs ids 1–140 by normalized text) with v7.2.
2. **Select** ~30 by uncertainty near v7.2's decision boundaries + diversity (embedding-cluster), **+ ~10 random
   control**, tagged so we can compare active-vs-random label value later.
3. **First pass:** I propose a label (👍/🤔/👎) + a confidence + one-line reason per item.
4. **Push to Notion** via `notion-cat` (chunked) into the calibration DB schema (`👍 tpot`/`🤔 borderline`/`👎 not tpot`,
   ids 141+); user verifies, focusing on my low-confidence items.
5. **Pull + merge** (`pull_calibration_labels.py` → `calibration_labels.json`), rebuild pairs/targets, **retrain the
   bake-off winner**, re-measure on a (now larger) held-out. Repeat.

### Round-3 batch — pushed 2026-06-22 (ids 141–180)

40 candidates via `build_round3_candidates.py` (20 boundary / 10 committee-disagree / 10 random control), deduped vs
ids 1–140, pushed by `push_round3_to_notion.py` with my first-pass proposal + confidence + reason in each page's body
callout and the **Taste select left empty** — verified contamination-safe (re-pull: 137 labeled / 40 blank). The
disagree bucket surfaced v7.2's residual blind spot: genuine-delight / absurdist tweets it scores −2 to −4 (#145
"concrete jungle wet dream tomato", #152 deadpan Louisiana-Purchase, #153 verb→noun "buttplug") that read as on-taste,
so those labels should be especially corrective. Close the loop after verification: `pull_calibration_labels.py` →
`build_human_pairs.py` → `train_scorer.py` → `eval_scorer_vs_human.py`, then re-run `bakeoff_scorer.py` to see whether
more labels lift the 0.67 ceiling or finally separate the arms.

---

## References

**Base models:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) ·
[SmolLM3](https://huggingface.co/blog/smollm3) · [Qwen3 report](https://arxiv.org/pdf/2505.09388) ·
[Creative Writing v3](https://llm-stats.com/benchmarks/creative-writing-v3) ·
[Ministral-3](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512) ·
[Gemma license](https://markaicode.com/gemma-3-apache-license-commercial-use-guide/) /
[Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) ·
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
**Reward modeling:** [regression>BT / base-choice +14% (2505.10775)](https://arxiv.org/pdf/2505.10775) ·
[Reusing Embeddings (2502.04357)](https://arxiv.org/html/2502.04357v1) ·
[LENS (2509.26074)](https://arxiv.org/pdf/2509.26074) · [Ettin (2507.11412)](https://arxiv.org/html/2507.11412v1) ·
[ModernBERT (2412.13663)](https://arxiv.org/pdf/2412.13663) · [RewardBench 2 (2506.01937)](https://arxiv.org/html/2506.01937v2) ·
[Skywork-Reward-V2 (2507.01352)](https://huggingface.co/papers/2507.01352) ·
[reward-bench repo](https://github.com/allenai/reward-bench) ·
[BT+MO complementary (2507.07375)](https://arxiv.org/abs/2507.07375) ·
[value bias (2601.20838)](https://arxiv.org/abs/2601.20838) ·
[DPO-with-ties / Davidson·Rao-Kupper (2409.17431)](https://arxiv.org/html/2409.17431) ·
[CORN/CORAL](https://github.com/Raschka-research-group/coral-pytorch) / [(2111.08851)](https://arxiv.org/pdf/2111.08851)
**Active learning:** [Nearly-optimal APL (2602.01581)](https://arxiv.org/pdf/2602.01581) ·
[ActPRM (2504.10559)](https://arxiv.org/html/2504.10559v1) ·
[Random Is Hard to Beat (2604.02766)](https://arxiv.org/pdf/2604.02766)
**Engineering:** [Unsloth benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks) ·
[Unsloth packing](https://unsloth.ai/docs/new/3x-faster-training-packing) ·
[12GB QLoRA gist](https://gist.github.com/Birch-san/57878c4a27cf34f57d3e861865a7d0a2) ·
[12GB guide 2026](https://www.buildmvpfast.com/blog/fine-tune-llm-laptop-qlora-local-gpu-2026) ·
[rsLoRA (2312.03732)](https://arxiv.org/abs/2312.03732) · [DoRA (2402.09353)](https://arxiv.org/abs/2402.09353) ·
[PiSSA](https://openreview.net/pdf?id=6ZBHIEtdP4) · [PEFT collapse in RL (2512.23165)](https://arxiv.org/html/2512.23165) ·
[liger-kernel (2410.10989)](https://arxiv.org/pdf/2410.10989)
