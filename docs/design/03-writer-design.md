# Phase 3 — Taste Writer: SFT data design

Companion to `01-system-design.md` (§3 reward, §4 eval, §5 plan) and the decision log (D4 model, D5
preference-opt, D10/D16 scorer). This doc covers the one genuinely **open** sub-decision for the Writer:
**how to turn a corpus of bare tweets into instruction→post SFT supervision**. Model choice (Qwen2.5-3B-Instruct,
Unsloth 4-bit QLoRA), reward, and eval are already settled — see references.

## 1. What the Writer is for

Three product deliverables map to two model roles:

| # | Deliverable | Role |
|---|---|---|
| 1 | Recommend post ideas | **Writer** — generate |
| 2 | Identify what's worth posting | **Scorer** — score (done: v4/v6) |
| 3 | Recommend how to improve a post | **Writer** (draft→improved), **Scorer** ranks candidates |

So the Writer needs two generation modes: **ideate** (∅/topic → post) and **improve** (weak draft → stronger post).
The Scorer is the judge/reward, not the generator.

## 2. The problem: no instructions in the data

The Community Archive is tweets, not (instruction, completion) pairs. SFT needs prompts. We must synthesize them.
Constraint: keep it faithful to *real tpot voice* (don't let a generic instruct-model's "helpful assistant" register
leak in), and reuse what we already built.

### APPROACH A: Unconditional style-SFT  *(minimal viable)*
  Summary: one fixed prompt ("Write a tpot-style post:") → a real curated good tweet as the completion. Pure
    voice/style cloning.
  Complexity: Low   Risk: Low
  Pros:
    - Smallest diff; data build is pure polars over the existing curated-good set (no generation).
    - Cleanest voice transfer; nothing for the model to misinterpret.
    - A strong base policy for DPO/GRPO (which supply the "what to write about" pressure via reward).
  Cons:
    - No controllability (can't ask for a topic); doesn't serve deliverable #3 (improve) at all.
    - Prompt is content-free, so the model learns an unconditional prior — weak as a *product*.
  Reuses: `build_deopt_pairs._good()` curation (high-z uploaders + high-fav liked, English/spam filtered).

### APPROACH B: Multi-task instruction-SFT  *(ideal architecture)*  ← recommended
  Summary: a small mix of instruction templates over real completions, covering both modes:
    (a) ideate-unconditional: "Write a tpot post." → good tweet;
    (b) ideate-topic: "Write a tpot post about {topic}." → good tweet, where {topic} is a 2-4 word phrase
        extracted from the tweet (cheap: local LLM or KeyBERT-style top noun-chunk; offline, one pass);
    (c) improve: "Improve this post for tpot:\n{weak}" → {good} — **reuse the deopt pairs directly**
        (rejected=weak draft, chosen=good). This is free supervision we already generated (D14-D16).
    (d) make-it-tpot: "Rewrite in tpot voice:\n{bland}" → {good} — also from deopt pairs (bland/corporate styles).
  Complexity: Med   Risk: Med
  Pros:
    - Directly trains both product modes (#1 ideate, #3 improve) in one model.
    - The improve/make-it-tpot tasks are *free* — they invert the deopt pairs we already built.
    - Topic-conditioning gives controllability for the CLI (`ideate --topic`).
  Cons:
    - Topic extraction adds one offline LLM pass (or a dep); noisy topics inject some label noise.
    - Mixing tasks needs balancing so improve doesn't dominate (20k pairs vs ~10k goods).
  Reuses: deopt pairs (`deopt_v6_*_pairs.parquet`), `_good()` curation, the local-Qwen batch-gen harness.

### APPROACH C: Retrieval/exemplar-conditioned SFT  *(creative/lateral)*
  Summary: condition each completion on k=3 retrieved *other* good tweets (by the same author or topic-neighbors
    via the MiniLM embeddings we already have) as in-context style anchors: "Here are tpot posts you like: {ex1..3}.
    Write another." Train-time in-context style transfer.
  Complexity: High   Risk: Med-High
  Pros:
    - Strong few-shot style grounding; at inference the user can steer by supplying exemplars they admire.
    - Leverages the embedding index already built for topic-controlled pairs (D12).
  Cons:
    - Longer sequences (3 exemplars + completion) → more VRAM, fewer examples per batch on 12 GB.
    - Risk of copy-paste/retrieval-leakage (model parrots an exemplar); needs dedup between exemplars and target.
    - Over-engineered for v1; the exemplar channel is better added later as an inference-time prompt, not baked into SFT.
  Reuses: MiniLM embeddings (`tpot_taste/data/embed.py`), author grouping.

**Recommendation: B, built in two stages.** Ship **A first** (fastest path to a coherent voice policy and a
DPO/GRPO base), then layer B's instruction mix on top (or train B directly if topic extraction is quick). B is the
right long-term shape because it serves deliverables #1 and #3 in one model and reuses the deopt pairs as free
improve-supervision. C's exemplar idea is deferred to an *inference-time* prompt option — no SFT cost, most of the upside.

## 3. Concrete spec (Approach B)

- **Completions (the "good" target):** curated good tweets from `_good()` (uploader z≥1 + liked fav≥50, English,
  substantive, 40-240 chars). ~10k train / ~1.5k test, already split temporally.
- **Tasks & mix (target ≈ balanced, improve capped so it can't dominate):**
  - ideate-unconditional: ~3k  (good tweets, fixed prompt)
  - ideate-topic: ~4k          (good tweets + extracted topic)
  - improve: ~4k               (sample from deopt pairs: rejected→chosen)
  - make-it-tpot: ~3k          (deopt pairs whose style ∈ {bland, corporate, dry, platitude})
- **Format:** Qwen chat template; system = a short tpot-voice persona; user = the instruction; assistant = the
  real tweet. Mask the prompt, train on the completion only (standard SFT).
- **Topic extraction:** offline single pass with the local Qwen ("Give a 2-4 word topic for this post:") or a
  lightweight noun-chunk heuristic — decide at build time; cache to parquet so it's one-time.
- **Train:** Unsloth 4-bit QLoRA, Qwen2.5-3B-Instruct (D4). LoRA on attn+MLP, r=16-32, 1-2 epochs, packing on,
  seq≤512. Same env/W&B project.
- **Leakage guard:** SFT completions come only from the **train** temporal window; the held-out window + unseen
  authors stay frozen for eval. Improve-pairs' `chosen` must not appear as an ideate completion (dedup by text).

## 4. Eval (per `01` §4)

- **Scorer-rated win-rate:** Writer vs base Qwen vs SFT-A, scored by the locked Scorer (v4/v6) on held-out prompts.
  (The Scorer's job: this is deliverable #2 closing the loop on #1/#3.)
- **LLM-as-judge** rubric (1-5, pairwise A/B, randomized, justify-then-verdict, 3-vote): tpot voice, not bait,
  specificity/insight, wit, not cringe. Calibrate to ~50 hand labels (κ) before trusting it.
- **Reward-hacking audit** (matters once RL starts): length, negativity, n-gram repetition drift vs SFT.
- **Qualitative:** a fixed probe set of (topic / weak-draft) inputs, eyeballed every iteration — the same
  discipline that caught the scorer's register (D11), length (D15), and bait (D16) confounds.

## 5. TDD test plan (write first, per repo convention)

`tests/test_writer_data.py`:
- format: every example renders through the chat template; assistant turn == the source tweet; prompt is masked.
- task coverage: all four task types present; mix within ±10% of target ratios.
- improve-supervision integrity: every improve/make-it-tpot example's target == a deopt-pair `chosen`, input == its `rejected`.
- leakage: no completion text appears in the held-out/unseen eval splits; ideate completions disjoint from improve `chosen`.
- filters: completions pass `_quality_english`; lengths within [40, 240].

## 6. Then Phase 4 (preference) — pointer

DPO (best-of-N vs the Scorer) → GRPO with the multi-objective reward + KL leash + length penalty (D5, `01` §3).
The locked Scorer (v4/v6) is the reward; the length penalty `ε` is the downstream mitigation for the scorer's mild
length lean noted in D15/D16 — defense in depth.
