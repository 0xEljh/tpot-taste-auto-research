# tpot-taste — System Design

> Post-training a small LM for tpot/tech-Twitter taste: **score** posts, **ideate**, **rewrite**.
> Target: near-SOTA on this narrow task on a single 12 GB RTX 3080 Ti (WSL2 + NixOS).
> Companion: `00-decision-log.md`. This doc is the handoff reference.

## 0. Problem framing

Three user-facing capabilities, mapped to two trainable models:

| Capability | "…" | Model |
|---|---|---|
| Decide if a draft is worth posting | score(post) → virality/fit | **Taste Scorer** |
| Recommend post ideas | ideate(topic/persona) → posts | **Taste Writer** |
| Improve a draft | rewrite(draft) → better post | **Taste Writer** (+ Scorer to verify) |

The Scorer is the keystone: a standalone product feature **and** the reward model the Writer is
optimized against. Build it first.

## 1. The data problem and our label (most important section)

**Source.** [Community Archive](https://www.community-archive.org/) — opt-in, public-domain full tweet
histories from tpot. Bulk via HF [`Rabrg/community-archive`](https://huggingface.co/datasets/Rabrg/community-archive)
(~11.5M tweets, 1.23 GB parquet). Live Supabase REST API for `account` metadata (follower counts) and freshness.

**HF mirror schema (per tweet):** `tweet_id, account_id, username, account_display_name, created_at,
full_text, retweet_count, favorite_count, reply_to_tweet_id, reply_to_user_id, reply_to_username,
quoted_tweet_id, conversation_id, avatar_media_url, archive_upload_id`.
Native engagement = `favorite_count` + `retweet_count`. `reply_count`/`quote_count`/impressions are NOT in
Twitter exports; reply volume is reconstructable from `reply_to_*` / `conversation_id`.

**Why raw likes are a bad target.** Engagement is dominated by (a) **follower count** (Million-Follower
Fallacy), (b) **tweet age** (likes accrue for years), (c) topic/timing. A model trained on raw likes learns
"who posted it", not "is it good".

**Our labels.**

1. **Pointwise — within-author normalized engagement** (regression target):
   - `e = log1p(favorite_count) + λ·log1p(retweet_count)` (RTs weighted; RT = amplification, like = approval).
   - Per author, z-score over the author's own history: `z = (e − μ_a) / σ_a`.
   - **Empirical-Bayes shrinkage** for thin authors: `μ_a ← (n_a·μ_a + k·μ_global)/(n_a + k)` (and σ likewise),
     require `n_a ≥ 20` else pool toward global prior.
   - **Time/era correction**: within-author normalize *within era* (e.g. per year) to absorb follower growth;
     v1 may use a per-author rolling baseline. Archived tweets are all mature, so no maturity window needed.

2. **Pairwise — same-author preference pairs** (BT target, *primary*):
   - A pair `(y_w ≻ y_l)` is valid iff: **same author**; **similar context** (shared URL/hashtag, or TF-IDF/embedding
     cosine > 0.7); within a bounded time/era window; **clear margin** (`e_w − e_l ≥ 1` author-z, or
     `fav_w ≥ 1.5·fav_l`); not near-duplicates of each other; deduped corpus.
   - Cap pairs/author (≤K) so prolific authors don't dominate; inverse-frequency weight.
   - Mirrors Tan/Lee/Pang (ACL 2014, same-author/same-URL wording study) + PopALM (top-3 vs less-liked).

**Splits (frozen, leakage-proof).** `test_temporal` (train < cutoff date, test ≥ cutoff → predict the
*future*), `test_unseen_authors` (author-disjoint), `test_pairs` (held-out same-author pairs).
Dedup first (MinHash/LSH + embedding > 0.9), drop pure retweets, strip bots. Report metrics stratified by
author and by reach bucket so "just predicts follower count" is exposed.

## 2. Key design decisions (alternatives)

### D1 — System architecture

```
APPROACH A: Single multi-task generator (minimal)
  Summary: one SFT+RL decoder does score/ideate/rewrite via prompting.
  Complexity: Low   Risk: Med-High
  Pros: one model; smallest footprint; fastest to ship.
  Cons: LLM self-scoring is uncalibrated + not trainable on pairwise engagement;
        no verifiable reward -> RL signal is weak; scoring eval is poor.
  Reuses: SFT/RL stack only.

APPROACH B: Two models — encoder Scorer + decoder Writer  [CHOSEN]
  Summary: encoder BT+regression Scorer = capability #2 AND the RL reward;
           decoder Writer (SFT->DPO->GRPO) = capabilities #1, #3.
  Complexity: Med   Risk: Low-Med
  Pros: calibratable, cheap, verifiable reward + standalone product; each
        independently benchmarkable; matches the pairwise data; ≈ PopALM/RePALM.
  Cons: two pipelines + two eval suites; encoder→decoder reward-family mismatch
        (managed with KL leash).
  Reuses: HF encoder FT (Scorer) + Unsloth/TRL (Writer).

APPROACH C: Three models — add LLM-backbone reward (value head) (maximal)
  Summary: B + a same-family LLM reward for tighter RL coupling; encoder kept for cheap eval.
  Complexity: High  Risk: Med
  Pros: best reward-policy coupling; can ensemble rewards.
  Cons: third pipeline; LLM reward heavy during 12 GB GRPO; diminishing returns for v1.
```
**Chosen: B.** C's LLM reward is a possible Phase-4b upgrade if encoder-reward RL underperforms.

### D2 — Scorer architecture

```
APPROACH A: Pointwise encoder regressor/classifier (minimal)
  Low/Low. Fast, full-FT in 12 GB, calibratable. Con: inherits confounds unless
  labels normalized; weaker as an RL reward.

APPROACH B: Encoder siamese Bradley-Terry (same-author pairs) + shared regression head  [CHOSEN]
  Med/Low. Confound-robust pairwise objective matches the data; multi-head shares a
  backbone (SMORM-style robustness); best ranking. Con: pair-mining overhead.

APPROACH C: LLM-backbone BT reward, QLoRA value head (maximal)
  High/Med. Same family as the policy (ideal RL reward) but heavy to run for
  large-scale ranking eval on 12 GB. Deferred.
```
**Chosen: B** (ModernBERT-base primary; DeBERTa-v3-base fallback).

### D3 — Preference optimization / RL

```
APPROACH A: Best-of-N -> DPO (minimal, robust)
  Low/Low, low VRAM, off-policy. Captures most of the gain. Bounded by SFT support.

APPROACH B: On-policy GRPO vs. Scorer (ceiling)
  Med-High/Med. 3B 4-bit LoRA, vLLM colocate, beta=0, num_gen=4, short completions.
  Highest ceiling. Con: 12 GB-tight; reward hacking; vLLM env complexity.

APPROACH C: Staged DPO -> GRPO from the DPO checkpoint  [CHOSEN]
  DPO gives a safe, shippable milestone and warm-starts GRPO; GRPO adds the final
  lift with KL to the DPO policy. Con: most total work.
```
**Chosen: C.**

## 3. Reward design (Writer RL)

Multi-objective, anti-reward-hacking (engagement-only → clickbait/outrage; Nature 2023: each negative
headline word +2.3% CTR):

```
R = R_scorer(post)                         # encoder BT/regression virality score
  + α · R_ground(sim-to-real-tpot-gold)    # PopALM ROUGE-anchor analogue (α≈0.5)
  − β · R_toxicity/negativity              # off-the-shelf classifier penalty
  − γ · R_engagement_bait                  # "RT if…", manufactured-outrage detector
  − δ · KL(π ‖ π_DPO)                       # leash; keeps policy where Scorer is calibrated
  − ε · length_penalty                     # decouple length (classic RM hack)
```
Early-stop on a held-out **LLM-judge** metric (not on R itself). Monitor length/negativity drift.

## 4. Evaluation

**Scorer (predictive ranking).** Pairwise accuracy on `test_pairs` (primary); per-author Spearman ρ &
Kendall τ; within-author/topic NDCG@{5,10,20}; calibration / ECE for the regression head. Report on
`test_temporal` and `test_unseen_authors` separately; stratify by reach bucket.

**Writer (generation).** (a) Scorer-rated win-rate of Writer vs. SFT vs. base on held-out prompts;
(b) **LLM-as-judge** rubric (1–5; pairwise A/B, randomized order, justify-then-verdict, 3-vote): *tpot voice*,
*not engagement-bait*, *specificity/insight*, *wit/aesthetic*, *not cringe*; calibrate judge to ~50 human
labels (Cohen's κ) before trusting it. (c) Reward-hacking audit: length, negativity, n-gram repetition drift.

**Benchmark deliverable.** Frozen `test_*` sets + a reproducible `scripts/eval_*` producing a single report
(W&B + markdown) with all metrics and qualitative examples.

## 5. Phased plan

- **P0 Env/scaffold** — flake, pinned deps, W&B, docs. *(this phase)*
- **P1 Data** — download; EDA; dedup; normalized labels; pair mining; freeze splits. **TDD**: tests for
  normalization, shrinkage, pair validity, dedup, leakage-checks *first*.
- **P2 Scorer** — train encoder BT+regression; ranking eval; ship `score()`.
- **P3 Writer SFT** — build instruction data (idea→post, topic→post, draft→improved via de-optimized
  synthetic drafts, make-it-tpot); Unsloth 4-bit QLoRA; perplexity + judge + Scorer-rated samples.
- **P4 Preference** — (a) best-of-N→DPO; (b) GRPO vs. reward+guardrails; win-rate + hacking audit.
- **P5 Integration** — `typer` CLI / API: `score`, `ideate`, `improve`; final benchmark report.

## 6. Stack & hardware notes

- **Models:** Writer `Qwen/Qwen2.5-3B-Instruct` (Unsloth 4-bit; A/B `Llama-3.2-3B-Instruct`).
  Scorer `answerdotai/ModernBERT-base` (fallback `microsoft/deberta-v3-base`).
- **3080 Ti = Ampere → no FP8.** All RL math is the 4-bit NF4 path. GRPO: 3B fits, 7B does not.
- **Deps** pinned to `negative-space-learning-v2/uv.lock` (cache hit). vLLM/sentence-transformers in
  separate extras. Never bump torch/triton/xformers piecemeal.
- **Env:** Nix devShell + uv; CUDA via WSL `libcuda.so` passthrough (see `flake.nix`).

## 7. Risks / open questions

| Risk | Mitigation |
|---|---|
| Few hundred authors → identity leakage | author-disjoint + temporal splits; cap pairs/author; per-author metrics |
| Follower growth over an author's history confounds within-author z | per-era normalization; optional Supabase follower history |
| Reward hacking (clickbait/outrage/length) | multi-objective reward; KL leash; judge early-stop; drift monitors |
| Encoder→decoder reward mismatch in RL | KL leash; optional Phase-4b LLM reward (D1-C) |
| GRPO instability / OOM at 12 GB | DPO warm-start; toggle `UNSLOTH_VLLM_STANDBY`; short completions; fall back to best-of-N+DPO |
| "tpot taste" ≠ engagement | separate judge axis + ground-anchor reward; don't optimize likes alone |

## 8. References

PopALM (LREC-COLING 2024, arXiv 2402.18950) · RePALM (Findings ACL 2024) · Tan/Lee/Pang (ACL 2014,
P14-1017) · "Negativity drives online news consumption" (Nature Human Behaviour 2023) · Unsloth GRPO /
memory-efficient RL docs · TRL GRPO/DPO docs. (Full URLs in `00-decision-log.md` research and agent reports.)
