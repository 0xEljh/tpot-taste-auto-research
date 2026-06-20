# Decision Log

Running log of decisions, rationale, and status. Newest status at top; decisions appended in order.
Companion to `01-system-design.md` (the full design).

---

## STATUS (living)

**Phase:** 0 — environment & scaffolding.
**Last updated:** 2026-06-20.

- [x] Recon: env, GPU, data, stack (4 parallel research agents).
- [x] Scaffold: `flake.nix`, `pyproject.toml` (pinned), dirs, `.env` (W&B), docs.
- [ ] Build env (`uv sync --extra train`) + verify `torch.cuda`.  ← in progress
- [ ] Phase 1: download Community Archive, EDA, normalized labels, pair mining, frozen splits.
- [ ] Phase 2: Taste Scorer (encoder BT + regression) + ranking eval.
- [ ] Phase 3: Taste Writer SFT (Qwen2.5-3B, Unsloth 4-bit).
- [ ] Phase 4: DPO, then GRPO vs. Scorer reward + guardrails.
- [ ] Phase 5: integration CLI + final benchmark report + reward-hacking audit.

**Pick-up notes for another engineer/agent:** read this file + `01-system-design.md`.
Proven dep versions live in `pyproject.toml [train]`. W&B project `tpot-taste`.
Environment reference: `/home/elijah/negative-space-learning-v2` (same Nix+uv+CUDA-on-WSL2 pattern).

---

## 2026-06-20 — D1: Target hardware-feasible scope = two models, encoder reward

**Decision.** Build (A) an **encoder Taste Scorer** and (B) a **decoder Taste Writer**, not a single
self-scoring generator. The Scorer is both a standalone deliverable (capability #2) and the RL reward.

**Alternatives** (see design doc §2 for full table): single multi-task generator (no verifiable reward,
weak scoring); three models adding an LLM-backbone reward (over-engineered for v1, heavy for 12 GB RL).

**Why.** A dedicated encoder reward is calibratable, cheap to run over 10⁴–10⁵ tweets for eval, fits 12 GB
at full fine-tune, and matches the same-author pairwise data design. Mirrors PopALM/RePALM (SOTA on the
closest published task). Cost: two pipelines + an encoder→decoder reward-family mismatch, managed with a
KL leash during RL.

## 2026-06-20 — D2: Core label = within-author normalized engagement + same-author pairs

**Decision.** Do **not** train on raw likes. Primary signals:
(1) **within-author z-score** of `log1p(engagement)` (empirical-Bayes shrinkage for low-history authors),
(2) **same-author preference pairs** (chosen ≻ rejected, same author, similar topic/time, clear margin).

**Why.** Absolute likes are dominated by follower count + tweet age (Million-Follower-Fallacy; engagement
accrues over time). Same-author differencing removes the reach confound — the Tan/Lee/Pang (ACL 2014)
move — leaving text-attributable signal. Follower counts are NOT in the HF mirror, but within-author
normalization doesn't need them. Likes-per-follower is a secondary signal if we pull the Supabase
`account` table.

**Risk flagged.** Only a few hundred opted-in authors → author-identity leakage. Mitigation: author-disjoint
splits + temporal split; cap pairs/author; report per-author metrics.

## 2026-06-20 — D3: Scorer = encoder Bradley-Terry + shared regression head

**Decision.** ModernBERT-base (fallback DeBERTa-v3-base), siamese **BT loss** on same-author pairs +
auxiliary **regression** head on normalized engagement (shared backbone). Full fine-tune (fits 12 GB).

**Alternatives:** pointwise regressor only (inherits confounds, weak reward); LLM-backbone value head
(best RL coupling but heavy for eval/12 GB — deferred to a possible Phase 4b).

## 2026-06-20 — D4: Writer = Qwen2.5-3B-Instruct, Unsloth 4-bit QLoRA

**Decision.** Primary `Qwen/Qwen2.5-3B-Instruct`; A/B against `meta-llama/Llama-3.2-3B-Instruct`.
Unsloth + 4-bit NF4 + LoRA.

**Why.** 3B is the largest that supports a real **on-policy GRPO** loop in 12 GB (Unsloth floor 5–7 GB
for ≤3B GRPO; 8B needs ~15 GB). The task bottleneck is style/punchiness, not deep reasoning, so 3B is apt.
**3080 Ti is Ampere → no FP8**; the "5 GB FP8 GRPO" path is unavailable, budget around 4-bit. Qwen2.5-7B
fits SFT/DPO only — kept as an upgrade path if we drop GRPO.

## 2026-06-20 — D5: Preference optimization = DPO first, then GRPO (staged)

**Decision.** Stage 4a **best-of-N → DPO** (off-policy, robust, low VRAM, reliable milestone), then
Stage 4b **GRPO** from the DPO checkpoint for the SOTA push (beta=0/no ref model, vLLM colocate,
num_generations=4, max_completion≈64–96). Reward = Scorer + guardrail penalties (toxicity/bait/length)
+ KL to the DPO policy.

**Why.** DPO de-risks and warm-starts; GRPO raises the ceiling but is 12 GB-tight and prone to reward
hacking. Staging gets a shippable model early and isolates GRPO instability.

## 2026-06-20 — D6: Pinned heavy deps to the local proven set

**Decision.** Pin `torch==2.10.0, transformers==5.5.0, unsloth==2026.4.8, trl==0.24.0, peft==0.18.1,
accelerate==1.13.0, bitsandbytes==0.49.2, datasets==4.3.0, numpy==2.2.6`.

**Why.** Exactly the versions in `negative-space-learning-v2/uv.lock`, already in the uv cache → fast,
first-try-clean resolve. vLLM and sentence-transformers kept in separate extras so they can't perturb the
proven set. Per the compat research: pick torch first, let it pin triton/xformers; never bump piecemeal.

## 2026-06-20 — D7: Env = Nix devShell + uv, CUDA via WSL passthrough

**Decision.** Reuse the NSL flake pattern: `python312` + `uv`, `LD_LIBRARY_PATH` to `/usr/lib/wsl/lib`
(host `libcuda.so`), `TRITON_LIBCUDA_PATH`, `stdenv.cc.cc.lib`/`zlib` for binary wheels, `.env` auto-load.
No nixGL, no Nixpkgs torch.
