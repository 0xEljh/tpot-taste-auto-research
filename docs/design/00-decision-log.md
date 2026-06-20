# Decision Log

Running log of decisions, rationale, and status. Newest status at top; decisions appended in order.
Companion to `01-system-design.md` (the full design).

---

## STATUS (living)

**Phase:** 0 — environment & scaffolding.
**Last updated:** 2026-06-20.

- [x] Recon: env, GPU, data, stack (4 parallel research agents).
- [x] Scaffold: `flake.nix`, `pyproject.toml` (pinned), dirs, `.env` (W&B), docs.
- [x] Build env + verify: torch 2.10.0+cu128 CUDA-OK, RTX 3080 Ti sm_86 (Ampere, no FP8), bf16, 11.6 GB free; unsloth/trl/peft/bnb import clean. flake.lock pinned to nixpkgs 2026-06-16.
- [x] Phase 1a: downloaded corpus (11.56M tweets, 1.9 GB), EDA + structure probe → `02-data-findings.md`.
- [x] Phase 1b: pipeline `tpot_taste/data/{clean,labels,pairs,splits}` (13 TDD tests pass); built + qualitatively
      validated + refined datasets. Final: 28,283 train / 12,589 temporal / 3,043 unseen-author pairs;
      1.26M labeled uploader tweets; 4.15M liked/taste tweets. Outputs in `data/processed` + `data/splits`,
      `data/processed/manifest.json`. Scripts: `build_dataset.py`, `inspect_pairs.py`, `eda.py`.
- [x] Phase 2: Taste Scorer v1 (Qwen2.5-3B + BT reward head, D10). Trained (W&B run vuuije2r) + evaluated +
      qualitatively assessed (D11). Held-out pairwise acc **0.611** temporal / **0.626** unseen-author;
      per-author Spearman ~0.10. Real but crude (register proxy). Adapter `outputs/scorer/qwen3b-bt-v1`;
      code `tpot_taste/scoring/`, `scripts/{train,eval,inspect}_scorer.py`, `outputs/scorer/eval.json`.
- [x] Phase 2b: Scorer v2 (topic-controlled pairs) **FAILED** — worse than v1 everywhere; revealed that
      within-author engagement-lift prediction has a **~0.61 ceiling** (same-topic pairs ≈ chance) [D13].
      v1 (`outputs/scorer/qwen3b-bt-v1`) remains the best scorer.
- [x] Phase 2c: Scorer v3 (taste) — qualitative **WIN** (corporate-cringe now lowest, purple-prose penalized,
      funny/aphorism high) but a **length/bait confound** (rewards "lol same"/"RT if"); synthetic 0.86,
      real-engagement ~0.50 (taste ⟂ engagement here) [D14]. Adapter `outputs/scorer/qwen3b-bt-v3-taste`.
- [x] Phase 2d: Scorer v4 (length-balanced degradations) — **bait/low-effort fix WORKED** ("lol same" +6.7→-6.0,
      "RT if" +4.3→-4.0; corporate-cringe stays lowest) but **over-corrected into a mild length bias**
      (pearson(score,chars) +0.03→+0.23; under-rates short aphorisms) [D15]. Adapter `outputs/scorer/qwen3b-bt-v4-taste`.
- [x] Phase 2e: Scorer v5 (length-MATCHED degradations) — **length bias FIXED** (real-data corr +0.23→-0.04) but
      **lost bait/corporate penalization** (corporate-cringe + engagement-bait became the TOP probes). The
      qualitative-over-quantitative lesson: the length *number* got perfect, the actual taste judgment got worse [D16].
- [x] **Scorer INCUMBENT = v4** (`outputs/scorer/qwen3b-bt-v4-taste`) — best on the axis that matters (bait/
      corporate/low-effort all lowest); mild length lean is downstream-mitigable (RL length penalty + judge-time norm).
- [~] Phase 2f: Scorer v6 — final challenger: train on **BOTH** degradation distributions (v4 ∪ v5 pairs, ~20k) so
      the model learns bait=bad robustly without a length shortcut. Beat v4 or v4 stands & LOCKS. HARD STOP. ← in progress
- [ ] Phase 3: Taste Writer SFT (Qwen2.5-3B-Instruct, Unsloth 4-bit) on curated good tpot tweets (deferred).
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

## 2026-06-20 — D16: Scorer v5 verdict — length fixed, bait penalization LOST → v4 is incumbent; v6 = combined data

**The qualitative-over-quantitative lesson, concretely.** v5's length-matched degradations did exactly what the
gate predicted: training-pair corr(is_chosen, length) +0.03 (v5) vs -0.26 (v4), and the trained scorer's
real-data length corr collapsed **+0.229 (v4) → -0.036 (v5)**. The *number* is now perfect.

But the *taste judgment got worse*. v5 probes (compressed, near-random): **corporate-cringe +2.59 (HIGHEST)**,
**engagement-bait +2.03 (2nd)**, earnest-vulnerable -0.95 (lowest). For an "is this worth posting?" tool, ranking
corporate-cringe and "RT if you agree" as the TOP picks is disqualifying. (Held-out top/bottom was fine — funny/
insight high, AI-poetry low — but the probe failure on the core bait/corporate axis is decisive.) Real transfer
~chance (0.517/0.510), like v3; synthetic 0.898.

**Why:** forcing bait to the SAME length as the good post made the bad-signal too subtle; at ~1/10 of pairs per
style the model didn't learn bait=bad. v4's bait was SHORT — a strong (if length-confounded) signal it latched
onto. Real tension: short-bait (v4) learns bait=bad via a length shortcut; same-length-bait (v5) is length-neutral
but doesn't learn bait=bad. Whack-a-mole across v3 (bait-high) → v4 (length-biased) → v5 (bait-high).

**Decisions.** (1) **v4 is the incumbent best scorer.** It most reliably flags the actionable "don't post this"
cases (bait/corporate/cringe/low-effort all lowest), its top held-out is substantive tpot, and its one flaw (mild
length lean under-rating short aphorisms) is downstream-mitigable (the RL reward already has an explicit length
penalty ε; judge-time length-normalize). Choosing v4 over v5 *despite v5's better length number* is the
qualitative call the user asked for. (2) One final **principled** challenger **v6 = train on v4 ∪ v5 pairs**
(~20k): the model sees bait-as-bad at BOTH short and matched length → should learn bait=bad robustly (from v4
pairs) without leaning on length (v5 pairs counterbalance). A different *mechanism* (multi-distribution), not
another style tweak. 1 epoch on 20k = same gradient steps as the 2-epoch 10k runs, each pair seen once (less
overfit). **Gate:** v6 must keep bait/corporate LOW (v4's win) AND |length corr| < ~0.15. If it doesn't beat v4,
**v4 stands and the scorer is LOCKED — no v7.** Then Phase 3 (Writer SFT).

## 2026-06-20 — D15: Scorer v4 verdict — bait fix worked, mild length bias → v5 (length-matched)

**v4 (length-balanced deopt, 9,966 pairs, 2 ep) vs v3 — measured + eyeballed.**

PRIMARY GOAL ACHIEVED — the v3 bait/low-effort confound is gone:
- "lol same" (low-effort): v3 **+6.66** (2nd-highest probe) → v4 **-6.00** (near lowest).
- "RT if you agree" (bait): v3 **+4.28** → v4 **-3.98**. rage-bait +4.88 → -2.03.
- corporate-cringe stays lowest (-6.31); generic-platitude low (-1.83).
- v4 top held-out = substantive tech/intellectual tpot (econ/AI/earnest); bottom = fragments, bare @mentions,
  "Find your audience". A coherent "is this worth posting?" signal — exactly deliverable #2.

NEW FLAW — a mild **length bias**: pearson(score, chars) = **+0.229** (v4) vs **+0.031** (v3, ≈0). Good SHORT
content got under-rated: funny-punchy +8.88→+1.30; aphorism +6.03→**-1.85** (now ≈ tied with generic-platitude).
Cause: v4 made the bait/low-effort degradations SHORT, coupling "short" with "bad" → the model leaned "longer =
better". Two non-English/unicode tweets also leaked to the very top (OOD failure; held-out set isn't language-filtered).

Real-engagement transfer (taste ⟂ engagement, per D13) actually improved slightly: test_pairs 0.515→**0.569**,
topic 0.496→**0.557** (still weak, as expected — we model taste, not engagement). Synthetic held-out: v4 **0.833**
on its own pairs; notably v3 scores **0.890** on v4's pairs (v3's craft signal is fine — its only gap was bait).

DIAGNOSIS → v5. v3 was already length-neutral (+0.03); its ONLY flaw was never seeing bait as a negative. v4
added bait-as-negative but coupled it to SHORTNESS. **v5 = bait/low-effort/vague as negatives BUT every
degradation kept length-MATCHED to the good** → length carries no signal (like v3) AND bait is penalized (like
v4). This also closes a reward-hacking vector before the scorer becomes the RL reward (length bias → verbose-post
hacking, a risk flagged up-front; defense-in-depth alongside the planned RL length penalty). **Gate:** adopt v5
only if it (a) collapses the length corr toward 0, (b) keeps bait/low-effort low, and ideally (c) separates
aphorism > platitude. If v5 regresses, lock the best of {v3, v4, v5} and STOP — no v6.

## 2026-06-20 — D14: Scorer v3 (taste) — qualitative WIN with a length/bait confound → v4

v3 (good-vs-degraded BT, 9,986 pairs): synthetic good-vs-degraded **0.861**; REAL engagement pairs **~0.50**
(chance). The 0.50 is fine — D13 showed engagement≈noise, and v3 measures **taste**, which is ~decorrelated
from engagement here (it down-ranks florid purple-prose tweets that DID get likes — a correct taste call).

**Probes: the v1/v2 failures are FIXED** — corporate-cringe **lowest**, generic-platitude low, funny-punchy +
aphorism **high**. Held-out top = substantive/voice-y; bottom = AI-poetry. ⇒ taste IS learnable and ≈ the
project's actual target ("tpot-**taste**"), in a way raw engagement never was.

**FLAW:** over-rewards terse/bait/low-effort ("lol same", "RT if you agree" score high). Cause: every v3
degradation style was length-preserving-or-LONGER → model learned "shorter = better" (a length confound, the
v3 analog of v1's register confound). **v4 fix:** length-balanced degradation styles (add terse / bait /
low-effort / vague-subtweet) so length doesn't predict quality and bait is an explicit negative. One change
(styles); same training.

## 2026-06-20 — D13: Topic-control (v2) FAILED — within-author craft signal is near-noise (key finding)

v2 (topic-matched pairs, 9.9k, 2 epochs) is **worse than v1 everywhere**: topic test_pairs 0.564 vs v1 0.580;
original test_pairs 0.562 vs v1 **0.611**. Probes scrambled (corporate-cringe highest, funny-punchy lowest).

**Finding (matters more than the run):** both models sit at **~0.56–0.59 on topic-matched pairs ≈ chance**.
Distinguishing the better-*written* version of the SAME idea from text is mostly noise — same-topic engagement
gaps are luck/timing, not extractable craft. v1's 0.61 came from the coarse **emotional-vs-technical register**
axis; remove it (topic-match) → chance. ⇒ **within-author engagement-lift prediction has a ~0.61 ceiling**;
the only strongly-learnable signal is coarse register/topic. More/tighter pairs = optimizing on noise.

**Pivot.** Stop chasing engagement-lift. Reframe the Scorer toward **tpot taste/craft** — more learnable and a
better fit for deliverable #2 ("is it worth posting") and #3 ("improve it"). Candidate v3:
**good-vs-deoptimized contrastive pairs** (LLM degrades a real tpot hit into a bland/cringe version; Scorer
learns crafted-tpot ≻ degraded). Keep `qwen3b-bt-v1` as the current best. Direction TBD with user.

## 2026-06-20 — D12: Iterate the Scorer first (user steer) — topic-controlled pairs (v2)

User: "my sense is to iterate on the scorer." Agreed — the Scorer is deliverable #2 *and* the downstream
reward; fix it before building the Writer on it. **Scorer v2 = topic-controlled same-author pairs**: embed
tweets (MiniLM mean-pool via transformers, no new dep), and for each high-z winner greedily match the most
**topically-similar** low-z loser from the same author (cosine ≥ τ) meeting the v1 margins. The pair then
differs in craft, not subject — directly attacks the D11 register-proxy. Change ONE thing vs v1 (pairs only;
same training) to attribute the effect. If insufficient → v3 multi-task (BT + regression + liked-taste head).
Writer SFT (briefly considered as the next step) deferred until the Scorer is solid.

## 2026-06-20 — D11: Scorer v1 is real but crude (a register proxy) — qualitative > numbers

**Eval.** Held-out pairwise acc 0.611 (temporal) / 0.626 (unseen authors) — above chance and generalizes to
new voices. Per-author Spearman ~0.10 (weak pointwise calibration).

**Qualitative (`inspect_scorer.py`) — the real story.** The Scorer learned an **emotional/relatable vs
technical-niche REGISTER axis**. Top-scored held-out tweets are vulnerability/feelings/relationship/spiritual;
bottom are technical/niche/terse. It *correctly* penalizes engagement-bait and "lol same", but **over-rates
generic platitudes** ("work hard, stay positive, believe in yourself" scored highest of all probes) and rates
a punchy aphorism lowest — and rates many 0–5♥ emotional tweets high (hence Spearman ~0.10). Within-author
pairs leaked a topic confound: the higher-engagement tweet was usually the more emotional one, so the model
learned register, not craft.

**Implication.** Usable as a *weak* v1 reward, but naive RL on it would push the Writer toward
emotional/platitude register (partial reward-hacking). The deferred D9 topic gate is now clearly justified:
Scorer v2 should mine **topic/register-controlled same-author pairs** so the model must learn craft, not topic;
optionally add a regression/taste head + the liked signal. Strengthen before using as the RL reward.

## 2026-06-20 — D10: Scorer uses the SAME base as the Writer (Qwen2.5-3B + reward head) [user pref]

**Supersedes D1/D3's separate-encoder choice.** Scorer = `Qwen/Qwen2.5-3B-Instruct` + a scalar reward head,
4-bit QLoRA, trained with **Bradley-Terry pairwise loss** (TRL `RewardTrainer`) on `train_pairs`
(chosen=winner tweet, rejected=loser tweet; prompt-free — we score standalone post quality).

**Why.** User preference + the decisive technical win: **reward–policy family match**. When we RL the Writer
(same base), the reward shares tokenizer + representations → removes the encoder→decoder mismatch (the main
con of two-model design D1-B) and enables a shared-base + swappable-LoRA-adapter end state (reward adapter =
Scorer, writer adapter = Writer).

**Trade-off (stated).** A 3B scorer is ~100× slower per tweet than a 110M encoder, and a from-scratch
encoder might calibrate ranking slightly better head-to-head. Accepted because we only score eval sets
(~30k) + best-of-N candidates, never the 1.26M corpus (z-labels come from raw engagement, no model). 3B
4-bit QLoRA reward training fits 12 GB easily (tweets are short, seq ≤256). Regression/taste auxiliary head
deferred to a later iteration — first iteration is BT-only, per "let's get a first iteration in".

## 2026-06-20 — D9: Pair-quality refinement from qualitative inspection (don't trust counts)

**Finding.** v1 pairs (33.5k) passed all structural checks (winner>loser likes 100%) but eyeballing
(`inspect_pairs.py`) showed ~half were confounded: media-only "winners" (image + 9 chars beating a
thoughtful 0-like tweet), no topic match, and 5-vs-0 noise. A scorer would learn "short + has image = good".

**Fix.** (1) Strip URLs + unescape HTML → `text_clean`; (2) require **substantive text** both sides
(`text_clean_len ≥ 15`) — kills media-only tweets; (3) absolute floor `min_winner_fav=10` — kills 5-vs-0
noise; (4) skip identical-text (repost) pairs. Result: 28,283 clean pairs; re-inspection shows
punchy/quotable/clear-hook beating obscure/rambly/mundane — real text-attributable tpot signal. fav_w p50 14→23.

**Why accepted.** Lost ~16% of pairs for much higher signal-to-noise. Topic-similarity gating deferred
(same-author+era+substantive-text already removes the dominant confounds); revisit if Scorer underfits.

## 2026-06-20 — D8: Data structure → uploader corpus vs. liked-taste signal (post-EDA)

**Finding.** `archive_upload_id` splits the mirror: **334 uploaders** (6.59M tweets, full timelines) vs.
**303,918 liked/context authors** (4.97M tweets, thin, already tpot-endorsed). Engagement is brutally
long-tailed (fav p50=2, 31% zero). Span 2008→2026. Details: `02-data-findings.md`.

**Decisions.**
- Within-author normalization + same-author BT pairs use **only uploaders** (the only unbiased full
  timelines). ~334 voices → use author-disjoint eval + within-author objective to fight identity leakage.
- Use the **4.97M liked tweets as a curated taste signal** (Scorer taste-positives + SFT voice exemplars).
- Normalize **per (author, era=year)** z-score (not just per-author) — follower growth over 2008→2026 is a
  real confound; EB-shrink thin cells.
- Favorites primary, retweets secondary head (RT 73% zero).
- Drop pure RTs (11.8%) + dedup `tweet_id` (12,859 dups). Maturity buffer: drop post-2026-01-17 from labels.
- Temporal split train<2025-06 / test 2025-06..2026-01-17; plus ~30 held-out uploaders.

**Why this is better.** The liked set turns a follower-confounded like-count problem into a partially
*human-curated taste* problem; uploaders give clean within-author relative-virality pairs. Trade-off:
few distinct voices — accepted, mitigated by held-out-author eval.

## 2026-06-20 — D7: Env = Nix devShell + uv, CUDA via WSL passthrough

**Decision.** Reuse the NSL flake pattern: `python312` + `uv`, `LD_LIBRARY_PATH` to `/usr/lib/wsl/lib`
(host `libcuda.so`), `TRITON_LIBCUDA_PATH`, `stdenv.cc.cc.lib`/`zlib` for binary wheels, `.env` auto-load.
No nixGL, no Nixpkgs torch.
