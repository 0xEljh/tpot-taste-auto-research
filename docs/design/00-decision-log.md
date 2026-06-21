# Decision Log

Running log of decisions, rationale, and status. Newest status at top; decisions appended in order.
Companion to `01-system-design.md` (the full design).

---

## STATUS (living)

**Phase:** 6 (REOPENED by user, D28) — go beyond the first pass. v6/DPO-v2 are the *incumbents* to beat. Threads:
(1) de-pollute the goods (curation), (2) demonstrative dipstick → Notion for human sensing, (3) self-play /
LLM-judge scorer ("same model in scoring"), (4) reframe objectives. **Root:** replace the engagement proxy with a
taste-vs-generic-virality signal. **Last updated:** 2026-06-21.

- [~] Phase 6a: goods audit DONE (`scripts/audit_goods.py`, D28) — pollution is real but subtle (platitudes 1.2%
      but +0.305 v6 lift; promo-creep is the bigger leak; v6's top is mostly genuine tpot). Naive heuristic filter
      rejected (low-yield + lossy). Next: demonstrative dipstick → Notion; then LLM-judge labeler (curation + scorer).
- [ ] Phase 6b: demonstrative dipstick eval set (scorer panel + writer panel) → `notion-cat` for human review.
- [ ] Phase 6c: LLM-judge / self-play scorer experiment (3A), calibrated on the dipstick; relabel pairs sans engagement.
- [ ] Phase 6d: re-curate goods with the calibrated judge (1B), retrain scorer, re-run audit + dipstick (before/after).

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
- [x] Phase 2f: Scorer v6 (combined v4∪v5 pairs, 1 ep) — clears the gate (length corr **+0.132** < 0.15) AND
      recovers punch (funny-punchy top) while keeping the key bait/corporate low; best all-round [D17].
- [x] **SCORER LOCKED = v6** → `outputs/scorer/qwen3b-bt-v6-combined` (alias `qwen3b-bt-taste-LOCKED`). No v7.
- [~] Phase 3: Taste Writer SFT (Qwen2.5-3B-Instruct) — design `03-writer-design.md`. **DONE:** TDD tests (7 pass,
      `tests/test_writer_data.py`) + SFT-data builder (`tpot_taste/writer/sft_data.py`, `scripts/build_writer_sft.py`)
      → built **19,970 records** (9,970 ideate + 10,000 improve) `data/splits/writer_sft_train.jsonl`.
      `scripts/train_writer.py` (TRL SFT, 4-bit QLoRA, full-text). **SFT v1 trained** (loss 6.5→1.1) **+ evaluated:
      win-rate 0.67 vs base** (improve 0.85), decisive tpot-voice transfer (base = assistant-slop scored -5..-6 by
      v6) [D18/D19]. Adapter `outputs/writer/qwen3b-sft-v1`; `scripts/eval_writer.py`.
- [x] Phase 4a: DPO DONE — v1 length-hacked (caught by the audit, D21); **v2 = length-penalized best-of-N**
      beats SFT **0.75** (v6) with the hack GONE (112 vs 123 chars) [D22]. **Writer LOCKED = DPO v2**
      (`outputs/writer/qwen3b-dpo-v2`, alias `qwen3b-writer-LOCKED`). Progression base→SFT(0.67)→DPO v2(0.75).
- [x] Phase 4b DONE: GRPO v1 tie (weak push, D25) → v2 (num_gen 8, stronger) **beats v6 0.65 BUT human read =
      platitude-drift** = the **v6 ceiling** (D27); guardrails held (no length/bait/repetition hack). Unsloth probed
      (works, but memory wasn't binding → deferred, D26). **Writer stays LOCKED = DPO v2**; GRPO v1/v2 kept as experiments.
- [x] Phase 5 DONE: `TasteEngine` (`tpot_taste/engine.py`) + `scripts/tpot.py` CLI (score/ideate/improve/repl,
      best-of-N: Writer generates → Scorer ranks), 8 TDD tests; `docs/benchmark.md` scorecard [D23]. End-to-end usable.
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

## 2026-06-21 — D33: Scorer v7.1 (judge-anchored deopt) WORKS — the platitude blind spot is fixed

**v7.1 (`qwen3b-bt-v71-judge-deopt`, 5,317 pairs = 4,320 judge-anchored degradations + 997 judge-taste, 1 ep,
lr 1e-4, r 16).** The fix from D32 landed — and it's the Phase-6 payoff:

| panel category | v6 | v7.1 |
|---|---|---|
| tpot_canon | +1.68 | **+3.08** (top) |
| aphorism | +1.59 | +2.22 |
| **generic_viral (platitude)** | **+2.66 (v6's #1)** | **+1.27 (now below tpot)** |
| promo | +0.91 | **−2.51** |
| bait | +0.46 | −0.76 |

- **The platitude blind spot — the whole reason for Phase 6 — is FIXED**: generic_viral fell from v6's top
  category to below both highs; promo/bait went sharply negative. HIGH-vs-LOW separation gap **+0.65 → +2.59 (~4×)**.
- **Held-out taste acc = 0.616** on *unrelated* real judge-labeled pairs (`taste_v7_test_pairs`) — vs v7 0.54 and
  v7b 0.49 (both chance). It **GENERALIZES**: the register confound is broken; it learned actual taste, not surface.
- **Confirms D32's diagnosis + fix:** the judge as a clean ANCHOR (D31) + v6's same-content, register-matched
  degradation contrast = a learnable taste signal on a de-polluted anchor. Both ingredients were necessary.

**Residual (honest):** `corporate` (+2.39) still edges aphorism (+2.22), so the panel is *technically* BLURRED by
0.17 — a narrow "tidy-takeaway / listicle-thread" weak spot, far smaller than the platitude problem. v7.2 lever:
up-weight corporate degradations / add harder corporate negatives.

**Decision: v7.1 is the new best scorer** (beats v6 on every pollution axis AND generalizes). Full dipstick
(`demo_eval --scorer v71`, pushed to Notion) confirmed it on real data: top real goods are now genuine tpot, the
OOD-non-English over-scoring is fixed (Russian tweets v6 rated high now score low), and the Writer's "work hard"
improve now de-platitudes (+3.05 "useless advice") where v6 amplified it. **LOCKED alias repointed v6→v7.1**
(reversible symlink, v6 preserved) — provisional, pending user review. v7/v7b kept as documented failures. **Next: Phase 6d — propagate the fix into the Writer** (still SFT'd on engagement-polluted
goods, D28): re-curate goods via the judge → re-SFT → re-DPO vs v7.1. The rot is two layers deep (D32 note).

## 2026-06-21 — D32: Scorer v7 (judge-distilled) FAILS — distilling unrelated pairs learns register, not taste

**v7 (`qwen3b-bt-v7-judge`, 997 judge-taste pairs, 2 ep, lr 1e-4).** Train acc **0.966** — but it's a mirage:
- **Held-out taste acc = 0.536 (chance)** on judge-labeled real pairs (`taste_v7_test_pairs`). No generalization.
- **Panel INVERTED** (`score_panel.py`): tpot_canon is v7's **lowest** category (−6.33), bait its highest (−1.68).
- The goods **audit looks better but SPURIOUSLY**: platitude delta flips +0.305 (v6) → −0.455 (v7). v7 demotes the
  whole polished/advice **register** — which includes genuine tpot prose — not taste. One metric (audit) fooled;
  the held-out pairwise + the panel caught it. The project's thesis once more: never trust a single number.

**Diagnosis.** Distilling the judge from ~1000 **unrelated** chosen/rejected real tweets is too hard for a LoRA
scalar head → it overfits a surface **register confound**: real tpot tweets are terser / lowercase / fragmentary,
real generic-viral (news/promo/platitude) is polished / complete, so the head learns *register*, not *taste* (it
even ranks my polished tpot archetypes like "rejected"). Contrast: v6 (deopt) generalized BECAUSE chosen/rejected
were the **same tweet degraded** — a consistent, learnable, register-matched contrast. The judge is fine (D31);
the *distillation target* was wrong.

**Fix — judge-ANCHORED deopt (the synthesis, → v7.1).** Use the validated judge to pick CLEAN tpot anchors
(score ≥ hi — fixes D28's polluted anchor) → degrade each into generic/platitude **at matched length/register**
(same-content contrast = learnable like v6; register-matched = kills the confound). Scales to 5–10k via multiple
degradations/anchor. Combine with the judge-taste pairs for semantic breadth. (Alt B: use the judge DIRECTLY as
the reward — slow but correct, works now. Alt C: just scale the unrelated judge-taste pairs — uncertain, likely
still hard.) First a cheap **diagnostic**: regularized retrain on the same 997 (1 ep, lower lr, r=8) — if held-out
stays ~chance, it's the method (→ judge-anchored deopt), not just overfit. v7 kept as a documented failure.

**Diagnostic result:** v7b (1 ep, lr 5e-5, r=8) → held-out taste acc **0.493 (still chance)**; the panel was less
broken (tpot_canon back on top, generic_viral down to +1.44) but the held-out pairwise stayed flat across BOTH
regularizations. **Confirmed: it's the METHOD, not overfit.** Unrelated-pair distillation can't learn
generalizable taste at this scale → proceeding with judge-anchored deopt (`build_taste_deopt_pairs.py` →
`taste_v71_train_pairs` = anchored degradations + judge-taste breadth → v7.1).

## 2026-06-21 — D31: Judge VALIDATED on real data — it corrects v6 both ways → it becomes the labeler (v7)

**`scripts/judge_vs_v6.py`, 236 real goods, judge vs v6.** Spearman = **+0.202** (they rank very differently —
expected if the judge fixes v6's systematic error). The eyeball is decisive — the judge is right in BOTH
directions on real, messy tweets, not just my archetypes:
- **v6-HIGH / judge-LOW (pollution the judge drops):** crypto shilling ("Jito will be the most valuable
  protocol…"), YC promo ("deadline for applying to the summer batch…"), news headlines ("BREAKING: Federal
  Reserve cuts rates…", avocado imports), political bait, and a **literal hex-hash string v6 scored +1.17**.
  All judge 0–2. These are exactly the engagement-selected non-tpot virality v6 over-rewards.
- **v6-LOW / judge-HIGH (tpot the judge rescues):** "DeepSeek is insane… an excuse to play Factorio with data
  centers" (−0.88→8), "i was procrastinating doing my taxes so we built cursor for excel" (−1.88→7), "two years
  ago I made a shrine to an anime man and then converted it into an altar" (+0.02→10), "enjoying Proofs and
  Refutations and how the characters elegantly diss each other" (−1.54→8). All genuine tpot v6 buried.

**Verdict: the judge is the taste signal we lacked.** It replaces engagement as the label source. The risk
(3B judge inherits the platitude prior) did not bind, on archetypes OR real data.

**Architecture = distillation (the generative judge is too slow to be the reward directly: ~1.5 s/item → hours
for the corpus).** Judge (BATCHED ~10 min / 3k items, D31) labels a pool → train the FAST scalar **BT scorer
v7** on judge-labeled taste pairs → v7 scales over the corpus + serves the RL reward at scalar-head speed. The
judge also doubles as a high-quality inference-time best-of-N ranker (8 cands × 1.5 s = OK interactively).
Added `judge_score_batch` (left-pad batched generation) to make labeling feasible.

**Next:** `scripts/build_taste_pairs.py` — sample a diverse pool (high-z goods to de-pollute + broad uploader
sample to recover buried tpot) → judge-score → form length-balanced taste pairs (chosen=judge-high,
rejected=judge-low, margin gate, per-author cap) → train scorer **v7** → re-run the dipstick + audit (before/after).

## 2026-06-21 — D30: The judge PASSES the panel — a rubric-driven base judge inverts v6's failure

**Result (`scripts/judge_eval.py`, base Qwen + the D29 rubric, no training).** On the SAME dipstick panel v6
fails, the judge cleanly separates taste from generic-virality:

| category | judge (0–10) | v6 (was) |
|---|---|---|
| tpot_canon | **7.80** | +1.68 |
| aphorism | **6.60** | +1.59 |
| generic_viral | **2.80** | **+2.66 (v6's top)** |
| promo / corporate / assistant_slop | **0.00** | +0.9…+2.2 |
| bait | 1.33 | +0.46 |

worst-HIGH 6.60 ≫ best-LOW 2.80 → **CLEAN** (v6 was BLURRED). Pairwise crux (tpot/aphorism vs platitude,
order-robust): **31/32 = 0.97**. **The feared failure mode didn't bind** — I worried a 3B judge would share the
mainstream platitude-loving prior; the rubric steers it out of that. So a *generative, prompted* judge has
better tpot-taste than our *trained* BT scorer. Big: it means the lever (a better taste signal) is reachable
without new human labels — the base model already knows, if asked correctly.

**Caveats (don't over-trust the number).** The panel is MY hand-authored archetypes, and the rubric names the
bad categories explicitly — shared vocabulary inflates the score. Two yellow flags: **18/50 crux pairs were
position-unstable** (36% — the judge flips when order swaps; `agree_winner` conservatively discards them) and
1 pointwise parse-miss. **Archetype-pass ≠ real-data-pass.** The decisive test is whether the judge corrects
v6's *known real errors* (the −5.09 "Earth as a fusion reactor" false-negative; the promo/platitude
false-positives from the audit).

**Next (running): `scripts/judge_vs_v6.py`** scores ~240 real goods with both; the disagreements ARE the
curation decisions (v6-high/judge-low = drop; v6-low/judge-high = rescue). If the judge's corrections eyeball
as right → it becomes the labeler: mine clean taste pairs (no engagement) → train scorer **v7** → re-run the
dipstick (before/after). If real-data is shakier than archetypes → strengthen rubric → few-shot → escalate to a
larger judge (API) for offline labels distilled into the 3B BT scorer. Also TODO: writer-as-judge variant
(D29-C, `--judge-adapter`), since the user suggested "the same model for writing in the scoring."

## 2026-06-21 — D28: Phase 6 reopened (user) — audit the goods for platitude pollution; the diagnosis refines

**User steer (Phase 6).** The first full loop is a good first pass; now go further. Four threads: (1) **data
curation** — some data may be polluting toward platitudes (which work on *broad* Twitter, not tpot); (2) a
**demonstrative dipstick** eval set pushed to Notion (`ncat`=`notion-cat`) for human qualitative sensing; (3) a
**self-play scorer** — "use the same model for writing in the scoring" / proposer-solver-judge; (4) iteratively
**reframe the research objectives**. Discuss-while-training to keep momentum.

**Reframe (the root).** Every wall traces to one fact we already proved — **taste ⟂ engagement** (~0.53 pairwise,
D13) — yet our goods are selected *entirely* by engagement (`z≥1.0` or `fav≥50`). Engagement-selection is a noisy
proxy that over-includes *generic virality* (platitudes/promo/motivational). It propagates **twice**: into the
scorer's `chosen` set AND (via `build_writer_sft.py:33`, ideate goods = `pairs.text_clean_w.unique()`) into the
Writer's imitation targets. So GRPO's platitude-drift (D27) was **rot already in the soil**, not an RL artifact.

**Audit (`scripts/audit_goods.py`, 10k train goods scored by v6).** The hypothesis holds but is *subtler* than
"goods are full of platitudes":
- **Heuristic platitudes are RARE (1.2%) but score HIGHER** — mean v6 **+0.843 vs +0.538** unmarked (**Δ +0.305**).
  So v6 *does* over-reward the motivational register (corroborates the D17/D27 blind spot is partly data-driven).
- **But v6's actual top-25 goods are mostly genuine tpot** — "Swedish word for speed is fart" (+7.62), "nobody is
  more committed to the bit" (+6.38), "how many states can a boolean be in? …at least three" (+4.38). The scorer
  is **not broadly broken**; its problem is the *boundary*, not the bulk.
- **The real leaks are promotional/announcement creep**, not bare clichés: "We're launching the FLF Incubator
  Fellowship 🚀" (+4.31), "Excited to announce the launch of my #AspiringAuthors community" (+3.44), "Wardley
  Mapping … 🗺🤯✅💫 Free for" (+4.31) — engagement-y promo that isn't tpot taste, scored high.
- **The heuristic has false positives**: it flags the genuinely-great "the trick to having a lot of good ideas is
  having even more terrible ideas and being able to tell the difference" (+3.33) as a platitude.

**Decision — do NOT ship a naive heuristic filter (1A).** Evidence kills it: 1.2% yield (removes almost nothing)
*and* lossy (discards real aphorisms). Instead the lever is a **semantic taste-vs-generic labeler** — which is the
**same tool** as the self-play judge (directive 3). So **directives 1 and 3 converge on one artifact** (an
LLM-judge that separates tpot-taste from generic-competent-virality), and the **demonstrative dipstick (2) is its
calibration anchor**. Sequence: dipstick → calibrate judge on it → judge both curates goods (1B) and relabels
pairs (3A). The reframed objective: **discriminate tpot-taste from generic virality**, replacing the engagement
proxy. `audit_goods.py` is the reusable before/after meter.

**Dipstick result (`scripts/demo_eval.py` + `tpot_taste/eval/demo_panel.py`, pushed to Notion for review).**
On clean hand-authored archetypes the failure is **systematic, not rare, and BIDIRECTIONAL**:
- **`generic_viral` (motivational platitudes) is the TOP category: +2.66** > tpot_canon +1.68 > aphorism +1.59
  → the spectrum is **BLURRED** (a "low" category outranks both "high" ones). The two sharpest aphorisms
  ("you don't find your taste, you notice it" / "most advice is autobiography in disguise") score the **lowest**
  of the highs (+0.77) — the scorer under-rates compressed insight and over-rates elaborated uplift.
- **False negatives on real goods**: the **lowest-scored of 600** sampled real goods is a delightful tpot tweet —
  "Thinking about Earth as a giant fusion reactor that spits out food" (**−5.09**); "occasional reminder that
  labmuffin is the scott alexander of skincare" (−3.09). It under-rates terse/deadpan/oblique tpot — *the core
  voice* — and is OOD-blind on non-English (scores Russian tweets +3.8).
- **The Writer trap sprang**: `ideate("discipline")` top candidate = "Disciplined people don't make excuses…"
  (**+4.59**, highest of the whole panel = a platitude); `improve` polishes a platitude draft instead of fixing it.

**Refined axis:** the scorer rewards **legible uplift / advice-shape** over **earned specificity / deadpan** — almost
exactly *anti*-tpot. This sharpens the judge target (D29): the rubric must DEMOTE uplift/advice/promo AND PROMOTE
terse/oblique/specific. The dipstick is now the constant ruler for every future scorer (re-run + diff).

## 2026-06-21 — D29: The taste judge — directives 1+3 converge on one artifact (design + plan)

**Decision.** Build a **taste-vs-generic judge** (`tpot_taste/scoring/judge.py`) — a model that judges *tpot taste*
rather than *engagement* — as the single artifact that serves BOTH curation (drop generic-viral goods) and the
scorer (relabel/mine preference pairs without the engagement proxy). This is the user's "use the same model in
scoring" / proposer-solver-judge, made concrete. Pure helpers TDD'd (`tests/test_judge.py`, 6 pass): rubric
prompt, robust `parse_verdict`/`parse_score`, and `agree_winner` (judge both orders, accept only if they agree →
kills position bias). Pairwise is primary (matches the BT scorer + less miscalibration); pointwise score is for
curation thresholds.

**Approaches considered (full text in the Phase-6 message).**
- **A — LLM-as-judge preference mining (minimal):** rubric-driven judge → train BT scorer on judge prefs not
  engagement. *Risk:* a 3B base judge may share the platitude-loving prior (the very failure). Cheap, testable.
- **B — self-play loop, human-anchored (ideal):** proposer (manufactures tpot-vs-generic contrasts) → solver
  (writer) → judge (calibrated on the dipstick) → prefs train both models → repeat. Unbounded clean data; risk
  of self-reward collapse without the human anchor.
- **C — unified generative reward model (lateral):** one Qwen that writes AND self-scores. Elegant, one artifact;
  role interference + slower scoring.

**Plan: A now, B as destination, C's generative-verifier as B's judge mechanism.** The non-negotiable guard:
the **dipstick is the judge's exam** — a judge is only trusted once it ranks the panel's tpot>platitude (which v6
fails). Next step: `scripts/judge_eval.py` grades the local-Qwen judge on the panel; if 3B is too weak, escalate
the rubric → few-shot → a larger judge (API) for offline labeling, distilled into the 3B BT scorer (v7).

## 2026-06-21 — D27: GRPO v2 beats v6 (0.65) but DRIFTS to platitudes — the v6 ceiling, qualitatively

**GRPO v2 (num_gen 8, lr 2e-6, rep_penalty 3.0, 300 steps) vs DPO v2, judged by raw v6 (n=60):** win-rate
**0.65** (v2 +1.78 vs +1.11); ideate 0.65, improve 0.65. Length 118 vs 126 (no hack); repetition controlled. So
the stronger push DID move the v6 score — v1 was under-trained, not at a hard ceiling.

**BUT the human read flips it.** GRPO v2's top "wins" are generic motivational advice / platitudes v6 over-rewards:
"If you're going to do something, go all in..." (+4.09), "you must read at least one book a week" (+3.98), "the
best way to avoid getting burned is not to get burned" (+1.67). v6 scores *elaborated* advice high — its D17
aphorism≈platitude blind spot (bare clichés score LOW on the probes, but dressed-up ones fool it). So pushing GRPO
hard on v6 drifts the Writer toward v6-pleasing genericness — which a human rates WORSE tpot. The +0.65 is largely
metric-gaming the blind spot, not a true-quality gain.

**The practical v6 ceiling — the project's thesis at the RL frontier:** once the policy is good (DPO v2), the only
way left to raise the v6 NUMBER is to exploit what v6 gets wrong. The length/bait/repetition guardrails held (they
stopped the easy hacks), but they can't cover the platitude blind spot, which is intrinsic to the scorer. **The
lever now is a BETTER SCORER** (human-labeled taste pairs / a calibrated LLM-judge ensemble), not more RL.

**Decision: Writer stays LOCKED = DPO v2.** GRPO v1/v2 kept as documented experiments (`qwen3b-grpo-v1/v2`)
demonstrating the ceiling. Final progression: base → SFT (0.67 vs base) → **DPO v2 (0.75 vs SFT) = shipped**;
GRPO +0.65 on v6 but a qualitative platitude-drift, so not adopted. **Phase 4b DONE — project complete.**

## 2026-06-21 — D26: Unsloth verdict (loads, but memory wasn't the constraint) + GRPO v2 (stronger push)

**Unsloth question (user-raised) — probed directly.** `FastLanguageModel` **loads cleanly on our pinned stack**
(unsloth 2026.4.8 patches Qwen2 on transformers 5.5, 4-bit OK) — viable, no version churn. But two findings
reframe it: (1) **vLLM isn't installed**, so Unsloth's headline win (vLLM-colocate generation) needs a heavy/risky
vllm install; (2) **memory was NOT the binding constraint** — plain TRL fits the stronger config (num_gen=8,
batch 8) on 12 GB, no OOM. So here Unsloth buys *speed* (the ~22 s/step generation), not *feasibility*.
**Decision: defer Unsloth** — adopt only if going bigger (7B) or installing vLLM for many-step GRPO. (Resolves
D18's "reserve Unsloth for where headroom is needed": at 3B / num_gen≤8, headroom wasn't needed.)

**GRPO v2 (stronger push, plain TRL):** num_gen 4→**8** (lower-variance group advantages), lr 1e-6→**2e-6**,
**rep_penalty 3.0** (closes the repetition blind spot v1 exposed; `repetition_ratio` + 2 tests), 200→**300** steps,
from DPO v2. Tests the v1 "weak push" hypothesis: still a tie → we're at the **v6 ceiling** (the lever is a better
scorer, not more RL); a win → v1 was under-trained. Judge on win-rate + length/bait/repetition audit + a human read.

## 2026-06-21 — D25: GRPO v1 ≈ DPO v2 (weak push) — Unsloth v2 to push harder / disambiguate

**GRPO v1 (200 steps, num_gen 4, β 0.04, lr 1e-6, reward = v6 − length − bait) vs DPO v2, judged by raw v6
(n=60):** overall **0.52** (GRPO +1.44 vs DPO v2 +1.31); ideate 0.60, improve 0.35. Length 113 vs 118 (no hack).
Reward stayed flat (~1.0–1.4), KL bounded (~1.9) all through training → the policy barely moved from its DPO v2
start. **Net: a tie.**

Reads: (1) the push was **weak** (low lr, 200 slow ~23 s/step steps, small groups) — can't tell ceiling from
under-trained. (2) Guardrails **held** (no length-hack; GRPO emits less bait/emoji than DPO v2 since its reward
penalizes bait — but the **raw-v6 eval can't see that**, so it under-credits GRPO). (3) Early **degeneration**:
v6 rewards repetition (GRPO's "I'm just going to tell you." ×3 → +1.99); pushing harder on v6 risks amplifying
its blind spots.

**Decision (answers the Unsloth question with data):** to know if GRPO can beat DPO v2, push harder — more steps +
bigger num_generations (lower-variance advantages), which is exactly Unsloth's win (vLLM-colocate speed + memory
headroom; plain TRL *fit* but is slow at num_gen 4). So **GRPO v2 on Unsloth**, + a **repetition penalty** in the
reward (close the v6 blind spot v1 exposed), judged on a **human read** (raw v6 is an unfair judge of a
guarded-reward policy). If Unsloth's FastLanguageModel fights transformers 5.5, fall back to plain-TRL GRPO with
num_gen bumped as far as fits. **Writer stays LOCKED = DPO v2** until something clearly beats it.

## 2026-06-21 — D24: GRPO (Phase 4b) — running; multi-objective reward + a QLoRA-generation dtype fix

GRPO from DPO v2, on-policy: reward = v6 taste score − length penalty (over 200 chars) − bait penalty
(`tpot_taste/writer/grpo_reward.py`, 7 tests). No vLLM (12 GB-tight) — transformers generation. Two blockers
fixed: (1) the TRL import bug → the D20 shim (covers GRPOTrainer too); (2) **QLoRA-generation dtype mismatch** —
kbit-training prep upcasts the frozen norms + lm_head to fp32, but GRPO *generates* (DPO never did), so bf16
hidden states hit an fp32 lm_head → `expected Float found BFloat16`. Fix (`scripts/train_grpo.py`): after trainer
setup, cast the 73 frozen fp32 params back to bf16 (trainable LoRA stays fp32 → training precision unaffected).

Smoke healthy and **fits 12 GB** (policy + reward scorer + generation, no OOM): rewards ~1.0–1.5, KL ~2 (leash
active), completions ~22–35 chars (length guardrail holding — no inflation). Full run: 200 steps, num_gen 4,
β 0.04, lr 1e-6, ~23 s/step (~75 min). **Next:** eval GRPO vs DPO v2 — win-rate + length/bait audit + a careful
HUMAN read (GRPO pushes hardest on v6, so it's the most prone to exploiting v6's blind spots).

## 2026-06-21 — D23: Phase 5 — integrated CLI + benchmark (system is end-to-end usable)

**TasteEngine** (`tpot_taste/engine.py`) lazily loads the locked Writer (DPO v2, CausalLM) + Scorer (v6,
SeqClassification) on separate 4-bit bases; ideate/improve do **best-of-N** (generate N → Scorer ranks → top-k),
so the two models work together — the actual product value. **CLI** `scripts/tpot.py`: score / ideate / improve /
repl (load-once interactive). 8 TDD tests on the pure helpers (build_prompt, rank_topk). Smoke: improve "lit a
fake cig" (-0.17) → ranked improvements +1.88 / +1.53 / +1.10. **Benchmark** `docs/benchmark.md` — the single
scorecard (scorer + writer metrics, the reward-hacking audit, reproduce + use).

**Design (per convention):** chose a unified engine over independent per-command loaders (the best-of-N
integration *is* the deliverable) and over a server (over-engineered for v1; added a thin `repl` for fast
interactive use). **Phase 5 DONE.** Remaining: Phase 4b GRPO — optional SOTA push, judge on a held-out human
read (it pushes harder on v6, so it's more prone to exploiting v6's blind spots).

## 2026-06-21 — D22: DPO v2 (length-balanced) — clean win, hack fixed → Writer LOCKED = dpo-v2

**DPO v2 (length-penalized best-of-N, 1,398 pairs, λ=0.005, 2 ep) vs SFT, judged by locked v6 (n=60):**
overall win-rate **0.75** (v2 +1.51 vs SFT +0.48); ideate 0.75, improve 0.75. **Length audit: 112 vs 123 chars**
— the v1 hack (175 vs 121, +45%) is GONE; v2 is even slightly terser than SFT.

Neutralizing length in best-of-N selection both KILLED the reward-hack AND raised the win-rate (0.67→0.75):
removing the length shortcut forced DPO to learn genuine quality, which generalizes better. v2 train acc 0.64
(< v1's 0.76) is the *expected* sign of a harder, shortcut-free task (cf. the v6 scorer). Qualitatively v2 is
terser, more aphoristic, tpot-voiced ("It seems that most people are only capable of a few emotions").

**Residual (inherited v6 blind spots, NOT DPO failures):** the scorer still occasionally rewards a
platitude-with-emoji ("...- Albert Einstein 🧠 #mindblown", +1.91) and once mis-ranked a good Matrix/Plato take
low — the D17 limits (aphorism≈platitude near the noise ceiling; fooled by some content). DPO faithfully
optimizes v6 so it inherits these; mitigable later via a stronger scorer / judge-time filters, and the GRPO
reward's bait penalty (§3).

**Writer LOCKED = DPO v2** (`outputs/writer/qwen3b-dpo-v2`, alias `qwen3b-writer-LOCKED`). Progression
base → SFT (0.67 vs base) → DPO v2 (0.75 vs SFT). Phase 4a DONE. Next: 4b GRPO (on-policy push) + Phase 5 CLI.

## 2026-06-21 — D21: DPO v1 improves the reward but LENGTH-HACKS — best-of-N length penalty → v2

**DPO v1 (best-of-N, 1,418 v6-judged pairs, 2 ep, lr 1e-5, β 0.1) vs SFT, judged by locked v6 (n=60):**
overall win-rate **0.67** (DPO +1.27 vs SFT +0.74); ideate 0.68, improve 0.65. DPO training was healthy
(rewards/accuracies 0.76, margins ~1.4) — DPO *did* raise the reward.

**But the length audit caught reward-hacking:** DPO outputs averaged **175 chars vs SFT's 121** (+45%).
Qualitative confirms partly length-gaming, not pure quality: DPO drifts to engagement-bait ("...What's up with
that? 🧐 Is it just me? 😂 #TryHardLife", -0.83) where SFT had a crisp aphorism (+2.03), and mangled a greentext
format (-4.75). Root cause: v6 has a mild +0.13 length lean (D17); best-of-N picks the highest-v6 candidate as
`chosen`, which skews long → DPO amplifies "longer = better". The 0.67 win-rate is real-quality + length-gaming.

**Fix (v2) — same lesson as the scorer's length saga (D15-D17):** neutralize length in best-of-N selection,
`adj_score = score - λ·len_chars` (λ=0.005), so `chosen` isn't just the longest. build_dpo_pairs now also dumps
raw candidates (dpo_candidates.parquet) so λ retunes without regenerating. **Gate:** v2 pairs' chosen-len ≈
rejected-len AND v2-beats-SFT keeps a quality win WITHOUT the length blow-up. (The RL reward's explicit length
penalty ε — design §3 — is the complementary guard for GRPO, Phase 4b.)

## 2026-06-21 — D20: TRL 0.24 × transformers 5.5 import bug — root-fixed in a compat shim

Phase 4 blocker: `from trl import DPOTrainer` crashes with `No module named 'mergekit'` (then
`llm_blender`). **Root cause:** transformers 5.5's `_is_package_available()` returns a `(bool, version)`
**tuple**, but TRL 0.24's `is_X_available()` helpers return it directly and guard imports with
`if is_X_available():` — a non-empty tuple is always truthy, so TRL eagerly imports absent optional deps.
**Fix:** `tpot_taste/writer/_trl_compat.py::patch_trl_availability()` coerces the cached
`trl.import_utils._*_available` tuples to plain bools before the trainer imports — zero env change, no
dependency risk (vs. installing mergekit/llm_blender, which could downgrade the pinned transformers==5.5.0,
D6). Call it before any `from trl import DPOTrainer/GRPOTrainer`. Also needed for Phase 4b GRPO.

## 2026-06-20 — D19: Writer SFT v1 works — tpot voice transfer confirmed (win-rate 0.67 vs base)

**Writer SFT v1 (Qwen2.5-3B + LoRA, full-text SFT on 19,970 ideate+improve records, 2 ep, loss 6.5→1.1) vs the
base model, judged by the locked v6 scorer (n=60):** overall win-rate **0.67** (writer +0.33 vs base -0.64);
ideate 0.57; **improve 0.85**.

**Qualitative (the real signal) — a decisive register shift.** Base outputs are textbook assistant-slop the scorer
scores -5..-6: "Absolutely! Here's a thought-provoking post for TPOT: ---", "Hey @tpot, have you ever pondered...",
"Hey there, tech geeks! 🚀", markdown/emojis/preambles. The SFT Writer dropped all of it and writes tpot-native
(lowercase, terse, earnest/wry, no preamble): "it's not even worth it anymore to be angry at anyone"; "the biggest
mistake in American politics is to believe that the economy is not a social good"; "if you meet someone who is
stupid then 99.9% chance they're stupid and not because of you". Voice transfer (deliverable #1 ideate, #3 improve)
clearly worked.

**Bonus — validates the v6 scorer in action:** it strongly penalizes assistant preamble/bait/cringe and rewards
tpot voice, exactly as intended. Known soft spots: (a) the scorer can still be fooled by substantive content that
carries an assistant preamble (a few base "wins" were "Absolutely! Here's... <substantive body>"); (b) some Writer
outputs are a bit generic/sappy or dry factoids. Neither undermines the result.

**Next (Phase 4):** best-of-N → DPO (then GRPO) using v6 as the reward + length/bait/KL guardrails (D5), to push
past SFT; then Phase 5 integration CLI (score/ideate/improve) + final benchmark. SFT v1 is a solid base policy.

## 2026-06-20 — D18: Writer SFT trainer = TRL SFTTrainer + 4-bit QLoRA (not Unsloth, for v1)

**Decision.** Train the Writer with **TRL `SFTTrainer` + peft LoRA + bitsandbytes 4-bit** — the proven scorer
stack on this box — rather than Unsloth. v1 trains **full-text** (render each record through the Qwen chat
template). Completion-only is **deferred**: TRL 0.24 has no importable `DataCollatorForCompletionOnlyLM`, and
`assistant_only_loss=True` needs `{% generation %}` tags Qwen's template lacks — and prompts are short, so
full-text is fine for v1 (revisit with a custom template if the model wastes capacity). Base Qwen2.5-3B-Instruct,
LoRA on attn+MLP (r=16, α=32), 2 epochs, seq≤512, bf16, paged-adamw-8bit, W&B `tpot-taste`. **Smoke-validated**
(load + chat-format + 5 steps, loss 6.5→4.0, no OOM at batch 8) before the full run.

**Why (vs Unsloth, D4's original pick).** Unsloth's win is speed/memory, which matters for **GRPO** (Phase 4,
12 GB-tight), not for 3B SFT (fits comfortably). The TRL+peft+bnb path minimises API risk and matches the scorer
pipeline; **reserve Unsloth for Phase 4** where the headroom is actually needed. Trade-off: slightly slower SFT —
negligible at this scale. **Smoke-test first:** the completion-only collator's response-template token-matching for
Qwen is a known fiddly spot — verify masking on 1–2 steps before the full run (don't burn an hour on a bad mask).

## 2026-06-20 — D17: Scorer LOCKED = v6 (combined v4∪v5 pairs) — best compromise; scorer phase done

**v6 (1 epoch on 19,942 v4∪v5 pairs) clears the D16 gate and is the best all-round scorer → LOCKED here, no v7.**

Probes (v6): funny-punchy **+2.67 (top)**, curious-question +1.60, specific-insight +1.45 | engagement-bait
**-0.92 (lowest)**, aphorism -0.73, earnest -0.62, platitude -0.61, corporate-cringe -0.49 | "lol same" +0.93 and
rage-bait +0.62 only mildly positive (the one soft spot). **Length corr +0.132** (gate <0.15 ✅; v4 was +0.229).

Why v6 over v4 (the incumbent): (1) **length-neutral enough** (+0.13 vs +0.23) — matters most for its downstream
role as the **RL reward** (less verbose-hacking pressure); (2) **recovers punch** — funny-punchy back on top, and
on held-out it correctly scores SHORT high-engagement injokes high ("slate star codesk" 145♥ → +3.73), which v4
under-rated; (3) **cleaner held-out top** — substantive tpot / AI takes, none of the non-English/unicode OOD
garbage v4 leaked to its very top; (4) nails the key bait case (engagement-bait lowest) + corporate/platitude
negative. v4 is more *decisive* (wider probe spread; "lol same"/rage-bait strongly negative) but its length lean +
punch under-rating + OOD-top-leak cost more than v6's milder "lol same"/rage-bait handling. As both a judge
(deliverable #2) and an RL reward, v6's *shape* is better.

Residual limits (documented, mitigated downstream): (a) "lol same"/rage-bait only mildly negative — a product
threshold (~+1.2) still ranks them below substantive content; (b) aphorism still under-rated — no version fixed
short-aphorism vs platitude (that fine-craft axis is near the D13 noise ceiling); (c) taste ⟂ engagement (real
pairs ~0.53), as expected — we model taste, not raw virality; (d) synthetic acc 0.72 (< v4/v5 — v6 doesn't overfit
either degradation shortcut; the least-important metric). All scorer adapters (v1,v3,v4,v5,v6) kept under
`outputs/scorer/` for reproducibility. **LOCKED = `qwen3b-bt-v6-combined`.** On to Phase 3 (Writer).

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
