# Phase-1 Data Findings (Community Archive)

EDA on `data/raw/community-archive.parquet` (11,557,482 tweets). Reproduce: `scripts/eda.py`.
These numbers set the concrete Phase-1 thresholds. See `01-system-design.md` §1 for the design.

## Corpus shape

- **11.56M tweets**, **303,938 distinct `account_id`**, span **2008→2026-02** (a few junk 1993/2006 stamps).
- Explosive recency: 2019:284k · 2020:675k · 2021:914k · 2022:1.08M · 2023:1.26M · 2024:1.39M · **2025:3.63M** · 2026:1.50M (capture ≈ 2026-02-17).
- Structure: **57.5% replies**, 10.3% quotes, **11.8% pure retweets** (`RT @…`), 27.4% contain a URL.
- Text: p50 = 100 chars / 16 words; p90 = 271 chars / 44 words; max 25k (long-form/threads).

## Engagement is brutally long-tailed (raw counts are NOT a usable target)

| metric | p50 | p90 | p99 | max | % zero |
|---|---|---|---|---|---|
| `favorite_count` | **2** | 117 | 16,526 | 6.58M | 31% |
| `retweet_count` | **0** | 17 | 2,712 | 2.98M | 73% |

→ Favorites are the denser/more-reliable signal; retweets are sparse (73% zero) → secondary head.
→ Virality is the rare upper tail; the signal is **within-author relative**, not absolute.

## The two sub-populations (decisive)

`archive_upload_id` cleanly separates the mirror:

| subset | rows | authors | median fav | role |
|---|---|---|---|---|
| **uploaders** (`archive_upload_id` not null) | 6.59M | **334** | 1.0 | opted-in tpot accounts; **full timelines** ⇒ the only unbiased within-author corpus |
| **liked/context** (`archive_upload_id` null) | 4.97M | 303,918 | 6.0 | tweets tpot users liked/quoted ⇒ **human-curated taste signal**, thin per author (195k authors ≤3 tweets) |

- Top uploaders (the canon): eigenrobot 290k · visakanv 262k · goblinodds 229k · DanielleFong 202k · ultimape 150k · mykola 148k · DRMacIver 144k · aliceisplaying 141k · nosilverv 85k · tasshinfogleman 78k …
- Author history depth (all authors): ≥100 tweets → 7,671 · ≥200 → 3,869 · ≥500 → 1,516 · ≥1000 → 775.
- 12,859 duplicate `tweet_id` rows (0.1%).

## Refined Phase-1 plan (thresholds)

1. **Dedup** on `tweet_id` (keep first); drop pure retweets (`full_text` starts `RT @`); drop null/empty `full_text` (9).
2. **Uploader corpus** = all tweets whose `account_id` ∈ {accounts with any non-null `archive_upload_id`} (≈334 authors, full timelines). This is the within-author normalization + same-author pairing source.
3. **Normalized engagement label** (regression head): `e = log1p(fav) + 0.5·log1p(rt)`; per-author z-score **within era** (per calendar year) to absorb follower growth over 2008→2026; empirical-Bayes shrinkage toward the author-global then global mean for thin (author,year) cells (use cells with ≥ ~10 tweets; else shrink).
4. **Same-author preference pairs** (BT head, primary): same uploader, same era window (≤180 days), similar topic (shared URL/hashtag or TF-IDF cosine > 0.5), margin `fav_w ≥ max(2·fav_l, fav_l+5)` **and** ≥ ~0.75 author-z gap; exclude near-dups; cap ≤ K pairs/author (inverse-frequency weight) so eigenrobot doesn't dominate.
5. **Taste signal** (Scorer aux + SFT exemplars): liked/context tweets (tpot-endorsed) as taste-positives; pair against engagement-matched uploader tweets that were *not* widely endorsed for a cross-population "taste" head (secondary, normalized).
6. **Maturity buffer**: drop tweets newer than ~30 days before capture (post-2026-01-17) from labels (engagement not saturated).
7. **Frozen splits**: temporal (train < 2025-06-01 · test 2025-06-01..2026-01-17, mature) + **author-disjoint** (hold out ~30 uploaders entirely) + `test_pairs`. Report per-author & per-reach-bucket.

### Caveat to carry forward
Only **~334 distinct uploader voices** → real author-identity-leakage risk for the Scorer and limited stylistic diversity for the Writer. Mitigations: author-disjoint eval, within-author pairwise objective (targets relative quality not identity), and the 4.97M liked tweets for aesthetic breadth. Flag in every Scorer eval whether per-author metrics hold on **held-out** authors.
