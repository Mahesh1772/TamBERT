# TamilBERT v0.1 — Corpus QA: Findings & Next Steps

## Context

I built out two QA passes over my TamilBERT training corpus before committing to tokenizer training: a **hygiene metrics** pass (H1 contamination, H2 encoding anomalies, H3 duplicate ratio, H4 non-alphabetic density, H5 invalid/short-line rate) and a **corpus profiling** pass (word count, sentence-length distribution, TTR, MTLD, unigram entropy). Both run per-source (Project Madurai, Tamil Wikipedia, Tamil CC100), and on the merged/train/test files.

## Hygiene Metrics (H1–H5) Results

| Metric | project_madurai | tamil_wiki | tamil_cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| H1 Contamination | 0% | 0% | 0% | 0% | 0% | 0% |
| H2 Encoding anomaly | 0.043% | 0.055% | 0.040% | 0.041% | 0.041% | 0.039% |
| H3 Duplicate ratio* | 16.4% | 5.9% | **53.1%** | **52.1%** | **51.1%** | 35.6% |
| H4 Non-alpha density | 2.5% | 4.0% | 2.6% | 2.7% | 2.7% | 2.7% |
| H5 Invalid line rate | **11.7%** | 1.7% | 2.9% | 3.0% | 3.0% | 3.0% |

*\*Caught a units bug: my hygiene function multiplies `encoding_anomaly_rate` and `invalid_line_rate` by 100 but leaves `duplicate_ratio` as a raw 0–1 fraction. Reading the raw CSV values naively would have badly understated this — I'm listing the corrected percentages above. Fixing this inconsistency in the code is on my next-steps list so it can't hide again.*

### What's clean
- **H1 is 0% everywhere** — confirms the contamination filter I ran during raw→cleaned actually stuck at full scale.
- **H2 sits at 0.04–0.06% everywhere** — well under my 1% threshold, and consistent with defining "anomaly" as replacement-char/orphaned-mark only (excluding NFC-mismatch, which turned out to be normal Tamil vowel-sign composition variance, not corruption).
- **H4 is 2.5–4% everywhere** — comfortably under my 10% threshold.

### What needs action

**H3 — Duplicate ratio.** Once I corrected for the units bug, tamil_cc100 is **53.1% exact-duplicate lines**. Since cc100 makes up ~92% of my merged corpus by token count, this single source is dragging merged/train/test all into the 35–52% range — deep in "Concerning" territory by my own thresholds. This is plausible given CC100/CommonCrawl-derived corpora are known to carry heavy duplication for lower-resource languages, but it needs fixing: an exact-hash dedup pass (not the Bloom filter I used for measurement — that's fine for estimating a rate, but a false positive during actual removal would silently discard real, unique content).

**H5 — project_madurai's invalid-line rate (11.7%).** I pulled actual flagged lines and found they're mostly single-token lines: recurring e-edition markers (மின்பதிப்பு), invocatory phrases (திருச்சிற்றம்பலம்), section headers, and — separately — bare punctuation or lone digits with no actual word (`-`, `,`, standalone numbers). My call: **keep the real-word headers** — they're genuine recurring convention in a classical-literature source, and project_madurai is only ~2.3% of the merged corpus anyway, so their influence on final vocabulary is small. **Remove the bare-punctuation/digit-only lines** — those carry zero linguistic content regardless of source, unlike a repeated real word.

## Corpus Profiling Metrics Results

| Metric | project_madurai | tamil_wiki | tamil_cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| Word count | 15.5M | 41.9M | 624.6M | 681.5M | 613.4M | 68.1M |
| TTR | 0.141 | 0.048 | 0.014 | 0.016 | 0.017 | 0.040 |
| MTLD | 135.4 | 63.0 | 218.0 | 190.8 | 203.3 | 548.3 |
| Unigram entropy (bits) | 15.62 | 13.85 | 14.04 | 14.19 | 14.19 | 14.12 |

- **train (613,365,438) + test (68,122,473) = merged (681,487,911) exactly** — confirms my split arithmetic is internally consistent, no lines lost or duplicated in the split step.
- **TTR decreases monotonically with source size** (project_madurai smallest → highest TTR; tamil_cc100 largest → lowest TTR) — matches Heaps' law exactly, which is a good sign my vocab-building code is behaving correctly, not a red flag.
- **MTLD is sensitive to sample size by design, not a bug in my code.** I confirmed this both empirically (running the same corpus at ~9.6K tokens gave MTLD=2211; the identical corpus at 2M+ lines gave 548) and in the literature: McCarthy and Jarvis (2010) introduced MTLD specifically because raw TTR mechanically decays with length, but even MTLD itself needs a minimum length to stabilize — McCarthy and Jarvis (2007) suggested 100–2,000 tokens as a rough floor, and other studies (e.g. Koizumi 2012) found MTLD is less affected by length than TTR specifically once texts reach at least ~100 tokens. My earlier "broken" 9.6K-token test run was simply below or near that stabilization floor — not a defect.
- **test.txt shows notably higher TTR/MTLD than train.txt** (0.040/548 vs 0.017/203). I initially flagged this as a possible train/test composition mismatch, but confirmed my split code does a proper random per-line shuffle (fixed `seed=42`) across the full merged corpus before assignment — so this isn't a sampling-methodology bug. Given test is ~9x smaller than train in absolute tokens, this divergence is much more likely the same length-sensitivity effect described above rather than a real distributional difference. To confirm rather than assume: I plan to subsample train down to test's exact token count and check whether the numbers converge.
- **tamil_wiki's sentence-length distribution has a serious outlier tail** — individual lines running up to ~70,000 tokens, while project_madurai and tamil_cc100 both cap naturally around 1,400–1,600 tokens with no such tail. This isolates a likely line-splitting failure in my Wikipedia XML extraction (probably paragraph/article breaks not being preserved as newlines during parsing) — needs inspection and a fix, not just a documentation note, since a 70K-token "line" will break anything downstream that assumes reasonable example sizes.

## Decisions Made Along the Way (for the record)

- Repeated punctuation runs (`,,,` `....` `''''` `----`) are normalized via a `tokenizers` normalizer at tokenizer-build time, not by rewriting the raw corpus files.
- Repeated exclamation/question marks (`!!!!` `??`) are intentionally **preserved** — they reflect genuine informal register in scraped news-comment-style content, not noise.
- Some dates lost their internal separator during earlier preprocessing (e.g. `13.08.2018` → `13.08 2018`). Left uncorrected rather than risk introducing false fixes elsewhere — documenting as a known limitation.
- NFC-mismatch alone is **not** treated as an encoding anomaly — confirmed this is normal Tamil vowel-sign composition variance (decomposed vs. precomposed vowel signs like `ெ`+`ா` vs `ொ`), not corruption. Fixed by normalizing, not by flagging/dropping.

## Next Steps (Action Plan)

1. Add a filter to the raw→cleaned step (all 3 sources) that drops lines consisting **only** of punctuation and/or digits — keep genuine real-word short lines (headers, section titles).
2. Isolate tamil_wiki lines exceeding ~2,500 tokens into a separate inspection file; manually determine a re-split rule (or a drop rule for anything unrecoverable); apply the fix to `tamil_wiki_extracted.txt`.
3. Fix the hygiene-metrics code's unit inconsistency — standardize `duplicate_ratio`, `contamination_rate`, and `non_alpha_rate` to the same 0–100 percentage scale as `encoding_anomaly_rate` and `invalid_line_rate`.
4. Re-run per-source cleaning **once**, with fixes #1 and #2 bundled together, so I only pay the reprocessing cost one time.
5. Re-merge the 3 cleaned sources.
6. Run exact-hash (blake2b digest) deduplication on the **merged** corpus, not per-source — this catches cross-source duplicates too (e.g. if cc100 scraped mirrors of wiki content), and is the actual fix for the 35–53% duplicate rate found above.
7. Re-split into train/test using the same shuffle code and `seed=42`, for reproducibility and comparability with this original split.
8. Re-run both hygiene metrics and corpus profiling metrics on all 5 files (3 sources, merged, train, test) as a final verification pass.
9. Confirm: duplicate ratio has dropped substantially; project_madurai's invalid-line rate reflects only genuine short content, not bare punctuation/digits; tamil_wiki's sentence-length distribution no longer shows the ~70K-token tail; TTR/MTLD trends still hold as expected.
10. Update this document with final post-fix numbers before considering the corpus locked for tokenizer training.

## Known Limitations (carry into final dataset documentation)

- Repeated punctuation runs are normalized at the tokenizer level, not in the source files.
- Repeated `!`/`?` are preserved intentionally as authentic informal register.
- Some dates are missing their internal separator (`DD.MM DDDD` instead of `DD.MM.DDDD`) — uncorrected by design.
- TTR, MTLD, and unigram entropy are **not directly comparable across sources of very different sizes** without controlling for token count — this is a documented property of these measures (Heaps' law for TTR; McCarthy & Jarvis 2007/2010 for MTLD's stabilization floor), not a defect in how they were computed here.