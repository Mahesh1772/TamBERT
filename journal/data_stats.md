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

## Corpus Profiling Metrics Results

| Metric | project_madurai | tamil_wiki | tamil_cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| Word count | 15.5M | 41.9M | 624.6M | 681.5M | 613.4M | 68.1M |
| TTR | 0.141 | 0.048 | 0.014 | 0.016 | 0.017 | 0.040 |
| MTLD | 135.4 | 63.0 | 218.0 | 190.8 | 203.3 | 548.3 |
| Unigram entropy (bits) | 15.62 | 13.85 | 14.04 | 14.19 | 14.19 | 14.12 |

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


Steps taken
- Cleaning project_madurai_extracted.txt and saving to data\cleaned\project_madurai_extracted_cleaned.txt
Lines written: 1479861, Lines skipped: 206943

| Source                              | Lines dropped | % of source dropped | Lines remaining |
| ----------------------------------- | ------------- | ------------------- | --------------- |
| tamil_wiki_extracted.txt            | 163,735       | 4.41% README.md     | 3,547,113       |
| tamil_wiki_extracted_long_lines.txt | 43            | 81.13% README.md    | 10              |
| project_madurai_extracted.txt       | 4,463         | 0.30% README.md     | 1,475,398       |
| tamil_cc100_extracted.txt           | 174,875       | 0.27% README.md     | 63,941,753      |
| Total corpus                        | 343,116       | 0.50% README.md     | 68,964,274      |