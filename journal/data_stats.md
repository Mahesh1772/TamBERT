# TamilBERT v0.1 — Data Pipeline QA Log

Sources: Project Madurai (classical literature), Tamil Wikipedia, Tamil CC100.
Metrics: hygiene (H1–H5) + corpus profiling (word count, sentence length, TTR, MTLD, entropy). Run per-source and on merged/train/test.

---

## Run 1 — Hygiene Metrics (pre-clean)

| Metric | madurai | wiki | cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| H1 Contamination | 0% | 0% | 0% | 0% | 0% | 0% |
| H2 Encoding anomaly | 0.043% | 0.055% | 0.040% | 0.041% | 0.041% | 0.039% |
| H3 Duplicate ratio | 16.4% | 5.9% | **53.1%** | **52.1%** | **51.1%** | 35.6% |
| H4 Non-alpha density | 2.5% | 4.0% | 2.6% | 2.7% | 2.7% | 2.7% |
| H5 Invalid line rate | **11.7%** | 1.7% | 2.9% | 3.0% | 3.0% | 3.0% |

H1/H2/H4 clean everywhere. H3 (cc100) and H5 (madurai) are the two real findings.

**Changed:**
- H3 — cc100 duplication (CommonCrawl-typical) propagates into merged/train/test via its 92% token share. Fix: exact-hash dedup (blake2b) on the **merged** corpus, not per-source, to also catch cross-source dupes.
- H5 — madurai's flagged lines are e-edition headers/invocatory phrases (real words, kept) mixed with bare punctuation/digit-only lines. Few lines of content dropped as universal fix is unfindable and since it was a small % of data decision was made to drop it.

---

## Run 1 — Corpus Profiling Metrics (pre-clean)

| Metric | madurai | wiki | cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| Word count | 15.5M | 41.9M | 624.6M | 681.5M | 613.4M | 68.1M |
| TTR | 0.141 | 0.048 | 0.014 | 0.016 | 0.017 | 0.040 |
| MTLD | 135.4 | 63.0 | 218.0 | 190.8 | 203.3 | 548.3 |
| Unigram entropy | 15.62 | 13.85 | 14.04 | 14.19 | 14.19 | 14.12 |

TTR inversely tracks source size (Heaps' law) — expected. Train/test TTR-MTLD divergence (0.017/203 vs 0.040/548) matches the ~9x token-count gap between them; `train + test = merged` exactly — split arithmetic verified.

**Changed:** none — values within expected range for sample size and language.

---

## Structural Noise Pass — Content-Ratio Filter

Wiki paragraph/sentence-boundary splitting (980 → 53 remaining long lines) surfaced infobox/table debris flattened into running text — no `.!?` or blank-line boundary exists inside a collapsed table to split on. Added `real_content_ratio`: share of tokens containing a Tamil character. 

Threshold 0.3, validated per-source before use. This could be set to any fraction, but as some text from Project Madurai (classical literature) has legitimately lesser tokens per line (such as Aathicudi) with a number at the end. This threshold seemed to be correctly validating most of the text while flagging a few valuable lines as `below threshold`, but as mentioned above it was drooped as this is a samll % of the data.  

| Source | Avg ratio | Lines dropped | % dropped |
|---|---|---|---|
| wiki | 0.956 | 163,735 | 4.41% |
| wiki_long_lines | 0.189 | 43 | 81.13% |
| madurai | 0.997 | 4,463 | 0.30% |
| cc100 | 0.997 | 174,875 | 0.27% |
| **Total** | — | 343,116 | 0.50% |

Below-threshold samples confirmed as table/encoding-reference charts (wiki), metadata/byline leakage (cc100), index/concordance entries (madurai). Attempted rescue of madurai's page-number-leader lines (`verse text - - - - - - 80`) via trailing dash-run strip — recovered 129/4,592 lines; diminishing returns, remainder dropped as-is.

**Changed:**
- Pipeline stage added: raw → clean → split (above/below 0.3) → only above-threshold feeds merge; below-threshold retained separately for audit.
- 10/53 wiki long-lines scored above 0.3 (legitimate long-form content) — merged back in rather than dropped with the rest.

---

## Merge + Deduplication

| Step | Lines |
|---|---|
| Merged (above-threshold sources only) | 68,964,274 |
| After exact-hash (blake2b) dedup | 34,093,187 |
| Dropped as duplicates | 34,871,087 (50.6%) |

~50% drop matches Run 1's H3 prediction: cc100 = 92.7% of merged lines at 53.1% measured dup rate → 63.9M × 0.531 ≈ 33.95M, in line with actual 34.87M dropped. Confirms the known H3 finding, not a new defect.

Train/test split (90/10, seed=42, unchanged shuffle logic) on deduped corpus:

| Split | Lines |
|---|---|
| train | 30,683,869 |
| test | 3,409,318 |

**Changed:** none — split logic unchanged, re-run on deduped input.

---

## Run 2 — Hygiene Metrics (post-clean) — template

| Metric | madurai | wiki | cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| H1 Contamination | | | | | | |
| H2 Encoding anomaly | | | | | | |
| H3 Duplicate ratio | | | | | | |
| H4 Non-alpha density | | | | | | |
| H5 Invalid line rate | | | | | | |

Expected: H3 → near 0% (post-dedup). H5 (madurai) → drop from 11.7% (headers/punctuation-only lines removed pre-merge).

**Changed:**

---

## Run 2 — Corpus Profiling Metrics (post-clean) — template

| Metric | madurai | wiki | cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| Word count | | | | | | |
| TTR | | | | | | |
| MTLD | | | | | | |
| Unigram entropy | | | | | | |

Expected: TTR up post-dedup (fewer repeated tokens per corpus size); word counts down ~50% in cc100/merged/train/test.

**Changed:**

---

## Known Limitations (carry into dataset card)

- Punctuation runs (`,,,` `....` `''''` `----`) normalized at tokenizer level, not in source files.
- `!`/`?` repeats preserved — informal register, not noise.
- Dates missing internal separator (`13.08 2018` vs `13.08.2018`) — left uncorrected.
- NFC-mismatch alone is not an anomaly — normal Tamil vowel-sign composition variance, fixed via normalization, not filtering.
- TTR/MTLD/entropy not directly comparable across differently-sized sources without controlling for token count (Heaps' law; McCarthy & Jarvis 2007/2010).