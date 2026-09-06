---
title: 2. Data Pipeline QA
nav_order: 2
---

# TamilBERT v0.1 — Data Pipeline QA Log

Sources: Project Madurai (classical literature), Tamil Wikipedia, Tamil CC100.  
Metrics used here fall into two groups:

- **Hygiene metrics** (H1–H5): these ask whether the text is structurally clean enough to trust.
- **Corpus profiling metrics**: these ask what the text looks like as language data, such as vocabulary spread, lexical diversity, and sentence shape.

All metrics were run per-source and, where relevant, on merged/train/test files.

---

## Metric definitions

This section is here so the later tables are easier to read. The goal is to make each number interpretable.

### Hygiene metrics

**H1. Contamination rate**

This measures how many disallowed characters appear relative to all characters seen in the file.

$$
\text{Contamination Rate} = \frac{\text{disallowed characters}}{\text{total characters}} \times 100
$$

A high value suggests the corpus contains too many unwanted characters, symbols, or script leakage.

**H2. Encoding anomaly rate**

This measures how many lines contain true encoding anomalies, such as a replacement character or an orphaned combining mark.

$$
\text{Encoding Anomaly Rate} = \frac{\text{anomalous lines}}{\text{total lines}} \times 100
$$

This is intentionally stricter than simple normalization mismatch. NFC mismatch alone is **not** treated as corruption.

**H3. Duplicate ratio**

This measures how many lines are duplicates of earlier lines in the same file.

$$
\text{Duplicate Ratio} = \frac{\text{duplicate lines}}{\text{total lines}} \times 100
$$

In the script this is estimated using a scalable Bloom filter during hygiene checks, then later verified more strictly with exact line hashing during merge-time deduplication.

**H4. Non-alphabetic density**

This measures how much of the corpus is made up of non-alphabetic characters.

$$
\text{Non-Alpha Density} = \frac{\text{non-alphabetic characters}}{\text{total characters}} \times 100
$$

This helps catch markup residue, punctuation-heavy junk, or formatting artifacts, though literary text can naturally score a little higher here.

**H5. Invalid line rate**

A line is considered invalid in this pipeline if it contains fewer than 2 tokens.

$$
\text{Invalid Line Rate} = \frac{\text{lines with token count} < 2}{\text{total lines}} \times 100
$$

This is a practical heuristic, not a universal truth. It catches many useless fragments, but it can also flag valid short poetic lines.

---

### Corpus profiling metrics

**Word count**

No special formula here. This is simply the total number of whitespace-split tokens in the corpus.

**Sentence length distribution**

For each line, compute:

$$
\text{Sentence Length} = \text{number of whitespace tokens in the line}
$$

This is useful for spotting line dumps, broken segmentation, or suspiciously long structural fragments.

**Type-Token Ratio (TTR)**

TTR measures how many unique word types exist relative to the total number of tokens.

$$
TTR = \frac{|V|}{N}
$$

where \(|V|\) is the number of unique word types and \(N\) is the total token count.

TTR is easy to compute, but it is highly sensitive to corpus size. Larger corpora almost always get lower TTR values.

**MTLD (Measure of Textual Lexical Diversity)**

MTLD is a lexical-diversity metric designed to be less sensitive to corpus length than TTR.

It works by scanning through the text and counting a “factor” whenever the running TTR falls below a threshold, here \(\theta = 0.72\).

$$
MTLD = \frac{\text{total tokens}}{\text{number of factors}}
$$

Higher MTLD generally means more sustained lexical variety across the text.

**Unigram entropy**

This measures how evenly the vocabulary is distributed.

$$
H = -\sum_{w \in V} p(w)\log_2 p(w)
$$

where

$$
p(w) = \frac{f(w)}{N}
$$

A higher entropy means token frequencies are spread out more evenly. A lower entropy suggests a smaller set of words dominates the corpus, which often happens with repetitive boilerplate or templated text.

---

### Extra metric added during cleanup

**Real content ratio**

This was introduced after inspection of long Wikipedia lines showed that some of them were mostly structural debris rather than useful Tamil text.

$$
\text{Real Content Ratio} = \frac{\text{tokens containing at least one Tamil character}}{\text{total tokens in the line}}
$$

A value near 1 means the line is mostly Tamil content. A low value usually means the line is dominated by markup residue, symbols, numbering, or mixed-content junk.

The threshold used here was **0.3**.

---

## Run 1 — Hygiene Metrics (pre-clean)

| Metric | madurai | wiki | cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| H1 Contamination | 0% | 0% | 0% | 0% | 0% | 0% |
| H2 Encoding anomaly | 0.043% | 0.055% | 0.040% | 0.041% | 0.041% | 0.039% |
| H3 Duplicate ratio | 16.4% | 5.9% | **53.1%** | **52.1%** | **51.1%** | 35.6% |
| H4 Non-alpha density | 2.5% | 4.0% | 2.6% | 2.7% | 2.7% | 2.7% |
| H5 Invalid line rate | **11.7%** | 1.7% | 2.9% | 3.0% | 3.0% | 3.0% |

H1, H2, and H4 were already calm across the board. The two real findings in this pass were H3 for CC100 and H5 for Project Madurai.

**Changed:**
- **H3** — CC100 duplication, which is typical of Common Crawl style data, propagated into merged/train/test because CC100 dominated the corpus by size. The fix was exact-hash deduplication with `blake2b` on the **merged** corpus rather than per-source, so cross-source duplicates could also be removed.
- **H5** — Madurai’s flagged lines were a mix of true junk and valid short literary lines, such as invocatory phrases, headers, and edition artifacts. Since no single universal fix could separate these perfectly and the percentage was small, the decision was to tolerate some over-flagging rather than aggressively delete short lines.

---

## Run 1 — Corpus Profiling Metrics (pre-clean)

| Metric | madurai | wiki | cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| Word count | 15.5M | 41.9M | 624.6M | 681.5M | 613.4M | 68.1M |
| TTR | 0.141 | 0.048 | 0.014 | 0.016 | 0.017 | 0.040 |
| MTLD | 135.4 | 63.0 | 218.0 | 190.8 | 203.3 | 548.3 |
| Unigram entropy | 15.62 | 13.85 | 14.04 | 14.19 | 14.19 | 14.12 |

TTR inversely tracks source size, which is expected from Heaps’-law-like behavior. The train/test TTR and MTLD divergence also makes sense because the test split is much smaller, so it naturally gets inflated lexical-diversity values relative to train.

`train + test = merged` exactly at this stage, so the split arithmetic is verified.

**Changed:** none — values were within expected range for source size and language mix.

---

## Structural Noise Pass — Content-Ratio Filter

Wikipedia paragraph and sentence-boundary splitting reduced the number of suspicious long lines from 980 to 53, but the remaining lines still looked wrong on inspection. Most were not meaningful long-form Tamil prose. They were collapsed infoboxes, tables, or formatting debris.

To separate these from real content, a new metric was added: **real content ratio**.

$$
\text{Real Content Ratio} = \frac{\text{tokens containing at least one Tamil character}}{\text{total tokens}}
$$

The threshold was set to **0.3**.

This could in theory be set to any fraction, but a stricter threshold would have hurt sources like Project Madurai, where some lines legitimately contain only a few tokens plus a number or structural marker. In practice, 0.3 caught the noisy structural lines while only flagging a very small amount of useful content.

| Source | Avg ratio | Lines dropped | % dropped |
|---|---|---|---|
| wiki | 0.956 | 163,735 | 4.41% |
| wiki_long_lines | 0.189 | 43 | 81.13% |
| madurai | 0.997 | 4,463 | 0.30% |
| cc100 | 0.997 | 174,875 | 0.27% |
| **Total** | — | 343,116 | 0.50% |

Below-threshold samples were manually checked and mostly matched expectations: table and reference-chart debris in Wikipedia, metadata and byline leakage in CC100, and index or concordance entries in Madurai.

An attempt was also made to rescue Madurai lines of the form `verse text - - - - - - 80` by stripping trailing dash runs. That recovered 129 out of 4,592 candidate lines, but the gain was too small to justify more special-case logic.

**Changed:**
- Pipeline stage added: `raw -> clean -> split (above/below 0.3) -> only above-threshold feeds merge`.
- Below-threshold lines were retained separately for audit.
- 10 out of the 53 remaining wiki long lines scored above 0.3 and were merged back in instead of being discarded.

---

## Merge + Deduplication

| Step | Lines |
|---|---|
| Merged (above-threshold sources only) | 68,964,274 |
| After exact-hash (`blake2b`) dedup | 34,093,187 |
| Dropped as duplicates | 34,871,087 (50.6%) |

The roughly 50% drop matches the earlier H3 signal almost exactly. CC100 made up about 92.7% of merged lines, and its duplicate ratio had already been measured at 53.1%, so the final dedup result confirms an existing diagnosis.

Train/test split was then run on the deduplicated corpus using a fixed 90/10 assignment with seed 42.

| Split | Lines |
|---|---|
| train | 30,683,869 |
| test | 3,409,318 |

**Changed:** none — split logic itself stayed unchanged, but it was applied to the deduplicated corpus rather than the noisier pre-dedup version.

---

## Run 2 — Hygiene Metrics (post-clean)

| Metric | madurai | wiki | cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| H1 Contamination | 0% | 0% | 0% | 0% | 0% | 0% |
| H2 Encoding anomaly | 0.050% | 0.004% | 0.041% | 0.065% | 0.065% | 0.067% |
| H3 Duplicate ratio | 6.8% | 19.3% | 53.0% | 0.19% | 0.18% | 0.07% |
| H4 Non-alpha density | 1.8% | 3.1% | 2.5% | 2.4% | 2.4% | 2.4% |
| H5 Invalid line rate | 4.4% | **9.5%** | 2.9% | 0.48% | 0.48% | 0.48% |

**Changed:**
- **H3** — duplicate ratio is near-zero on train (0.18%) and test (0.07%), which validates the exact-hash deduplication step. Per-source rates still look higher because deduplication is only applied at merge time, not written back into each individual source file.
- **H5** — Wikipedia rose to 9.5%, which looks concerning numerically, but manual inspection showed that many flagged lines were valid short section headers such as references and bibliography labels, along with alphabet-entry-like items. Since final train/test files already sit at 0.48%, no extra Wikipedia-specific filter was added.

---

## Run 2 — Corpus Profiling Metrics (post-clean)

| Metric | madurai | wiki | cc100 | merged | train | test |
|---|---|---|---|---|---|---|
| Word count | 14.9M | 39.6M | 623.4M | 437.4M | 393.6M | 43.7M |
| TTR | 0.147 | 0.050 | 0.014 | 0.025 | 0.026 | 0.055 |
| MTLD | 374.3 | 103.9 | 222.3 | 224.7 | 242.2 | 859.6 |
| Unigram entropy | 15.92 | 14.07 | 14.04 | 14.42 | 14.42 | 14.31 |

**Changed:**
- TTR rose across the board after filtering and deduplication, as expected. Train rose from 0.017 to 0.026 and test from 0.040 to 0.055, which is consistent with exact duplicates being removed faster than vocabulary items disappear.
- MTLD rose more than raw word-count reduction alone would predict for Madurai and Wikipedia. That suggests the removed lines were locally repetitive in a way that depressed MTLD disproportionately.
- Token-level word-count drop in CC100, merged, train, and test is smaller than the line-level dedup drop of 50.6%. This implies duplicate lines were shorter than average, which fits the expectation of repeated short headlines, boilerplate, and metadata fragments.

---

## Tests not run yet

The checks above were enough to justify the current pipeline, but a few useful tests from the broader evaluation plan have **not** been run yet. These are worth recording so the dataset card and later versions can clearly separate “tested” from “planned.”

### 6. KL divergence / JS divergence between corpora

This was not run yet, but it would be useful when deciding whether a **new** source adds genuinely new distributional signal or just repeats what is already present.

KL divergence is defined as:

$$
D_{KL}(P \parallel Q) = \sum_{w} P(w)\log\frac{P(w)}{Q(w)}
$$

Since KL is asymmetric and can be unstable when probabilities are missing, the more practical version here is Jensen–Shannon divergence:

$$
D_{JS}(P \parallel Q) = \frac{1}{2}D_{KL}(P \parallel M) + \frac{1}{2}D_{KL}(Q \parallel M), \quad M = \frac{P + Q}{2}
$$

Interpretation:

- \(D_{JS} \approx 0\): the new source looks very similar to the existing pool.
- larger \(D_{JS}\): the new source is distributionally different and may add useful variety.

This would be especially useful before adding future corpora, because it gives a pre-tokenizer signal of whether a source is likely to be redundant.

### 7. Bigram overlap / coverage

This was also not run yet.

$$
\text{Bigram Overlap}(A, B) = \frac{|Bigrams(A) \cap Bigrams(B)|}{|Bigrams(A) \cup Bigrams(B)|}
$$

This is just the Jaccard overlap over bigram sets. High overlap would suggest that a new source adds little new co-occurrence information even if it increases raw token count.

This is a simpler and more interpretable companion to JS divergence when evaluating new corpora.

### 8. Pseudo-perplexity per source

This was not run yet because it needs a trained masked language model.

$$
PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i \mid \text{context})\right)
$$

High pseudo-perplexity on a given source would suggest that the model has not learned that domain well. This is useful later for checking whether literary, encyclopedic, and noisy web text are all being modeled equally well.

### 9. Embedding-space clustering

This was not run yet either.

This test does not need the final TamilBERT to exist. A baseline encoder such as IndicBERT or mBERT can already be used to embed a few hundred sentences per source, after which UMAP or KMeans can show whether the sources occupy meaningfully different regions of embedding space.

Interpretation:

- if sources cluster very tightly together, a new source may not be adding much domain variety;
- if they separate clearly, the corpus mixture is likely bringing in real stylistic or topical diversity.

This is not a hard acceptance criterion, but it is a useful qualitative check before committing to larger-scale data additions.

---

## Known limitations

- Punctuation runs such as `,,,`, `....`, `''''`, and `----` are normalized later at tokenizer level rather than rewritten in source files.
- Repeated `!` and `?` are preserved because they may reflect informal but real usage.
- Dates with imperfect internal separators are left unchanged.
- NFC mismatch alone is not treated as an encoding anomaly.
- TTR, MTLD, and unigram entropy should not be compared too literally across corpora with very different token counts.

---

## Final note

This QA pass was mainly about making the corpus more defensible before tokenizer training. The biggest confirmed fixes were the content-ratio filter for structural noise and exact-hash deduplication for line repetition.

The remaining planned checks, especially JS divergence for new-source selection and post-tokenizer fertility/OOV analysis, are less about basic cleanup and more about deciding whether future additions improve the corpus in a meaningful way. 