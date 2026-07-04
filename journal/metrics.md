## When Each Check Runs

| Check | Needs tokenizer? | Needs LM/encoder? | Run when |
|---|---|---|---|
| Token/word count per source | ❌ | ❌ | Before everything |
| TTR / MTLD | ❌ (whitespace split OK) | ❌ | Before tokenizer |
| Unigram entropy | ❌ | ❌ | Before tokenizer |
| N-gram KL/JS divergence | ❌ | ❌ | Before tokenizer |
| Corpus fertility | ✅ | ❌ | After tokenizer training |
| OOV / coverage rate | ✅ | ❌ | After tokenizer training |
| LM perplexity per domain | ✅ | ✅ (small LM) | After MLM pretraining |
| Embedding clustering | ✅ | ✅ (mBERT/IndicBERT) | Before or after tokenizer |

---

## Test Reference

### Corpus Statistics (Pre-tokenizer)

**1. Raw word/sentence count per source**
- No formula — just whitespace split and count.
- SDK: `wc -w file.txt` or `len(text.split())` in Python.
- Purpose: ensure no single source dominates token budget.

**2. Sentence length distribution**
- Compute mean/median/std of `len(sentence.split())` per source.
- Short mean (< 5 words) signals phrase dumps or boilerplate; long mean (> 60) signals unsegmented text.

---

### Lexical Diversity (Pre-tokenizer)

**3. Type-Token Ratio (TTR)**

$$TTR = \frac{|V|}{N}$$

where $|V|$ is unique word types, $N$ is total tokens (whitespace split).

- Sensitive to corpus length — only compare sources at the same $N$.
- SDK: compute in a sliding window manually; no canonical library.

**4. MTLD (Measure of Textual Lexical Diversity)**

Walks the text left-to-right; counts a "factor" every time TTR drops below a threshold $\theta = 0.72$:

$$MTLD = \frac{\text{total tokens}}{\text{number of factors}}$$

- Length-independent unlike TTR; safe to compare corpora of different sizes.
- SDK: `pip install lexical-diversity` → `from lexical_diversity import lex_div as ld; ld.mtld(tokens)`

**5. Unigram Entropy**

$$H = -\sum_{w \in V} p(w) \log_2 p(w)$$

where $p(w) = \frac{f(w)}{N}$.

- Higher entropy = more evenly distributed vocabulary.
- Low entropy means a few words dominate (likely boilerplate/templates).
- SDK: `scipy.stats.entropy(list(word_freq.values()))`

---

### Distributional Comparison (Pre-tokenizer)

**6. KL Divergence between corpora**

$$D_{KL}(P \| Q) = \sum_{w} P(w) \log \frac{P(w)}{Q(w)}$$

- Asymmetric; use JS divergence for a symmetric, bounded version:

$$D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{P + Q}{2}$$

- $D_{JS} \approx 0$: corpora look the same (redundant addition). $D_{JS} \approx 1$: maximally different.
- SDK: `scipy.spatial.distance.jensenshannon(p_freq, q_freq)`

**7. Bigram overlap / coverage**

$$\text{Bigram overlap}(A, B) = \frac{|Bigrams(A) \cap Bigrams(B)|}{|Bigrams(A) \cup Bigrams(B)|} \quad (\text{Jaccard})$$

- High overlap → new corpus adds little new co-occurrence signal.
- SDK: `collections.Counter` for bigrams, then set intersection/union.

---

### Post-tokenizer Checks

**8. Fertility per source**

$$Fertility = \frac{\text{total subword tokens}}{\text{total whitespace words}}$$

- Target ≤ 2.5 for Tamil; if a specific source spikes (e.g., 4+), it has unusual script mixing or noise.
- SDK: run your trained `BertWordPieceTokenizer` on a sample of each source.

**9. OOV rate per source**

$$OOV\% = \frac{\text{tokens mapped to [UNK]}}{\text{total tokens}} \times 100$$

- Should be < 1% on clean Tamil; a source with 5%+ OOV likely has encoding issues, non-Tamil characters, or very domain-specific jargon.
- SDK: count `[UNK]` in tokenizer output.

---

### Post-model Checks

**10. Pseudo-perplexity per source** (after your small MLM is trained)

$$PPL = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid \text{context})\right)$$

- High PPL on a domain = model hasn't learned that domain's patterns well.
- SDK (with HuggingFace): loop masked token predictions and average log-probability.

**11. Embedding-space clustering** (using mBERT or IndicBERT before you train your own)

- Embed 500–1000 random sentences per source using `ai4bharat/indic-bert` or `bert-base-multilingual-cased`.
- Run KMeans or UMAP + visual inspection.
- SDK: `sentence-transformers` + `sklearn.cluster.KMeans` or `umap-learn`.
- If sources cluster cleanly apart → good diversity; if all blend → you're adding the same domain.

---
| Metric                              | Acceptable                                    | Investigate | Concerning          | How to tackle                                                                                                                                  | Notes                                                                                                                                                                                                                                                  |
| ----------------------------------- | --------------------------------------------- | ----------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| H1. Non-Tamil contamination         | 0–3%                                          | 3–10%       | >10%                | Filter/flag lines by contamination % per source; strip stray Latin/symbols if isolated, or drop the whole source if systemic                   | Some contamination is legit (loanwords, numerals, code-mixing) — check which chars are flagging before deleting, don't blanket-strip                                                                                                                   |
| H2. Encoding anomaly rate           | <1%                                           | 1–5%        | >5%                 | Re-decode with correct source encoding, run NFC normalization pass, drop lines with U+FFFD (unrecoverable)                                     | A high rate often means one source was scraped/decoded wrong — isolate by source before fixing globally, don't patch line-by-line                                                                                                                      |
| H3. Exact duplicate ratio           | <5% aclanthology                              | 5–15%       | >15% youtubegopenai | Streaming hash-set dedup (blake2b/MD5), drop repeats, keep first occurrence                                                                    | Web-scrape corpora typically show 3–14% exact dup rate even in "clean" datasets like C4 aclanthology — some redundancy is normal, don't panic at low single digits                                                                                     |
| H3b. Bloom filter estimate          | Same bands as above, but treat as directional | —           | —                   | If flagged rate is borderline, rerun with exact hash-set on a subsample to confirm                                                             | Bloom filter's error_rate only bounds false positives (reporting new lines as duplicate) — it will never under-report; error_rate=0.001 means ~0.1% of unique lines could be wrongly flagged as dupes, so your true rate is ≤ reported rate, not exact |
| H4. Non-alphabetic density          | <10%                                          | 10–25%      | >25%                | Strip boilerplate/markup patterns (repeated symbols, ad text signatures), inspect high-density lines manually                                  | Punctuation-heavy poetry/scripture (Thirukkural-style) can look "concerning" here without being noise — check content, not just the number                                                                                                             |
| H5. Empty/truncated/short-line rate | <2%                                           | 2–8%        | >8%                 | Drop empty/whitespace-only lines outright; for short lines, spot-check whether they're legit (proverbs) vs. scraping artifacts before dropping | Your corpus has legitimate short lines (Aathichudi) — don't set the word-count threshold too aggressively or you'll delete real content, not noise                                                                                                     |

---
## Practical Order

**Before tokenizer**
1. Word/sentence count per source
2. Sentence length distribution
3. Unigram entropy per source
4. MTLD per source
5. JS divergence: each new source vs. existing pool (for adding new source)
6. Bigram Jaccard overlap

**After tokenizer**

7. Fertility per source
8. OOV rate per source

**After first MLM run**

9. PPL per domain
10. Embedding clustering

Steps 1–6 together give you a strong, fully pre-model signal of whether a corpus is worth including. If a new source has low MTLD, high JS similarity to your existing pool, and high bigram overlap — you can safely skip it without training anything. LDCIL and EMILLE should score distinctly on JS divergence vs. CC-100 precisely *because* they are curated and institutional.
