## When Each Check Runs

No — most of these checks should be run **before** the tokenizer is trained. The point is to audit your raw text corpus *first*, so you catch imbalance, redundancy, or noise before it bakes into your vocabulary. Only the perplexity and embedding clustering checks need a model.

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

## Full Collated Test Reference

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

## Practical Order for Your Pipeline

**Before tokenizer**
1. Word/sentence count per source
2. Sentence length distribution
3. Unigram entropy per source
4. MTLD per source
5. JS divergence: each new source vs. existing pool
6. Bigram Jaccard overlap

**After tokenizer**

7. Fertility per source
8. OOV rate per source

**After first MLM run**

9. PPL per domain
10. Embedding clustering

Steps 1–6 together give you a strong, fully pre-model signal of whether a corpus is worth including. If a new source has low MTLD, high JS similarity to your existing pool, and high bigram overlap — you can safely skip it without training anything. LDCIL and EMILLE should score distinctly on JS divergence vs. CC-100 precisely *because* they are curated and institutional.
