# Tamil BERT — Personal Roadmap to SOTA

> A note to self. Written so that getting lost mid-project is not an option.
> **The goal:** A Tamil BERT that beats IndicSBERT-STS's Spearman score of **0.82** on the Tamil STS benchmark.
> 
> **Model Artifacts:** [Google Drive Folder](https://drive.google.com/drive/folders/1-fFuIsgYCrKbmJWZA-sdtO_UPyk6Fi1W?usp=sharing)

***

## What Is Being Built Here?

A **BERT-style encoder model for Tamil** that produces high-quality sentence embeddings. When two semantically similar Tamil sentences go in, the cosine similarity of their embedding vectors should be high. When two unrelated sentences go in, it should be low.

This is NOT a generative model — it won't write Tamil. It's a **semantic understanding backbone** that the Tamil NLP community can fine-tune for any of these downstream tasks:

- Semantic search & document retrieval
- Text classification (sentiment, topic, spam)
- Named Entity Recognition (NER)
- Question Answering (extractive span prediction)
- Natural Language Inference (NLI)
- POS tagging & dependency parsing
- Machine Translation (as encoder)
- Text Summarization (encoder-decoder)

Every one of these is a **downstream task** that consumes this BERT as its backbone. Build the foundation once — the community builds on top.

***

## The Full Pipeline at a Glance

```
[Step 1] Gather Tamil Corpus (unlabelled text)
              ↓
[Step 2] Train Tamil Tokenizer  ← uses the unlabelled corpus
              ↓
[Step 3] MLM Pre-training       ← uses the same unlabelled corpus
              ↓
[Step 4] NLI Fine-tuning        ← uses IndicNLI Tamil (labelled)
              ↓
[Step 5] STS Fine-tuning        ← uses indic_sts Tamil TRAIN split only
              ↓
[Step 6] Evaluate               ← uses indic_sts Tamil TEST split (never touched before)
              ↓
         Beat Spearman 0.82 → SOTA Tamil Semantic BERT ✅
```

***

## Step 1 — Gather the Tamil Corpus

Data comes first because the tokenizer needs Tamil text to build its vocabulary, and the model needs Tamil text to pre-train on. Same corpus serves both purposes.

### Where to Get Tamil Text

| Source | What it contains | Link |
|--------|-----------------|------|
| Tamil Wikipedia dump | Encyclopedia articles, clean formal Tamil | [dumps.wikimedia.org/tawiki](https://dumps.wikimedia.org/tawiki/) |
| CC-100 Tamil | Common Crawl filtered for Tamil | [data.statmt.org/cc-100](https://data.statmt.org/cc-100/) |
| Samanantar Tamil | 50M+ Tamil sentences, AI4Bharat | `ai4bharat/samanantar` on HuggingFace |
| Project Madurai | Public domain Tamil literature | [projectmadurai.org](https://www.projectmadurai.org) |

### One Important Split to Make Early

Before doing anything with this corpus — **hold out 10% now**:

```
Full Tamil Corpus
    ├── 90% → tokenizer training + MLM pre-training
    └── 10% → held-out for tokenizer COVERAGE CHECK only
               (can fold back into MLM after coverage is measured)
```

Measuring coverage on training data is pointless (it'll always be ~100%). The held-out chunk gives a real signal.

***

## Step 2 — Train a Tamil-Specific Tokenizer

### Why Not Just Use mBERT's Tokenizer?

Tamil is agglutinative — words compound heavily. mBERT only gives Tamil ~2k vocabulary slots out of 120k total, so it breaks Tamil words into 4–13 pieces. The goal is ~1.5–2.0 pieces per word.

> Think of it this way: imagine English text where "running" gets split into ["r", "un", "nin", "g"]. That's what mBERT does to Tamil. A good Tamil tokenizer should produce ["running"] or at worst ["run", "ning"].

### Training Code

```python
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer()
tokenizer.train(
    files=["tamil_corpus_90pct.txt"],
    vocab_size=32000,               # standard for monolingual BERT
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)
tokenizer.save_model("./tamil-tokenizer/")
```

### Measuring Fertility (on held-out 10%)

```python
words = open("held_out_tamil.txt").read().split()
total_tokens = sum(len(tokenizer.encode(w).tokens) for w in words)
fertility = total_tokens / len(words)
print(f"Fertility: {fertility:.2f}")  # want ≤ 2.5
```

### Measuring Coverage (on held-out 10%)

```python
unknown_count = sum(
    1 for w in words if "[UNK]" in tokenizer.encode(w).tokens
)
coverage = 1 - (unknown_count / len(words))
print(f"Coverage: {coverage:.2%}")  # want > 95%
```

> **Note:** Never check train coverage — it's always ~100% and tells nothing. Only the held-out set matters.

### Tokenizer Targets to Hit

| Metric | Target | Reference |
|--------|--------|-----------|
| Fertility score | ≤ 2.5 tokens/word | Sarvam-30B: 2.35 / Nandi-Mini: 2.05 |
| Vocabulary size | 30k–50k tokens | Standard for monolingual BERT |
| Coverage on held-out | > 95% | % of words not falling back to [UNK] |

***

## Step 3 — MLM Pre-training (on Unlabelled Corpus)

This is where the model learns Tamil — grammar, vocabulary, context. It randomly masks 15% of tokens and learns to predict them from surrounding context.

### Two Approaches

**Option A: Continue pre-training LaBSE** ← Recommended starting point

LaBSE (Language-Agnostic BERT Sentence Embeddings by Google) already covers 109+ languages including Tamil. Continuing pre-training on a large Tamil corpus sharpens its Tamil representations without starting from zero.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("sentence-transformers/LaBSE")
# Swap in the custom Tamil tokenizer, then run MLM training
```

**Option B: Train from scratch**

Full control over vocabulary and architecture. Takes ~24+ hours on a single GPU (based on the Cramming paper). Only worth it if Option A plateaus.

### No hard metric here — just watch MLM loss drop below 2.0.

### On Training Data Overlap

Overlap between the pre-training corpus and fine-tuning data is **totally fine**. The model sees unlabelled text here (predicting masked tokens) and labelled pairs later (learning similarity). Different objectives, same text — no problem. The only thing that must NEVER leak into training is the `sts_test` evaluation split.

***

## Step 4 — NLI Fine-tuning

Raw BERT embeddings are not great for semantic similarity right out of pre-training. NLI fine-tuning teaches the model to map sentences that *entail* each other close together in embedding space, and sentences that *contradict* each other far apart.

This uses the SBERT siamese network approach — the same model encodes both sentences, and a classification head predicts entailment / contradiction / neutral.

### Data

```python
from datasets import load_dataset
nli = load_dataset("ai4bharat/IndicNLI", "ta")
```

Data format:
```
Premise:    "ஒரு மனிதன் வெளியில் நடக்கிறான்"
Hypothesis: "ஒருவர் நடைப்பயிற்சி செய்கிறார்"
Label:      entailment (0)
```

### Training

```python
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer("sentence-transformers/LaBSE")

train_loss = losses.SoftmaxLoss(
    model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=3
)
```

### Expected Result After This Step

Spearman ~0.72–0.74 on Tamil STS. Below target, but that's expected — Step 5 closes the gap.

***

## Step 5 — STS Fine-tuning

The final fine-tuning step. Sentence pairs with human similarity scores (0–5) teach the model to align its cosine similarity with human judgement.

### Data — Train Split Only

```python
from datasets import load_dataset
sts_train = load_dataset("jaygala24/indic_sts", name="en-ta", split="train")
# DO NOT LOAD sts test split here. Not yet.
```

### Training

```python
from sentence_transformers import losses, InputExample

train_examples = [
    InputExample(
        texts=[row["en_sentence"], row["ta_sentence"]],
        label=row["score"] / 5.0   # normalise 0–5 to 0–1 for cosine loss
    )
    for row in sts_train
]

train_loss = losses.CosineSimilarityLoss(model)
```

***

## Step 6 — Evaluation (The Moment of Truth)

**Only now does the test split get loaded.**

```python
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import util
import numpy as np

sts_test = load_dataset("jaygala24/indic_sts", name="en-ta", split="test")

sentences_a = sts_test["en_sentence"]
sentences_b = sts_test["ta_sentence"]

embeddings_a = model.encode(sentences_a, convert_to_tensor=True)
embeddings_b = model.encode(sentences_b, convert_to_tensor=True)

cosine_scores = util.cos_sim(embeddings_a, embeddings_b).diagonal().cpu().numpy()
human_scores = np.array(sts_test["score"])

corr, _ = spearmanr(human_scores, cosine_scores)
print(f"Spearman Correlation: {corr:.4f}")
```

### The Leaderboard to Beat

| Model | Spearman (Tamil STS) |
|-------|---------------------|
| mBERT (vanilla) | 0.49 |
| TamilBERT (vanilla) | 0.59 |
| MuRIL (vanilla) | 0.60 |
| LaBSE (vanilla) | 0.72 |
| L3Cube TamilBERT after NLI | 0.72 |
| IndicSBERT-NLI | 0.74 |
| L3Cube TamilBERT after NLI+STS | 0.80 |
| L3Cube MuRIL after NLI+STS | 0.80 |
| **IndicSBERT-STS (current SOTA)** | **0.82 ← beat this** |

*Source: L3Cube arXiv 2304.11434*

### MTEB Formal Evaluation

To appear on the community leaderboard officially:

```python
import mteb
task = mteb.get_task("TamilNewsClassification")
evaluator = mteb.MTEB([task])
evaluator.run(model, output_folder="results/")
```

***

## Things That Must Not Be Forgotten

### Always Use Mean Pooling — Not pooler_output

`pooler_output` is trained for Next Sentence Prediction (NSP), not semantic similarity. It consistently underperforms mean pooling on STS tasks across all Indic languages per the L3Cube paper.

```python
# ✅ CORRECT — mean pooling over last_hidden_state
import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# ❌ WRONG for semantic similarity tasks
embedding = model_output.pooler_output
```

### Test Set Contamination — Never

If any sentence from `indic_sts` test split appears in pre-training or fine-tuning data, the Spearman score is meaningless. Filter it out explicitly before any training begins.

### Evaluation Is on Sentence Pairs, Not Words

The STS benchmark provides sentence pairs. Encode each sentence separately, compute cosine similarity for the pair, then compare that score to the human label. This is not word-level embedding evaluation.

***

## Baseline Models to Know

| Model | HuggingFace ID | Notes |
|-------|---------------|-------|
| Tamil BERT (L3Cube) | `l3cube-pune/tamil-bert` | Monolingual Tamil BERT base |
| Tamil SBERT NLI | `l3cube-pune/tamil-sentence-bert-nli` | After NLI fine-tuning |
| Tamil SBERT STS | `l3cube-pune/tamil-sentence-similarity-sbert` | Spearman 0.80 on Tamil |
| IndicSBERT-STS (SOTA) | `l3cube-pune/indic-sentence-bert-nli` | Spearman 0.82 — the target |
| LaBSE (backbone) | `sentence-transformers/LaBSE` | Recommended starting backbone |
| MuRIL | `google/muril-base-cased` | Strong multilingual Indic baseline |

***

## Tools & Libraries

| Purpose | Library | Install |
|---------|---------|---------|
| Tokenizer training | `tokenizers` | `pip install tokenizers` |
| Model training | `transformers` + `sentence-transformers` | `pip install transformers sentence-transformers` |
| Dataset loading | `datasets` | `pip install datasets` |
| Spearman / Pearson | `scipy` | `pip install scipy` |
| MTEB evaluation | `mteb` | `pip install mteb` |
| General NLP metrics | `evaluate` | `pip install evaluate` |

***

## All Targets — Quick Reference

| Step | Metric | Target |
|------|--------|--------|
| Tokenizer | Fertility score | ≤ 2.5 tokens/word |
| Tokenizer | Held-out coverage | > 95% |
| Tokenizer | Vocabulary size | 30k–50k tokens |
| MLM pre-training | MLM loss | < 2.0 |
| After NLI fine-tuning | Spearman (Tamil STS) | > 0.74 |
| **After STS fine-tuning** | **Spearman (Tamil STS)** | **> 0.82 ← current SOTA** |

***

## The Bigger Picture

This BERT is step one of a longer journey:

```
Tamil BERT (this repo)
        ↓
Tamil SBERT — sentence embeddings for search & retrieval
        ↓
Task-specific fine-tunes — NER, QA, classification heads
        ↓
Encoder-Decoder — mBART-style summarization & translation
        ↓
Tamil LLM — decoder-only generative model (LLaMA-style)
```

Each stage stands alone as a real contribution. The BERT alone matters — ship it, open source it, let the community use it.

***

*For Tamil. From scratch. One step at a time.*