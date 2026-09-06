---
title: Home
layout: default
nav_order: 0
---

# TamBERT — Build Journal

A Tamil BERT-style encoder, trained from scratch, aiming to beat IndicSBERT-STS's Spearman
**0.82** on the Tamil STS benchmark. Not a generative model — a semantic-understanding backbone.

These are the build notes: what was decided, what was measured, and what broke. Each entry covers
one stage of the pipeline, in order, and records the reasoning rather than just the outcome —
including the approaches that were tried and abandoned.

```
corpus  →  tokenizer  →  MLM pre-training  →  NLI fine-tune  →  STS fine-tune  →  eval
  1–2        3–6              8–9                 10             not yet         not yet
```

## Read in order

| # | Entry | What it covers |
|---|---|---|
| 1 | [Data](journal/01_data.md) | Sourcing and cleaning Tamil text from Wikipedia, CC-100 and Project Madurai |
| 2 | [Data Pipeline QA](journal/02_data_stats.md) | The hygiene metrics — contamination, encoding anomalies, duplicate ratio |
| 3 | [Tokenizer Basics](journal/03_tokenizer_basics.md) | Normalizers, pre-tokenizers, and the four preprocessing experiments |
| 4 | [Sandhi Aware Splitting](journal/04_tokenizer_sandhi_split.md) | Marking Tamil phonological boundaries without rewriting the text |
| 5 | [Grapheme Aware Splitting](journal/05_tokenizer_grapheme_split.md) | A fix that silently did nothing, and the placeholder scheme that replaced it |
| 6 | [Tokenizer Training](journal/06_tokenizer_training.md) | How BPE, WordPiece and Unigram actually build a vocabulary |
| 7 | [Tokenizer Evaluation](journal/07_tokenizer_evaluation.md) | Fertility and OOV across all 12 variants, and picking the winner |
| 8 | [MLM Pre-training](journal/08_mlm_pretraining.md) | Masked language modelling, the architecture, and every training argument |
| 9 | [The MLM Training Run](journal/09_mlm_training_run.md) | Losing 420,000 steps of history, and the two numbers that looked like bugs |
| 10 | [NLI Fine-tuning](journal/10_nli_finetuning.md) | Entailment, cross-encoder vs siamese, and a mismatch found while writing it up |

## Where the project stands

| Stage | Status |
|---|---|
| Corpus | Done — 30.7M train / 3.4M test lines, deduplicated |
| Tokenizer | Done — 12 variants trained. Winner `03_sandhi_codepoint_bert`, fertility **1.3414** |
| MLM pre-training | Early-stopped at step 548,000 (19% of the schedule). `eval_loss` **3.7678**, perplexity 43.3 |
| NLI fine-tuning | Written, not yet run |
| STS fine-tuning | Not started |
| Evaluation | Not started |

The MLM target is a loss below 2.0, so there is still **1.77 nats** to go. Entry 9 argues that the
plateau is an artifact of stopping while the learning rate was still near peak, rather than a
capacity ceiling — the decay never annealed.

## The leaderboard being chased

| Model | Spearman (Tamil STS) |
|---|---|
| mBERT (vanilla) | 0.49 |
| TamilBERT (vanilla) | 0.59 |
| LaBSE (vanilla) | 0.72 |
| L3Cube TamilBERT after NLI | 0.72 |
| L3Cube TamilBERT after NLI + STS | 0.80 |
| **IndicSBERT-STS (current SOTA)** | **0.82** |

*Source: L3Cube, [arXiv 2304.11434](https://arxiv.org/abs/2304.11434)*

---

Code and full run artifacts: [github.com/Mahesh1772/TamBERT](https://github.com/Mahesh1772/TamBERT)
