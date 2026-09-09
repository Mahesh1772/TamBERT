---
title: Home
layout: default
nav_order: 0
---

# TamBERT — Build Journal

A Tamil BERT-style encoder, trained from scratch, aiming to beat the best published Tamil STS score —
Spearman **0.80**. Not a generative model — a semantic-understanding backbone.

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
| STS fine-tuning | Blocked — no Tamil–Tamil STS dataset exists |
| Evaluation | Blocked — downstream of STS |

The MLM target is a loss below 2.0, so there is still **1.77 nats** to go. Entry 9 argues that the
plateau is an artifact of stopping while the learning rate was still near peak, rather than a
capacity ceiling — the decay never annealed.

Stages 5 and 6 are not simply unwritten. The dataset they need does not appear to exist: `indic_sts`
is `en-XX` only with no `ta-ta` config and no train split, and the leaderboard's own Tamil STS-B is
English STS-B put through Google Translate. The
[README](https://github.com/Mahesh1772/TamBERT#the-missing-piece-a-tamiltamil-sts-set) sets out what
has been ruled out and what would qualify — pointers very welcome.

## The leaderboard being chased

Tamil column only, from L3Cube Tables 1 and 3.

| Model | Vanilla | + NLI | + NLI + STS |
|---|---|---|---|
| mBERT | 0.49 | 0.65 | 0.75 |
| TamilBERT (L3Cube) | 0.59 | 0.72 | **0.80** |
| MuRIL | 0.60 | 0.72 | **0.80** |
| LaBSE | 0.72 | — | — |
| IndicSBERT | — | 0.74 | **0.80** |

The top of the Tamil table is a three-way tie at **0.80**. An earlier version of this page chased
0.82 and credited it to IndicSBERT-STS on Tamil; that was a mis-transcription — 0.82 is Bengali and
Kannada, and Tamil is 0.80 across the board.

*Source: L3Cube, [arXiv 2304.11434](https://arxiv.org/abs/2304.11434)*

---

Code and full run artifacts: [github.com/Mahesh1772/TamBERT](https://github.com/Mahesh1772/TamBERT)
