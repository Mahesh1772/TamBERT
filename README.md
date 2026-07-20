# Tamil(+English) BERT — Personal Roadmap to SOTA

> A note to self. Written so that getting lost mid-project is not an option.
> **The goal:** A Tamil-first BERT that beats the Tamil STS SOTA of **0.82 Spearman**.

***

## v1.1 Pivot Note — read this before anything else

v1 was scoped as Tamil-only (corpus, tokenizer, model). Mid-build, three things came up that changed the plan:

1. **The actual benchmark being targeted (`jaygala24/indic_sts`, `en-ta` config) is cross-lingual** — every record pairs an English sentence with its Tamil translation and scores their similarity. A model with zero English exposure can't produce a meaningful score on it: the English half of every pair falls back to `[UNK]`/byte fragments, so the cosine similarity is noise regardless of how good the Tamil side is.
2. **This does not mean the project needs GPT-scale anything.** Cross-lingual sentence similarity is an encoder task, not a generation task — LaBSE does it well at 471M params with zero generative capability. Small dedicated encoders are the standard tool here, not a weaker substitute for something bigger. Bilingual ≠ lower ambition.
3. **A monolingual path does technically exist, and probably produced the 0.82 number in the first place.** The L3Cube papers behind most of the leaderboard below (`L3Cube-IndicSBERT`, arXiv 2304.11434) build their Tamil STS data by machine-translating the English STS-Benchmark (STSb, 8,628 pairs) into Tamil — monolingual Tamil-Tamil pairs, a *different* dataset from `jaygala24/indic_sts`. **Open item:** confirm which exact setup produced the 0.82 figure before treating it as directly comparable to whatever this project measures. Until confirmed, treat the leaderboard table below as directional, not exact.

**Decision for v1.1:** go bilingual (Tamil-majority, English-supporting). Rationale: it's the only path that produces a score on the benchmark as currently coded, and cross-lingual capability is independently useful (retrieval, code-mixed text, MT-adjacent use cases) — not purely a leaderboard-chasing move.

***

## What Is Being Built Here?

A **BERT-style encoder model, Tamil-primary with English support**, producing high-quality sentence embeddings. Two semantically similar sentences (Tamil-Tamil or English-Tamil) should have high cosine similarity; unrelated ones should be low.

This is NOT a generative model. It's a **semantic understanding backbone** the Tamil NLP community can fine-tune for:

- Semantic search & document retrieval (monolingual and cross-lingual)
- Text classification (sentiment, topic, spam)
- Named Entity Recognition (NER)
- Question Answering (extractive span prediction)
- Natural Language Inference (NLI)
- POS tagging & dependency parsing
- Machine Translation (as encoder)
- Text Summarization (encoder-decoder)

***

## The Full Pipeline at a Glance

```
[Step 1] Gather Tamil + English Corpus (unlabelled text)
              ↓
[Step 2] Train Bilingual Tokenizer  ← uses the unlabelled corpus
              ↓
[Step 3] MLM Pre-training           ← uses the SAME unlabelled corpus (Tamil+English, same design as before)
              ↓
[Step 4] NLI Fine-tuning            ← Divyanshu/indicxnli (Tamil, labelled) — dataset ID corrected, see below
              ↓
[Step 5] STS Fine-tuning            ← jaygala24/indic_sts TRAIN split only — confirmed cross-lingual en-ta
              ↓
[Step 6] Evaluate                   ← jaygala24/indic_sts TEST split — never touched before
              ↓
         Beat Spearman 0.82 → SOTA Tamil(+EN) Semantic BERT ✅ (pending benchmark-source confirmation above)
```

***

## Status as of this version

- [x] Step 1 — Corpus gathered, cleaned, deduplicated (Madurai + Wiki + CC-100). QA'd across two full hygiene passes.
- [x] Step 2 — Tamil-only tokenizer trained and evaluated. **Fertility < 1.6 tokens/word on held-out test set** (target was ≤ 2.5 — well clear).
- [ ] Step 2.1 — Extend tokenizer to bilingual vocab (this version's main open task — see Tokenizer section below).
- [ ] Step 3 — MLM pre-training not yet run.
- [ ] Steps 4–6 — not started.

***

## Step 1 — Gather the Corpus (now Tamil + English)

Data comes first because the tokenizer needs text to build its vocabulary, and the model needs text to pre-train on. Same corpus serves both purposes — this doesn't change with English added, it's the same design, just a second language flowing through the same pipeline shape.

### Tamil sources (already collected, see `data.md` for full pipeline detail)

| Source | What it contains | Link |
|--------|-----------------|------|
| Project Madurai | Public-domain literary Tamil | [projectmadurai.org](https://www.projectmadurai.org) |
| Tamil Wikipedia | Formal, edited, informational prose | [dumps.wikimedia.org/tawiki](https://dumps.wikimedia.org/tawiki/) |
| CC-100 Tamil | Common Crawl filtered for Tamil | [data.statmt.org/cc-100](https://data.statmt.org/cc-100/) |

### English sources — to add

| Source | What it adds | Notes |
|--------|--------------|-------|
| English Wikipedia dump | Formal, encyclopedic English | Same extraction shape as Tamil Wikipedia (namespace filter, template/markup strip, long-line split) |
| CC-100 English (or a filtered slice) | Scale, web-register variety | English doesn't need the full CC-100 English volume — this is a *supporting* language, not the primary one. Cap it, don't match Tamil's CC-100 volume 1:1 |

**Target mix: 80–90% Tamil / 10–20% English by volume**, not 50/50. Two reasons:
- The project's whole point is a Tamil-efficient backbone. A 50/50 corpus, combined with a 50/50 vocab split, recreates the exact problem this project exists to fix — mBERT gives Tamil ~2k of 120k vocab slots because Tamil is one of 100+ languages sharing a budget. Diluting your own budget the same way defeats the purpose.
- Cross-lingual alignment (the actual capability needed for Steps 5–6) needs *enough* English exposure to be non-noise, not parity with Tamil.

### Preprocessing — needs a new piece, not a reused one

The existing `real_content_ratio` filter (see `data_stats.md`) checks specifically for **Tamil** characters. It will silently zero out all English lines if reused as-is. Needs a parallel filter checking for Latin/English-word content, run separately on the English source(s), same threshold logic (≈0.3), same audit-file pattern (`above_threshold` / `below_threshold`).

Everything else in the existing pipeline — dedup via `blake2b` hashing, 90/10 train/test split with fixed seed, hygiene metrics (H1–H5) — applies unchanged to the merged Tamil+English corpus.

***

## Step 2 — Tokenizer (bilingual vocab)

### Current state

Trained and measured on Tamil-only text: fertility < 1.6 tokens/word at vocab_size = 32,000 (31,000 learned + special tokens, 1,000 reserved unused). This is a strong result — for reference, most large multilingual models allocate a small fraction of their vocab to Tamil specifically (mBERT: ~2k of 120k).

### v1.1 target: 40,000 total vocab

Multiples of 8 (ideally 64) matter for real reasons, not superstition — NVIDIA's Tensor Core kernels (the ones actually doing the matmuls during training) get meaningfully better throughput when embedding-table dimensions are divisible by 8, with 64-alignment being the safer target for Ampere/Ada-class GPUs. 40,000 clears both (40,000 ÷ 8 = 5,000 exact; ÷ 64 = 625 exact).

| Allocation | Tokens | Notes |
|---|---|---|
| Tamil | 28,000 | ~90% of the original 31k budget kept — still ~14x what mBERT-style models give Tamil |
| English | 10,000 | Enough for common English word/subword coverage for the alignment task — this is a supporting-language budget, not a full monolingual-English-BERT budget (those run ~30k) |
| Special tokens | 5 | `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` |
| Reserved / unused | 1,995 | Same purpose as before — slots for downstream fine-tuning tasks |
| **Total** | **40,000** | |

### How to actually hit this split

Joint BPE/WordPiece training on a mixed Tamil+English corpus does **not** give reliable control over the per-language split — the algorithm just picks the globally most frequent pairs, and won't respect a target ratio on its own. To hit 28k/10k reliably: train two separate vocabularies (Tamil corpus → 28k tokens, English corpus → 10k tokens), then merge into one vocab file, deduplicating shared tokens (digits, punctuation, symbols will overlap).

### Expect fertility to move slightly

Tamil's slice shrank from 31k → 28k, so fertility will likely tick up a bit from the measured 1.6. Re-measure after retraining — don't assume it holds exactly. Still expect comfortably under the 2.5 target given the current margin.

***

## Step 3 — MLM Pre-training

Same corpus as the tokenizer — Step 1's output feeds both Step 2 and Step 3 by design, same as the original plan. Adding English doesn't change this shape; the bilingual corpus just flows through both steps the same way the Tamil-only corpus was going to.

### Two Approaches

**Option A: Continue pre-training LaBSE** — reconsidered and currently **not** the pick for this project. LaBSE is 471M params, ~385M of which is an embedding table tied to its own 500k-token vocab. Swapping in a custom 32k–40k vocab discards that table entirely — not a warm start, closer to a vocabulary transplant onto a pretrained transformer body. Also tight on an 8GB laptop GPU: full LaBSE needs ~7.5GB just for Adam optimizer state before any batch data.

**Option B: Train from scratch** — the pick. BERT-base scale (~110M params) with the custom bilingual vocab is the coherent choice: no embedding mismatch, comfortable on 8GB VRAM, and any downstream gain is attributable to this project's own tokenizer/data choices rather than residual LaBSE knowledge.

### Hardware note (RTX 4060 laptop, 8GB VRAM)

Not comparable in speed to the tokenizer-training step — that was Rust-based CPU string processing (`tokenizers` library); MLM pre-training is GPU tensor math through PyTorch/cuDNN/CUTLASS, bottlenecked by VRAM and memory bandwidth, not CPU threads. The Cramming paper's "24 hours" reference used a desktop RTX 2080Ti (11GB, ~616GB/s bandwidth, ~250W) — the 4060 laptop (8GB, ~256–288GB/s, 35–115W) is behind on the two things that matter most here. Plan for a fixed compute-time budget (Cramming-style: short sequences ~128, mixed precision, gradient accumulation/checkpointing to fit 8GB) rather than a full epoch over the whole corpus, and treat MLM loss < 2.0 as the stopping signal, not epoch count. Realistic expectation: multiple days on this hardware, not one — a cloud GPU rental for this step specifically is worth considering if a faster turnaround matters.

### On Training Data Overlap

Overlap between the pre-training corpus and fine-tuning data (Steps 4–5) is fine — different objectives, same text, no problem. The only thing that must never leak into training is the `sts_test` evaluation split (Step 6).

***

## Step 4 — NLI Fine-tuning

### Data — corrected dataset ID

```python
from datasets import load_dataset
nli = load_dataset("Divyanshu/indicxnli")  # NOT ai4bharat/IndicNLI — that repo ID doesn't exist
# check the dataset card for exact language filtering/config syntax before running
```

`Divyanshu/indicxnli` is English XNLI machine-translated into 11 Indic languages including Tamil — genuinely monolingual Tamil premise/hypothesis pairs, same shape as originally planned. Worth remembering it's MT-Tamil (translationese), same caveat that already applies to CC-100.

Everything else about this step (SBERT-style siamese fine-tuning, `SoftmaxLoss`, expected ~0.72–0.74 Spearman after this step) is unchanged.

***

## Step 5 — STS Fine-tuning

### Data — Train Split Only

```python
from datasets import load_dataset
sts_train = load_dataset("jaygala24/indic_sts", name="en-ta", split="train")
# Confirmed cross-lingual: en_sentence (English) paired with ta_sentence (Tamil)
# DO NOT LOAD the test split here.
```

Unchanged from the original plan, now with the cross-lingual nature explicit rather than assumed.

***

## Step 6 — Evaluation

**Only now does the test split get loaded.** Code unchanged from v1 — see original Spearman correlation snippet against `jaygala24/indic_sts` test split.

### Before trusting this number against the 0.82 target

Confirm whether the L3Cube leaderboard figures were measured on this same `en-ta` cross-lingual setup or on their own translated-STSb monolingual setup (see Pivot Note above). If they're different benchmarks, the numbers aren't directly comparable and this section needs a second target defined explicitly.

### MTEB — confirmed valid

`mteb.get_task("TamilNewsClassification")` is a real, monolingual Tamil task (14.5k train / 2k test) — no issue here, usable as-is for the formal leaderboard step.

***

## The Leaderboard to Beat

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
| **IndicSBERT-STS (current SOTA)** | **0.82 ← target, pending benchmark-source confirmation** |

*Source: L3Cube arXiv 2304.11434*

***

## Things That Must Not Be Forgotten

- **Mean pooling, never `pooler_output`** — unchanged from v1, still applies with the bilingual model.
- **Test set contamination — never.** Filter `indic_sts` test split out of every training stage explicitly.
- **The new English content-ratio filter is not a reuse of the Tamil one** — it needs its own character-detection logic (Step 1).
- **Don't 50/50 the corpus or the vocab** — Tamil stays majority in both, by design, not by accident.
- **Re-measure fertility after the vocab change** — 1.6 was measured at 31k Tamil-only; 28k Tamil-in-a-bilingual-vocab is a different number until proven otherwise.

***

## Baseline Models to Know

| Model | HuggingFace ID | Notes |
|-------|---------------|-------|
| Tamil BERT (L3Cube) | `l3cube-pune/tamil-bert` | Monolingual Tamil BERT base |
| Tamil SBERT NLI | `l3cube-pune/tamil-sentence-bert-nli` | After NLI fine-tuning |
| Tamil SBERT STS | `l3cube-pune/tamil-sentence-similarity-sbert` | Spearman 0.80 on Tamil |
| IndicSBERT-STS (SOTA) | `l3cube-pune/indic-sentence-bert-nli` | Spearman 0.82 — the target (verify benchmark source) |
| LaBSE (reference, not backbone) | `sentence-transformers/LaBSE` | Cross-lingual reference point, not used as starting checkpoint in v1.1 |
| MuRIL | `google/muril-base-cased` | Strong multilingual Indic baseline |

*Not independently re-verified this pass — worth a check before final writeup.*

***

## All Targets — Quick Reference

| Step | Metric | Target |
|------|--------|--------|
| Tokenizer | Fertility score (Tamil) | ≤ 2.5 tokens/word (achieved: <1.6 at 31k Tamil-only; re-measure at 28k) |
| Tokenizer | Vocab allocation | 28k Tamil / 10k English / 5 special / 1,995 unused = 40,000 total |
| Tokenizer | Held-out coverage | > 95% |
| MLM pre-training | MLM loss | < 2.0 |
| After NLI fine-tuning | Spearman (Tamil STS) | > 0.74 |
| **After STS fine-tuning** | **Spearman (Tamil STS)** | **> 0.82 ← pending benchmark-source confirmation** |

***

*For Tamil (and enough English to make the benchmark honest). From scratch. One step at a time.*
