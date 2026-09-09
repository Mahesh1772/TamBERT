# TamBERT

A Tamil BERT-style encoder, trained from scratch, aimed at the best published Tamil STS score —
Spearman **0.80**. Not a generative model; a semantic-understanding backbone that downstream tasks
(retrieval, classification, NER, QA, reranking) can be built on.

The build log lives in [`journal/`](journal/) and is published at
**[mahesh1772.github.io/TamBERT](https://mahesh1772.github.io/TamBERT)**. Each entry covers one stage
and records the reasoning, the measured numbers, and the approaches that were tried and dropped. This
file is the roadmap: what the plan was, what was actually built, and what is still open.

> **On the 0.80 target.** Earlier versions of this README chased **0.82** and attributed it to
> IndicSBERT-STS on Tamil. That was a mis-transcription. In L3Cube's paper
> ([arXiv 2304.11434](https://arxiv.org/abs/2304.11434)) the Tamil column tops out at 0.80, shared
> three ways; 0.82 belongs to Bengali and Kannada. The corrected leaderboard is
> [below](#the-leaderboard-being-chased).

---

## Where the project stands

| Stage | Status |
|---|---|
| 1. Corpus | **Done** — 30.7M train / 3.4M test lines, deduplicated |
| 2. Tokenizer | **Done** — 12 variants trained. Winner `03_sandhi_codepoint_bert`, fertility **1.3414** |
| 3. MLM pre-training | **Done, short of target** — early-stopped at step 548,000. `eval_loss` **3.7678** (perplexity 43.3) against a target of < 2.0 |
| 4. NLI fine-tuning | **Written, not run** — blocked on an MLM `final/` export that the run never produced |
| 5. STS fine-tuning | **Blocked on data** — see [the missing piece](#the-missing-piece-a-tamiltamil-sts-set) |
| 6. Evaluation | **Blocked** — downstream of stage 5 |

```
corpus  →  tokenizer  →  MLM pre-train  →  NLI fine-tune  →  STS fine-tune  →  eval
 done        done            done             written          no data        blocked
```

Stages 1–3 landed roughly where the plan wanted them, with one substitution: the plan recommended
continuing pre-training from LaBSE and treated from-scratch as the fallback. From-scratch is what was
built, and LaBSE was never touched. Stages 5 and 6 are not "not yet written" — they are waiting on a
dataset that does not appear to exist.

---

## Step 1 — Corpus

Three sources, merged and deduplicated. Samanantar was on the original shortlist and never got used —
these three alone produced more Tamil than the MLM run managed to read even once, so the shortfall was
never compute-limited by corpus size.

| Source | Words (after cleaning) | What it contributes |
|---|---|---|
| [CC-100 Tamil](https://data.statmt.org/cc-100/) | 623.4M | Web-scale volume, and the noisiest input by far |
| [Tamil Wikipedia](https://dumps.wikimedia.org/tawiki/) | 39.6M | Clean formal encyclopedic prose |
| [Project Madurai](https://www.projectmadurai.org) | 14.9M | Public-domain literature; scraped live |

CC-100's duplicate ratio measured **53.0%**, which is most of the reason the merge step exists at all.
After a blake2b dedup pass the combined corpus is **437.4M words / 34.1M lines**, split 90/10 on seed
42 into **30,683,869 train** and **3,409,318 test** lines. Post-merge duplicate ratio: 0.19%,
disallowed-character contamination 0.0%, encoding anomalies 0.065% of lines.

The 10% held-out split was originally meant as a throwaway — measure tokenizer coverage, then fold it
back into pre-training. It stayed held out instead, and became the MLM evaluation set as well. Keeping
it is what makes every fertility and `eval_loss` number on this page mean something.

Two thresholds carry more weight than their size suggests: a **0.3 real-content ratio** filter
(deliberately conservative, so numbered literary lines from Project Madurai survive) and a
**2,500-token long-line cutoff** that catches Wikipedia table remnants. Lines that fail either are
written to audit files rather than dropped silently.

**Run:** `scripts/data_pipeline/01_extract_wiki.py` … `06_calculate_corpus_metrics.py`, in order.
**Read:** [1. Data](journal/01_data.md) · [2. Data Pipeline QA](journal/02_data_stats.md)

---

## Step 2 — Tokenizer

The plan was one WordPiece tokenizer at 32k vocab. What got built is a **12-way ablation**: three
algorithms (WordPiece, BPE, Unigram) crossed with four preprocessing variants — plain, grapheme
placeholders, sandhi markers, and both stacked. The grid is driven by one YAML file per algorithm, so
adding a variant means editing config, not code.

Every variant clears the original targets with room to spare:

| Metric | Target | Best achieved |
|---|---|---|
| Fertility | ≤ 2.5 tokens/word | **1.3414** |
| Held-out coverage | > 95% | **99.98%** (OOV 0.016%) |
| Vocabulary | 30k–50k | 32,000 |

Against the baseline that motivated the whole stage:

| Tokenizer | Fertility | OOV |
|---|---|---|
| mBERT (`bert-base-multilingual-cased`) | 3.3413 | 0.698% |
| `03_sandhi_codepoint_bert` | **1.3414** | 0.016% |

mBERT needs 2.5× the tokens for the same Tamil and produces ~44× the `[UNK]` rate. Spending all 32,000
slots on one language is the entire difference.

Three results the plan did not anticipate:

- **Algorithm mattered ~7× more than preprocessing.** Every Unigram variant lands in 1.497–1.516;
  every BPE and WordPiece variant in 1.341–1.367. The gap between families is ~0.15 tokens/word; within
  a family, ~0.02.
- **Grapheme marking did not help.** It was the most involved piece of work in the stage and it made
  BPE and WordPiece slightly *worse*. Preventing splits inside a grapheme cluster costs more in merge
  flexibility than the linguistic tidiness buys.
- **Sandhi marking helped slightly, and consistently** — it won all six of its pairings, always in the
  third decimal place. Small, but the direction never flipped.
- **In-representation fertility is not raw-text fertility.** Re-measured on unmarked text,
  `03_sandhi_codepoint_bert` goes 1.3414 → 1.4141 (1,253 of its 32,000 entries contain the `⟂` marker
  and become unreachable), and the grapheme variants collapse to ~6.03. Any stage feeding this
  tokenizer must apply the same marking the tokenizer trained with — `scripts/nli/train_nli.py` derives
  that from the variant's own config rather than trusting anyone to remember.

**Run:** `sandhi_precompute.py` → `grapheme_precompute.py` → `train_{bert,bpe,unigram}_tokenizer.py`
→ `tokenizer_stats.py` (all under `scripts/tokenizer/`).
**Read:** [3. Basics](journal/03_tokenizer_basics.md) ·
[4. Sandhi](journal/04_tokenizer_sandhi_split.md) ·
[5. Grapheme](journal/05_tokenizer_grapheme_split.md) ·
[6. Training](journal/06_tokenizer_training.md) ·
[7. Evaluation](journal/07_tokenizer_evaluation.md)

---

## Step 3 — MLM pre-training

`RobertaForMaskedLM` from random initialization — 6 layers, 12 heads, hidden 768 (DistilRoBERTa-sized),
32k vocab, `max_position_embeddings` 516. Effective batch 64, peak LR 1e-4 with linear decay, 15%
masking, on the sandhi-marked corpus.

```
eval_loss   3.7678   (perplexity 43.3)   full 3.4M-line test set
target      < 2.0    (perplexity 7.4)
```

**1.77 nats short**, and stopped at **19.0%** of the 6-epoch ceiling (step 548,000 of 2,879,892).
Early stopping fired, but the linear decay never annealed — LR was still ~8.1e-5 of its 1e-4 peak. Most
of MLM's final quality comes from the LR decaying toward zero, so "stopped improving" here means
stopped improving *at high LR*, not out of capacity. The cheap next experiment is to resume from
`checkpoint-548000` with the epoch ceiling cut to ~1.5 so the decay actually completes.

Two numbers from this run looked like bugs and were not. The logged training `loss` reads 4× high — a
`gradient_accumulation_steps=4` artifact; divided by 4, step 500 gives 9.72 against ln(32000) = 10.37
for a random-init model, exactly right. And warmup ran ~2,002 steps rather than the configured ~172,800,
which changes how a resume must be set up.

The expensive lesson: **this run trained inside a OneDrive-synced folder**, and five of its six
checkpoints came back gutted — the ~270 MB weights and ~540 MB optimizer files missing, the kilobyte
files intact. `checkpoint-548000` has since been recovered whole. `checkpoint-518000`, which is the
*best* model and the one that produced the 3.7678 above, has not. Everything in
`scripts/training_utils.py` — resume validation that checks for `optimizer.pt` and not just a plausible
directory name, a rotation-proof copy of the metric history — exists because of this run.

**Run:** `scripts/mlm/train_roberta.py` (resumes automatically from a validated checkpoint).
**Read:** [8. MLM Pre-training](journal/08_mlm_pretraining.md) ·
[9. The Training Run](journal/09_mlm_training_run.md) ·
[full post-mortem](mlm/03_sandhi_codepoint_bert/README.md)

---

## Step 4 — NLI fine-tuning

Written and reviewed; **not yet run**. It needs `mlm/03_sandhi_codepoint_bert/final/`, which does not
exist — that MLM run predates the weights-only export step, so `scripts/mlm/export_backbone.py` has to
rebuild it from a recovered checkpoint first.

The plan called for the SBERT siamese recipe: `SentenceTransformer` with `losses.SoftmaxLoss` on a LaBSE
backbone. What is written is a **cross-encoder** — `RobertaForSequenceClassification` over the
from-scratch backbone with a fresh 3-way head, on plain `transformers.Trainer`. No
`sentence-transformers` dependency anywhere in the project. The siamese setup produces exactly the
pooled-embedding artifact that STS wants, and is the better choice on paper; the cross-encoder went
first because it exercises the whole export → load → new head → train → export handoff end to end,
which is the chain most likely to be quietly broken. [Entry 10](journal/10_nli_finetuning.md) argues
this out properly and does not fully settle it.

Data is **IndicXNLI** Tamil (~393k train / 3.2k dev pairs), not the `ai4bharat/IndicNLI` the plan named,
and it arrives as a zip via `gdown` rather than through `datasets.load_dataset`. Three epochs, LR 2e-5,
batch 16, max_length 128, selecting on accuracy — the labels are balanced 1:1:1.

Given a perplexity-43 backbone, expectations here should be modest. The plan's "Spearman 0.72–0.74 after
this step" was a number for a LaBSE-based siamese model and does not transfer.

**Run:** `scripts/nli/train_nli.py`.
**Read:** [10. NLI Fine-tuning](journal/10_nli_finetuning.md)

---

## Step 5 — STS fine-tuning

Not implemented. The blocker here is data, not code. The plan was:

```python
load_dataset("jaygala24/indic_sts", name="en-ta", split="train")
```

That call cannot succeed, for two independent reasons, and finding out why is what stalled the stage.

**`indic_sts` has no train split.** It ships `test` only — 1,044 rows for `en-ta`. There is nothing
there to fine-tune on.

**`indic_sts` has no Tamil–Tamil configuration.** Every config is `en-XX`: English paired with one of
twelve Indic languages. Training on those pairs optimizes *cross-lingual alignment*, which is a
different objective from monolingual Tamil similarity — and a particularly bad fit here, since this
backbone's 32k vocabulary is entirely Tamil. An English sentence fed through it comes out as `[UNK]`s
and fragments.

So where do the leaderboard's 0.80s come from? From the L3Cube paper's own translated data: *"To make
the dataset accessible for all ten Indian languages used in this study, we translate it using Google
Translate."* The Tamil STS-B those numbers are measured on is machine-translated English STS-B, with
similarity scores inherited from the English pairs rather than judged by Tamil speakers — and it does
not appear to have been publicly released.

That leaves the bar itself resting on MT output. Beating it with a monolingual model is still a
meaningful result, but the measurement deserves better than translated English.

### The missing piece: a Tamil–Tamil STS set

This is the one thing the project cannot build its way around, and it is where an outside pointer would
help more than any amount of extra training.

Ruled out so far:

| Candidate | Why it doesn't fit |
|---|---|
| `jaygala24/indic_sts`, `mteb/indic_sts` | `en-XX` only, no `ta-ta`; test split only (1,044 rows for `en-ta`) |
| L3Cube's translated Tamil STS-B | Google Translate output of English STS-B; scores inherited from English; not released |
| English `stsb` + MT | Reproduces exactly the problem above, one step further from the source |

What would qualify:

- **Tamil–Tamil sentence pairs** — both sides originally Tamil, not translated into it.
- **Human similarity judgements from Tamil speakers**, on any consistent scale (0–5, 0–1, ordinal).
- Enough rows for a real test split. A few hundred well-annotated pairs would be genuinely useful; a
  train split on top would unblock the stage completely.
- Any license that permits research use.

Adjacent things that would also move this forward: Tamil paraphrase corpora with graded rather than
binary labels, Tamil question-duplicate pairs, or an existing annotation effort that stalled before
release.

If you know of something that fits — or have annotated pairs sitting unpublished, or are thinking about
building a set like this — please [open an issue or a discussion](https://github.com/Mahesh1772/TamBERT/issues).
Getting one good file into the open would unblock the last two stages here, and it would give everyone
else working on Tamil semantics a bar that isn't machine-translated. That seems worth doing together.

---

## Step 6 — Evaluation

Downstream of step 5, so also open. Two decisions are already fixed for whenever it happens.

**Mean pooling over `last_hidden_state`, never `pooler_output`.** The pooler was trained for Next
Sentence Prediction and underperforms mean pooling on STS across Indic languages, per the L3Cube paper.

**The evaluation split is the project's one hard boundary.** Overlap between the pre-training corpus and
NLI or STS *training* data is fine and expected — different objectives over the same text. Test pairs
reaching any training stage would make every number on this page meaningless, so whatever set step 5
ends up using, its test split gets filtered out of training explicitly rather than assumed absent. No
such filter exists yet, because there is nothing yet to filter.

### The leaderboard being chased

Tamil column only, from L3Cube [arXiv 2304.11434](https://arxiv.org/abs/2304.11434) Tables 1 and 3.
Scores are embedding cosine similarity vs. human judgement (Spearman) on their translated Tamil STS-B.

| Model | Vanilla | + NLI | + NLI + STS |
|---|---|---|---|
| mBERT | 0.49 | 0.65 | 0.75 |
| TamilBERT (L3Cube) | 0.59 | 0.72 | **0.80** |
| MuRIL | 0.60 | 0.72 | **0.80** |
| LaBSE | 0.72 | — | — |
| IndicSBERT | — | 0.74 | **0.80** |

The top of the Tamil table is a **three-way tie at 0.80** — monolingual TamilBERT, MuRIL and
IndicSBERT-STS all land there. That is the bar.

| Reference model | HuggingFace ID |
|---|---|
| Tamil BERT (L3Cube) | `l3cube-pune/tamil-bert` |
| Tamil SBERT, NLI | `l3cube-pune/tamil-sentence-bert-nli` |
| Tamil SBERT, STS — 0.80 | `l3cube-pune/tamil-sentence-similarity-sbert` |
| IndicSBERT, NLI — 0.74 | `l3cube-pune/indic-sentence-bert-nli` |
| IndicSBERT, STS — 0.80 | `l3cube-pune/indic-sentence-similarity-sbert` |
| MuRIL | `google/muril-base-cased` |
| LaBSE | `sentence-transformers/LaBSE` |

---

## Running it

```bash
conda create -n tambert_env python=3.11 -y && conda activate tambert_env
pip install -r requirements.txt
pip install -e .      # mandatory — see below
```

The editable install is **not optional**. `pyproject.toml` declares `scripts/` as the package root,
which is the only reason absolute imports like `from paths import Paths` resolve; nothing in `scripts/`
runs without it.

`requirements.txt` targets **cu126** (torch 2.13.0). A Blackwell / RTX 50-series box needs `cu128`,
which caps torch at 2.11.0 — change the index URL and the pin together, and keep the pin at **≥ 2.6**
for the reason documented in that file.

Two things that will bite on Windows: pass `encoding='utf-8'` to every `open()` (cp1252 cannot decode
Tamil at all), and **never set an `output_dir` inside a cloud-synced folder** — see step 3.

On Windows, [`bat_files/`](bat_files/) wraps every stage as a double-clickable script that activates
conda itself — `setup_env.bat`, then `01_data_pipeline.bat` through `05_nli_finetune.bat` in order.
Each one lists the underlying `python scripts/…` commands in its header comment, and
[`bat_files/README.md`](bat_files/README.md) covers the env vars and the pre-flight checks.

### Repo layout

| Path | What it is |
|---|---|
| `scripts/` | All source. Package root — `data_pipeline`, `tokenizer`, `mlm`, `nli` are importable |
| `scripts/paths.py` | Single source of truth for every path in the project. No script hardcodes a filename |
| `journal/` | Per-stage design logs. Read the relevant one before changing a stage |
| `tokenizer/`, `mlm/`, `nli/` | **Output** directories, same names as the code packages under `scripts/`. Check the prefix before editing |
| `data/` | Corpus, in four parallel representations (raw, sandhi-marked, grapheme-marked, both) |

There is no test suite and no linter. Each stage validates itself by writing a metrics JSON or CSV;
that file is the check.

---

## The bigger picture

```
Tamil BERT (this repo)
   → Tamil SBERT — sentence embeddings for search and retrieval
   → task heads — NER, QA, classification
   → encoder-decoder — summarization, translation
   → Tamil LLM — decoder-only
```

Each stage stands alone as a contribution. The encoder alone is worth shipping.

*For Tamil. From scratch. One step at a time.*
