# TamilBERT v1.0 — NLI Fine-tuning

The MLM stage produced a model that understands Tamil in a general way — it can fill in blanks sensibly. What it
cannot do is tell you whether two Tamil sentences mean the same thing, which is the entire point of the project.
Raw pre-trained embeddings are famously mediocre at similarity: nothing in the masked-token objective ever asked
the model to place two differently-worded sentences with the same meaning near each other.

This stage is the first that pushes the model toward that. The script is `scripts/nli/train_nli.py`. As of
writing it has **never been run**, so this entry documents the design and its reasoning rather than results.

---

## What natural language inference is

Given two sentences — a **premise** and a **hypothesis** — decide which of three relationships holds:

| Label | ID | Meaning |
|---|---|---|
| entailment | 0 | If the premise is true, the hypothesis must be true |
| neutral | 1 | The hypothesis might be true; the premise doesn't settle it |
| contradiction | 2 | If the premise is true, the hypothesis must be false |

```
Premise:    ஒரு மனிதன் வெளியில் நடக்கிறான்      ("a man is walking outside")
Hypothesis: ஒருவர் நடைப்பயிற்சி செய்கிறார்      ("someone is taking a walk")
Label:      entailment
```

The `ID2LABEL` mapping in the script follows the XNLI convention, and the script's comment records that the
order was confirmed against sampled examples from the notebook rather than assumed. That check is worth doing,
because a silently permuted label order produces a model that trains to convergence and is wrong about
everything, with no error anywhere.

### Why NLI comes before STS

The task the project is judged on is STS — rating sentence pairs on a 0–5 similarity scale and correlating those
ratings against human judgement. There is Tamil STS training data available, so it's fair to ask why this stage
exists at all.

The answer is in the leaderboard the README is chasing:

| Setup | Spearman on Tamil STS |
|---|---|
| TamilBERT, no fine-tuning | 0.59 |
| TamilBERT + NLI only | 0.72 |
| TamilBERT + NLI + STS | 0.80 |
| IndicSBERT-STS (the target) | 0.82 |

NLI alone moves the score by 0.13. NLI followed by STS gets to 0.80. The reason is data volume: the Tamil STS
training split is small, and IndicXNLI Tamil has roughly 393,000 labelled pairs. NLI does the heavy lifting of
teaching the embedding space a coarse notion of "these two sentences are about the same thing", using two orders
of magnitude more supervision than STS can offer. STS then calibrates that space against human scores.

Entailment is a useful proxy for this because entailed pairs are, by construction, semantically close, and
contradictory pairs are semantically related but opposed — which is a much harder and more informative signal
than randomly-paired unrelated sentences.

---

## The data

IndicXNLI Tamil, which is XNLI machine-translated into Indic languages by AI4Bharat. The script downloads it as
a zip from Google Drive via `gdown`, rather than through `datasets.load_dataset`:

```python
NLI_GDRIVE_ID = '196RTClD2_Yg7xpXhQ3Mn0FVSD9vBeuBo'
```

It extracts to `data/nli/indicxnli_tamil/` as three JSON files — `train.json`, `dev.json`, `test.json` — and
skips the download entirely if all three are already present.

One quirk is handled explicitly and is exactly the kind of thing that wastes an afternoon:

```python
_SPLIT_KEYS = {'train': 'train', 'dev': 'validation', 'test': 'test'}
```

`dev.json` stores its rows under a top-level key called `validation`, not `dev`. The filename and the key inside
the file disagree, so the mapping has to be stated rather than derived from the filename.

The loader renames columns to `sentence1` / `sentence2` / `gold_label`, and every file is opened with
`encoding='utf-8'` — the project-wide rule, since Windows defaults `open()` to cp1252, which cannot decode Tamil
at all.

The script then prints row counts and a per-label breakdown for each split before doing anything else. The
label distribution is worth seeing, because XNLI is balanced roughly 1:1:1 across the three classes, and that
balance justifies a decision further down about which metric to optimize.

---

## Loading the backbone

```python
backbone_dir = paths.mlm_run_generator(BACKBONE_RUN) / 'final'
```

`BACKBONE_RUN` is `'03_sandhi_codepoint_bert'` — the MLM run from the previous entry. The script checks the
directory exists and raises a clear error naming the fix if it doesn't, because **as of writing it does not
exist**. That MLM run predates the export step, so `export_backbone.py` has to be run first. This is the one
hard prerequisite for this stage.

```python
model = RobertaForSequenceClassification.from_pretrained(
    str(backbone_dir),
    num_labels=len(ID2LABEL),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
```

The MLM model was a `RobertaForMaskedLM`: an encoder plus a head that predicts a token from the 32,000-word
vocabulary at every position. Loading those same weights as `RobertaForSequenceClassification` keeps the encoder
and discards the language-modelling head, replacing it with a freshly initialized classifier that reads one
vector and emits three logits.

Transformers prints a warning about newly initialized weights when this happens. **That warning is the intended
behaviour, not a problem** — a randomly initialized head is exactly what a new task needs. Worth internalizing,
because the same warning appearing when you *didn't* expect it means the encoder weights failed to load, which
is a real disaster wearing the same clothes.

Passing `id2label` and `label2id` writes the label order into the exported `config.json`, so the saved model
documents its own convention instead of leaving a future reader to guess whether 0 meant entailment.

---

## Cross-encoder, not siamese — a deliberate divergence

This is the most consequential design decision in the stage, and the script's own docstring flags it.

The project README's Step 4 describes the **SBERT siamese** recipe: `SentenceTransformer` with
`losses.SoftmaxLoss`. The script implements a **cross-encoder** instead. These are genuinely different, not two
spellings of the same thing.

| | Cross-encoder (what the script does) | Siamese / SBERT (what the README describes) |
|---|---|---|
| Input | both sentences concatenated into one sequence | each sentence encoded separately |
| Attention | premise tokens attend directly to hypothesis tokens | no cross-sentence attention at all |
| Output | 3 logits from the pair | two sentence vectors, combined then classified |
| Produces sentence embeddings? | not directly | yes, that's the point |
| Accuracy on NLI itself | higher | lower |
| Usable for similarity search | no | yes |

A cross-encoder is the stronger NLI classifier, because letting the two sentences attend to each other is a
large advantage for a task that's fundamentally about their relationship. But it doesn't produce a sentence
embedding — you cannot encode a million Tamil documents once and then compare them by cosine similarity, because
the model only ever scores *pairs*.

The siamese setup is weaker at NLI and produces exactly the artifact STS needs: an encoder whose pooled output
is shaped so that similar sentences land nearby. `SoftmaxLoss` trains through the pooling operation, so gradient
actually reaches the embedding geometry.

**The open question this leaves.** Cross-encoder fine-tuning does train the backbone on Tamil entailment, and
the encoder that comes out is better at Tamil than the MLM one was. What it does not do is directly optimize the
pooled sentence vector, which is the thing STS will consume. So it's not obvious that the 0.72 in the
leaderboard table transfers — that number comes from the siamese recipe.

The practical reasons for going this way anyway are reasonable, and worth stating plainly rather than dressing
up:

- It needs no `sentence-transformers` dependency, working with the same `Trainer` plumbing as the MLM stage.
- It matches `notebooks/TamBERT_NLI.ipynb`, which is where the code came from.
- It is the **first real downstream signal** of any kind. Accuracy on IndicXNLI Tamil is a number that says
  whether a perplexity-43 encoder learned anything usable, and there is currently no such number at all.
- It exercises the whole handoff — export, load, fresh head, train, export again — before STS, where a broken
  contract would be more expensive to discover.

Given a perplexity-43 backbone, expectations should be modest either way. If STS later disappoints, revisiting
this as a siamese fine-tune is the first thing to try.

---

## Tokenization

```python
encoded = tokenizer(batch['sentence1'],
                    batch['sentence2'],
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_token_type_ids=False)
```

Passing two arguments invokes the tokenizer's pair template, which produces:

```
[CLS] premise tokens [SEP] hypothesis tokens [SEP]
```

The script's comment records that this was verified rather than assumed. It matters that the boundary marker is
`[SEP]` — a token the MLM already saw and has a trained embedding for — rather than something new.

### `return_token_type_ids=False` is load-bearing

This is the single most important line in the script, and the reason is set up two entries back.

The pair template assigns a token type ID of **1** to sentence B, the standard BERT way of distinguishing
segments. But the MLM config was built with `type_vocab_size=1`, so the model's `token_type_embeddings` table
has exactly **one row**, at index 0. An ID of 1 is an out-of-bounds embedding lookup.

On CPU that surfaces as `IndexError: index out of range in self`, which is at least findable. On GPU it becomes
an opaque device-side assertion that names neither the tensor nor the line — one of the least pleasant errors in
the ecosystem to debug. Which is presumably why the script sets `os.environ['CUDA_LAUNCH_BLOCKING'] = '1'` at
the top: it forces CUDA to report errors at the originating call instead of whenever it next synchronizes.

There is a wrinkle that makes this easy to get wrong. The comment records that transformers 5.14 happens **not**
to emit `token_type_ids` for this particular tokenizer, so the default is currently safe *by accident*. Stating
the flag explicitly makes it safe on purpose, and immune to a library version bumping. RoBERTa is
single-token-type by design anyway, so nothing is lost.

### Sequence length and padding

`MAX_LENGTH = 128`, inherited from the notebook. Rather than trust that, the script measures what it costs:

```python
print(f"{name:5s}: median {int(np.median(lengths))} tokens, p99 {int(np.percentile(lengths, 99))}, "
      f"truncated at {MAX_LENGTH}: {truncated:,} ({truncated / len(ds):.2%})")
```

If the truncation rate comes back high, 128 was wrong and the printout says so instead of hiding it. This is the
same instinct as the token-length histograms in the tokenizer evaluation stage — state the cost of a threshold
rather than assuming it's generous.

The measurement has a small piece of engineering behind it. Nothing is padded during tokenization; a
`DataCollatorWithPadding` pads each batch to its own longest member at load time instead, so a batch of short
pairs doesn't waste compute on 128 positions of `[PAD]`. That means the token counts aren't recoverable from a
uniform tensor shape afterwards, so `num_tokens` rides along as its own integer column and is dropped once the
statistic is printed. Reading a few MB of ints beats materializing every `input_ids` list for all ~393,000
training rows.

`preserve_index=False` on `Dataset.from_pandas` stops `datasets` from smuggling in an `__index_level_0__`
column, which `Trainer` would later complain about.

---

## Training configuration

| Setting | Value | Why |
|---|---|---|
| `num_train_epochs` | 3 | The standard BERT fine-tuning budget. Fine-tuning needs far fewer passes than pre-training |
| `per_device_train_batch_size` | 16 | Notebook value. **No gradient accumulation**, so unlike the MLM stage the logged loss is not inflated |
| `learning_rate` | 2e-5 | Five times lower than the MLM stage's 1e-4. The encoder is already trained; the goal is to nudge it, not overwrite it |
| `weight_decay` | 0.01 | Unlike MLM's 0. A 393k-row supervised set on an already-trained model can overfit, so mild regularization is warranted |
| `warmup_ratio` | 0.06 | Converted to an absolute step count before use — see below |
| `eval_steps` / `save_steps` | 2000 | About 12 evaluations per epoch |
| `save_total_limit` | 3 | Same rationale as MLM |
| `metric_for_best_model` | `eval_accuracy` | Not loss — see below |
| `fp16` | True | GPU only; fails on a CPU-only torch build |
| `seed` | 42 | Matching the corpus split seed |

### Warmup as a count, not a ratio

```python
steps_per_epoch = math.ceil(len(train_ds) / TRAIN_BATCH_SIZE)
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = round(WARMUP_RATIO * total_steps)
```

This is the MLM stage's warmup lesson applied preemptively. `LambdaLR` restores its step counter on resume but
rebuilds its LR function from whatever the code passes at construction time, so a schedule specified as a
*ratio* can silently shift if anything about the dataset size or epoch count changes between runs. Computing the
count once and passing the count makes the schedule reproducible across a resume. The script prints the
resulting schedule so it's on the record.

### Evaluating every 2,000 steps, on the whole dev set

Evaluation is by steps rather than by epoch because 3 epochs would give only 3 evaluation points, far too coarse
for early stopping to mean anything.

Unlike the MLM stage, there is **no subsampling**. The MLM test set was 3.4M lines, which forced a 20k
subsample; the NLI dev set is around 3,200 rows, which is about 100 forward passes at an eval batch size of 32.
Cheap enough to evaluate in full every time, so the metric has no sampling noise in it at all.

`eval_steps` equals `save_steps` for the reason established in the MLM stage: `load_best_model_at_end=True` is
required by `EarlyStoppingCallback` and needs a fresh evaluation at every save.

### Optimizing accuracy, not loss

```python
metric_for_best_model='eval_accuracy',
greater_is_better=True,
```

The MLM stage tracked `eval_loss`, since there was no other option — perplexity is the metric. Here there is a
choice, and accuracy is the right one for two reasons. It's the number the task is actually judged on, and the
labels are balanced roughly 1:1:1, so accuracy isn't distorted by class imbalance the way it would be on a
skewed dataset.

Loss and accuracy can diverge late in fine-tuning — a model can become better calibrated (lower loss) while
getting no more answers right, or grow overconfident on the ones it already had right while accuracy plateaus.
Selecting on the metric you'll report avoids picking a checkpoint that looks better only on the metric you
won't.

`compute_metrics` returns accuracy, macro-F1, and per-class F1, computed with plain numpy rather than pulling in
scikit-learn for three formulas. Per-class F1 is the useful diagnostic here: a model that has quietly learned to
never predict `neutral` — the hardest of the three classes — shows up immediately as a near-zero
`f1_neutral` while overall accuracy still looks respectable at around 66%.

The F1 computation guards against a zero denominator, which happens exactly in that never-predicts-this-class
case, when the metric would otherwise crash rather than report the problem.

### Early stopping

```python
EarlyStoppingCallback(early_stopping_patience=5)
```

Patience 5 rather than the MLM stage's 15. At `eval_steps=2000` that's a 10,000-step no-improvement window,
about 0.4 of an epoch. Tighter is appropriate because the total run is only ~73,500 steps — a 15-evaluation
window would be 30,000 steps, over 40% of the entire run, which would make early stopping nearly inoperative.

### The end of the run

```python
export_final_model(trainer, tokenizer, nli_run_dir)
trainer.remove_callback(early_stopping_callback)
final_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix='test')
save_run_artifacts(trainer, nli_run_dir, final_metrics)
```

Same shape as the MLM stage — export the weights before the final evaluation, so a killed evaluation doesn't cost
the model — with one addition. `remove_callback` detaches the early-stopping callback before the test-set
evaluation, because that callback inspects every evaluation result and would otherwise try to interpret a
one-off test measurement as another point in the training curve it's monitoring.

The dev set drives training decisions; the test set is touched exactly once, at the end, and only for reporting.

---

## An issue found while writing this up

The tokenizer basics entry stated the rule plainly: a precomputed marking scheme carries forward to inference,
and any new text has to go through the same sandhi marking before being handed to the tokenizer, or it won't
line up with what the tokenizer learned.

**`train_nli.py` does not do this.** It passes `batch['sentence1']` and `batch['sentence2']` — raw IndicXNLI
Tamil — straight to a tokenizer whose vocabulary was learned from `⟂`-marked text, and the model it feeds was
pre-trained entirely on `⟂`-marked text.

Measured on the corpus test set, this costs:

| | Sandhi-marked input | Unmarked input |
|---|---|---|
| Fertility | 1.3412 | 1.4141 |
| Marker-bearing vocab entries used | 1,207 | **0** |

1,253 of the 32,000 vocabulary entries contain the marker. On unmarked text not a single one of them is
reachable, so about 3.9% of the embedding table is dead weight — rows the MLM stage spent compute training that
can never activate again. Sequences also run about 5% longer, which nibbles at the 128-token budget.

The fix is small. `scripts/tokenizer/core/sandhi.py` exposes `sandhi_mark_boundaries(text, lang='ta')`, the same
function `sandhi_precompute.py` used to build the corpus files, so it's a matter of applying it to `sentence1`
and `sentence2` inside `tokenize_function`.

Two honest caveats before treating this as settled. The magnitude is unknown — 5% more tokens and a 4% smaller
effective vocabulary is a real distribution shift, but whether it's worth measurable accuracy is an empirical
question, and fine-tuning may simply adapt around it. And IndicXNLI is machine-translated text, so the sandhi
rules may fire differently on it than on the Wikipedia and Common Crawl prose they were tuned against. The right
move is to run it both ways and compare, which is cheap at this scale.

Flagging it rather than silently fixing it, because it's the same category of bug as the `initial_alphabet`
no-op from the grapheme stage: correct-looking code, no error anywhere, and a quiet degradation that only shows
up if someone goes looking. That one was found by reading library documentation carefully. This one was found by
writing down what the pipeline was supposed to do and comparing it against what the code did.

---

## Running it

```bash
python scripts/mlm/export_backbone.py 03_sandhi_codepoint_bert   # required first — final/ does not exist yet
python scripts/nli/train_nli.py
```

Outputs land in `nli/03_sandhi_codepoint_bert/`, mirroring the MLM layout: rotating `checkpoint-*` directories, a
run-root `trainer_state.json` kept safe from rotation, `final_metrics.json`, and `final/` — the weights-only
export that STS fine-tuning will consume.

---

## What to expect, and what comes next

A 6-layer encoder at perplexity 43 is a weak starting point, and this is a cross-encoder rather than the siamese
setup the leaderboard's 0.72 was measured with. So the honest expectation is a modest accuracy number whose main
value is being the first downstream measurement to exist at all, plus a working end-to-end check of the handoff
chain before STS.

Two things to carry into the next stage:

**Mean pooling, never `pooler_output`.** For any similarity task, the sentence vector comes from mean-pooling
`last_hidden_state` across the attention mask. `pooler_output` was designed for next-sentence prediction — an
objective this model never trained on, since `type_vocab_size=1` — and it underperforms mean pooling across Indic
languages per the L3Cube paper the README cites.

**The `indic_sts` test split is the one hard boundary in this project.** Overlap between the pre-training corpus
and NLI or STS *training* data is fine, and expected — different objectives over the same text. The evaluation
split is different in kind: if any of it has been seen during training, the Spearman score at the end measures
nothing, and every number in this journal becomes unfalsifiable.
