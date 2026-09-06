# TamilBERT v1.0 — MLM Pre-training

The tokenizer can now turn Tamil text into token IDs and back, but it has no idea what any of those IDs mean.
`▁மலர்` and `▁வான்` are just numbers 8,412 and 3,109 as far as it's concerned, with no sense that one is closer
in meaning to a flower than to the sky. This stage is where meaning gets learned, and it's the single most
expensive step in the whole project.

The script is `scripts/mlm/train_roberta.py`. This entry walks through what it does and why each number in it
is the number it is. What actually happened when it ran — including a near-disaster with the checkpoints — is
the next entry, [09_mlm_training_run.md](09_mlm_training_run.md).

---

## What masked language modelling is

The model gets shown a sentence with some of the words hidden, and has to guess what was hidden. That's the
whole objective.

```
Input:   ▁வான் [MASK] ▁உலகம் ▁வழங்கி ▁வரு ##தலால்
Target:            ▁நின்று
```

To fill that blank correctly, the model has to build up an enormous amount of implicit knowledge: which Tamil
words plausibly follow which, how case endings agree, what kinds of nouns take what kinds of verbs, which words
tend to co-occur in the same topic. Nobody labels any of this. The supervision signal comes free from the text
itself, because the answer was always sitting there before it got hidden — which is why the entire 30.7M-line
corpus can be used without a single human annotation.

15% of tokens get masked on each pass. That fraction is the BERT paper's value and effectively the field
standard. Mask too few and most of each forward pass is wasted, since the model only learns from the positions
it has to predict. Mask too many and there isn't enough surviving context left to make a sensible guess, so the
task becomes noise.

### Why masking, and not next-word prediction

A generative model predicts the next word from the words before it, so it only ever sees leftward context. A
masked model sees the whole sentence with holes in it, so it reads **bidirectionally** — the words after the
blank inform the guess just as much as the ones before.

That's exactly the property this project needs. The goal is sentence embeddings where semantically similar
Tamil sentences land close together, and a good sentence representation has to account for the whole sentence at
once, not just a left-to-right prefix. It's also why this model can never generate Tamil, which the project
README states as a deliberate scope decision rather than a limitation to fix later.

### Why an encoder, sized this way

The architecture is `RobertaForMaskedLM`: a RoBERTa encoder with a language-modelling head bolted on top to
produce vocabulary-sized predictions at each masked position. RoBERTa rather than BERT because RoBERTa is
essentially BERT with the next-sentence-prediction objective removed and the training recipe improved, and NSP
is of no use here — the project README notes that `pooler_output`, the vector NSP trains, consistently
underperforms mean pooling for similarity work across Indic languages.

| Setting | Value | Reasoning |
|---|---|---|
| `num_hidden_layers` | 6 | DistilRoBERTa depth. Half of BERT-base's 12, which roughly halves both training time and memory, and fits a single-GPU budget |
| `hidden_size` | 768 | BERT-base width, kept rather than reduced — embedding quality depends more on width than depth |
| `num_attention_heads` | 12 | 768 / 12 = 64 dimensions per head, the standard ratio |
| `vocab_size` | 32,000 | Read off the trained tokenizer with `get_vocab_size()`, never hardcoded |

Depth was the thing traded away rather than width, because the output of this stage is a vector per token that
later gets mean-pooled into a sentence vector, and that vector's expressiveness is bounded by `hidden_size`. A
6-layer model at 768 dimensions gives up some of the compositional reasoning a 12-layer model would have, while
keeping the representation width the downstream similarity task cares about.

---

## Wiring the tokenizer to the model

The tokenizer trained in the previous stage assigned these IDs:

| Token | ID |
|---|---|
| `[UNK]` | 0 |
| `[CLS]` | 1 |
| `[SEP]` | 2 |
| `[PAD]` | 3 |
| `[MASK]` | 4 |

Two things get built from the same loaded tokenizer object. First a `RobertaConfig`, which needs the IDs so the
model knows which positions to ignore and which special tokens mean what. Then a `PreTrainedTokenizerFast`
wrapper, which is what HuggingFace's `Trainer` and the data collator expect to be handed.

That wrapper is built by naming every special token explicitly:

```python
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
                                    cls_token=BOS_TOKEN,
                                    sep_token=EOS_TOKEN,
                                    pad_token=PAD_TOKEN,
                                    unk_token=UNK_TOKEN,
                                    mask_token=MASK_TOKEN)
```

The obvious alternative, `AutoTokenizer.from_pretrained(...)`, reads the special-token map out of a
`tokenizer_config.json` file on disk. That file is written by whatever code saved the tokenizer, and if it ever
disagrees with what the model was trained with, nothing complains — the model just quietly starts treating the
wrong ID as padding. Naming the tokens from `tokenizer.core.constants` makes the map impossible to drift,
because it comes from the same constants the tokenizer training used. The same pattern is repeated in
`export_backbone.py` and `train_nli.py` for the same reason.

### Two config values that look wrong and aren't

**`max_position_embeddings=512 + pad_id + 1`**, which works out to 516.

A transformer has no inherent sense of word order, so position information is added by looking up a learned
embedding per position — position 0, position 1, and so on. RoBERTa has a quirk here: it reserves the low
position IDs and starts real positions at `pad_id + 1`, so with `pad_id=3` the first actual token sits at
position 4 rather than 0. A sequence of 512 tokens therefore needs position IDs up to 515, and the table has to
have 516 rows.

Hardcoding 516 would work today and break silently the moment a differently-trained tokenizer assigned a
different `[PAD]` ID. Computing it from the `pad_id` that was just looked up keeps the two in lockstep. Getting
this wrong produces an out-of-bounds lookup, which on GPU surfaces as an opaque device-side assertion rather
than a readable error.

**`type_vocab_size=1`**.

BERT can be fed two segments at once — a sentence pair — and distinguishes them with a token type ID of 0 or 1,
looked up in a small `token_type_embeddings` table. RoBERTa dropped this along with NSP, so this model gets a
table with exactly **one** row, because it only ever sees single-segment input during pre-training.

This is worth remembering, because it becomes a live trap two stages later. NLI fine-tuning feeds the model
sentence *pairs*, and if anything hands it a token type ID of 1, that's a lookup into row 1 of a one-row table.
The NLI entry covers how that's prevented.

---

## Feeding it the corpus

```python
data_files = {
    "train": str(paths.train_sandhi_marked),
    'test': str(paths.test_sandhi_marked)
}
```

Note *which* files. The chosen tokenizer, `03_sandhi_codepoint_bert`, learned its vocabulary from sandhi-marked
text, so it gets sandhi-marked text here too. The previous entry measured what mixing representations costs:
1,253 of the 32,000 vocabulary entries contain the `⟂` marker, and feeding unmarked text leaves every one of
them unused while pushing fertility from 1.3414 up to 1.4141. Keeping the representation consistent is the
whole reason these particular file paths appear.

### Truncation is mandatory, not defensive

```python
raw_tokenizer.enable_truncation(max_length=512)
```

The token length distribution from the evaluation stage showed the test set's longest line hitting 2,421
tokens. The position embedding table has 516 rows. Without this line, that one line is an out-of-bounds lookup
and the run dies — somewhere in the middle of a multi-day job, having consumed real GPU hours.

Setting it to 512 costs almost nothing in return: 99.9% of lines are 127 tokens or shorter, so the cut only
touches the Wikipedia table remnants the corpus stage already identified as junk.

### Dynamic masking

```python
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
```

The collator picks the masking positions fresh every time a batch is assembled, rather than masking the corpus
once up front. So the same line seen in epoch 2 has different words hidden than it did in epoch 1, which
multiplies the effective amount of training signal available from a fixed corpus. This is one of the concrete
changes RoBERTa made over BERT, which used a fixed mask.

The collator also implements BERT's 80/10/10 split of what "masking" means. Of the 15% of positions selected:

- **80%** are replaced with `[MASK]`.
- **10%** are replaced with a random token from the vocabulary.
- **10%** are left exactly as they were.

All three still count as prediction targets. The reason for the odd split is that `[MASK]` never appears in real
text — it only exists during pre-training. If the model only ever had to predict positions marked with
`[MASK]`, it would learn "produce a real word wherever you see that token" and would have no reason to build
useful representations of positions that *aren't* masked. The 10% random and 10% unchanged cases force it to
maintain a meaningful representation of every position, because it can't tell from the input alone which
positions it will be graded on.

---

## The training configuration

Each of these values has a specific reason, and several of them were changed from an earlier version of the
script after something went wrong.

| Setting | Value | Why |
|---|---|---|
| `learning_rate` | 1e-4 | Standard peak LR for from-scratch BERT/RoBERTa. Five times higher than a fine-tuning LR, because the weights start random |
| `lr_scheduler_type` | `linear` | Ramp up, then decay linearly toward zero. Matches the original RoBERTa recipe |
| `warmup_ratio` | 0.06 | About 6% of steps spent ramping the LR from 0 to peak. A randomly-initialized transformer hit with the full LR immediately can diverge in the first few hundred steps |
| `per_device_train_batch_size` | 16 | What fits in GPU memory at 512 sequence length |
| `gradient_accumulation_steps` | 4 | Effective batch size 64 — see below |
| `weight_decay` | 0 | No regularization. With 30.7M lines and a 6-layer model, overfitting isn't the binding constraint |
| `fp16` | True | Mixed precision, for throughput and memory headroom. **Requires a GPU** — this fails outright on a CPU-only torch build |
| `num_train_epochs` | 6 | A ceiling, not a target. Early stopping is expected to fire long before this |

### Gradient accumulation, and the logging trap it sets

A batch of 16 sequences is small enough to make the gradient estimate noisy. The fix without more GPU memory is
gradient accumulation: run four batches of 16, add up their gradients, and only then take one optimizer step.
The model updates as though it saw a batch of 64.

The cost is bookkeeping. One "step" now means four forward passes, so wall-clock per step is four times longer,
and — this is the part that caused real confusion later — **the loss HuggingFace logs is the sum across the four
accumulated batches, not the average**. The logged number is roughly 4× the true per-token loss. The next entry
covers the several hours spent thinking the run had diverged because of this.

### Evaluating against a subsample

```python
eval_subset_size = min(20000, len(tokenized_datasets['test']))
eval_subset = tokenized_datasets['test'].shuffle(seed=42).select(range(eval_subset_size))
```

The test set is 3.4M lines. At an eval batch size of 16 that's about 213,000 forward passes for one full
evaluation — comparable to a meaningful fraction of a training epoch, and it would have to run at every single
checkpoint. Doing that would mean spending more compute measuring the model than training it.

So the frequent evaluations run against a fixed, shuffled 20,000-line subsample. Fixed matters: the same 20k
lines every time, via `seed=42`, so successive `eval_loss` values are comparable to each other. A fresh random
sample each time would add noise that looks like the model getting better or worse.

The full test set is then evaluated exactly once, after training finishes, and that's the number worth
reporting. Losing sight of which of the two any given loss value came from is an easy mistake — the numbers
differ substantially.

### Checkpointing, and the constraint that sets its frequency

```python
eval_strategy='steps',   eval_steps=10000,
save_strategy='steps',   save_steps=10000,
save_total_limit=3,
load_best_model_at_end=True,
metric_for_best_model='eval_loss',
```

Both strategies are `steps` rather than `epoch` for a blunt reason: one epoch here is around 480,000 steps.
Saving once per epoch means a crash can cost days of work.

`save_steps=10000` is a compromise against disk. Each checkpoint is roughly 800 MB — about 260 MB of weights
plus 520 MB of Adam optimizer state, which is twice the weight size because Adam tracks two moment estimates per
parameter. An earlier version used 2,000, which meant around 240 saves per epoch and something like 190 GB,
which is unmanageable to keep or transfer. At 10,000 there are about 48 saves per epoch, so a crash costs at
most ~2% of an epoch.

**`eval_steps` must equal `save_steps`.** This is a hard requirement rather than a stylistic choice.
`load_best_model_at_end=True` is required by `EarlyStoppingCallback`, and it works by comparing the
`metric_for_best_model` recorded at each save. If a save happens at a step with no fresh evaluation, there's no
metric to compare and the mechanism breaks.

`save_total_limit=3` keeps only the three newest checkpoints and deletes the rest. This one has a sharp edge
that the next entry is largely about: rotation deletes a checkpoint's `trainer_state.json` along with its
weights, and that file is where the entire metric history lives.

### Early stopping

```python
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=15)
```

Stop when `eval_loss` hasn't improved for 15 consecutive evaluations. The unit is **evaluation calls, not
epochs**, which is easy to misread. At `eval_steps=10000` that's a 150,000-step window of no improvement before
the run gives up — around 0.3 of an epoch.

15 is deliberately loose. The asymmetry is that too loose only wastes some compute, while too tight kills a run
that was still improving. The script's comment suggests tightening it to about 5 once there's real evidence for
how noisy `eval_loss` is step to step.

One thing to know when reading the run history: the run that actually happened used `eval_steps=2000`, not
10,000. So its early-stopping window was 15 × 2,000 = 30,000 steps, which is how the best checkpoint gets
located at 30,000 steps before the stopping point. The script's current value is 10,000.

---

## What the schedule works out to

With an effective batch size of 64 over roughly 30.7M training lines, the run recorded:

| | |
|---|---|
| Steps per epoch | 479,982 |
| Total steps for the 6-epoch ceiling | 2,879,892 |
| Warmup at `warmup_ratio=0.06` | ~172,800 steps |

Nearly 2.9 million optimizer steps is the scale of what the ceiling asks for, and it's worth registering how
large that is before reading what the run actually managed. It stopped at step 548,000.

---

## What comes out

The script ends with three things, in a deliberate order:

1. **`export_final_model(...)`** writes `mlm/<run>/final/` — config, weights, and tokenizer, with no optimizer
   state. This is the artifact the NLI stage consumes. It runs *before* the final evaluation, because that
   evaluation is a multi-hour job over 3.4M lines and if it gets killed the weights should already be safe on
   disk.
2. **The full test-set evaluation**, with perplexity attached alongside the loss.
3. **`save_run_artifacts(...)`** writes `final_metrics.json` and the complete log history to the run root.

Perplexity is worth explaining since it's the number usually quoted. It's just `exp(loss)`, and it converts a
cross-entropy loss in nats into something interpretable: roughly, how many equally-likely options the model is
effectively choosing between at each masked position. A perplexity of 1 would be perfect certainty. Random
guessing across a 32,000 vocabulary would be 32,000, corresponding to a loss of `ln(32000) = 10.37` — which is
a useful sanity check for a freshly initialized model's very first loss value.

The README's target is a loss below 2.0, or a perplexity around 7.4.

The next entry covers what the run actually achieved, and the checkpoint disaster it survived.
