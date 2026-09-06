---
title: 7. Tokenizer Evaluation
nav_order: 7
---

# TamilBERT v1.0 — Tokenizer Evaluation

Twelve tokenizers came out of the previous stage: three algorithms (BPE, WordPiece, Unigram) crossed with four
preprocessing variants (plain codepoint, grapheme, sandhi, sandhi plus grapheme). Only one of them gets used
for the rest of the project, so this stage is about measuring them and picking the winner on evidence rather
than on which one sounded most promising.

The measuring happens in two places. Each variant writes its own `metrics.json` at the end of training, inside
`tokenizer/<variant>/`. Then `scripts/tokenizer/core/tokenizer_stats.py` reads all twelve of those files back,
builds the comparison table, plots the charts, and adds an mBERT baseline to compare against.

---

## The two metrics

### Fertility

Fertility is the average number of tokens the tokenizer produces per word:

$$
\text{fertility} = \frac{\text{total tokens}}{\text{total words}}
$$

Lower is better. A fertility of 1.0 would mean every single word became exactly one token, which is not
achievable for a language like Tamil with a 32k vocabulary, and not even desirable — the model would have no
shared pieces to generalize across related word forms. A fertility of 4.0 would mean the average Tamil word is
being shredded into four fragments, which wastes sequence length and forces the model to reassemble meaning
from pieces that carry very little on their own.

The implementation is deliberately blunt. `calculate_fertility` in `scripts/tokenizer/core/metrics.py` counts
words with `line.split()` and tokens with `tokenizer.encode(line)`. It never inspects what characters make up a
line, which is what makes the same function reusable across all four preprocessing variants.

### OOV rate

OOV stands for out-of-vocabulary. It's the fraction of tokens that came out as `[UNK]`, meaning the tokenizer
had no way to represent that piece of text and gave up:

$$
\text{oov rate} = \frac{\text{count of } [UNK] \text{ tokens}}{\text{total tokens}}
$$

Lower is better, and for a monolingual subword tokenizer trained on a large corpus it should be very close to
zero. Any subword tokenizer that keeps individual characters in its vocabulary can fall back to spelling an
unfamiliar word out character by character, so `[UNK]` should only appear for genuinely foreign characters that
never showed up in training at all.

The project README frames the same idea as coverage, which is just the other side of it: coverage equals
`1 - oov_rate`.

### Why both are measured on the held-out split

Both metrics run against the test file, never the training file. Measuring coverage on the same text the
tokenizer trained on is meaningless — it will always come back at essentially 100%, because the tokenizer built
its vocabulary from exactly those words. The 90/10 split made back in the corpus stage exists precisely so
there is 3.4M lines of Tamil the tokenizer has never seen, and that's the only text where these numbers say
anything real.

---

## The results

All twelve variants, sorted by fertility, best first. Every one was trained to a 32,000 vocabulary, so the
vocab column is constant and left out.

| Variant | Fertility | OOV rate |
|---|---|---|
| `03_sandhi_codepoint_bert` | **1.3414** | 0.016% |
| `01_metaspace_codepoint_bert` | 1.3420 | 0.014% |
| `03_sandhi_codepoint_bpe` | 1.3431 | 0.000% |
| `01_metaspace_codepoint_bpe` | 1.3451 | 0.000% |
| `02_metaspace_grapheme_bpe` | 1.3477 | 0.015% |
| `04_sandhi_grapheme_bpe` | 1.3485 | 0.016% |
| `02_metaspace_grapheme_bert` | 1.3657 | 0.017% |
| `04_sandhi_grapheme_bert` | 1.3667 | 0.018% |
| `02_metaspace_grapheme_unigram` | 1.4970 | 0.014% |
| `04_sandhi_grapheme_unigram` | 1.4988 | 0.014% |
| `03_sandhi_codepoint_unigram` | 1.5149 | 0.000% |
| `01_metaspace_codepoint_unigram` | 1.5157 | 0.000% |

Three things fall out of this table immediately.

**The algorithm matters more than the preprocessing.** Every Unigram variant lands between 1.497 and 1.516,
while every BPE and WordPiece variant lands between 1.341 and 1.367. The gap between algorithm families is
around 0.15 tokens per word; the gap between preprocessing variants inside a family is around 0.02. Whichever
preprocessing trick gets applied, Unigram stays roughly 11% worse than the merge-based algorithms on this
corpus.

**Grapheme marking did not help.** This is worth sitting with, because the previous stage put real work into
building the placeholder scheme. Within BPE, the grapheme variants (1.3477, 1.3485) are *worse* than the plain
codepoint variants (1.3431, 1.3451). Within WordPiece the same, and by a wider margin. Only in Unigram does
grapheme marking help, and there it only claws back some of Unigram's own deficit. Preventing the tokenizer
from splitting inside a grapheme cluster turns out to cost more in lost merge flexibility than it gains in
linguistic tidiness — the splits it was preventing apparently weren't hurting much.

**Sandhi marking helped slightly, and consistently.** In every family, the sandhi variant beats its
non-sandhi counterpart at the same grapheme setting: 1.3414 vs 1.3420, 1.3431 vs 1.3451, 1.5149 vs 1.5157. The
margins are tiny, in the third decimal place, but the direction is the same in all six pairings, which is more
convincing than any single pair would be. Marking a phonological boundary gives the merge algorithm a genuine
hint about where a compound joins.

The OOV column is effectively a pass/fail, and everything passes. The worst variant is 0.018%, or roughly one
`[UNK]` in every 5,500 tokens. The BPE variants report a flat zero. Coverage in the README's terms is above
99.98% everywhere, against a target of 95%.

---

## Against the baseline and the targets

`tokenizer_stats.py` also loads `bert-base-multilingual-cased` and runs the identical metric function on the
identical raw test file, which is the comparison that justifies the whole stage:

| Tokenizer | Fertility | OOV rate |
|---|---|---|
| mBERT (`bert-base-multilingual-cased`) | 3.3413 | 0.698% |
| `03_sandhi_codepoint_bert` (this project) | 1.3414 | 0.016% |

mBERT needs 2.5 times as many tokens to express the same Tamil text, and produces around 44 times the `[UNK]`
rate. The README's explanation for this holds up: mBERT spreads 120k vocabulary slots across 100-plus
languages, so Tamil gets a couple of thousand of them and its words come apart into fragments. Spending all
32,000 slots on one language is what buys the difference.

Every target the README set for this stage is met with room to spare:

| Metric | Target | Achieved |
|---|---|---|
| Fertility | ≤ 2.5 tokens/word | 1.3414 |
| Coverage on held-out | > 95% | 99.98% |
| Vocabulary size | 30k–50k | 32,000 |

---

## Token length distributions

Fertility is a per-word average, which says nothing about how long a whole line gets. That matters for the next
stage, because a transformer has a fixed maximum sequence length and anything longer gets cut off. So
`tokenizer_stats.py` also encodes the entire 3.4M-line test set with each variant and records the distribution
of tokens per line.

For `03_sandhi_codepoint_bert`, measured on its own sandhi-marked test file:

| Statistic | Tokens per line |
|---|---|
| mean | 17.2 |
| median | 13 |
| 75th percentile | 23 |
| 90th percentile | 36 |
| 95th percentile | 45 |
| 99.9th percentile | 127 |
| max | 2,421 |

The shape here is a long right tail on a small body. Half of all lines are 13 tokens or shorter, and 99.9% fit
inside 127 tokens, but the single longest line in the test set is 2,421 tokens. Those outliers are the
Wikipedia table remnants that the corpus stage's 2500-token cutoff was written to catch — the cutoff bounded
them rather than removing them.

This is the number that sets the truncation limit in the MLM stage. A 512-token sequence length covers well
beyond the 99.9th percentile, so almost nothing real gets cut, but the 2,421-token outliers would overflow the
model's position embeddings entirely if truncation weren't switched on explicitly.

---

## The representation caveat, and why it changes the answer

Here is the part that took the longest to understand, and it nearly led to picking the wrong tokenizer.

Every fertility number in the table above was measured with the tokenizer and the test file **in the same
representation**. A grapheme variant was measured against `test_grapheme_marked`, where multi-codepoint
clusters have already been swapped for Private Use Area placeholders, while the tokenizer's vocabulary was
still in placeholder form too. That's internally consistent, and the previous stage's notes explain at length
why it's the correct way to measure — the word count in the denominator is unaffected by placeholder
substitution, so the numbers are directly comparable across variants.

What it does *not* tell you is how the saved tokenizer behaves on ordinary raw Tamil text, which is what every
downstream stage will actually feed it. `journal/tokenizer_notes.md` flagged exactly this as an unfinished
follow-up: reload each saved tokenizer and re-run the metrics against the original raw test file, as an
independent check that the numbers survive.

Running that check is what this table shows. The right column re-measures fertility on unmarked `test.txt`,
using a sample of every 68th line so the whole file is represented evenly:

| Variant | Reported | On raw text |
|---|---|---|
| `01_metaspace_codepoint_bert` | 1.3420 | 1.3417 |
| `01_metaspace_codepoint_bpe` | 1.3451 | 1.3447 |
| `01_metaspace_codepoint_unigram` | 1.5157 | 1.5154 |
| `03_sandhi_codepoint_bert` | 1.3414 | 1.4141 |
| `03_sandhi_codepoint_bpe` | 1.3431 | 1.4262 |
| `03_sandhi_codepoint_unigram` | 1.5149 | 1.5834 |
| `02_metaspace_grapheme_unigram` | 1.4970 | 4.4321 |
| `04_sandhi_grapheme_unigram` | 1.4988 | 4.9219 |
| `02_metaspace_grapheme_bpe` | 1.3477 | 6.0299 |
| `04_sandhi_grapheme_bpe` | 1.3485 | 6.0302 |
| `02_metaspace_grapheme_bert` | 1.3657 | 6.0343 |
| `04_sandhi_grapheme_bert` | 1.3667 | 6.0346 |

Three tiers show up, and each has a different explanation.

**The plain codepoint variants match to four decimal places.** Their training and test files were raw text to
begin with, so re-measuring on raw text is measuring the same thing twice. Getting 1.3417 against a reported
1.3420 is a useful confirmation that nothing about the measurement pipeline is broken.

**The sandhi variants degrade a little.** `03_sandhi_codepoint_bert` goes from 1.3414 to 1.4141, about 5%
worse. This is the trade-off the tokenizer basics entry warned about: the tokenizer learned its vocabulary from
text with `⟂` markers in it, so feeding it text without markers means it can't use the pieces it learned. Of
its 32,000 vocabulary entries, 1,253 (3.9%) contain the marker. On sandhi-marked text, 1,207 of those get used;
on unmarked text, **zero** of them do. Just under 4% of the vocabulary becomes unreachable, and fertility rises
to match.

**The grapheme variants collapse completely,** from around 1.35 to around 6.03 — a 4.5x degradation. For the
BPE and WordPiece grapheme variants the reason is direct: `relabel_vocab_after_save` is `false` for those two
families, so their saved vocabularies are still full of Private Use Area placeholder codepoints. Counting the
entries that contain at least one placeholder character makes the scale of it obvious:

| Variant | `relabel_vocab_after_save` | Vocab entries holding placeholder codepoints |
|---|---|---|
| `02_metaspace_grapheme_bert` | false | 29,790 of 32,000 (**93.1%**) |
| `02_metaspace_grapheme_bpe` | false | 29,877 of 32,000 (**93.4%**) |
| `02_metaspace_grapheme_unigram` | true | 0 |
| `01_metaspace_codepoint_bert` | false | 0 |

Raw Tamil text contains no Private Use Area characters at all, so 93% of those vocabularies is unreachable and
the tokenizer falls back to spelling everything out character by character. Encoding a four-word Tamil line
with `02_metaspace_grapheme_bert` produces 27 tokens, almost all single characters.

The Unigram grapheme variants land in between, at 4.43 and 4.92, because Unigram is the one family where
`relabel_vocab_after_save` is `true`. The zero in that table confirms the relabel genuinely worked — not one
placeholder codepoint survives, and the entries are readable Tamil again — which also answers the open question
`tokenizer_notes.md` left about whether relabelling was safe. It was. What it doesn't do is restore the original
fertility: 4.43 against a reported 1.4970, still more than three times worse than the codepoint variants. The
vocabulary is real Tamil, but the *pieces* it learned were shaped by placeholder text, and they don't segment
raw Tamil the way they segmented the substituted corpus.

None of this means the grapheme variants are broken. It means they are only usable with the placeholder map
applied to the input first, exactly as the tokenizer basics entry predicted: a precomputed marking scheme
carries forward to inference, and any new text has to go through the same transformation before being handed to
the tokenizer. What the re-measurement establishes is that they are not drop-in tokenizers, and that the cost
of forgetting the marking step is severe rather than marginal.

The same warning applies to the sandhi variants, just at a much smaller magnitude — 5% rather than 450%.

---

## Sandhi marker usage

One more diagnostic, only meaningful for the four sandhi variants. The `⟂` marker is a real character in the
text, so the tokenizer has to decide what to do with it: either give it its own token, or absorb it into a
larger token alongside surrounding Tamil characters. `tokenizer_stats.py` counts both cases across the test
set.

| Variant | Marker isolated | Marker fused into a larger token |
|---|---|---|
| `03_sandhi_codepoint_bert` | 0 | 2,959,086 |
| `03_sandhi_codepoint_bpe` | 2,519 | 2,956,886 |
| `03_sandhi_codepoint_unigram` | 4,269 | 2,957,112 |
| `04_sandhi_grapheme_bert` | 0 | 2,960,327 |
| `04_sandhi_grapheme_bpe` | 2,162 | 2,957,497 |
| `04_sandhi_grapheme_unigram` | **2,957,990** | 162,498 |

Fused is what you want. An isolated marker costs a full extra token per boundary and contributes nothing but
"a boundary happened here", whereas a fused marker rides along inside a token that was going to exist anyway,
making the boundary information free.

Five of the six variants fuse almost every occurrence — WordPiece manages it 100% of the time. `04_sandhi_grapheme_unigram`
is the exception and it is inverted: 95% of markers stand alone as their own token. That single fact explains
why it has the worst fertility of any Unigram variant (1.4988) and the worst raw-text fertility of the two
Unigram grapheme variants (4.92 vs 4.43). Stacking both markings gave Unigram's pruning step so many
placeholder characters to account for that it never learned pieces combining a marker with its neighbours.

---

## The pick

**`03_sandhi_codepoint_bert`** — WordPiece, sandhi-marked, no grapheme substitution.

It wins on the headline number, with the lowest fertility of all twelve at 1.3414, though only by 0.0006 over
`01_metaspace_codepoint_bert`, which is too thin a margin to decide anything by itself. The reasons that
actually matter are the ones from the caveat section:

- It is a **codepoint** variant, so its vocabulary is real Tamil text and it degrades gracefully — 5% worse on
  unmarked input rather than 450% worse.
- Its sandhi marker is **fused 100% of the time**, so the boundary information costs nothing.
- Its OOV rate of 0.016% is far below the 5% the target allows, and the difference between that and BPE's flat
  zero amounts to one extra `[UNK]` per 6,000 tokens.

The one thing to carry forward: 1.3414 is the number for sandhi-marked input. Anything downstream that feeds
this tokenizer raw Tamil without running `sandhi_mark_boundaries` first gets 1.4141 instead, and leaves 1,253
vocabulary entries permanently unused. That turns out to matter at the NLI stage — see
[10_nli_finetuning.md](10_nli_finetuning.md).

The next stage takes this tokenizer and trains a language model on top of it.
