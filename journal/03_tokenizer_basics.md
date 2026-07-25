# TamilBERT v1.0 — Tokenizer Basics

Once the corpus was ready, the next problem was turning Tamil text into something a model can actually learn from. That job belongs to the tokenizer, and it turned out to involve almost as many small decisions as building the corpus did.

Most of the comparisons in this project use HuggingFace's `tokenizers` library, since it ships as native Rust code. That speed difference matters a lot once the training file runs into hundreds of millions of words.

---

## What a tokenizer does

A tokenizer sits between raw text and the model. Its job is to convert text into numbers (IDs) that a computer can actually process. To do this well, a tokenizer goes through its own training step, where it scans a large amount of text, learns which patterns show up most often, and assigns a new ID to each recurring pattern.

This project compares three ways of building that vocabulary: BPE (Byte Pair Encoding), which repeatedly merges the most frequent pair of symbols it finds; WordPiece, the approach BERT uses, which works similarly to BPE but picks each merge based on likelihood rather than raw frequency; and Unigram, which starts from a large pool of candidate subwords and prunes it down to whichever set best explains the training text. All three are standard approaches used across modern language models.

A tokenizer is doing a good job when long words and common phrases end up represented by only a handful of IDs, rather than being broken down letter by letter. A compound Tamil word, for example, should ideally come out as two or three familiar pieces instead of a long chain of single characters. Getting there depends on training the tokenizer on a large, varied corpus and giving it a vocabulary big enough to hold those common pieces.

---

## Tokenizer vocabulary and special tokens

`VOCAB_SIZE` is the total number of rows in the tokenizer, where each row maps one subword to one ID.

A small number of these rows are set aside for special tokens. These carry a fixed meaning and are locked in from the start, so they never get overwritten by anything learned during training. That means the vocabulary actually available for real Tamil subwords is `VOCAB_SIZE` minus the number of special tokens.

Normally, a Unigram tokenizer trained with SentencePiece would use a different naming style for these, things like `<unk>` and `<pad>`, following the convention most Unigram examples use. This project does something different: every tokenizer variant, BPE, WordPiece, and Unigram alike, uses the same BERT style bracketed tokens ([UNK], [CLS], [SEP], [PAD], [MASK]). That's a deliberate choice made for this project, not an attempt to match outside convention. Keeping the naming identical across variants makes it much easier to compare BPE, WordPiece, and Unigram directly, since none of their results are shaped by a naming quirk the others don't share.

The special tokens used in this project are:

- **unk (unknown)**: assigned to any word or fragment the tokenizer has never seen before. A Tamil monolingual tokenizer, for instance, would fall back to `unk` for stray English or Spanish characters that slip into a sentence.
- **cls**: added during post processing, and generally marks the start of a sequence. This convention comes from BERT specifically, but is applied across every model in this project for consistency.
- **sep**: also added during post processing, and marks the separation between two segments of text inside the same sequence, using the same BERT convention.
- **pad**: used to even out batch lengths. Shorter sentences get extra `pad` tokens appended until they match the length of the longest sentence in the batch. These tokens are never attended to, so they don't distort what the model actually learns.
- **mask**: used during masked language model training to hide certain tokens so the model has to guess what belongs there. This is how the model ends up learning Tamil grammar and meaning, by repeatedly filling in blanks.
- **unusedN**: a block of empty, reserved slots kept aside for future downstream tasks such as fine tuning.

---

## The parts of a tokenizer

A tokenizer feels like one single step, but it's actually built from a few smaller stages that can each be configured on their own. This project experiments with two of those stages in detail: normalization and pre-tokenization.

### Normalization

Normalization functions run over every line of input before anything else happens. Their job is to sanitize and standardize the text so that whatever comes next, starting with the pre-tokenizer, sees something consistent. A line might pass through just one normalizer, or a short sequence of them, depending on the setup.

The normalizers available in HuggingFace's `tokenizers` library, fall into three groups: Unicode normalization, text cleaning normalizers, and special purpose normalizers.

#### Unicode normalization

Unicode normalization tells the tokenizer how to handle characters that can be represented in more than one way while still looking identical on screen.

Take the French letter é. It can be stored as a single, composed codepoint, or as two separate codepoints, a plain `e` followed by a combining accent mark:

- Composed: é = 1 codepoint (U+00E9)
- Decomposed: e + ́ = 2 codepoints (U+0065 + U+0301)

`NFC` forces every character into its composed, single codepoint form. `NFD` does the opposite, forcing everything into its split, decomposed form.

Some characters also carry a stylistic variant, things that render in a special way like italics or circled digits, but which really mean the same underlying character underneath. `NFKC` handles this: it's `NFC` plus folding for these compatibility variants. It's a more aggressive transformation than plain `NFC`, so it isn't a safe default whenever the stylistic difference actually carries meaning.

| Input | NFC output | NFKC output |
|---|---|---|
| ﬁ (single ligature codepoint) | ﬁ (unchanged) | fi (split into two plain letters) |
| ① (circled digit one) | ① (unchanged) | 1 (plain digit) |
| ＡＢＣ (full width Latin) | ＡＢＣ (unchanged) | ABC (normal width) |
| ² (superscript two) | ² (unchanged) | 2 (plain digit) |

`NFKD` combines both aggressive behaviors: `NFD`'s splitting plus `NFKC`'s compatibility folding. It's the strongest of the four normalizers.

| Form | What it does | Splits composed into decomposed? | Folds compatibility variants? | Example |
|---|---|---|---|---|
| NFC | Composes decomposed characters into their single canonical form | No, it recomposes | No | é (decomposed) becomes é (composed) |
| NFD | Decomposes composed characters into a base character plus combining marks | Yes | No | é becomes e + ́ |
| NFKC | Same as NFC, but also folds compatibility characters like ligatures and special forms into their canonical equivalents | No, it recomposes | Yes | ﬁne café becomes fine café |
| NFKD | Same as NFD, but also decomposes compatibility characters | Yes | Yes | ﬁne becomes f + i + n + e |

#### Text cleaning normalizers

- **LowerCase**: converts text to lowercase.
- **StripAccents**: removes accent characters left behind after an `NFD` split.
- **Strip**: removes surrounding whitespace.
- **Replace**: swaps out a string, or a regex pattern, for something else.
- **Nmt**: a prebuilt cleanup step from the SentencePiece paper. It strips control characters, normalizes whitespace, and replaces a handful of problematic Unicode characters.

#### Special purpose normalizers

These are ready made sequences of normalization steps that were first introduced in specific papers:

- **ByteLevel**: from the GPT-2 paper. Converts every byte into a visible Unicode character, which means the tokenizer never needs an `[UNK]` token at all.
- **BertNormalizer**: released alongside BERT. Cleans control characters, adds spacing around CJK characters, and can optionally strip accents or lowercase text.
- **Precompiled**: the SentencePiece normalizer. Loads a normalization map that was computed ahead of time rather than built on the fly.

### The pre-tokenizer step

The pre-tokenizer decides how the corpus gets split before any merging begins, and that first split has an outsized effect on the final vocabulary. Since a good tokenizer represents a real word using as few IDs as possible, the most frequent phrases and words need to end up with their own dedicated IDs. The merge algorithm works by counting how often each element in the vocabulary occurs, so the starting point it counts from, meaning the pre-tokenizer's output, ends up shaping every merge decision that follows.

Different models tend to favor different pre-tokenizers depending on the training corpus and how their particular subword algorithm behaves. For this project, the comparison is kept intentionally narrow: four pre-tokenizer setups are tested, across 3 models (Unigram, BPE, BERT).

#### Types of pre-tokenizers

Most modern tokenizers actually run a sequence of several pre-tokenizers chained together rather than just one. Many of the options used here come as Rust backed built-ins from the `tokenizers` library, which is part of why they were preferred, they're simply faster than an equivalent step written in Python.

| Pre-tokenizer | What it does |
|---|---|
| BertPreTokenizer | Splits on spaces and on punctuation, treating each punctuation character as its own token |
| ByteLevel | Splits into words, then re-represents every byte as a visible character (the GPT-2 approach, using a 256 character alphabet) |
| CharDelimiterSplit | Splits on a single character you provide, similar to `.split(delimiter)` |
| Digits | Splits digits away from surrounding text, either grouped together ("123") or one at a time ("1", "2", "3") |
| Metaspace | Replaces whitespace with a marker character (▁ by default) and splits on that marker. This is the one used in this project's scripts. |
| Punctuation | Splits on punctuation characters individually, with configurable behavior (isolated, removed, merged, contiguous) |
| Sequence | Chains several pre-tokenizers together and applies them in order |
| Split | Splits on a custom string or regex pattern, with configurable behavior around the delimiter |
| UnicodeScripts | Splits at boundaries between different Unicode script families (Latin, Han, and so on), similar to how SentencePiece's Unigram implementation works |
| Whitespace | Splits using the pattern `\w+\|[^\w\s]+` |
| WhitespaceSplit | Splits purely on whitespace, similar to `.split()` |

#### The four experiments

This project tests four different pre-tokenizer setups. Each one trains on its own version of the corpus, built specifically to simulate that scenario: the raw corpus for the unicode split baseline, a sandhi-marked file for the sandhi split, a grapheme-marked file for the grapheme split, and a combined sandhi-plus-grapheme-marked file for the last one. That's four distinct text files in total, one per experiment, generated from `sandhi_precompute.py` and `grapheme_precompute.py` (which produces the other two, grapheme-only and sandhi+grapheme), so each pre-tokenizer's effect shows up cleanly rather than being blended into one shared file:

1. **Unicode split**: split text into individual Unicode characters.
2. **Grapheme split**: split text into graphemes, the language specific characters that can be made up of more than one Unicode character.
3. **Sandhi split**: mark boundaries at points where Tamil sound change (sandhi) rules predict a natural word or morpheme join, without rewriting or altering the underlying text.
4. **Sandhi + Grapheme**: combine both, marking sandhi boundaries and splitting into graphemes at the same time, to see how the two behave together.

Two of these four, sandhi split and grapheme split, needed real custom work beyond what the library ships with out of the box. Building a proper custom PreTokenizer class for either one turned out to be too slow in practice, since a custom class runs in Python while every built-in pre-tokenizer runs as compiled Rust code. The shortcut used instead was to precompute the marking ahead of time: generate train and test files where the sandhi boundaries or grapheme placeholders are already baked into the text, then let the tokenizer train on that file directly using its normal, built-in pre-tokenizer. That trade-off carries forward to inference as well. Since the tokenizer itself was never taught to do this marking, any new text has to go through the same sandhi marking or grapheme placeholder substitution first, before it's handed to the tokenizer, otherwise it won't line up with what the tokenizer actually learned. The next two sections walk through each one, along with the problems that came up while building them.
