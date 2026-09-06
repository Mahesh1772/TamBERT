---
title: 6. Tokenizer Training
nav_order: 6
---

# TamilBERT v1.0 — Tokenizer Training

## What training actually produces

Training a tokenizer produces a single `.json` file. At its core, that file is a map between the most frequent pieces of text found in the corpus and the token IDs, plain numbers, that the model actually reads. "Most frequent" is really only the deciding factor for BPE; WordPiece and Unigram, the other two methods covered here, use their own criteria to decide what goes into that map, even though all three end up producing the same kind of file.

This map gets built by scanning the training corpus over and over, refining the vocabulary a little more with each pass, until it reaches whatever size was requested ahead of time. Three different methods for building that map are covered in this project: BPE, WordPiece, and Unigram. Each one has its own advantages, and each one arrives at a similarly sized vocabulary through a genuinely different process.

## BPE (Byte Pair Encoding)

BPE starts as small as possible. The starting vocabulary is just the smallest atomic units the text can be broken into, individual characters or bytes, nothing bigger. From there, the corpus gets scanned to count how often every adjacent pair of units shows up next to each other. Whichever pair shows up the most gets merged into a single new unit and added to the vocabulary. That merged unit is now treated exactly like any other atomic unit going forward, which means it's eligible to be merged again in a later round. This counting and merging cycle repeats, one merge at a time, until the vocabulary reaches its target size.

For example, imagine a small corpus made up mostly of the words "low", "lower", and "lowest", each appearing many times. Early on, the most frequent adjacent pair is likely to be `l` and `o`, so `lo` gets merged into a single unit. On a later pass, `lo` and `w` might turn out to be the most frequent pair, so `low` gets merged next. The process keeps going like this, always finding whatever pair is most common at that moment and locking it in as a new unit, until the vocabulary is full.

## WordPiece

WordPiece is the tokenizer behind the original BERT model. It follows the same general shape as BPE: start from a vocabulary of small units, repeatedly merge adjacent pairs into a new unit, and add that new unit to the vocabulary. The real difference between the two is the criterion used to decide which pair to merge next.

Where BPE just picks whichever pair occurs most often, WordPiece scores every candidate pair using this formula:

$$
\text{score}(a, b) = \frac{\text{freq}(a, b)}{\text{freq}(a) \times \text{freq}(b)}
$$

This score is highest for pairs that show up together far more often than you'd expect from how common each piece is on its own. That has a useful side effect: it favors merging pairs that form a genuinely meaningful new unit, rather than merging two units just because they're both extremely common individually and therefore bump into each other a lot by coincidence. On every pass, the pair with the highest score gets merged and added to the vocabulary, and the cycle repeats until the vocabulary reaches its target size.

For example, in a corpus full of English text, the units `un` and `happy` might individually be very common, but if they only actually appear next to each other a modest number of times, their score stays low. A pair like `play` and `ing`, which shows up together disproportionately often relative to how common each piece is alone, would score higher and get merged sooner.

## Unigram

Unigram works in the opposite direction from BPE and WordPiece. Instead of starting small and merging upward, it starts with a huge candidate vocabulary, essentially every character plus a large number of candidate subwords pulled from the corpus, and prunes that vocabulary down until it reaches the target size.

Every candidate subword in this starting vocabulary gets assigned a probability, initially based on how often it shows up in the corpus. Once every subword has a probability, any given word can be split in several different ways, and each of those ways (called a segmentation) has its own combined probability, found by multiplying together the probabilities of its individual pieces:

$$
P(\text{segmentation}) = \prod_{i} P(\text{piece}_i)
$$

Training then alternates between two steps, repeated over and over, an approach generally known as the EM algorithm. First, using the current probabilities, the algorithm works out how much each subword is actually being relied on across every plausible way the corpus could be segmented, not just the single best guess for each word, but a properly weighted mix of all the reasonable ways it could be split. Second, it uses that information to re-estimate every subword's probability. These two steps pull each other toward consistency, and repeating them enough times settles the probabilities into a stable state.

Once the probabilities settle, the vocabulary gets trimmed. The subwords removed are the ones that would hurt the corpus's overall likelihood the least if they disappeared, not simply the ones that occur the least often. That distinction matters: a rare subword can still be essential if it happens to be the only way certain words could be represented, so pruning by pure frequency would throw away the wrong things. This train-then-prune cycle repeats until the vocabulary lands on its target size.

For example, take a word like "playing" and imagine the vocabulary contains both whole pieces like `play` and `ing`, as well as smaller fragments like `p`, `l`, `a`, `y`. Splitting the word as `play` + `ing` multiplies together only two, reasonably high, probabilities. Splitting it letter by letter multiplies together seven much smaller probabilities. Since each additional piece in a segmentation multiplies in another number smaller than one, the shorter, more meaningful split usually wins out, unless the individual pieces of the longer split happen to be unusually common. At the end of training, a word only needs its single most probable segmentation, and that best split gets found efficiently using a technique called Viterbi decoding, which works through the word once from left to right rather than checking every possible way to split it.

---

These three methods, BPE, WordPiece, and Unigram, are the ones trained and compared throughout the rest of this project.