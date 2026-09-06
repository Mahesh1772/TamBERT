---
title: 5. Grapheme Aware Splitting
nav_order: 5
---

# TamilBERT v1.0 — Grapheme Aware Splitting

## What a grapheme cluster is

A Tamil grapheme cluster is one visual letter, the single unit a reader actually perceives, even though it's often built from more than one Unicode codepoint underneath. A base consonant combines with a dependent vowel sign, and sometimes a virama mark on top of that. கா looks like a single character on screen, but it's stored as two separate codepoints.

Left on its own, a tokenizer works at the codepoint level. Nothing stops it from learning a token boundary that falls right inside a grapheme cluster, a split that has no linguistic meaning and shouldn't exist in the first place.

## The first attempt, and why it silently failed

The first attempt to prevent this was straightforward on paper: collect every distinct grapheme cluster found in the corpus using `regex.findall(r"\X", ...)`, then hand that full list to the trainer through its `initial_alphabet` parameter. The hope was that each cluster would get seeded as one atomic starting unit before training began.

This failed, and it failed silently. The `tokenizers` library documents `initial_alphabet` as keeping only the first character of any multi-character string passed into it. So instead of seeding கா as one atomic two codepoint unit, it silently kept only க. The entire grapheme aware seeding step was a no-op, and nothing about the training run signaled that anything had gone wrong. This is almost certainly the same root cause behind an earlier vocabulary cap that had been hit and not yet explained, around 1,492 entries.

## How it was actually fixed

`initial_alphabet` simply cannot hold multi-codepoint entries. There's no parameter level fix available for that. So the fix happens to the text itself, before the tokenizer ever sees it:

1. Scan the corpus for every distinct grapheme cluster that spans more than one codepoint.
2. Assign each one a single placeholder codepoint, pulled from the Unicode Private Use Area, a reserved range with no assigned meaning of its own, and safe to repurpose this way.
3. Rewrite the corpus, swapping every occurrence of a multi-codepoint cluster for its one codepoint placeholder.

Since a placeholder is exactly one codepoint by construction, the trainer physically cannot split it any further.

This is a different fix from making each grapheme cluster its own pre-tokenizer split, an idea that was considered and rejected separately. Pre-tokenizer boundaries are hard walls that the model can never merge back across, so that approach would have capped every token at exactly one grapheme cluster, and never let multiple clusters combine into a larger subword token, which defeats the entire point of running BPE, WordPiece, or Unigram in the first place. The placeholder swap avoids this problem because it happens at the character level. Merging across grapheme boundaries within a word still works exactly as it always did, just on placeholder codepoints instead of raw ones.

## How this affects the metrics

`calculate_fertility`, `calculate_oov_rate`, and `calculate_tokenizer_metrics` don't look at what characters actually make up a line. They call `line.split()` to get the word count, and `tokenizer.encode(line)` to get the token count.

Placeholder substitution never touches whitespace, only the multi-codepoint clusters that sit between spaces. That means the word count, the denominator used in the fertility metric, comes out identical whether it's measured on the raw text or on the placeholder substituted version.

The one rule that has to hold is that the tokenizer and the file being measured stay in the same representation, both in placeholder form, or both in real text form, never mixed. As long as that holds, the numbers stay correct and comparable. That's exactly how the scripts are structured: `train_metrics = calculate_tokenizer_metrics(...)` runs against `paths.test_grapheme_marked` (or the sandhi plus grapheme equivalent) while the tokenizer sitting in memory is still in its placeholder form. Everything stays consistent, right up until anything gets relabeled back to real text.

## How the decoder keeps working

- **Grapheme only setup**: uses `decoders.Metaspace()`. Its whole job is turning `▁` marked tokens back into properly spaced text, and that logic doesn't care whether a token's characters are placeholder codepoints (during training) or real Tamil text (after relabeling). The reassembly works identically either way.
- **Sandhi plus grapheme setup**: uses `decoders.Sequence([Metaspace(), Replace(Regex("⟂"), "")])`. Same Metaspace reasoning as above, plus a literal find and strip of the ⟂ marker. Since the sandhi marker is only ever a single codepoint, it was never a candidate for placeholder substitution in the first place, only multi-codepoint clusters get remapped. It passes through the whole pipeline untouched, in both the training data and the relabeled vocabulary alike, so the strip step keeps working without any changes of its own.

## What the new files contain

- **`grapheme_placeholder_map.json`**: the reversible lookup table. Each real multi-codepoint grapheme cluster is mapped to the placeholder codepoint assigned to it. This table is what makes the whole scheme undoable after training finishes.
- **`train_grapheme_marked` / `test_grapheme_marked`**: the plain corpus with every multi-codepoint cluster swapped for its placeholder. Word boundaries, punctuation, digits, and any character that was already a single codepoint stay untouched, only genuine multi-codepoint clusters change. Most fonts will render the placeholders as blank boxes if the file is opened directly, which is expected, Private Use Area codepoints have no glyphs assigned to them.
- **`train_sandhi_grapheme_marked` / `test_sandhi_grapheme_marked`**: the same swap applied on top of the already sandhi marked files, so both kinds of marking exist together in one file, built for the sandhi plus grapheme experiment.

These files come out of `grapheme_precompute.py`, which produces two paired sets: one built from plain codepoint text, and one built from the sandhi split text, so the grapheme swap can run over both.

None of these files are meant to stick around as permanent parts of the project. They're training time intermediates. The only piece meant to be used going forward is the final saved tokenizer, once `restore_vocab_in_place` relabels its vocabulary back to real text. That tokenizer works directly on ordinary raw Tamil text, with no placeholder map needed at inference time.

## Final note

The tokenizer stage ended up teaching the same lesson as the data stage: check assumptions rather than trusting them. The `initial_alphabet` failure is the clearest example, a fix that looked correct on paper did nothing at all, and only close inspection of how the library actually behaves caught it. Between the sandhi rules and the grapheme placeholder scheme, this stage ends with four pre-tokenizer setups, unicode split, grapheme split, sandhi split, and sandhi plus grapheme, all trained under matching conditions, so their fertility and OOV numbers can be compared fairly against each other.
