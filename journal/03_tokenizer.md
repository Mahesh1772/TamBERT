# TamilBERT v1.0 — Tokenizer

> This project predominantly uses the `tokenizer` and it's related classes to execute the comparisons detailed in the README as it is natively run on `RUST` and are quite fast.

## What is a Tokenizer

Tokenizer is a separate component that sits before any model, which converts data in text from to digits form for computer processing.

To efficiently represent text in digits, it undergoes 'traning' to learn the most commonly occuring patters within text and assign a new ID to this recurring pattern (apart from the IDs available to the sub-units). 

>Insert example here

A tokenizer is considered elite when long words and common jargon are represented using few 'token IDs' (ID for different sub-words/words). This gets facilitated by training the tokenizer on a large corpus of rich text and having a sizeable vocabulary

## Tokenizer Vocabulary

`VOCAB_SIZE` - The total number of rows in the tokenizer, where each row represents a mapping from a subword -> ID.

Some rows are reserved for special token/words and are called `special tokens`. These have special meaning and are blocked out from the start to ensure they do not get replaced. Hence the effective size would be `VOCAB_SIZE` - `special tokens`.

> All tokenizer variants (BPE, Unigram) use identical BERT-style bracketed special tokens ([UNK], [CLS], [SEP], [PAD], [MASK]) for consistency across the project, rather than following Unigram's typical SentencePiece-style <unk>/<pad> convention seen in HuggingFace's own examples. This is a deliberate deviation, chosen for cross-variant consistency within TamBERT rather than external convention-matching.

Some of the special tokens used in the project include:
- unk: Unknown token, this is the ID assigned to all the words the tokenizer encounters which are not in its vocabulary. For example, a Tamil trained monolingual tokenizer would use `unk` to represent characters from Spanish or English.
- cls: Token added in the post-processing stage. Usually depicts the start of a sequence of text (specific to BERT but used on all models for consistency).
- sep: Token added in the post-processing stage. Usually depicts the sperator between segments of a sequence of text (specific to BERT but used on all models for consistency). 
- pad: Used to fix batch length mismatches. Adds extra tokens to the end of smaller sentences to reach maximum sentence length (additionally added tokens will have this token ID). These tokens are never attended to, so they don't distort model learning.
- mask: Used to replace some tokens during `step 3: MLM training` to make the model learn what words fit there (model learns tamil grammer and semantics)
- unusedN: Reserved and unused slots which would be occupied during future downstream tasks like fine-tuning.

## Parts of a tokenizer

Eventhough a single mechanism there are inherent parts to it which make configurable.

### Normalization

Functions applied to every line of input, which are used to sanitize and standardize the input before it reaches the subsequent parts. Every sentence is first ran through a sequnce(or just one) normalizations before feeding into `pre-tokenizer`.
Some of the common normalization techiques are Unicode, Text Cleaning, Special purpose and custom built.

#### 1. Unicode normalization

Informs tokenizer how to handle special characters. 

Some characters can be stored in 2 different ways but will look the same when displayed on screen. `NFC` forces everything into a composed codepoint (single codepoint represents the entire visual character) and `NFD` forces everything into split form.
Example with French letter which render as `é`
- Composed: é = 1 codepoint (U+00E9)
- Decomposed: e + ´ = 2 codepoints (U+0065 + U+0301)

Some character which have a stylistic element to them have a slightly additional codepoint added to make it appear that way (like italic, bold). But in essence these can be boiled to their equivalent form. This is what `NFKC` does, it is `NFC` + compatibile variants of unicode points. It is a more agressive version of `NFC`, hence not a good default if specific semantic meaning is carried within the style.

| Input                         | NFC output      | NFKC output                       |
| ----------------------------- | --------------- | --------------------------------- |
| ﬁ (single ligature codepoint) | ﬁ (unchanged)   | fi (split into two plain letters) |
| ① (circled digit one)         | ① (unchanged)   | 1 (plain digit)                   |
| ＡＢＣ (full-width Latin)        | ＡＢＣ (unchanged) | ABC (normal-width)                |
| ² (superscript two)           | ² (unchanged)   | 2 (plain digit)                   |

`NFKD` is just applying `NFD` splitting behaviour + `NFKC` compatibility folding. It is the most agressive out of all 4 normalizers.


| Form | What it does                                                                                          | Splits composed → decomposed? | Folds compatibility variants? | Example                       |
| ---- | ----------------------------------------------------------------------------------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| NFC  | Composes decomposed characters into their canonical single form                                       | No (recomposes)               | No                            | é (decomposed) → é (composed) |
| NFD  | Decomposes composed characters into base + combining marks                                            | Yes                           | No                            | é → e + ́                     |
| NFKC | Like NFC, but also folds compatibility characters (ligatures, special forms) to canonical equivalents | No (recomposes)               | Yes                           | ﬁne café → fine café          |
| NFKD | Like NFD, but also decomposes compatibility characters                                                | Yes                           | Yes                           | ﬁne → f + i + n + e           |


#### 2. Text Cleaning Normalizers

- LowerCase
- StripAccents: Run after NFD to remove special chars split from alphabets
- Strip: Remove whitespace
- Replace: Used to replace string or a pattern within string encapsulated in `Regex`
- Nmt: Prebuilt preprocessing from SentPiece paper (strips control characters, normalizes whitespace, replaces certain problematic Unicode characters).

#### 3. Special purpose Normalizers

Pre-Designed sequence of normalizers introduced in papers
- ByteLevel: Gpt2 paper - convert every byte to a unicode character so there is no need for a [UNK] token
- BertNormalizer: Released with BERT - cleans control characters, adds spaces around CJK characters, optionally strips accents, optionally lowercases
- Precompiled: SentPeice normalizer - loads and precomputed normalization map, is not built.

### PreTokenizer

This step defines how the corpus is split (initially) so that subsequent merges can be learnt to build up tokenizer vocabulary. Since a good tokenizer uses lesser IDs to represent a human readable 'word', we need proper configurations to draw out that result.

To achieve a good tokenizer we need the most commonly occuring phrases/words to have their own unique IDs. Since algorithms use theh frequence of every element in the vocab to figure out the next best token to add to the vocabulary, the starting point of elements is quite an important decision which will impact the quality of subsequent merges and hence the tokenizer.

In practice, different models may prefer different pre-tokenizers depending on the training corpus and the merge behavior of the subword model. For this project, the experiment is kept intentionally narrow: four pre-tokenizer setups are tested, all using the BPE model, so their effect on the assembled Tamil corpus can be compared directly.

##### What does it do?

Pretokenizer splits the text into smallest atomic units from which the 'merging' responsible for tokenizer vocabulary starts. There are many types available, splitting on white space, puncutaion, byte level, certain character specified, digits, graphemes or on space replaced by some character.

There exists certain preconfigured sequence of pretokenizer actions done in a particular order maybe found imperically or specified in a paper. But to keep the comparisons relatable and meaningful, we stick to easily configurable and understandable pre-tokenizers to decipher each ones effect on the corpus/tokenizer performance.


##### Types of Pretokenizers (used in the project)

The exhaustive list of pretokenizers would be endless. Most models in modern architecture use a 'sequence' or a combination of different individual ones. Many used in this project are `tokenizer` pre-built versions which run in `Rust` hence are faster and were preferred. 

The broad available ones in the library are:
| Pre-tokenizer      | What it does                                                                                                                        |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| BertPreTokenizer   | Splits on spaces and on punctuation, treating each punctuation character separately                                                 |
| ByteLevel          | Splits into words and re-represents every byte as a visible character (GPT-2 style, 256-char alphabet)                              |
| CharDelimiterSplit | Splits on a single provided character, like .split(delimiter)                                                                       |
| Digits             | Splits digits from surrounding text, either grouped ("123") or individual ("1","2","3")                                             |
| Metaspace          | Replaces whitespace with a marker character (default ▁) and splits on it — the one your scripts use                                 |
| Punctuation        | Splits on punctuation characters individually, with configurable behavior (isolated, removed, merged, contiguous)                   |
| Sequence           | Composes multiple pre-tokenizers and applies them in order                                                                          |
| Split              | Splits on a custom string or regex pattern, with configurable delimiter behavior and optional inversion                             |
| UnicodeScripts     | Splits on boundaries between different Unicode script families (Latin, Han, etc.), mimicking SentencePiece's Unigram implementation |
| Whitespace         | Splits using the regex \\w+\|[^\\w\\s]+                                                                                             |
| WhitespaceSplit    | Splits purely on whitespace, like .split()                                                                                          |

---
> This project experiments with different types of pretokinzer configurations. The 4 experiments are as follows:
>
> 1. **Unicode split:** Split text into inidividual unicode characters
> 2. **Grapheme split:** Split text into graphemes (language based characters which can be more than a single unicode character)
> 3. **Sandhi split:**  Mark boundaries at points where Tamil phonological (sandhi) rules predict a natural word or morpheme join, without rewriting or altering the underlying text
> 4. **Sandhi + Grapheme:** Mark boundaries using sandhi rules and split into graphemes to tbe the lowest split level to see how combination of both performs.
---

### **Custom PreTokenizer for Tamil**

To impelemt a **sandhi split:** Tamil specific splitting, an approach was followed to build a custom `PreTokenizer` inheriting class. But that was too slow for practicallity as it was a python class and hence all the processing slowed down compared to `Rust`. 

The fix used instead was the following
1. Generate new train and test files with the simulated **sandhi split** and used a special marker `⟂` to demarkate the grapheme boundaries.
2. Train tokenizer and test the fertility metric on the new generated file
3. In the decoder add a additional step to filter out this special char and replace it with nothing (to get back normal text)

This warrants and additional step that was computed by the file - `sandhi_precompute.py`.

#### **What is Sandhi Split**

This is a custom pattern which is used to segment/split the input text so that tamil text can be partitioned correctly into the 'samllest' lingustically alligned word (as certain words in tamil can be decomposed to contain multiple subwords).

This module is adopted from [Aagathiyam: Sandhi aware tokenization for Tamil](https://github.com/RoshiniPriya05/Agathiyam-Tamil/blob/main/Agathiyam-%20Sandhi%20aware%20tokenization%20for%20Tamil%20Language/core/sandhi.py) research paper directly.

The module uses well documented and known tamil word/vowel appreance use cases to properly split at the correct character such that downstream tasks can learn tamil morphology properly. The exhaustive list of splits are detailed in the table below.

| Category                                       | Split rule                                                                           | Better representative example   | Boundary marked                       |
| ---------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------- | ------------------------------------- |
| **Vowel + vowel joins (A)**                    | Boundary before the second vowel when two vowels meet across a word boundary         | மலை அது (மலை + அது)             | Before **அ** in அது                   |
|                                                |                                                                                      | தீ எரிந்தது (தீ + எரிந்தது)     | Before **எ**                          |
|                                                |                                                                                      | பூ ஒளிர்கிறது (பூ + ஒளிர்கிறது) | Before **ஒ**                          |
| **Glide insertion cues (B)**                   | Boundary where a ய்/வ் glide would naturally appear between adjacent vowels          | கிளி அது (கிளி + அது → கிளியது) | Between **ி** and **அ**               |
|                                                |                                                                                      | தெரு அது (தெரு + அது → தெருவது) | Between **உ** and **அ**               |
|                                                |                                                                                      | நீ அவர் (நீ + அவர் → நீயவர்)    | Between **ஈ** and **அ**               |
| **Nasal + stop assimilation (C)**              | Boundary between a nasal and its matching stop consonant                             | தங்கை → தங் | கை                | Between **ங்** and **க**              |
|                                                |                                                                                      | பந்தம் → பந் | தம்              | Between **ந்** and **த**              |
|                                                |                                                                                      | கம்பம் → கம் | பம்              | Between **ம்** and **ப**              |
| **Gemination / doubling (D)**                  | Boundary between identical doubled consonants                                        | பக்கம் → பக் | கம்              | Between **க்** and **க**              |
|                                                |                                                                                      | வெள்ளை → வெள் | ளை              | Between **ள்** and **ள**              |
|                                                |                                                                                      | அன்னை → அன் | னை                | Between **ன்** and **ன**              |
| **திரிதல் mutation cues (E)**                  | Boundary at classic consonant mutation environments                                  | கல் சிலை (ல் + ச)               | Between **ல்** and **ச**              |
|                                                |                                                                                      | பேர் ராஜா (ர் + ர)              | Between **ர்** and **ர**              |
|                                                |                                                                                      | பொன் தட்டு (ன் + த)             | Between **ன்** and **த**              |
| **கெடுதல் final-consonant-loss (F)**           | Boundary immediately after a consonant that commonly elides before a following vowel | மரக் கிளை                       | After **க்**                          |
|                                                |                                                                                      | வரும் அவன்                      | Between **ம்** and **அ**              |
|                                                |                                                                                      | படித் அவர்                      | Between **த்** and **அ**              |
| **Case suffix / postposition joins (G)**       | Boundary between a stem and a following case suffix or postposition                  | மரம் ஐ                          | Before **ஐ**                          |
|                                                |                                                                                      | வீடு க்கு                       | Before **க்கு**                       |
|                                                |                                                                                      | வீடு இல்                        | Before **இல்**                        |
|                                                |                                                                                      | அவன் உடன்                       | Before **உடன்**                       |
| **Verbal participle & auxiliary joins (H)**    | Boundary between a participle and its auxiliary verb                                 | படி இரு                         | Between **படி** and **இரு**           |
|                                                |                                                                                      | செய்து விடு                     | Between **செய்து** and **விடு**       |
|                                                |                                                                                      | எடுத்து கொள்                    | Between **எடுத்து** and **கொள்**      |
| **Numeral + classifier / suffix (I)**          | Boundary between a numeral and a following ordinal or case suffix                    | 10 ஆம்                          | Before **ஆம்**                        |
|                                                |                                                                                      | 5 ஐ                             | Before **ஐ**                          |
|                                                |                                                                                      | 12 ஆண்டு                        | Before **ஆண்டு** (ordinal expression) |
| **Whitespace boundaries (separate mechanism)** | Boundary at every space ↔ non-space transition                                       | அம்மா வீட்டில்                  | அம்மா | ␠ | வீட்டில்                  |
---

#### **How the boundary-marking works, step by step**
- The script scans the raw Tamil text using all the regex rules (Sections A–I) — it does not rewrite or change any text at this stage, it only looks for pattern matches.
- Every time a rule matches, it records just one number: the position right before the second part of the match. This is the "cut point."
- All these cut points get collected into a set (duplicates removed automatically since it's a set).
- Cut points sitting at the very start (position 0) or very end of the text are dropped, since marking a boundary there wouldn't do anything.
- If no cut points remain after that, the original text is returned unchanged — no marker added.
- If cut points do exist, the text gets sliced at each cut point, one by one, from left to right.
- Between each slice, the marker symbol ⟂ gets inserted.
- The final result is: original text, but with ⟂ dropped in at every sandhi boundary — nothing else about the text is touched or reordered.
- Because ⟂ is only ever inserted (never replacing or deleting anything), removing all the ⟂ symbols afterward gives back the exact original text, with zero information lost.
- This marked version (original text + ⟂ symbols) is what gets fed into the tokenizer training scripts, so the tokenizer can learn to treat these sandhi points as meaningful split locations.

Above proecess found in `sandhi.py`.

### What a "grapheme split" is, and why the first attempt was silently wrong

A Tamil grapheme cluster — one visual "letter" a reader perceives as a single unit, which is often *more than one Unicode codepoint* under the hood: a base consonant plus a dependent vowel sign, sometimes plus a virama. `கா` looks like one character but is two codepoints. The tokenizer, left to its own devices, works at the codepoint level, so nothing stops it from learning a token boundary that falls *inside* a grapheme cluster that are semantically meaningless, splits that shouldn't exist.

#### Failed first attempt to enforce grapheme to be the smallest unit

The first attempt to prevent this: collect every distinct grapheme cluster in the corpus (`regex.findall(r"\X", ...)`) and hand that list to the trainer via `initial_alphabet`, hoping it would seed each cluster as one atomic starting unit.

This failed silently. The `tokenizers` library documents `initial_alphabet` as keeping only the *first character* of any multi-character string passed in. So instead of seeding `கா` as one atomic 2-codepoint unit, it silently kept just `க`. The "grapheme-aware" seeding was a complete no-op, and nothing in the run would have told us that. This is almost certainly the same root cause as the ~1,492-entry vocab cap hit earlier.

#### How it's actually fixed

`initial_alphabet` can't hold multi-codepoint entries — there's no parameter-level fix. Instead, the fix happens to the *text itself*, before the tokenizer ever sees it:

1. Scan the corpus for every distinct grapheme cluster that spans more than one codepoint.
2. Assign each one a single placeholder codepoint, pulled from the Unicode Private Use Area (an unused range with no assigned meaning of its own — safe to repurpose).
3. Rewrite the corpus, swapping every occurrence of a multi-codepoint cluster for its one-codepoint placeholder.

Since a placeholder is exactly one codepoint by construction, the trainer can't split it. Crucially, this is different from trying to isolate each grapheme cluster as its own pre-tokenizer split: that approach was floated and rejected separately, because pre-tokenizer boundaries are hard walls the model can never merge across. That would cap every token at exactly one grapheme cluster, never letting multiple clusters combine into a bigger subword token, which is the entire point of running BPE/Unigram in the first place. The placeholder swap avoids this because it happens at the character level, so merging across (now single-codepoint) grapheme boundaries within a word works exactly as it always did.

#### Why we regenerated the sandhi-marked files

Sandhi rules to split the text have been updated to include more commonly found sandhi joins and frequently used unique joins like with directions. Another fix includes the removal of `\s*` from the rules. This is what caused false marker insertions across spaces. Now the files created are de-boated and only contain the marks we would essentially need. 

Since one of our combinations require grapheme on top of sandhi split file, we make 2 pairs of files in the `grapheme_precompute.py`. One for codepoint and one for sandhi split text, to merge grapheme wise.

#### How this affects the metrics

`calculate_fertility` / `calculate_oov_rate` / `calculate_tokenizer_metrics` don't inspect *what* characters make up a line, they call `line.split()` for the word count and `tokenizer.encode(line)` for the token count. Placeholder substitution never touches whitespace, only the multi-codepoint clusters between spaces, so the word count (the fertility denominator) is identical whether you measure on raw text or on the placeholder-substituted version. As long as the tokenizer and the test file being measured are in the *same* representation — both placeholder-form, or both real-text-form, never mixed: the numbers are correct and comparable. That's exactly how the scripts are structured: `train_metrics = calculate_tokenizer_metrics(...)` runs against `paths.test_grapheme_marked` (or the sandhi+grapheme equivalent) while the in-memory tokenizer is still in placeholder form — consistent, before anything gets relabelled.

#### How the decoder stays the same and keeps working

- `03` (grapheme only): `decoders.Metaspace()`. Its whole job is turning `▁`-marked tokens back into correctly spaced text. It doesn't matter whether a token's characters are placeholder codepoints (during training) or real Tamil text (after relabelling) — the reassembly logic is identical either way.
- `05` (sandhi + grapheme): `decoders.Sequence([Metaspace(), Replace(Regex("⟂"), "")])`. Same Metaspace reasoning, plus a literal find-and-strip of `⟂`. The sandhi marker is a single codepoint, so it was never a candidate for placeholder substitution (only multi-codepoint clusters get remapped). It passes through the whole pipeline, training data and relabelled vocab alike, completely unchanged. The strip step keeps working without modification because there's nothing about it that depended on the grapheme fix in the first place.

#### What the new files actually contain

- `grapheme_placeholder_map.json` — the reversible lookup table: each real multi-codepoint grapheme cluster mapped to its assigned placeholder codepoint. This is what makes the whole scheme undoable after training.
- `train_grapheme_marked` / `test_grapheme_marked` — the plain corpus with every multi-codepoint cluster swapped for its placeholder. Word boundaries, punctuation, digits, and any already-single-codepoint Tamil characters are untouched. Only genuine multi-codepoint clusters change. Most fonts will render the placeholders as blank boxes if you open the file directly, this is expected since Private Use Area codepoints have no defined glyphs.
- `train_sandhi_grapheme_marked` / `test_sandhi_grapheme_marked` — the same swap applied on top of the already sandhi-marked files, so both markings coexist in one file for the `04` variant.

None of these are meant to be permanent artifacts of the project — they're training-time intermediates. The final saved tokenizer, after `restore_vocab_in_place` relabels its vocab, is the only piece meant to be used going forward, and it works on ordinary raw Tamil text with no placeholder map required at inference time.
