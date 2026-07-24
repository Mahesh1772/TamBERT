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

#### **Custom PreTokenizer for Tamil**

To impelemt a **sandhi split:** Tamil specific splitting, an approach was followed to build a custom `PreTokenizer` inheriting class. But that was too slow for practicallity as it was a python class and hence all the processing slowed down compared to `Rust`. 

The fix used instead was the following
1. Generate new train and test files with the simulated **sandhi split** and used a special marker `⟂` to demarkate the grapheme boundaries.
2. Train tokenizer and test the fertility metric on the new generated file
3. In the decoder add a additional step to filter out this special char and replace it with nothing (to get back normal text)

This warrants and additional step that was computed by the file - `sandhi_precompute.py`.

#### **What is Sandhi Split**

This is a custom pattern which is used to segment/split the input text so that tamil text can be partitioned correctly into the 'samllest' lingustically alligned character (even if it spans multiple unicode characters).

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