# TamilBERT v1.0 — PreTokenizer

This is the first configuration of `Tokenizer` in huggingface `tokenizer` API. This step defines how the corpus is split (initially) so that subsequent merges can be learnt to build up tokenizer vocabulary. Since a good tokenizer uses lesser IDs to represent a human readable 'word', we need proper configurations to draw out that result.

To achieve a good tokenizer we need the most commonly occuring phrases/words to have their own unique IDs. Since algorithms use theh frequence of every element in the vocab to figure out the next best token to add to the vocabulary, the starting point of elements is quite an important decision which will impact the quality of subsequent merges and hence the tokenizer.

In practice, different models may prefer different pre-tokenizers depending on the training corpus and the merge behavior of the subword model. For this project, the experiment is kept intentionally narrow: four pre-tokenizer setups are tested, all using the BPE model, so their effect on the assembled Tamil corpus can be compared directly.

## What does it do?

Pretokenizer splits the text into smallest atomic units from which the 'merging' responsible for tokenizer vocabulary starts. There are many types available, splitting on white space, puncutaion, byte level, certain character specified, digits, graphemes or on space replaced by some character.

There exists certain preconfigured sequence of pretokenizer actions done in a particular order maybe found imperically or specified in a paper. But to keep the comparisons relatable and meaningful, we stick to easily configurable and understandable pre-tokenizers to decipher each ones effect on the corpus/tokenizer performance.


## Types of Pretokenizers (used in the project)

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

### Custom PreTokenizer for Tamil

To impelemt a **sandhi split:** Tamil specific splitting, an approach was followed to build a custom `PreTokenizer` inheriting class. But that was too slow for practicallity as it was a python class and hence all the processing slowed down compared to `Rust`. 

The fix used instead was the following
1. Generate new train and test files with the simulated **sandhi split** and used a special marker `⟂` to demarkate the grapheme boundaries.
2. Train tokenizer and test the fertility metric on the new generated file
3. In the decoder add a additional step to filter out this special char and replace it with nothing (to get back normal text)

This warrants and additional step that was computed by the file - `sandhi_precompute.py`.




## Special Tokens

The sequence of special tokens mean different things to the tokenizer. Some of them map unknown words to a fixed `unk_token`'s  ID 


> All tokenizer variants (BPE, Unigram) use identical BERT-style bracketed special tokens ([UNK], [CLS], [SEP], [PAD], [MASK]) for consistency across the project, rather than following Unigram's typical SentencePiece-style <unk>/<pad> convention seen in HuggingFace's own examples. This is a deliberate deviation, chosen for cross-variant consistency within TamBERT rather than external convention-matching.