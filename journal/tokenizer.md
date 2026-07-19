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

Some of the special tokens used in the project include:
- unk: Unknown token, this is the ID assigned to all the words the tokenizer encounters which are not in its vocabulary. For example, a Tamil trained monolingual tokenizer would use `unk` to represent characters from Spanish or English.
- cls: Token added in the post-processing stage. Usually depicts the start of a sequence of text (specific to BERT but used on all models for consistency).
- sep: Token added in the post-processing stage. Usually depicts the sperator between segments of a sequence of text (specific to BERT but used on all models for consistency). 
- pad:
- mask:
- unusedN: Reserved and unused slots which would be occupied during future downstream tasks like fine-tuning.