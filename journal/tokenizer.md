# Notebook 2: TamBERT Tokenizer

## Vocabulary size decision = 32k (with unused tokens) [30+2]

The size of mapping from code point -> token IDs.

Increasing vocab size:
- Every ultra rare elements and sequences will have a unique token ID associated with it.
- The llm can attend to more parts of the input sequence (**this is good**)
- Might encode entire phrases as a single token, this might not give the llm enough information to process during a forward pass

Special tokens in vocabulary:
1. Need a `<|end-of-text|>` token to mark end of all input to llm.
    - special token handled during tokenizer traing, not result of BPE merges
2. `Fill in the Middle` tokens from [research paper](https://arxiv.org/pdf/2207.14255)
3. `<|end-of-prompt|>`

As Since embedding_table.shape = [vocab_size, embed_dim], and the final layer of a llm (transformer) would be a linear layer (fully connected layer) which will predict the logits for each token ID in the vocabulary (predict the next word/sub-word to generate next, has a `vocab_size` neurons. So increase in vocab_size will increase the size of the table and the nn.Dense() layer for logit production.

## Tokenizer method [BPE (Byte Piece Encoder)]
- Successively mint new tokens from the most commonly occuring pair of elements

SDKs that use BPE
1. `tiktoken` from openAI, scripts for inference and trained gpt-4 and gpt-2 tokenizers available. No option for training the tokenizers.
    a. chars -> unicode bytes -> token IDs  
2. `sentencepiece` from google provides scirpts for training and inference.
    a. chars -> token IDs.
    b. Rare chars that only appear a few times are [UNK] or are turned to bytes then utf-8 encoded into raw bytes instead -> token IDs

## SentencePiece Training config

- No normalization (keep the data as is in raw form, to be representative of the original text - `Andrej Karpathy`)
- Sentence related parameters should be turned off. Set the sentence length and stuff to high numbers
- Treatment of rare words: enable byte fallback
- merge rules: split_digits, split_by_whiespace, split_by_number, split_by_unicode_script, max_sentencepiece_length, add_dummy_prefix, allow_whitespace_only_pieces
- Andd in the special token IDs (UNK, pad_id, bos_id, eos_id)
- System resources: num_threads
- Enable prefix space. This is to ensure, words at the start of a sentence would be considered a repetition to the same word that appears in the middle of a chunk of text
```
'word' = 'hello| world'
```

## Token sequence:
1. special tokens
2. byte tokens
3. merged tokens
4. individual code point tokens (the coverage hyperparameter controls how many times a single point should be present within the training text to be added as to the token seq)

Tokenizer behaviour when encoutering new chars not part of train set:
1. If byte fallback is enabled: The characters get token ID of one of the byte_fallback bytes within the vocabulary
2. If byte fallback is disabled: The chunk of new chars becomes UNK (entire sequence of new chars become UNK)

### Fine tuning : Increasing Vocabulary Size
- Reseize the embedding table to extend to new vocabulary size to introduce new tokens (for web search, different languages) and the rows will be initialized with random weights, and change the llm head to be able to predict logis for this new token.
- We can freeze the pre-trained model and only work with the newly added rows to embeding table and retrain the head with the new data introduced.

## Common Tokenization failures
- Why can't LLM spell words? **Tokenization**.
- Why can't LLM do super simple string processing tasks like reversing a string? **Tokenization**.
- Why is LLM worse at non-English languages (e.g. Japanese)? **Tokenization**.
- Why is LLM bad at simple arithmetic? **Tokenization**.
- Why did GPT-2 have more than necessary trouble coding in Python? **Tokenization**.
- Why did my LLM abruptly halt when it sees the string "<|endoftext|>"? **Tokenization**.
- What is this weird warning I get about a "trailing whitespace"? **Tokenization**.
- Why the LLM break if I ask it about "SolidGoldMagikarp"? **Tokenization**.
- Why should I prefer to use YAML over JSON with LLMs? **Tokenization**.
- Why is LLM not actually end-to-end language modeling? **Tokenization**.
- What is the real root of suffering? **Tokenization**.


## Workflow for Tokenizer Notebook

| Test | Goal                | Fixed                                                        | Varied              | Options tested                                                                                 | # Runs | Metric                                                 | Depends on              |
| ---- | ------------------- | ------------------------------------------------------------ | ------------------- | ---------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------ | ----------------------- |
| 1    | Best normalization  | Pre-tokenizer = Whitespace; Algo = BPE; merge count          | Normalization       | NFC vs. None                                                                                   | 2      | OOV, Fertility                                         | —                       |
| 2    | Best pre-tokenizer  | Normalization = Test 1 winner; Algo = BPE; merge count       | Pre-tokenizer stack | (a) Whitespace + codepoint-BPE, (b) Whitespace + grapheme-BPE, (c) Sandhi-split + grapheme-BPE | 3      | OOV, Fertility                                         | Test 1 result           |
| 3    | Best tokenizer algo | Normalization = Test 1 winner; Pre-tokenizer = Test 2 winner | Tokenizer algorithm | BPE, Unigram, WordPiece(BERT), GPE (grapheme-only, reference floor)                            | 4      | OOV, Fertility (+ note GPE is non-comparable baseline) | Test 1 + Test 2 results |