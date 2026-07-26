# Notebook 2: TamBERT Tokenizer

## Vocabulary size decision = 32k (with unused tokens) [31+1]

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

As Since embedding_table.shape = [vocab_size, embed_dim], and the final layer of a llm (transformer) would be a linear layer (fully connected layer) which will predict the logits for each token ID in the vocabulary (predict the next word/sub-word) to generate next, has a `vocab_size` neurons. So increase in vocab_size will increase the size of the table and the nn.Dense() layer for logit production.

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

### Folder structure

```
tambert/
├── data/
│   ├── raw/
│   │   ├── tamil_wiki/
│   │   ├── cc100/
│   │   └── project_madurai/
│   ├── cleaned/
│   │   ├── tamil_wiki.csv
│   │   ├── cc100.csv
│   │   └── project_madurai.csv
│   └── corpus/
│       ├── corpus_90pct.txt
│       └── corpus_10pct.txt
│
├── notebooks/
│   ├── 01_fetch_and_clean.ipynb       ← experimental, messy, exploratory
│   ├── 02_train_tokenizer.ipynb
│   ├── 03_mlm_pretrain.ipynb
│   ├── 04_nli_finetune.ipynb
│   └── 05_sts_finetune_and_eval.ipynb
│
├── scripts/
    ├── paths.py
    ├── utils.py
    ├── data_pipeline/
    │   ├── 01_extract_wiki.py
    │   ├── 02_filter_cc100.py
    │   ├── 03_scrape_madurai.py
    │   ├── 04_clean_data.py
    │   ├── 05_merge_corpus.py
    │   └── 06_calculate_corpus_metrics.py
    └── tokenizer/
        ├── sandhi.py                          # shared sandhi-split module (imported, not run standalone)
        ├── grapheme_pretokenizer.py           # shared GraphemePreTokenizer class
        ├── metrics.py                         # calculate_fertility + calculate_oov (shared by everything below)
        ├── pretokenizer/
        │   ├── 01_whitespace_codepoint.py
        │   ├── 02_whitespace_grapheme.py
        │   ├── 03_sandhi_codepoint.py
        │   └── 04_sandhi_grapheme.py
        └── training/
            ├── 01_train_bpe.py
            ├── 02_train_unigram.py
            ├── 03_train_wordpiece.py
            ├── 04_train_grapheme_only.py
            └── 05_evaluate_all.py             # loads each saved tokenizer, runs OOV+fertility, outputs comparison CSV
│
├── tokenizer/                         ← saved tokenizer artifacts go here
├── checkpoints/                       ← model checkpoints
├── results/                           ← eval outputs, MTEB results
└── README.md
```

# Journal — grapheme-atomic Unigram training fix

## What I was doing

Comparing four tokenizer configurations (whitespace vs. sandhi pre-tokenization × codepoint vs. grapheme handling) across BPE and Unigram, to see which setup gives the best fertility on Tamil. The BPE variants were already trained and treated as the working reference. Moving on to the Unigram variants (`03_train_unigram_grapheme.py`, `04_train_unigram_sandhi.py`, `05_train_unigram_sandhi_grapheme.py`) before running them for real.

## What a "grapheme split file" is, and why the first attempt was silently wrong

A Tamil grapheme cluster — one visual "letter" a reader perceives as a single unit — is often *more than one Unicode codepoint* under the hood: a base consonant plus a dependent vowel sign, sometimes plus a virama. `கா` looks like one character but is two codepoints. The tokenizer, left to its own devices, works at the codepoint level, so nothing stops it from learning a token boundary that falls *inside* a grapheme cluster — semantically meaningless, splits that shouldn't exist.

The first attempt to prevent this: collect every distinct grapheme cluster in the corpus (`regex.findall(r"\X", ...)`) and hand that list to the trainer via `initial_alphabet`, hoping it would seed each cluster as one atomic starting unit.

This failed silently. The `tokenizers` library documents `initial_alphabet` as keeping only the *first character* of any multi-character string passed in — no error, no warning. So instead of seeding `கா` as one atomic 2-codepoint unit, it silently kept just `க` — which the trainer would have picked up on its own anyway, since single codepoints are common. The "grapheme-aware" seeding was a complete no-op, and nothing in the run would have told us that. This is almost certainly the same root cause as the ~1,492-entry vocab cap hit earlier — a hidden truncation, not a crash.

## How it's actually fixed

`initial_alphabet` can't hold multi-codepoint entries — there's no parameter-level fix. Instead, the fix happens to the *text itself*, before the tokenizer ever sees it:

1. Scan the corpus for every distinct grapheme cluster that spans more than one codepoint.
2. Assign each one a single placeholder codepoint, pulled from the Unicode Private Use Area (an unused range with no assigned meaning of its own — safe to repurpose).
3. Rewrite the corpus, swapping every occurrence of a multi-codepoint cluster for its one-codepoint placeholder.

Since a placeholder is exactly one codepoint by construction, the trainer can't split it — atomicity falls out automatically, no special training parameter needed. Crucially, this is different from (and doesn't repeat the mistake of) trying to isolate each grapheme cluster as its own pre-tokenizer split: that approach was floated and rejected separately, because pre-tokenizer boundaries are hard walls the model can never merge across — it would have capped every token at exactly one grapheme cluster, never letting multiple clusters combine into a bigger subword token, which is the entire point of running BPE/Unigram in the first place. The placeholder swap avoids this because it happens at the character level, *before* pre-tokenization — ordinary word-level Metaspace splitting still applies on top, so merging across (now single-codepoint) grapheme boundaries within a word works exactly as it always did.

## Why we regenerated the sandhi-marked files

Not a re-do of the sandhi marking itself — the existing `train_sandhi_marked` / `test_sandhi_marked` files (from the original sandhi precompute step) are untouched and still correct. What's new is a second, layered precompute pass (`grapheme_precompute.py`) that reads those *already sandhi-marked* files and applies the placeholder swap on top, producing `train_sandhi_grapheme_marked` / `test_sandhi_grapheme_marked`. Two markings stacked on the same text: the sandhi boundary marker (⟂) from before, plus the new grapheme placeholders. The `04_train_unigram_sandhi.py` (sandhi only, no grapheme handling) is unaffected by any of this — it never used `initial_alphabet` in the first place, so there was nothing to fix there.

## How this affects the metrics

`calculate_fertility` / `calculate_oov_rate` / `calculate_tokenizer_metrics` don't inspect *what* characters make up a line — they call `line.split()` for the word count and `tokenizer.encode(line)` for the token count. Placeholder substitution never touches whitespace, only the multi-codepoint clusters between spaces, so the word count (the fertility denominator) is identical whether you measure on raw text or on the placeholder-substituted version. As long as the tokenizer and the test file being measured are in the *same* representation — both placeholder-form, or both real-text-form, never mixed — the numbers are correct and comparable. That's exactly how the scripts are structured: `train_metrics = calculate_tokenizer_metrics(...)` runs against `paths.test_grapheme_marked` (or the sandhi+grapheme equivalent) while the in-memory tokenizer is still in placeholder form — consistent, before anything gets relabelled. No changes needed to `metrics.py` itself; it was already representation-agnostic.

Worth doing as a follow-up, not yet in the script: after the saved tokenizer gets relabelled back to real text, reload it fresh and re-run `calculate_tokenizer_metrics` against the *original* raw test file (`paths.test`, not the marked one) as a second, independent confirmation that the relabel didn't change anything. The script's current sanity check only confirms one sample sentence encodes correctly — a full metrics re-run on the real test file would confirm the fertility number itself survived the relabel intact.

## How the decoder stays the same and keeps working

Neither decoder needed to change, and for the same underlying reason in both cases: decoding only cares about the whitespace marker (`▁`) and, for the sandhi variant, the literal `⟂` character — never about what characters make up the rest of a token string.

- `03` (grapheme only): `decoders.Metaspace()`. Its whole job is turning `▁`-marked tokens back into correctly spaced text. It doesn't matter whether a token's characters are placeholder codepoints (during training) or real Tamil text (after relabelling) — the reassembly logic is identical either way.
- `05` (sandhi + grapheme): `decoders.Sequence([Metaspace(), Replace(Regex("⟂"), "")])`. Same Metaspace reasoning, plus a literal find-and-strip of `⟂`. The sandhi marker is a single codepoint, so it was never a candidate for placeholder substitution (only multi-codepoint clusters get remapped) — it passes through the whole pipeline, training data and relabelled vocab alike, completely unchanged. The strip step keeps working without modification because there's nothing about it that depended on the grapheme fix in the first place.

## What the new files actually contain

- `grapheme_placeholder_map.json` — the reversible lookup table: each real multi-codepoint grapheme cluster mapped to its assigned placeholder codepoint. This is what makes the whole scheme undoable after training.
- `train_grapheme_marked` / `test_grapheme_marked` — the plain corpus with every multi-codepoint cluster swapped for its placeholder. Word boundaries, punctuation, digits, and any already-single-codepoint Tamil characters are untouched; only genuine multi-codepoint clusters change. Most fonts will render the placeholders as blank boxes if you open the file directly — expected, since Private Use Area codepoints have no defined glyphs.
- `train_sandhi_grapheme_marked` / `test_sandhi_grapheme_marked` — the same swap applied on top of the already sandhi-marked files, so both markings coexist in one file for the `05` variant.

None of these are meant to be permanent artifacts of the project — they're training-time intermediates. The final saved tokenizer, after `restore_vocab_in_place` relabels its vocab, is the only piece meant to be used going forward, and it works on ordinary raw Tamil text with no placeholder map required at inference time.