# TamilBERT v1.0 — Tokenizer Training

## What does training entail

Training a tokenizer gives us a .json file which contains a map between the most frequent (eventho freq is only the merging criteria for bpe) and token IDs which are numbers the machine understands. This map is built by looking at the corpus time and time again or from information extracted from the corpus. There are different methods of merging rules which can be employed and each has it's advantages. Unigram, BPE and WordPiece.

## BPE (Byte Pair Encoding)

The smallest atomic units (SAU) the text can be broken down into are the starting point (vocabulary) for this algorithm. We then scan the entire corpus and find the frequence of subsequent pairs of SAU. This map of frequence is then used to find the most frequent pair, this pair of SAU are then merged and added to the 'vocabulary'. After it's addition now this merged entity is considered a SAU on it's own. Now the step above is repeated again to find the most frequently occuring subsequent SAU which will then be merged and added to the vocabulary. This continues until the capacity is filled.

> Example here

## WordPiece

Tokenizer for the BERT transformer model for NLP. It follows the same scheme as BPE in the sense that, vocabulary is split into seqeunce of SAUs and subsequent SAUs are merged together to form a new SAU added to the vocabulary. The only difference here is that, the merging criteria is maximum likelihood.

> Insert formula

The max. liklihood of the subsequent SAUs are computed and the pair with the highest liklihood is chosen to be merged. The cycle repeats until the vocabulary_size is reached.

> Example here

## Unigram

