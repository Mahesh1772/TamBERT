# TamBERT v1.0 - Workflow

| Stage               | What happens to the model                                                  | What comes out                               | What it's now capable of                                                                                   |
| ------------------- | -------------------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1. Corpus           | Nothing yet — just collecting raw Tamil text                               | A big cleaned text file                      | Nothing on its own — it's just fuel                                                                        |
| 2. Tokenizer        | Learns how to chop Tamil text into subword pieces                          | A tokenizer file (vocab + merge/split rules) | Can turn Tamil text into token IDs and back — but still no "understanding"                                 |
| 3. MLM pre-training | Model reads huge amounts of Tamil text, guesses hidden words               | A base language model                        | Understands Tamil grammar, word relationships, context — but doesn't yet do any specific task              |
| 4. NLI fine-tuning  | Model learns to judge if two sentences agree, contradict, or are unrelated | An NLI-tuned model                           | Pushes similar-meaning sentences close together in vector space — first "usable" step for similarity tasks |
| 5. STS fine-tuning  | Model learns exact similarity scores (0–5) between sentence pairs          | A sentence-embedding model                   | Can rate how similar any two Tamil sentences are, with human-calibrated precision                          |
| 6. Evaluation       | Nothing changes in the model — just tested                                 | A single benchmark score (Spearman)          | Tells you how good the model actually is, compared to others                                               |

| Model                                                   | Setup          | Tamil score (Spearman/cosine) |
| ------------------------------------------------------- | -------------- | ----------------------------- |
| Vanilla monolingual TamilBERT                           | No fine-tuning | 0.59                          |
| TamilBERT + NLI only (1-step)                           | Monolingual    | 0.72                          |
| TamilBERT + NLI + STS (2-step, fully monolingual ta-ta) | Monolingual    | 0.80                          |
| IndicSBERT-STS (multilingual, zero-shot on Tamil)       | Multilingual   | 0.82                          |

[English STS link](https://huggingface.co/datasets/sentence-transformers/stsb)


[NLI Fine Tuning Data](huggingface.co/datasets/Divyanshu/indicxnli/viewer/ta)