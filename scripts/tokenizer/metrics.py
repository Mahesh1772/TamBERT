from pathlib import Path

SAMPLE_TEXT = "அம்மா வீட்டில் இருக்கிறார்!!! நான் பள்ளிக்கு, 12. போகிறேன்."

# --- BERT-style constants ---
UNK_TOKEN = '[UNK]'
VOCAB_SIZE = 5
BPE_SPECIAL_TOKENS = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']

def calculate_fertility(tokenizer,
                        file_path: Path,
                        unknown_token: str = UNK_TOKEN):
    """Calculate the fertility of a tokenizer on a given text file.

    Args:
        tokenizer (_type_): tokenizer object with a `tokenize` method that takes a string and returns a list of tokens.
        file_path (Path): path to the text file for which to calculate fertility.
        unknown_token (str, optional): The token used to represent unknown words. Defaults to '[UNK]'.
    """

    unk_id = tokenizer.token_to_id(unknown_token)
    if unk_id is None:
        raise ValueError(f"Unknown token '{unknown_token}' not found in tokenizer vocabulary.")

    total_words = 0
    total_subword_tokens = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            words = line.split()
            total_words += len(words)
            total_subword_tokens += len(tokenizer.encode(line).tokens)  # Assuming the tokenizer has an `encode` method that returns a list of tokens
            
    fertility = total_subword_tokens / total_words if total_words > 0 else 0
    return fertility

def calculate_oov_rate(tokenizer, file_path: Path, unknown_token: str = UNK_TOKEN):
    """Calculate the out-of-vocabulary (OOV) rate of a tokenizer on a given text file.

    Args:
        tokenizer (_type_): tokenizer object with a `tokenize` method that takes a string and returns a list of tokens.
        file_path (Path): path to the text file for which to calculate OOV rate.
        unknown_token (str, optional): The token used to represent unknown words. Defaults to '[UNK]'.
    """
    
    unk_id = tokenizer.token_to_id(unknown_token)
    if unk_id is None:
        raise ValueError(f"Unknown token '{unknown_token}' not found in tokenizer vocabulary.")

    total_tokens = 0
    oov_words = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            token_ids = tokenizer.encode(line).ids  # Assuming the tokenizer has an `encode` method that returns a list of token IDs
            total_tokens += len(token_ids)
            oov_words += sum(1 for token_id in token_ids if token_id == unk_id)
                    
    oov_rate = oov_words / total_tokens if total_tokens > 0 else 0
    return oov_rate * 100  # Return as a percentage

def calculate_tokenizer_metrics(tokenizer, file_path: Path, unknown_token: str = UNK_TOKEN):
    """Calculate both fertility and OOV rate for a given tokenizer on a specified text file.

    Args:
        tokenizer (_type_): tokenizer object with a `tokenize` method that takes a string and returns a list of tokens.
        file_path (Path): path to the text file for which to calculate metrics.
        unknown_token (str, optional): The token used to represent unknown words. Defaults to '[UNK]'.
    """
    
    unk_id = tokenizer.token_to_id(unknown_token)
    if unk_id is None:
        raise ValueError(f"Unknown token '{unknown_token}' not found in tokenizer vocabulary.")

    total_tokens = 0 # also total subword tokens
    oov_words = 0
    total_words = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            words = line.split()
            total_words += len(words)
            
            token_ids = tokenizer.encode(line).ids  # Assuming the tokenizer has an `encode` method that returns a list of token IDs
            total_tokens += len(token_ids)
            oov_words += sum(1 for token_id in token_ids if token_id == unk_id)
    
    fertility = total_tokens / total_words if total_words > 0 else 0
    oov_rate = oov_words / total_tokens if total_tokens > 0 else 0
    
    return {
        'fertility': fertility,
        'oov_rate': oov_rate*100  # Return OOV rate as a percentage
    }