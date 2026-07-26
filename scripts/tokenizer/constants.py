from tokenizers import Regex, normalizers

SAMPLE_TEXT = "அம்மா வீட்டில் இருக்கிறார்!!! நான் பள்ளிக்கு, 12. போகிறேன்."

# --- CONSTANTS ---

UNUSED_TOKENS = [f'[unused{i}]' for i in range(1000)] # 1000 unused tokens for BERT-style special tokens
VOCAB_SIZE = 31000 + len(UNUSED_TOKENS)  # 31k is the BERT base vocab size, plus 1000 unused tokens 

# BPE Model constants
UNK_TOKEN = '[UNK]'
BPE_SPECIAL_TOKENS = [UNK_TOKEN, '[CLS]', '[SEP]', '[PAD]', '[MASK]'] + UNUSED_TOKENS

# Unigram Model constants
# UNIGRAM_UNK_TOKEN = '<unk>'
# UNIGRAM_SPECIAL_TOKENS = [UNIGRAM_UNK_TOKEN, '<cls>', '<sep>', '<pad>', '<mask>', '<s>', '</s>'] + UNUSED_TOKENS

standard_normalizer = normalizers.Sequence([
    normalizers.Strip(),
    normalizers.NFC(),
    normalizers.Replace(Regex(r",+"), ","),
    normalizers.Replace(Regex(r"\.+"), "."),
    normalizers.Replace(Regex(r"'+"), "'"),
    normalizers.Replace(Regex(r"-+"), "-")])