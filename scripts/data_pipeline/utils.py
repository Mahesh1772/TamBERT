import random, re, regex
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `paths.py` in scripts/ is importable
from paths import Paths

# Define Vocabulary parameters
ENGLISH_WORDS  = re.compile(r'[A-Za-z]+')   # removes ALL Latin, including single chars
TAMIL_VOCAB = re.compile(
    r'[\u0B80-\u0BFF]+(?:\.[\u0B80-\u0BFF]+)*'  # Tamil + dot-abbreviations: கி.மீ
    r'|[0-9]+(?:\.[0-9]+)?'                       # integers + decimals: 3.14
    r'|[,!?"\-%\']'                           # prose punctuation — human-typed
    # = [ ] { } are intentionally excluded from training text, reserved using [unused] tokens
)
TAMIL_CHARACTERS = re.compile(r'[\u0B80-\u0BFF]')  # Tamil characters only
REPLACEMENT_CHAR = '\ufffd'
DISALLOWED = re.compile(r'[^\u0B80-\u0BFF\s0-9,!?"\-%\'.]')
NON_ALPHA = regex.compile(r'[\p{N}\p{P}\p{S}]')
NEW_LINE = '\n'

# Helper function to extract Tamil text from the Wikipedia dump
def extract_tamil_words(text:str) -> list[str]:
    """
    Extracts Tamil words from the given text using the defined regex patterns.
    
    Args:
        text (str): The input text from which to extract Tamil words.
        
    Returns:
        list: A list of extracted Tamil words.
    """
    # Remove English words and punctuation
    text = ENGLISH_WORDS.sub(' ', text)
    # Extract Tamil words
    tamil_words = TAMIL_VOCAB.findall(text)
    return tamil_words

def reservoir_sample(input_file, k=5, encoding='utf-8'):
    """
    Performs reservoir sampling to randomly select k lines from the input file.
    
    Args:
        input_file (str): Path to the input text file.
        k (int): Number of lines to sample (default is 5).
        encoding (str): Encoding of the input file (default is 'utf-8').
    """
    reservoir = []
    with open(input_file, 'r', encoding=encoding) as f:
        for n, line in enumerate(f):
            if n < k:
                reservoir.append(line.strip())
            else:
                j = random.randint(0, n)
                if j < k:
                    reservoir[j] = line.strip()
    return reservoir