import random, re
from pathlib import Path

# Define Vocabulary parameters
ENGLISH_WORDS  = re.compile(r'[A-Za-z]+')   # removes ALL Latin, including single chars
TAMIL_VOCAB = re.compile(
    r'[\u0B80-\u0BFF]+(?:\.[\u0B80-\u0BFF]+)*'  # Tamil + dot-abbreviations: கி.மீ
    r'|[0-9]+(?:\.[0-9]+)?'                       # integers + decimals: 3.14
    r'|[,!?"\-%\']'                           # prose punctuation — human-typed
    # = [ ] { } are intentionally excluded from training text, reserved using [unused] tokens
)

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

def create_directories():
    """
    Sets up the environment by creating necessary directories.
    """
    # Setup the Paths for data storage
    data = Path('data')
    data.mkdir(exist_ok=True, parents=True)
    print('Data folder created...')
    raw_data = data / Path('raw')
    raw_data.mkdir(exist_ok=True, parents=True)
    print('Raw folder created...')
    cleaned_data = data / Path('cleaned')
    cleaned_data.mkdir(exist_ok=True, parents=True)
    print('Cleaned folder created...')
    
    return raw_data, cleaned_data