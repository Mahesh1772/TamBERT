import re, lzma
from pathlib import Path
import sys
from tqdm import tqdm
from urllib.request import urlretrieve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `paths.py` in scripts/ is importable
from paths import Paths
from utils import extract_tamil_words, reservoir_sample

def setup_environment(cc_url='https://data.statmt.org/cc-100/ta.txt.xz', cc100_file=None):
    """
    Sets up the environment by creating necessary directories and downloading the Tamil Wikipedia dump.
    """
    
    if cc100_file.exists():
        print(f"CC-100 file already exists at {cc100_file}. Skipping download.")
    else:
        print(f"Downloading CC-100: Monolingual Dataset data from {cc_url}...")
        urlretrieve(cc_url, cc100_file)
        print(f"Download completed. File saved as {cc100_file}")


def clean_cc100_text(text:str) -> str:
    """
    Cleans the CC-100 text by removing unwanted characters and formatting.
    
    Args:
        text (str): The input CC-100 text to clean.
        
    Returns:
        str: The cleaned CC-100 text.
    """
    if not text:
        return ''

    # 1. Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # 2. Strip HTML entities and tags
    text = re.sub(r'&(?:[a-zA-Z]+|#[0-9]+|#x[0-9a-fA-F]+);', '', text)
    # 3. Remove English words and retain only Tamil words
    tamil_words = extract_tamil_words(text)
    return ' '.join(tamil_words)

def parse_tamil_cc100(xz_path, out_path):
    """
    Parses the Tamil CC-100 file, extracts Tamil text, and saves it to an output file.
    
    Args:
        xz_path (str): Path to the compressed CC-100 file (.xz).
        out_path (str): Path to the output text file where extracted Tamil text will be saved.
        keep_ns (tuple): Tuple of namespace numbers to keep (default is ('0',) for main articles).
    """
    num_pages = 0
    
    with lzma.open(xz_path) as lzma_file:
        with open(out_path, 'w', encoding='utf-8', buffering=1024*1024) as out_file:
            with tqdm(total=68_237_343, unit='lines', unit_scale=True, desc='Processing CC-100') as pbar: # Size ws observed from trial runs, adjust if needed
                for line_num, line in tqdm(enumerate(lzma_file), desc='Processing lines'):
                    pbar.update(1)
                    line = line.decode('utf-8').strip()
                    cleaned_line = clean_cc100_text(line)
                    if cleaned_line:
                        out_file.write(cleaned_line + '\n')
                        num_pages += 1
                        
                    if (line_num + 1) % 1000000 == 0:
                        print(f"Processed {line_num + 1} lines, kept {num_pages} lines with Tamil text.")

    print()
    print(f"Finished processing. Total pages: {num_pages}. Output saved to {out_path}\n")
    

def main():
    # Setup the environment and download the WebText CC-100 Tamil dataset
    paths = Paths()
    setup_environment(cc100_file=paths.cc100_file)
    
    # Parse the Tamil CC-100 file and extract Tamil text
    parse_tamil_cc100(paths.cc100_file, paths.tamil_cc100)
    
    # Perform reservoir sampling to get a few random lines from the extracted text
    sampled_lines = reservoir_sample(paths.tamil_cc100, k=5)
    
    print("Sampled lines from the extracted Tamil text:\n ")
    for line in sampled_lines:
        print(line)
        
main()