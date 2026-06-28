import shutil, bz2, os, re, random, string
from lxml import etree
from pathlib import Path
from tqdm import tqdm
from urllib.request import urlretrieve
from utils import create_directories, extract_tamil_words, reservoir_sample

def setup_environment():
    """
    Sets up the environment by creating necessary directories and downloading the Tamil Wikipedia dump.
    """
    # Setup the Paths for data storage
    raw_data, cleaned_data = create_directories()

    # Download the Wikipedia dump file
    wiki_dump_url = 'https://dumps.wikimedia.org/tawiki/latest/tawiki-latest-pages-articles.xml.bz2'
    wiki_dump_file = raw_data / 'tamil_wiki_dump.bz2'
    print(f"Downloading Wikipedia dump from {wiki_dump_url}...")
    urlretrieve(wiki_dump_url, wiki_dump_file)
    print(f"Download completed. File saved as {wiki_dump_file}")
    
    # Create the output file for extracted Tamil text
    extracted_text_file = cleaned_data / 'tamil_wiki_extracted.txt'

    return wiki_dump_file, extracted_text_file

def clean_wiki_text(text:str) -> str:
    """
    Cleans the Wikipedia text by removing unwanted characters and formatting.
    
    Args:
        text (str): The input Wikipedia text to clean.
        
    Returns:
        str: The cleaned Wikipedia text.
    """
    if not text:
        return ''

    # --- 1. Remove {{template}} blocks --- which are used for metadata and formatting, not article content.
    # Run 3 times because templates can be nested one level inside another
    # article wikitext rarely nests more than 1-2 levels.
    for _ in range(3):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)

    # --- 2. Remove <ref>...</ref> citation blocks ---, e.g. "<ref>cited from XYZ, 2020</ref>"
    # DOTALL so '.' also matches newlines (refs can span multiple lines).
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    # Self-closing refs that just point to an earlier-defined ref: <ref name="x"/>
    text = re.sub(r'<ref[^>]*/>', '', text)

    # --- 3. Remove HTML comments ---, e.g. "<!-- editor note -->"
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # --- 4. Remove any remaining raw HTML tags ---
    # e.g. "<b>", "<small>", "<div>" — formatting, strip the tags but
    text = re.sub(r'<[^>]+>', '', text)

    # --- 5. Resolve wiki links, keep the display text ---
    # "[[Pahang|பகாங்கு]]" -> "பகாங்கு"  (piped link: show 2nd part)
    # "[[பகாங்கு]]"        -> "பகாங்கு"  (simple link: show as-is)
    text = re.sub(r'\[\[([^\]|]*\|)?([^\]]*)\]\]', r'\2', text)

    # --- 6. Resolve external links, keep the link's label text ---
    # "[https://example.com நன்றி]" -> "நன்றி"   (url + display text)
    text = re.sub(r'\[https?://\S+\s+([^\]]*)\]', r'\1', text)
    # "[https://example.com]" -> ""              (bare url, no label, just delete)
    text = re.sub(r'\[https?://\S+\]', '', text)

    # --- 7. Remove bold/italic markup ---, e.g. "'''bold'''" and "''italic''"
    text = re.sub(r"'''|''", '', text)

    # --- 8. Remove section heading markers ---
    # "==Section Title==" -> "Section Title" (keep the heading text, drop the =)
    text = re.sub(r"^[=]+\s*|\s*[=]+$", '', text, flags=re.MULTILINE)

    return text

def parse_tamil_wiki_dump(bz2_path, out_path, keep_ns=('0',)):
    """
    Parses the Tamil Wikipedia dump file, extracts Tamil text, and saves it to an output file.
    
    Args:
        bz2_path (str): Path to the compressed Wikipedia dump file (.bz2).
        out_path (str): Path to the output text file where extracted Tamil text will be saved.
        keep_ns (tuple): Tuple of namespace numbers to keep (default is ('0',) for main articles).
    """
    with bz2.open(bz2_path) as bz2_file:
        ctx = etree.iterparse(bz2_file, events=('end',), tag='{*}page')
        n_pages = n_kept = 0
        
        with open(out_path, 'w', encoding='utf-8', buffering=1024*1024) as out_file:
            for _, page in tqdm(ctx, desc='Processing pages'):
                n_pages += 1
                
                # Extract main article namespace only
                ns_el = page.find('{*}ns')
                if keep_ns and (ns_el is None or ns_el.text not in keep_ns):
                    page.clear()
                    continue
                
                # pull raw text from the <revision><text> element
                text_el = page.find('{*}revision/{*}text')
                raw = text_el.text if text_el is not None and text_el.text else ''
                
                # Strip markup and clean the text
                tamil_words = extract_tamil_words(clean_wiki_text(raw))
                if tamil_words:
                    out_file.write(' '.join(tamil_words) + '\n')
                    n_kept += 1
                
                # Free memory by clearing the processed page element
                page.clear()
                while page.getprevious() is not None:
                    del page.getparent()[0]
                
                if n_pages % 5000 == 0:
                    print(f"Processed {n_pages} pages, kept {n_kept} pages with Tamil text.")
        del ctx  # free memory
    print(f"Finished processing. Total pages: {n_pages}, kept pages with Tamil text: {n_kept}. Output saved to {out_path}")
    

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

def main():
    # Setup the environment and download the Tamil Wikipedia dump
    wiki_dump_file, extracted_text_file = setup_environment()
    
    # Parse the Tamil Wikipedia dump and extract Tamil text
    parse_tamil_wiki_dump(wiki_dump_file, extracted_text_file)
    
    # Perform reservoir sampling to get a few random lines from the extracted text
    sampled_lines = reservoir_sample(extracted_text_file, k=5)
    
    print("Sampled lines from the extracted Tamil text:")
    for line in sampled_lines:
        print(line)