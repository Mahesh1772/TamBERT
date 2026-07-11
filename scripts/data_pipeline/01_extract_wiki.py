import shutil, bz2, os, re, random, string
import sys
from lxml import etree
from pathlib import Path
from tqdm import tqdm
from urllib.request import urlretrieve
from utils import create_directories, extract_tamil_words, reservoir_sample
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `paths.py` in scripts/ is importable
from paths import Paths

def setup_environment():
    """
    Sets up the environment by creating necessary directories and downloading the Tamil Wikipedia dump.
    """
    # Setup the Paths for data storage
    _, raw_data, cleaned_data = create_directories()

    # Download the Wikipedia dump file
    wiki_dump_url = 'https://dumps.wikimedia.org/tawiki/latest/tawiki-latest-pages-articles.xml.bz2'
    wiki_dump_file = raw_data / 'tamil_wiki' / 'tawiki-latest-pages-articles.xml.bz2'
    
    if not wiki_dump_file.exists():
        print(f"Downloading Wikipedia dump from {wiki_dump_url}...")
        urlretrieve(wiki_dump_url, wiki_dump_file)
        print(f"Download completed. File saved as {wiki_dump_file}")
    else:
        print(f"Wikipedia dump already exists at {wiki_dump_file}. Skipping download.")
    
    # Create the output file for extracted Tamil text
    extracted_text_file = cleaned_data / 'tamil_wiki_extracted.txt'
    long_lines_path = cleaned_data / 'tamil_wiki_extracted_long_lines.txt'

    return wiki_dump_file, extracted_text_file, long_lines_path

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

def parse_tamil_wiki_dump(bz2_path, out_path, long_lines_path, keep_ns=('0',)):
    with bz2.open(bz2_path) as bz2_file:
        ctx = etree.iterparse(bz2_file, events=('end',), tag='{*}page')
        n_pages = n_kept = n_long = 0

        with open(out_path, 'w', encoding='utf-8', buffering=1024*1024) as out_file, \
            open(long_lines_path, 'w', encoding='utf-8', buffering=1024*1024) as long_lines_file:
            for _, page in tqdm(ctx, desc='Processing pages'):
                n_pages += 1
                ns_el = page.find('{*}ns')
                if keep_ns and (ns_el is None or ns_el.text not in keep_ns):
                    page.clear()
                    continue

                text_el = page.find('{*}revision/{*}text')
                raw = text_el.text if text_el is not None and text_el.text else ''
                cleaned = clean_wiki_text(raw)

                # Split BEFORE word extraction
                chunks = re.split(r'\n{2,}|(?<=[.!?])\s+', cleaned)

                for chunk in chunks:
                    tamil_words = extract_tamil_words(chunk)
                    if tamil_words:
                        line = ' '.join(tamil_words) + '\n'
                        if len(line.split()) > 2500:
                            long_lines_file.write(line)
                            n_long += 1
                        else:
                            out_file.write(line)
                            n_kept += 1

                page.clear()
                while page.getprevious() is not None:
                    del page.getparent()[0]
                
                if n_pages % 50000 == 0:
                    print(f"Processed {n_pages} pages, kept {n_kept} pages with Tamil text, and {n_long} long lines.")
        del ctx  # free memory
    print()
    print(f"Finished. Total pages: {n_pages}, kept lines: {n_kept}, long lines: {n_long}.")

def main():
    # Setup the environment and download the Tamil Wikipedia dump
    # wiki_dump_file, extracted_text_file, long_lines_path = setup_environment()
    paths = Paths()
    
    # Parse the Tamil Wikipedia dump and extract Tamil text
    parse_tamil_wiki_dump(paths.wiki_dump, paths.tamil_wiki, paths.tamil_wiki_long_lines)
    
    # Perform reservoir sampling to get a few random lines from the extracted text
    sampled_lines = reservoir_sample(paths.tamil_wiki, k=5)
    
    print("Sampled lines from the extracted Tamil text:\n")
    for line in sampled_lines:
        print(line)
        
if __name__ == '__main__':
    main()