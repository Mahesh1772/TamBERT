import shutil, bz2, os, re, random, string, lzma, requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from urllib.request import urlretrieve
from utils import create_directories, extract_tamil_words, reservoir_sample

def setup_environment():
    """
    Sets up the environment by creating necessary directories and downloading the Tamil Wikipedia dump.
    """
    # Setup the Paths for data storage
    _, cleaned_data = create_directories()

    # Download the Project Madurai dump htmls
    project_madurai_url = 'https://www.projectmadurai.org/pmworks.html'
    r = requests.get(project_madurai_url)
    project_madurai_soup = BeautifulSoup(r.text, 'html.parser')
    html_links = {}
    
    print(f"Downloading Project Madurai data from {project_madurai_url}...")
    
    for link in project_madurai_soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/pm_etexts/utf8/') and href.endswith('.html'):
            # Example : https://www.projectmadurai.org/pm_etexts/utf8/pmuni1081_02.html
            filename = link.text.strip()
            full_url = f"https://www.projectmadurai.org{href}"
            html_links[filename] = full_url
            
    print(f"Extracted {len(html_links)} HTML links.")
    
    # Create the output file for extracted Tamil text
    extracted_text_file = cleaned_data / 'tamil_project_madurai_extracted.txt'

    return html_links, extracted_text_file

def clean_project_madurai_text(text:str) -> str:
    """
    Cleans the Project Madurai text by removing unwanted characters and formatting.

    Args:
        text (str): The input Project Madurai text to clean.
        
    Returns:
        str: The cleaned Project Madurai text.
    """
    if not text:
        return ''

    # 1. Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # 2. Strip emails
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    # 3. Strip HTML entities (&oldid=, &amp, etc)
    text = re.sub(r'&(?:[a-zA-Z]+|#[0-9]+|#x[0-9a-fA-F]+);', '', text)
    # 4. Remove English words and retain only Tamil words
    tamil_words = extract_tamil_words(text)
    return tamil_words

def fetch_and_clean_page(item):
    """
    Fetches the HTML content from the given URL, cleans it, and returns the cleaned Tamil text.

    Args:
        item (tuple): A tuple containing the filename and URL of the HTML page.
        
    Returns:
        str: The cleaned Tamil text extracted from the HTML page.
    """
    _, url = item
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
    
        if not soup.body:
            return []

        lines = soup.body.get_text(separator='\n').splitlines()
        return [' '.join(c) for line in lines for c in [clean_project_madurai_text(line)] if c]
    
    except Exception as e:
        print(f"Error fetching or cleaning page {url}: {e}")
        return []

def parse_project_madurai(html_links, out_path):
    """
    Parses the Project Madurai HTML links, extracts Tamil text, and saves it to an output file.
    
    Args:
        html_links (dict): A dictionary mapping filenames to URLs of HTML pages.
        out_path (str): Path to the output text file where extracted Tamil text will be saved.
    """
    with open(out_path, 'w', encoding='utf-8', buffering=1024*1024) as out_file:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {executor.submit(fetch_and_clean_page, item): item for item in html_links.items()}
            for i, future in enumerate(tqdm(as_completed(future_to_url), total=len(future_to_url), desc='Processing pages')):
                for line in future.result():
                    out_file.write(line + '\n')
                
                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1} pages.")

    print()
    print(f"Finished processing. Total pages: {len(html_links)}. Output saved to {out_path}\n")
    

def main():
    # Setup the environment and download the Tamil Wikipedia dump
    html_links, extracted_text_file = setup_environment()
    
    # Parse the Project Madurai HTML links and extract Tamil text
    parse_project_madurai(html_links, extracted_text_file)
    
    # Perform reservoir sampling to get a few random lines from the extracted text
    sampled_lines = reservoir_sample(extracted_text_file, k=5)
    
    print("Sampled lines from the extracted Tamil text:\n ")
    for line in sampled_lines:
        print(line)
                
main()