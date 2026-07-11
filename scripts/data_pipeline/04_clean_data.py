from collections import defaultdict
from pathlib import Path
import sys
from tqdm.asyncio import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `paths.py` in scripts/ is importable
from paths import Paths
from utils import create_directories, reservoir_sample, TAMIL_CHARACTERS

def setup_environment():
    """
    Sets up the environment by creating necessary directories and downloading the Tamil Wikipedia dump.
    """
    # Setup the Paths for data storage
    _, _, cleaned_data = create_directories()

    # Setup paths for Merged text file and the train/test files
    project = cleaned_data / 'tamil_wiki_extracted.txt'
    long_lines = cleaned_data / 'tamil_wiki_extracted_long_lines.txt'
    madurai = cleaned_data / 'project_madurai_extracted.txt'
    tamil_cc100 = cleaned_data / 'tamil_cc100_extracted.txt'

    file_paths = [project, long_lines, madurai, tamil_cc100]

    return file_paths

def real_content_ratio(line):
    tokens = line.split()
    if not tokens:
        return 0.0
    tamil_tokens = sum(1 for t in tokens if TAMIL_CHARACTERS.search(t))
    return tamil_tokens / len(tokens)

def content_ratio_distribution(file_path, threshold=0.3):
    file_path = Path(file_path)
    below_path = file_path.with_name(f"{file_path.stem}_below_threshold{file_path.suffix}")
    above_path = file_path.with_name(f"{file_path.stem}_above_threshold{file_path.suffix}")

    below = above = 0

    with open(file_path, encoding='utf-8') as f, \
         open(below_path, 'w', encoding='utf-8') as below_file, \
         open(above_path, 'w', encoding='utf-8') as above_file:

        for line in f:
            stripped = line.rstrip('\n')
            ratio = real_content_ratio(stripped)

            if ratio < threshold:
                below += 1
                below_file.write(line)
            else:
                above += 1
                above_file.write(line)

    print(f"Total lines: {below + above}| Lines below threshold ({threshold}): {below}| Lines above threshold ({threshold}): {above}")
    return below, above, below + above, above_path

def main():
    # Setup environment and get paths for the files to process
    content_ratios = defaultdict(list)
    # file_paths = setup_environment()
    paths = Paths()
    above_file_paths = []

    for file_path in [paths.tamil_wiki, paths.tamil_wiki_long_lines, paths.tamil_madurai, paths.tamil_cc100]:
        print(f"Processing file: {file_path.name}")
        below, above, total, above_file_path = content_ratio_distribution(file_path, threshold=0.3)
        content_ratios[file_path.name].append(above / total if total > 0 else 0)
        above_file_paths.append(above_file_path)

    print("Content Ratios:")
    for file_name, ratios in content_ratios.items():
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        print(f"{file_name}: Average Content Ratio = {avg_ratio:.4f}\n")
    
    for above_file_path in above_file_paths:
        print(f"\nSample lines from {above_file_path.name}:")
        sampled_test_lines = reservoir_sample(above_file_path, sample_size=5)
        for line in sampled_test_lines:
            print(line)
        
if __name__ == "__main__":
    main()