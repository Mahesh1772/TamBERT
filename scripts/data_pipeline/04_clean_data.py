from collections import defaultdict
from pathlib import Path
from tqdm.asyncio import tqdm
from paths import Paths
from utils import reservoir_sample, TAMIL_CHARACTERS

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
    paths = Paths()
    above_file_paths = []

    for file_path in [paths.tamil_wiki, paths.tamil_wiki_long_lines, paths.project_madurai, paths.tamil_cc100]:
        print(f"Processing file: {file_path.name}")
        _, above, total, above_file_path = content_ratio_distribution(file_path, threshold=0.3)
        content_ratios[file_path.name].append(above / total if total > 0 else 0)
        above_file_paths.append(above_file_path)

    print("Content Ratios:")
    for file_name, ratios in content_ratios.items():
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        print(f"{file_name}: Average Content Ratio = {avg_ratio:.4f}\n")
    
    for above_file_path in above_file_paths:
        print(f"\nSample lines from {above_file_path.name}:")
        sampled_test_lines = reservoir_sample(above_file_path, k=5)
        for line in sampled_test_lines:
            print(line)
        
if __name__ == "__main__":
    main()