# One-time offline pass (run once, save forever)
from pathlib import Path
import sys
from tqdm import tqdm
from sandhi import sandhi_split, sandhi_mark_boundaries
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import Paths

paths = Paths()

TRAIN_LINES = 30_683_869
TEST_LINES = 3_409_318

def create_sandhi_marked_file(input_path, output_path, total_lines):
    """Create a sandhi-marked file from the input file."""
    print(f"Creating sandhi-marked file from {input_path}...")
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_lines, desc=f"Processing {input_path.name}", unit="lines"):
            marked_line = sandhi_mark_boundaries(line.strip(), lang="ta")
            fout.write(marked_line + "\n")
    print(f"Sandhi-marked file created: {output_path}")
    
def main():
    create_sandhi_marked_file(paths.train, paths.train_sandhi_marked, TRAIN_LINES)
    create_sandhi_marked_file(paths.test, paths.test_sandhi_marked, TEST_LINES)

if __name__ == "__main__":
    main()