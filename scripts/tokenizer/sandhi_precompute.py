# One-time offline pass (run once, save forever)
from pathlib import Path
import sys
from tqdm import tqdm
from sandhi import sandhi_split
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import Paths

paths = Paths()

TRAIN_LINES = 30_683_869
TEST_LINES = 3_409_318

print("Creating sandhi-marked training and test files...")
# Create a new file with sandhi marked text, using the sandhi_split function
with open(paths.train, encoding="utf-8") as fin, open(paths.train_sandhi_marked, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, total=TRAIN_LINES, desc="train", unit="lines"):
        chunks = sandhi_split(line.strip(), lang="ta")
        marked_line = "⟂".join(tok for tok, _ in chunks)
        fout.write(marked_line + "\n")
print(f"Sandhi-marked training file created: {paths.train_sandhi_marked}")


print("Creating sandhi-marked test file...")        
# Create a new file with sandhi marked text, using the sandhi_split function        
with open(paths.test, encoding="utf-8") as fin, open(paths.test_sandhi_marked, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, total=TEST_LINES, desc="test", unit="lines"):
        chunks = sandhi_split(line.strip(), lang="ta")
        marked_line = "⟂".join(tok for tok, _ in chunks)
        fout.write(marked_line + "\n")
print(f"Sandhi-marked test file created: {paths.test_sandhi_marked}")