"""Download only; experiments and measurements live in notebook/*.ipynb."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_baselines.data import download_wikipedia

if __name__ == "__main__":
    print(download_wikipedia(ROOT), flush=True)
