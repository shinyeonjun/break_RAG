"""Command-line entry to the same functions used by the measurement notebook."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from dotenv import load_dotenv
from rag_baselines.data import read_json
from rag_baselines.validation_study import run_c_development

if __name__ == '__main__':
    load_dotenv(ROOT / '.env', encoding='utf-8-sig', override=False)
    output = Path((ROOT / 'results/current_staged_run.txt').read_text(encoding='utf-8'))
    summary, freeze = run_c_development(ROOT, output, read_json(ROOT / 'config/validation_study.json'))
    print(summary.to_string(index=False))
    print('Frozen C policy:', freeze['selected_arm'])
