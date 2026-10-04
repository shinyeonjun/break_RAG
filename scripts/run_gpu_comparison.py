from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from dotenv import load_dotenv
from rag_baselines.data import read_json
from rag_baselines.validation_execution import run_comparison, freeze_d_policy

if __name__ == '__main__':
    load_dotenv(ROOT/'.env', encoding='utf-8-sig', override=False)
    output = Path((ROOT/'results/current_staged_run.txt').read_text(encoding='utf-8'))
    split = sys.argv[1] if len(sys.argv)>1 else 'dev'
    candidate = split == 'd_candidate'
    summary = run_comparison(ROOT, output, read_json(ROOT/'config/validation_study.json'),
                             split='dev' if candidate else split, include_c=split=='eval', d_candidate=candidate)
    print(summary.to_string(index=False))
    if candidate:
        table, frozen = freeze_d_policy(ROOT, output)
        print(table.to_string(index=False))
        print('Frozen D policy:', frozen['selected_arm'])
