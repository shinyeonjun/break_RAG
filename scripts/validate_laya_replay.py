"""Offline replay of captured model inputs; never calls Jev or a generator."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # Windows native-DLL import order
from transformers import AutoTokenizer
from rag_baselines.data import read_json, read_jsonl, write_json, file_sha256
from rag_baselines.laya_local import prepare_laya_inputs, audit_laya_input


def validate_replay(source, output):
    config = read_json(ROOT / 'config/d_laya.json')
    snapshot = Path.home() / '.cache/huggingface/hub/models--convaiinnovations--laya-multilingual/snapshots'
    tokenizer = AutoTokenizer.from_pretrained(snapshot / config['model']['revision'] / 'tokenizer', local_files_only=True)
    rows, failures = [], []
    for record in read_jsonl(source / 'system.jsonl'):
        retrieval = record['retrieval']
        if retrieval['method'] != 'D':
            continue
        for index, call in enumerate(retrieval['trace']['calls']):
            request = call['request']
            for item in prepare_laya_inputs(request['state'], request['questions']):
                row = {'question_id': retrieval['question_id'], 'call_index': index,
                       'stage': call['stage'], 'decision_id': item['question_id']}
                try:
                    audit = audit_laya_input(tokenizer, item['state'], item['question'],
                                            max_len=config['max_len'], head_max_len=config['head_max_len'])
                    row.update({k: v for k, v in audit.items() if k != 'input_ids'})
                    row['accepted'] = True
                except ValueError as error:
                    row.update(accepted=False, error=str(error))
                    failures.append(row)
                rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'd_input_replay.csv', index=False)
    summary = {'source': str(source.relative_to(ROOT)), 'source_sha256': file_sha256(source / 'system.jsonl'),
               'decisions': len(rows), 'accepted': sum(r['accepted'] for r in rows), 'failures': failures,
               'input_protocol': 'per-target-state-v1', 'inference_performed': False}
    write_json(output / 'd_input_replay_summary.json', summary)
    return summary


if __name__ == '__main__':
    output = Path((ROOT / 'results/current_staged_run.txt').read_text(encoding='utf-8'))
    print(json.dumps(validate_replay(ROOT / 'results/20261004T052947_1e5333aa', output), ensure_ascii=False))
