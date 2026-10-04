"""Replay only the 16 relevance cases with a reversed ordinal rubric."""
from pathlib import Path
from copy import deepcopy
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from rag_baselines.data import read_json, read_jsonl, write_json
from rag_baselines.laya_local import LayaDecisionClient
from scripts.check_laya_decisions import append

if __name__ == '__main__':
    output = Path((ROOT / 'results/current_staged_run.txt').read_text(encoding='utf-8'))
    cases = [c for c in read_jsonl(output / 'decision_cases.jsonl') if c['stage'] == 'scoring']
    client = LayaDecisionClient(read_json(ROOT / 'config/d_laya.json'), device='cpu')
    calls, rows = [], []
    for case in cases:
        question = deepcopy(case['question'])
        question['criteria'].reverse()
        answer = client.decide(case['state'], {case['id']: question}, calls=calls, max_calls=len(cases), stage='scoring',
                               on_record=lambda c: append(output / 'score_order_calls.jsonl', c))[case['id']]
        rows.append({'case_id': case['id'], 'query_id': case['query_id'], 'expected': case['expected'],
                     'native_reversed_score': answer['score'], 'semantic_score': len(question['criteria']) - 1 - answer['score']})
    frame = pd.DataFrame(rows)
    baseline = pd.read_csv(output / 'decision_validation.csv').query("stage == 'scoring'")
    paired = frame.merge(baseline[['case_id', 'value']], on='case_id', validate='one_to_one')
    paired['semantic_score_difference'] = paired.semantic_score - paired.value
    paired.to_csv(output / 'score_order_validation.csv', index=False)
    ranking = frame.pivot(index='query_id', columns='expected', values='semantic_score')
    summary = {'case_count': len(frame), 'correct_positive_negative_rank_pairs': int((ranking[True] > ranking[False]).sum()),
               'rank_pairs': len(ranking), 'mean_absolute_score_shift': float(paired.semantic_score_difference.abs().mean()),
               'scope': 'Paired order test on the same eight inspected development questions, no threshold fitting.'}
    write_json(output / 'score_order_summary.json', summary)
    print(summary)
