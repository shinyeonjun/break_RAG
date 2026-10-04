"""A bounded 48-case diagnostic, evaluated against inspected source facts."""
from pathlib import Path
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # Native DLL import order on Windows
from rag_baselines.data import read_json, read_jsonl, write_json, write_jsonl
from rag_baselines.laya_local import LayaDecisionClient

QIDS = [
    'nq:2725109744985341457', 'nq:-8415517321967947495',
    'nq:-5101130977596327361', 'nq:-5121074461515558304',
    'hotpotqa:5ac415ed5542991943173907', 'hotpotqa:5ae687de5542991bbc976119',
    'hotpotqa:5a7f2faf5542993067513683', 'hotpotqa:5adc026655429947ff1738b5',
]
PRIMARY = ['44819466:0', '176555:0', '5022853:0', '5720175:0',
           '1977912:0', '29289213:0', '8636109:0', '25122684:0']
NOTES = [
    'Season 2 premiered May 30, 2017.', 'The park is located in southern Oregon.',
    'SSDI is explicitly payroll-tax-funded.', 'The song is a single by Cher released in 1966.',
    'Focus was published in 1964 and Thurn and Taxis in 2006; both dates are needed.',
    'The play premiered at Royal Exchange; a second source names the theatre and shopping centre.',
    'Spike Slawson is explicitly a member of Me First and the Gimme Gimmes; Grant Hart source identifies other bands.',
    'Bruckenthalia has one species; Baeckea has about 75. Both counts are needed.',
]


def build_cases(source):
    originals = {r['retrieval']['question_id']: r['retrieval'] for r in read_jsonl(source / 'system.jsonl')
                 if r['retrieval']['method'] == 'C'}
    labels = {r['id']: r for r in read_jsonl(ROOT / 'data/prepared/labels_dev.jsonl')}
    passages = {p['id']: p for p in read_jsonl(ROOT / 'data/prepared/corpus_2000.jsonl')}
    cases = []
    for index, (qid, primary, note) in enumerate(zip(QIDS, PRIMARY, NOTES)):
        retrieval = originals[qid]
        calls = retrieval['trace']['calls']
        navigation = [(call, key, q) for call in calls if call['stage'] == 'navigation'
                      for key, q in call['request']['questions'].items()]
        positive = next((call, key, q) for call, key, q in navigation
                        if primary in [c['id'] for c in q['instructions']['branch'].get('cards', [])])
        # Negatives use a different, inspected subject (e.g. biology vs entertainment).
        negative_id = 'media' if index in (1, 2, 7) else 'science'
        negative = next((call, key, q) for call, key, q in navigation if key == negative_id)
        for expected, (call, key, question) in ((True, positive), (False, negative)):
            cases.append({'id': f'{index}-navigation-{int(expected)}', 'query_id': qid, 'stage': 'navigation',
                'state': copy.deepcopy(call['request']['state']), 'question': copy.deepcopy(question),
                'expected': expected, 'basis': note if expected else 'This branch describes a different subject without a needed source link.',
                'source_call_id': call.get('call_id'), 'source_decision_id': key})
        score_template = next(q for call in calls if call['stage'] == 'scoring'
                              for q in call['request']['questions'].values())
        unrelated = PRIMARY[(index + 3) % len(PRIMARY)]
        for expected, pid in ((True, primary), (False, unrelated)):
            question = copy.deepcopy(score_template)
            question['instructions']['passage'] = {'title': passages[pid]['title'], 'text': passages[pid]['content']}
            cases.append({'id': f'{index}-scoring-{int(expected)}', 'query_id': qid, 'stage': 'scoring',
                'state': {'query': retrieval['question']}, 'question': question, 'expected': expected,
                'basis': note if expected else 'The inspected passage addresses a different entity/subject, with no answer fact or source link.',
                'source_passage_id': pid})
        coverage_template = next(q for call in calls if call['stage'] == 'coverage'
                                 for q in call['request']['questions'].values())
        group = labels[qid]['evidence_groups'][0]
        omitted = {'hotpotqa:5ae687de5542991bbc976119': '959572:0',
                   'hotpotqa:5a7f2faf5542993067513683': '8636109:0'}.get(qid, group[-1])
        for expected, ids in ((True, group), (False, [pid for pid in group if pid != omitted])):
            cases.append({'id': f'{index}-coverage-{int(expected)}', 'query_id': qid, 'stage': 'coverage',
                'state': {'query': retrieval['question'], 'evidence': [{'id': pid, 'text': passages[pid]['content']} for pid in ids]},
                'question': copy.deepcopy(coverage_template), 'expected': expected, 'basis': note,
                'source_passage_ids': ids, 'omitted_required_passage': None if expected else omitted})
    return cases


def run_checks(source, output, device='cpu'):
    cases = build_cases(source)
    write_jsonl(output / 'decision_cases.jsonl', cases)
    config = read_json(ROOT / 'config/d_laya.json')
    config['batch_size'] = 4
    client = LayaDecisionClient(config, device=device)
    rows, calls = [], []
    for case in cases:
        answers = client.decide(case['state'], {case['id']: case['question']}, calls=calls,
                                max_calls=len(cases), stage=case['stage'],
                                on_record=lambda call: append(output / 'decision_calls.jsonl', call))
        answer = answers[case['id']]
        key = 'score' if case['stage'] == 'scoring' else 'noul'
        value = answer[key]
        threshold = 2.0 if key == 'score' else .5
        rows.append({'case_id': case['id'], 'query_id': case['query_id'], 'stage': case['stage'],
                     'expected': case['expected'], 'value': value, 'predicted': value >= threshold,
                     'correct': (value >= threshold) == case['expected'], 'seconds': calls[-1]['seconds'],
                     'input_tokens': calls[-1]['response']['usage']['input_tokens']})
        print(f"{case['id']}: {key}={value:.4f}, expected={case['expected']}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'decision_validation.csv', index=False)
    summary = frame.groupby('stage').agg(cases=('correct', 'size'), accuracy=('correct', 'mean'),
                                       mean_seconds=('seconds', 'mean')).reset_index().to_dict('records')
    write_json(output / 'decision_validation_summary.json', {'runtime': client.runtime_info(), 'stages': summary,
               'case_count': len(rows), 'human_label_basis': 'Source facts and branch descriptions inspected; labels are not Jev predictions.',
               'scope': 'Eight development questions, balanced 48 diagnostic cases; not an unbiased accuracy estimate.'})
    print(json.dumps(summary), flush=True)
    return rows


def append(path, record):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    output = Path((ROOT / 'results/current_staged_run.txt').read_text(encoding='utf-8'))
    run_checks(ROOT / 'results/20261004T052947_1e5333aa', output,
               device=sys.argv[1] if len(sys.argv) > 1 else 'cpu')
