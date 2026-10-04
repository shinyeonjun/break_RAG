"""GPU A/B/D validation and a frozen A/B/C/D evaluation run."""
from copy import deepcopy
from pathlib import Path
import gc
import json

import pandas as pd  # Windows native-DLL import order
import torch
import httpx
from openai import OpenAI, DefaultHttpxClient

from .data import read_json, read_jsonl, write_json, file_sha256
from .models import Baselines
from .store import DocumentStore
from .c_tree import load_tree, digest
from .c_retrieval import JevRetrieval, ComparativeRetrieval
from .laya_local import LayaDecisionClient
from .jev import JevClient
from .raw_logging import RawLog
from .evaluation import runtime_manifest, measure_system
from .validation_study import selected_questions, summary_rows, aggregate, append_record


def run_comparison(root, output, study, *, split, include_c=True, d_candidate=False):
    root, output = Path(root), Path(output)
    if d_candidate and (split != 'dev' or include_c):
        raise ValueError("D candidate tuning is restricted to development.")
    if not study['save_raw_logs'] or study['repeats'] != 1:
        raise ValueError("This study requires raw logging and one declared repetition.")
    # Refuse CPU fallback or a knowingly oversubscribed GPU for timed comparisons.
    if not torch.cuda.is_available():
        raise RuntimeError("This hardware comparison requires CUDA for A/B/D.")
    # Notebook 05 executes development and evaluation in the same process.
    # Release inactive allocations from the previous arm before the free-memory gate.
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    if free < 4 * 1024**3:
        raise RuntimeError(f"GPU comparison needs free memory for three pinned models; free={free/1024**3:.2f} GiB. Close idle notebook kernels first.")
    freeze = read_json(output / 'frozen_c_policy.json') if include_c else None
    d_freeze = read_json(output / 'frozen_d_policy.json') if split == 'eval' else None
    if split == 'eval' and freeze is None:
        raise ValueError("Freeze the C development policy before evaluation.")
    questions = selected_questions(root, split, study['questions_per_task'][split])
    methods = ('D',) if d_candidate else (('A', 'B', 'C', 'D') if include_c else ('A', 'B', 'D'))
    if len(questions) * len(methods) > study['max_llm_calls']:
        raise ValueError("Comparison would exceed the configured LLM call budget.")
    arm = 'D_candidate' if d_candidate else ('ABCD_frozen' if include_c else 'ABD_input_fix')
    directory = output / split / arm
    directory.mkdir(parents=True, exist_ok=False)
    c_config = freeze['c_config'] if freeze else read_json(root / 'config/c.json')
    d_source = root / ('config/d_laya_candidate.json' if d_candidate else 'config/d_laya.json')
    d_config = d_freeze['d_config'] if d_freeze else read_json(d_source)
    generation = read_json(root / 'config/generation.json')
    prepared = read_json(root / 'data/prepared/manifest.json')
    entry = prepared['corpora'][str(study['corpus_size'])]
    store = DocumentStore(root / entry['path'])
    if store.sha256 != entry['sha256']:
        raise ValueError("Prepared corpus integrity check failed.")
    tree, preparation = load_tree(root, store, read_json(root / 'config/c_taxonomy.json'), read_json(root / 'config/c.json'))
    # Freeze source signatures and policy before loading evaluation labels.
    evaluation_freeze = {'split': split, 'methods': methods, 'c_policy': freeze, 'd_policy': d_freeze, 'd_config': d_config,
                         'generation': generation, 'corpus_sha256': store.sha256,
                         'tree_sha256': digest(tree), 'evaluation_labels_loaded': False}
    write_json(directory / 'policy_before_labels.json', evaluation_freeze)
    labels = read_jsonl(root / f'data/prepared/labels_{split}.jsonl')
    answer_audit = read_json(root / 'config/answer_audit.json')
    audit = RawLog(root, manifest={'phase': f'{split}_comparison', 'study': study,
        'frozen_policy': evaluation_freeze, 'notebook': '05_measure_abcd.ipynb'},
        prices={'jev': c_config['prices'], 'openai': generation['prices']})
    for path in [root/'notebook/05_measure_abcd.ipynb', root/'config/validation_study.json', d_source, root/'config/generation.json',
                 root/'config/answer_audit.json', store.path, directory/'policy_before_labels.json',
                 root/f'data/prepared/questions_{split}.jsonl', root/f'data/prepared/labels_{split}.jsonl',
                 *sorted((root/'rag_baselines').glob('*.py'))]:
        audit.snapshot(path)
    with httpx.Client(timeout=c_config['timeout_seconds'], event_hooks=audit.http_hooks) as jev_http, \
         DefaultHttpxClient(timeout=180, event_hooks=audit.http_hooks) as llm_http, \
         OpenAI(max_retries=0, timeout=180, http_client=llm_http) as generator:
        baseline = Baselines(root, device='cuda', retrieve_k=50, evidence_k=5)
        baseline.raw_log = audit
        baseline.build_index(store.path)
        laya = LayaDecisionClient(d_config, device=baseline.device)
        jev = JevClient(c_config, http_client=jev_http)
        c = JevRetrieval(root, baseline.store, tree, preparation, c_config, client=jev)
        d = JevRetrieval(root, baseline.store, tree, preparation, d_config, client=laya, method='D', provider='laya')
        engine = ComparativeRetrieval(baseline, c, d)
        # The chosen C arm must match its development tree/model/policy exactly.
        if freeze and c.signature != freeze['search_signature']:
            raise ValueError("Frozen C search signature differs from the development-selected implementation.")
        if d_freeze and d.signature != d_freeze['search_signature']:
            raise ValueError("Frozen D search signature differs from the development-selected implementation.")
        meta = runtime_manifest(root, engine)
        meta.update({'mode': 'live_system_abcd', 'methods': list(methods), 'split': split,
            'repeats': 1, 'generation': generation, 'study': study, 'c_selected_arm': freeze['selected_arm'] if freeze else None,
            'answer_audit': answer_audit, 'd_selected_arm': d_freeze['selected_arm'] if d_freeze else arm,
            'question_ids': [q['id'] for q in questions], 'questions_file_sha256': file_sha256(root/f'data/prepared/questions_{split}.jsonl'),
            'labels_file_sha256': file_sha256(root/f'data/prepared/labels_{split}.jsonl'),
            'raw_log': str(audit.directory.relative_to(root)), 'gpu_free_bytes_before_load': free,
            'd_policy_note': d_config.get('policy_note', 'Original D thresholds preserved; per-target input protocol fixed.')})
        write_json(directory/'manifest.json', meta)
        results = measure_system(engine, questions, labels, client=generator, generation_config=generation,
            max_calls=len(questions)*len(methods), methods=methods, audit=audit,
            on_result=lambda r: append_record(directory/'system.jsonl', r))
    frame = pd.DataFrame(summary_rows(results, arm, answer_audit))
    frame.to_csv(directory/'questions.csv', index=False)
    summary = aggregate(frame)
    summary.to_csv(directory/'summary.csv', index=False)
    write_json(directory/'completed.json', {'records': len(results), 'system_sha256': file_sha256(directory/'system.jsonl')})
    summary_name = 'dev_d_candidate' if d_candidate else f'{split}_comparison'
    write_json(output/f'{summary_name}.json', {'directory': str(directory.relative_to(root)),
               'summary': summary.to_dict('records'), 'raw_log': str(audit.directory.relative_to(root))})
    audit.event('session','comparison_complete', {'directory': str(directory.relative_to(root))})
    return summary


def freeze_d_policy(root, output):
    """Select only between two completed development arms, without eval labels."""
    root, output = Path(root), Path(output)
    if (output/'frozen_d_policy.json').exists():
        raise ValueError("D policy is already frozen; start a new study to tune again.")
    rows, manifests, sources = [], {}, {}
    for arm in ('ABD_input_fix', 'D_candidate'):
        directory = output/'dev'/arm
        completed = read_json(directory/'completed.json')
        if file_sha256(directory/'system.jsonl') != completed['system_sha256']:
            raise ValueError("D development result integrity check failed.")
        manifests[arm] = read_json(directory/'manifest.json')
        if manifests[arm]['split'] != 'dev':
            raise ValueError("D policy selection requires development results.")
        records = read_jsonl(directory/'system.jsonl')
        if len(records) != completed['records']:
            raise ValueError("D development result count disagrees with completion receipt.")
        d_records = [r for r in records if r['retrieval']['method'] == 'D']
        keys = [(r['retrieval']['question_id'],r['retrieval']['repeat']) for r in d_records]
        if len(keys) != len(set(keys)) or {key[0] for key in keys} != set(manifests[arm]['question_ids']):
            raise ValueError("D development questions are missing, repeated or unexpected.")
        audit_config = manifests[arm].get('answer_audit')
        if audit_config is None:
            snapshots = list((root/manifests[arm]['raw_log']/'sources').glob('*_answer_audit.json'))
            if len(snapshots) != 1:
                raise ValueError("Need the original answer audit snapshot for D policy selection.")
            audit_config = read_json(snapshots[0])
        restored = aggregate(pd.DataFrame(summary_rows(d_records,arm,audit_config)))
        sources[arm] = file_sha256(directory/'summary.csv')
        table = pd.read_csv(directory/'summary.csv')
        stored = table[table.method == 'D'].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(restored,stored,check_dtype=False,atol=1e-10)
        except AssertionError as error:
            raise ValueError("D development summary disagrees with the original result JSONL.") from error
        row = restored.iloc[0].to_dict()
        row['arm'] = arm
        rows.append(row)
    first, second = manifests.values()
    if set(first['question_ids']) != set(second['question_ids']) or first['generation'] != second['generation']:
        raise ValueError("D development arms must share questions and generation settings.")
    if first['d']['signature']['tree_sha256'] != second['d']['signature']['tree_sha256']:
        raise ValueError("D development arms must share the same tree.")
    table = pd.DataFrame(rows).sort_values(['answer_em','complete_evidence_hit','online_api_cost_usd'],
                                          ascending=[False,False,True])
    choice = table.iloc[0]['arm']
    selected = manifests[choice]['d']
    frozen = {'selected_arm': choice, 'd_config': selected['config'], 'search_signature': selected['signature'],
              'selection_rule': 'highest development answer_em, then complete_evidence_hit, then lowest API cost',
              'dev_summary_sha256': sources,
              'note': 'Exploratory two-policy development selection. Multiple policy changes are not a one-variable ablation.'}
    table.to_csv(output/'dev_d_summary.csv', index=False)
    write_json(output/'frozen_d_policy.json', frozen)
    return table, frozen
