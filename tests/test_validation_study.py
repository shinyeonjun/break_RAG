import json
from pathlib import Path

import pytest

from rag_baselines.validation_study import selected_questions, summary_rows


def test_requested_question_count_is_task_balanced_and_never_clamped_silently(tmp_path):
    directory = tmp_path/'data/prepared'
    directory.mkdir(parents=True)
    rows = [{'id': f'{task}:{i}', 'task': task, 'question': 'What?'} for task in ('nq','hotpotqa') for i in range(3)]
    (directory/'questions_dev.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    assert [q['id'] for q in selected_questions(tmp_path, 'dev', 2)] == ['nq:0','nq:1','hotpotqa:0','hotpotqa:1']
    with pytest.raises(ValueError, match='Too few'):
        selected_questions(tmp_path, 'dev', 4)
    with pytest.raises(ValueError, match='positive'):
        selected_questions(tmp_path, 'dev', True)


def test_reviewed_joint_answer_and_ungrounded_answer_do_not_replace_legacy_scores():
    record = {
        'retrieval': {'question_id':'joint','task':'nq','method':'D','repeat':0,'evidence':[],
                      'metrics':{'complete_evidence_hit':0,'complete_candidate_hit':0},
                      'seconds':{'retrieval_total':.1},'model_tokens':{},'trace':{'calls':[]}},
        'generation': {'answer':'English language','citations':[],'valid_output':True,
                       'seconds':{'prompt_prepare':.01,'api_request':.2},'api_usage':{'input_tokens':100}},
        'answer_metrics': {'answer_em':1.,'answer_f1':1.}, 'online_total_seconds':.32,'online_api_cost_usd':.0001}
    audit = {'joint_answers': {'joint':{'components':['English language','history']}},
             'gold_semantics_needing_review':{}}
    row = summary_rows([record], 'control', audit)[0]
    assert row['answer_em'] == 1 and row['reviewed_answer_em'] == 0
    assert row['component_coverage'] == .5 and row['grounding_status'] == 'unsupported_no_evidence'
    assert not row['protocol_compliant'] and not row['citation_entailment_verified']
