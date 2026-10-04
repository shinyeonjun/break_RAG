import json

import pandas as pd
import pytest

from rag_baselines.data import file_sha256, write_json
from rag_baselines.validation_study import summary_rows, aggregate
from rag_baselines.validation_execution import freeze_d_policy


def completed_arm(output, arm, *, em, cost):
    directory = output/'dev'/arm
    directory.mkdir(parents=True)
    audit = {'joint_answers': {}, 'gold_semantics_needing_review': {}}
    record = {
        'retrieval': {'question_id':'nq:1','task':'nq','method':'D','repeat':0,'evidence':[],
                     'metrics':{'complete_evidence_hit':0,'complete_candidate_hit':0},
                     'seconds':{'retrieval_total':.1},'model_tokens':{},'trace':{'calls':[]}},
        'generation': {'answer':'Unknown','citations':[],'valid_output':True,
                      'seconds':{'prompt_prepare':.01,'api_request':.2},'api_usage':{'input_tokens':100}},
        'answer_metrics':{'answer_em':em,'answer_f1':em},'online_total_seconds':.32,'online_api_cost_usd':cost}
    (directory/'system.jsonl').write_text(json.dumps(record)+'\n',encoding='utf-8')
    aggregate(pd.DataFrame(summary_rows([record],arm,audit))).to_csv(directory/'summary.csv',index=False)
    write_json(directory/'completed.json',{'records':1,'system_sha256':file_sha256(directory/'system.jsonl')})
    write_json(directory/'manifest.json',{'split':'dev','question_ids':['nq:1'],'generation':{'model':'fixed'},
        'answer_audit':audit,'d':{'config':{'policy':arm},'signature':{'tree_sha256':'same','policy':arm}}})
    return directory


def test_d_freeze_uses_completed_dev_results_and_cannot_be_overwritten(tmp_path):
    completed_arm(tmp_path,'ABD_input_fix',em=0,cost=.001)
    completed_arm(tmp_path,'D_candidate',em=1,cost=.002)
    _, frozen = freeze_d_policy(tmp_path,tmp_path)
    assert frozen['selected_arm']=='D_candidate'
    assert frozen['d_config']['policy']=='D_candidate'
    with pytest.raises(ValueError,match='already frozen'):
        freeze_d_policy(tmp_path,tmp_path)


def test_d_freeze_rejects_summary_that_disagrees_with_original_jsonl(tmp_path):
    completed_arm(tmp_path,'ABD_input_fix',em=0,cost=.001)
    directory=completed_arm(tmp_path,'D_candidate',em=1,cost=.002)
    table=pd.read_csv(directory/'summary.csv')
    table['answer_em']=0
    table.to_csv(directory/'summary.csv',index=False)
    with pytest.raises(ValueError,match='summary'):
        freeze_d_policy(tmp_path,tmp_path)
    assert not (tmp_path/'frozen_d_policy.json').exists()


def test_d_freeze_rejects_different_question_sets(tmp_path):
    completed_arm(tmp_path,'ABD_input_fix',em=0,cost=.001)
    directory=completed_arm(tmp_path,'D_candidate',em=1,cost=.002)
    manifest=json.loads((directory/'manifest.json').read_text())
    manifest['question_ids']=['nq:2']
    write_json(directory/'manifest.json',manifest)
    with pytest.raises(ValueError,match='questions'):
        freeze_d_policy(tmp_path,tmp_path)
