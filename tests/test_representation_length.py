from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import json

import pandas as pd
import pytest
import httpx

from rag_baselines.c_tree import assemble_tree, digest
from rag_baselines.data import read_json, write_json, write_jsonl
from rag_baselines.parallel_routing import prepare_groups, reuse_description_checkpoints
from rag_baselines.representation_length import validate_length_settings, summarize_lengths, length_reference
from rag_baselines.store import DocumentStore


@pytest.fixture
def tiny_source(tmp_path):
    passages = [{'id':str(i),'title':f'Title {i}','content':f'Original fact {i}',
                 'source_spans':[],'wikipedia_id':str(i)} for i in range(3)]
    write_jsonl(tmp_path/'corpus.jsonl', passages)
    store = DocumentStore(tmp_path/'corpus.jsonl')
    taxonomy = {'id':'root','title':'Records','description':'Records','children':[
        {'id':'leaf','title':'Leaf','description':'Leaf'},
        {'id':'other','title':'Other','description':'Other'}]}
    tree = assemble_tree(store,taxonomy,{p['id']:['leaf'] for p in passages},
                         {'group_size':1,'descriptor_words':8})
    return store,tree


def test_new_length_protocol_isolated_from_old_checkpoints(tmp_path,tiny_source):
    store,_=tiny_source
    source,target=tmp_path/'cache/old',tmp_path/'cache/new'
    source.mkdir(parents=True);target.mkdir()
    common={'corpus_sha256':store.sha256,'tree_sha256':'tree','group_size':1,
            'description_policy':{'target':180},'generation':{'model':'pinned'}}
    descriptor={'summary':'Fact','entities':[],'qa':[]}
    write_json(source/'group_0000.json',{'signature':common,'passage_ids':['0'],
        'descriptor':descriptor,'descriptor_sha256':digest(descriptor)})
    reuse_description_checkpoints(target,{**common,'description_protocol':'new-band'},
                                 [{'id':'group_0000','passage_ids':['0']}],store)
    assert not (target/'group_0000.json').exists()


def test_concurrent_description_receipts_are_unique_and_cacheable(tmp_path,tiny_source):
    store,tree=tiny_source
    lock,captured,stages=Lock(),[],[]
    def create(**kwargs):
        passages=json.loads(kwargs['input'].split('\n',1)[1])
        descriptor={'summary':passages[0]['text'],'entities':[],'qa':[]}
        with lock:captured.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(descriptor),model_dump=lambda **_: {
            'status':'completed','usage':{'input_tokens':10,'output_tokens':5}})
    def result(stage,payload):
        with lock:stages.append(stage)
    generator=SimpleNamespace(responses=SimpleNamespace(create=create))
    audit=SimpleNamespace(result=result)
    settings={'descriptor_target_tokens':180,'descriptor_max_output_tokens':3200,
              'descriptor_input_tokens_limit':32768,'descriptor_length_mode':'target_band',
              'descriptor_preparation_workers':2}
    generation=read_json(Path(__file__).resolve().parents[1]/'config/generation.json')
    groups,receipt,path=prepare_groups(tmp_path,store,tree,1,settings,generation,generator,audit,progress=None)
    assert len(captured)==len(set(stages))==3
    assert receipt['preparation_workers']==2
    assert receipt['signature']['description_policy']['descriptor_length_mode']=='target_band'
    assert all('144 to 180' in r['input'] for r in captured)
    assert [g['passage_ids'] for g in groups]==[['0'],['1'],['2']]
    again,_,again_path=prepare_groups(tmp_path,store,tree,1,settings,generation,generator,audit,progress=None)
    assert again==groups and path==again_path and len(captured)==3
    legacy={k:v for k,v in settings.items() if k!='descriptor_length_mode'}
    _,legacy_receipt,legacy_path=prepare_groups(tmp_path,store,tree,1,legacy,generation,generator,audit,progress=None)
    assert len(captured)==6 and legacy_path!=path
    assert 'descriptor_length_mode' not in legacy_receipt['signature']['description_policy']


def test_length_settings_preflight_bounds_calls_and_fixed_treatment():
    settings=read_json(Path(__file__).resolve().parents[1]/'config/representation_length.json')
    validate_length_settings(settings)
    too_small={**settings,'max_llm_calls':1}
    with pytest.raises(ValueError,match='call budget'):validate_length_settings(too_small)
    with pytest.raises(ValueError,match='distinct'):validate_length_settings({**settings,'descriptor_token_targets':[180,180]})
    with pytest.raises(ValueError,match='common target-band'):validate_length_settings({**settings,'descriptor_length_mode':'upper_bound'})


def test_length_summary_keeps_zero_mass_conditional_unknown():
    rows=[]
    for target,hits in ((180,[0,0]),(360,[1,0])):
        for hit in hits:
            rows.append({'descriptor_target_tokens':target,'macro_complete':hit,'complete_evidence_hit':hit,
                'macro_tp':hit,'macro_positive':1,'macro_fp':1-hit,'macro_negative':2,
                'macro_seconds':.1,'micro_seconds':.2,'retrieval_seconds':.3,'decision_input_tokens':100,
                'decision_api_cost_usd':.001,'physical_http_requests':2,'input_truncated':False})
    preparation={str(t):{'descriptor_mean_tokens_proxy':t,'descriptor_min_tokens_proxy':t,
        'descriptor_max_tokens_proxy':t,'descriptor_target_band_fraction':1} for t in (180,360)}
    summary=summarize_lengths(pd.DataFrame(rows),preparation,.875).set_index('descriptor_target_tokens')
    assert pd.isna(summary.loc[180,'micro_complete_given_macro'])
    assert summary.loc[360,'micro_complete_given_macro']==1
    assert not summary.target_met.any()
    assert summary.loc[360,'macro_tpr']==.5


def test_reference_loader_never_reads_evaluation_and_checks_artifact_hashes(tmp_path):
    ref=tmp_path/'results/previous'
    manifest={'dev_question_ids':['nq0','hot0']}
    frozen={'settings':{'corpus_size':2}}
    labels={'nq0':{'evidence_groups':[['a']]},'hot0':{'evidence_groups':[['b']]}}
    baseline=[{'question_id':qid,'method':'B','evidence':[{'id':pid}],'candidates':[{'id':pid}],
               'question':'query','task':'nq' if qid=='nq0' else 'hotpotqa',
               'corpus_sha256':'source','corpus_count':2,'metrics':{'placeholder':1}}
               for qid,pid in (('nq0','a'),('hot0','b'))]
    from rag_baselines.evaluation import retrieval_metrics
    for row in baseline:row['metrics']=retrieval_metrics(row,labels[row['question_id']])
    for name,value in [('manifest.json',manifest),('frozen_policy.json',frozen),('preparation.json',{}),
                       ('development_labels.json',labels),('baseline_development_quality.json',baseline)]:
        write_json(ref/name,value)
    from rag_baselines.data import file_sha256
    write_json(ref/'completed.json',{'file_sha256':{p.name:file_sha256(p) for p in ref.iterdir()}})
    write_jsonl(tmp_path/'data/prepared/questions_dev.jsonl',[
        {'id':'nq0','task':'nq','question':'query'},{'id':'hot0','task':'hotpotqa','question':'query'}])
    write_json(tmp_path/'data/prepared/manifest.json',{'corpora':{'2':{'sha256':'source'}}})
    settings={'reference_run':'results/previous','corpus_size':2,'questions_per_task':1}
    loaded=length_reference(tmp_path,settings)
    assert len(loaded[1])==2 and loaded[4]==1
    write_jsonl(tmp_path/'data/prepared/questions_dev.jsonl',[
        {'id':'nq0','task':'nq','question':'changed query'},{'id':'hot0','task':'hotpotqa','question':'query'}])
    with pytest.raises(ValueError,match='identical question text'):length_reference(tmp_path,settings)
    write_jsonl(tmp_path/'data/prepared/questions_dev.jsonl',[
        {'id':'nq0','task':'nq','question':'query'},{'id':'hot0','task':'hotpotqa','question':'query'}])
    write_json(tmp_path/'data/prepared/manifest.json',{'corpora':{'2':{'sha256':'changed source'}}})
    with pytest.raises(ValueError,match='identical original corpus'):length_reference(tmp_path,settings)
    write_json(tmp_path/'data/prepared/manifest.json',{'corpora':{'2':{'sha256':'source'}}})
    write_json(ref/'development_labels.json',{})
    with pytest.raises(ValueError,match='changed'):length_reference(tmp_path,settings)


def test_completed_report_reconciles_cached_preparation_and_physical_charges(tmp_path,tiny_source):
    from rag_baselines.jev import JevClient
    from rag_baselines.raw_logging import RawLog
    from rag_baselines.parallel_routing import ParallelRouting, routing_quality
    from rag_baselines.representation_length import load_length_report, length_question_row
    from rag_baselines.data import file_sha256
    store,tree=tiny_source
    repo=Path(__file__).resolve().parents[1]
    generation,c_config=read_json(repo/'config/generation.json'),read_json(repo/'config/c.json')
    def generate(**kwargs):
        passage=json.loads(kwargs['input'].split('\n',1)[1])[0]
        descriptor={'summary':passage['text'],'entities':[],'qa':[]}
        return SimpleNamespace(output_text=json.dumps(descriptor),model_dump=lambda **_: {
            'status':'completed','usage':{'input_tokens':10,'output_tokens':5}})
    generator=SimpleNamespace(responses=SimpleNamespace(create=generate))
    groups,preparation={},{}
    for target in (180,360):
        group,prepared,path=prepare_groups(tmp_path,store,tree,1,{
            'descriptor_target_tokens':target,'descriptor_max_output_tokens':3200,
            'descriptor_input_tokens_limit':32768,'descriptor_length_mode':'target_band'},
            generation,generator,SimpleNamespace(result=lambda *args:None),progress=None)
        groups[target]=group
        actual=[g['descriptor_text_tokens_proxy'] for g in group]
        preparation[str(target)]={**prepared,'directory':str(path.relative_to(tmp_path)),
            'descriptor_mean_tokens_proxy':sum(actual)/len(actual),'descriptor_min_tokens_proxy':min(actual),
            'descriptor_max_tokens_proxy':max(actual),'descriptor_target_band_fraction':0}
    audit=RawLog(tmp_path,manifest={},prices={'jev':c_config['prices'],'openai':generation['prices']})
    def respond(request):
        body=json.loads(request.content)
        answers={qid:({'type':'noul','noul':.9} if q['type']=='noul' else
            {'type':'score','score':3,'confidence':1,'probabilities':{'0':0,'1':0,'2':0,'3':1}})
            for qid,q in body['questions'].items()}
        return httpx.Response(200,json={'model':c_config['model'],'answers':answers,
            'usage':{'input_tokens':100,'output_tokens':0}})
    label={'evidence_groups':[['0']]}
    policy={'selection_k':2,'evidence_k':2,'max_calls_per_retrieval':8,'http_workers':1}
    directory=tmp_path/'results/length_fixture'
    directory.mkdir()
    write_json(tmp_path/'data/prepared/manifest.json',{'corpora':{'3':{'path':'corpus.jsonl','sha256':store.sha256}}})
    records=[]
    with httpx.Client(transport=httpx.MockTransport(respond),event_hooks=audit.http_hooks) as http:
        client=JevClient(c_config,api_key='fixture-key',http_client=http)
        for target,group in groups.items():
            engine=ParallelRouting(store,group,client,'jev',policy,directory/'decision_calls.jsonl',audit)
            with audit.scope(phase='length_development',method='C-parallel',question_id='query',descriptor_target_tokens=target):
                retrieved=engine.retrieve('Original fact')
            records.append({'question_id':'query','task':'nq','descriptor_target_tokens':target,
                'retrieval':retrieved,'metrics':routing_quality(retrieved,group,label,store)})
    write_json(directory/'manifest.json',{'settings':{'corpus_size':3,'descriptor_token_targets':[180,360]},
        'corpus_sha256':store.sha256,'generation':generation,'c_model':c_config,
        'question_ids':['query'],'development_target':1,'raw_log':str(audit.directory.relative_to(tmp_path))})
    write_json(directory/'development_labels.json',{'query':label})
    write_json(directory/'preparation.json',preparation)
    write_json(directory/'fixed_conditions.json',{'policy':policy,
        'group_membership_sha256':digest([(g['id'],g['passage_ids']) for g in groups[180]]),
        'groups_sha256':{str(target):digest(group) for target,group in groups.items()}})
    write_jsonl(directory/'retrievals.jsonl',records)
    frame=pd.DataFrame([length_question_row(r) for r in records])
    frame.to_csv(directory/'questions.csv',index=False)
    summarize_lengths(frame,preparation,1).to_csv(directory/'summary.csv',index=False)
    write_json(directory/'completed.json',{'records':2,'physical_api_attempts':{'openai':0,'jev':4},
        'file_sha256':{p.name:file_sha256(p) for p in directory.iterdir() if p.is_file()}})
    restored,summary,_,_=load_length_report(tmp_path,directory)
    assert len(restored)==2 and summary.target_met.all()
    ledger_path=audit.directory/'api_calls.jsonl'
    ledger=[json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines()]
    body_path=audit.directory/ledger[0]['response']['path']
    original_body=body_path.read_bytes()
    body_path.write_bytes(original_body+b' ')
    with pytest.raises(ValueError,match='body integrity'):load_length_report(tmp_path,directory)
    body_path.write_bytes(original_body)
    ledger[0]['api_cost_usd']+=.01
    write_jsonl(ledger_path,ledger)
    with pytest.raises(ValueError,match='Physical API charges'):load_length_report(tmp_path,directory)
