"""Eight-query GPU retrieval probe of a separately named development policy."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from rag_baselines.data import read_json,read_jsonl,write_json
from rag_baselines.store import DocumentStore
from rag_baselines.c_tree import load_tree
from rag_baselines.c_retrieval import JevRetrieval
from rag_baselines.laya_local import LayaDecisionClient
from rag_baselines.evaluation import retrieval_metrics
from rag_baselines.raw_logging import RawLog,query_scope
from scripts.check_laya_decisions import QIDS,append

if __name__=='__main__':
    output=Path((ROOT/'results/current_staged_run.txt').read_text(encoding='utf-8'))
    config=read_json(ROOT/'config/d_laya_candidate.json')
    store=DocumentStore(ROOT/'data/prepared/corpus_2000.jsonl')
    c_config=read_json(ROOT/'config/c.json')
    tree,preparation=load_tree(ROOT,store,read_json(ROOT/'config/c_taxonomy.json'),c_config)
    questions={q['id']:q for q in read_jsonl(ROOT/'data/prepared/questions_dev.jsonl')}
    labels={q['id']:q for q in read_jsonl(ROOT/'data/prepared/labels_dev.jsonl')}
    audit=RawLog(ROOT,manifest={'phase':'D_candidate_probe','question_ids':QIDS,'config':config},prices={})
    audit.snapshot(ROOT/'config/d_laya_candidate.json')
    client=LayaDecisionClient(config,device='cuda')
    engine=JevRetrieval(ROOT,store,tree,preparation,config,client=client,method='D',provider='laya')
    rows=[]
    for qid in QIDS:
        q=questions[qid]
        with query_scope(audit,'decision_probe',q,'D'):
            result=engine.retrieve(q['question'])
            result.update(question_id=qid,task=q['task'],repeat=0)
            result['metrics']=retrieval_metrics(result,labels[qid])
            result['raw_log']=audit.reference()
            audit.result('retrieval',result,label=labels[qid])
            append(output/'d_candidate_probe.jsonl',result)
        row={'question_id':qid,**result['metrics'],'evidence_count':len(result['evidence']),
             'candidates':len(result['candidates']),'seconds':result['seconds']['retrieval_total']}
        rows.append(row)
        print(row,flush=True)
    pd.DataFrame(rows).to_csv(output/'d_candidate_probe.csv',index=False)
    write_json(output/'d_candidate_probe_summary.json',{'rows':rows,'runtime':client.runtime_info(),
        'raw_log':str(audit.directory.relative_to(ROOT)),'scope':'Development probe on the same eight inspected questions; no generator and no generalization claim.'})
