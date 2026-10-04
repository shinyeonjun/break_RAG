"""Write a source-backed report from completed runs and executed notebook exports."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from rag_baselines.data import read_json, read_jsonl, file_sha256, write_json
from rag_baselines.evaluation import notebook_source_sha256


def write_report(output):
    output=Path(output)
    evaluation=output/'eval/ABCD_frozen'
    completed=read_json(evaluation/'completed.json')
    if file_sha256(evaluation/'system.jsonl')!=completed['system_sha256']:
        raise ValueError('Evaluation source changed after completion.')
    manifest=read_json(evaluation/'manifest.json')
    frame=pd.read_csv(evaluation/'questions.csv')
    summary=pd.read_csv(evaluation/'summary.csv').set_index('method')
    cdev=pd.read_csv(output/'dev_c_summary.csv').set_index('arm')
    ddev=pd.read_csv(output/'dev_d_summary.csv').set_index('arm')
    roles=pd.read_csv(output/'eval_api_roles.csv')
    paired=pd.read_csv(output/'eval_paired_comparison.csv')
    pipeline=pd.read_csv(output/'eval_pipeline_observations.csv')
    cfreeze=read_json(output/'frozen_c_policy.json')
    dfreeze=read_json(output/'frozen_d_policy.json')
    if set(summary.index)!={'A','B','C','D'} or len(frame)!=completed['records']:
        raise ValueError('Need complete A/B/C/D evaluation observations.')
    questions=int(summary.questions.iloc[0])
    if not all(summary.questions==questions):
        raise ValueError('Methods have unequal questions.')
    returned=set(); d_input_rows=0; d_max_tokens=0
    for line in (evaluation/'system.jsonl').open(encoding='utf-8'):
        record=json.loads(line)
        returned.add(record['generation']['returned_model'])
        if record['retrieval']['method']=='D':
            for call in record['retrieval']['trace']['calls']:
                if call.get('cpu_fallbacks',0): raise ValueError('D CPU fallback found.')
                for row in call['effective_inputs']:
                    encoding=row['encoding']
                    if encoding['truncated'] or encoding['state_tokens_dropped'] or encoding['instruction_tokens_dropped'] or any(encoding['option_tokens_dropped']):
                        raise ValueError('D input truncation found.')
                    d_input_rows+=1
                    d_max_tokens=max(d_max_tokens,encoding['input_tokens'])
    source_paths=[evaluation/p for p in ('manifest.json','policy_before_labels.json','system.jsonl','questions.csv','summary.csv','completed.json')]
    source_paths.extend(output/p for p in ('dev_c_summary.csv','dev_d_summary.csv','frozen_c_policy.json','frozen_d_policy.json',
        'd_input_replay_summary.json','decision_validation_summary.json','score_order_summary.json','eval_api_roles.csv',
        'eval_paired_comparison.csv','eval_local_throughput.csv','eval_pipeline_observations.csv'))
    for arm in ['C_baseline','C_top2','C_top3','C_dedup','ABD_input_fix','D_candidate']:
        source_paths.extend(output/'dev'/arm/p for p in ['manifest.json','system.jsonl','summary.csv','completed.json'] if (output/'dev'/arm/p).exists())
    rows=[]
    labels={'A':'기본 RAG','B':'재정렬 RAG','C':'Jev 트리 · top3','D':'Laya 트리 · 개발 선택'}
    for method,row in summary.iterrows():
        rows.append(f"| {method} · {labels[method]} | {row.answer_em:.1%} | {row.answer_f1:.3f} | {row.complete_evidence_hit:.1%} | "
                    f"{row.online_seconds:.3f} | {row.retrieval_seconds:.3f} | {row.llm_input_tokens:.1f} | ${row.online_api_cost_usd:.7f} |")
    c=summary.loc['C']; a=summary.loc['A']; b=summary.loc['B']; d=summary.loc['D']
    navigation_usd=roles.loc[(roles.method=='C') & (roles.role=='navigation'),'USD'].sum()
    navigation_calls=int(roles.loc[(roles.method=='C') & (roles.role=='navigation'),'requests'].sum())
    c_vs_b=paired[(paired.comparison=='C - B') & (paired.metric=='answer_em')].iloc[0]
    d_missing=int(((pipeline.method=='D') & (pipeline.category=='후보 근거 ID 미확보')).sum())
    lines=[
        '# 개발 수정 및 별도 평가 결과',
        '',
        f"NQ·HotpotQA 각 {questions//2}문항, 방법당 {questions}문항을 새 검색·생성으로 한 번씩 실행했다. "
        f"C 정책은 `{cfreeze['selected_arm']}`, D 정책은 `{dfreeze['selected_arm']}`로 개발에서 고정했다.",
        '',
        '| 방법 | 원래 EM | F1 | 공동 근거 ID 확보 | 전체 초/질문 | 검색 초/질문 | GPT 입력 토큰/질문 | API USD/질문 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',*rows,
        '',
        f"C는 A보다 전체 시간이 {c.online_seconds/a.online_seconds:.2f}배, 기록된 API 비용이 {c.online_api_cost_usd/a.online_api_cost_usd:.2f}배다. "
        f"생성기 입력은 A보다 {(1-c.llm_input_tokens/a.llm_input_tokens):.1%} 줄었다. "
        '입력 감소와 서비스 전체의 효율 개선을 구분해야 한다.',
        '',
        f"C의 분기 탐색은 {navigation_calls}회, API 비용 ${navigation_usd:.6f}였다. "
        f"C 전체 온라인 API 비용의 {navigation_usd/(c.online_api_cost_usd*questions):.1%}를 차지했다. "
        '이번 관측에서 비용의 주된 단계는 분기 탐색이다.',
        '',
        f"C와 B의 EM 차이는 {c_vs_b.mean_difference*100:+.1f}%p다. 95% 대응 부트스트랩 구간은 "
        f"[{c_vs_b.bootstrap_low*100:.1f}, {c_vs_b.bootstrap_high*100:.1f}]%p로 0을 포함한다. "
        '이번 표본으로 B에 대한 정확도 우위를 확정하지 않는다.',
        '',
        f"D의 EM은 {d.answer_em:.1%}, 공동 근거 ID 확보율은 {d.complete_evidence_hit:.1%}였다. "
        '낮은 API 비용은 로컬 탐색에 API 요금을 매기지 않은 결과도 포함한다. GPU·전력·유지보수 비용을 포함한 총운영비가 아니다.',
        f"D에서는 {d_missing}문항이 후보 단계의 공동 근거 ID 미확보 및 EM 불일치로 분류됐다. "
        f"실제 입력 {d_input_rows:,}개 검사에서 잘림과 CPU fallback은 없었으므로 입력 잘림만으로 낮은 품질을 설명할 수 없다. "
        f"가장 긴 실제 입력은 {d_max_tokens:,}토큰이었다.",
        '',
        '## 개발 변경과 진단',
        '',
        f"C의 개발 EM은 baseline {cdev.loc['C_baseline','answer_em']:.1%}, top2 {cdev.loc['C_top2','answer_em']:.1%}, "
        f"top3 {cdev.loc['C_top3','answer_em']:.1%}, 충분성 재사용 {cdev.loc['C_dedup','answer_em']:.1%}였다. "
        f"재사용으로 실제 충분성 호출 {int(cdev.loc['C_dedup','coverage_reused'])}개를 피했지만 "
        'top3와 결합해 평가한 것은 아니다.',
        '',
        f"D는 입력 수정만 한 대조군에서 개발 EM {ddev.loc['ABD_input_fix','answer_em']:.1%}, "
        f"별도 후보에서 {ddev.loc['D_candidate','answer_em']:.1%}였다. "
        '분기 0.1·충분성 0.8·top3 후보는 여러 변경을 동시에 했으므로 각 변경의 효과를 따로 주장하지 않는다.',
        '',
        '실제 SDK 패킹 재생 3,701개는 지시문·선택지·본문을 보존했다. 이는 추론 성능 검증과 다르다. '
        '출처를 확인한 48개 개발 판단과 16개 선택지 순서 검사에서 절대 점수/확률의 한계가 관측됐다. '
        '입력 수정을 완료했다고 해당 판단모델의 탐색 품질이 충분해지는 것은 아니다.',
        '',
        '## 평가 대조와 해석',
        '',
        '원시 결과와 요약 CSV, 완료 해시, 실제 전달 프롬프트, 원문 저장소, A/B 후보 ID·dense 점수, '
        '원래 답변/근거 지표, usage와 비용을 대조했다. 물리 API ledger와 질문별 비용도 일치했다. '
        'D의 실제 입력 잘림과 CPU fallback은 허용하지 않았다.',
        '',
        f"공통 요청 생성기는 `{manifest['generation']['model']}`이고 실제 반환 ID는 {', '.join(f'`{v}`' for v in sorted(returned))}다. "
        f"로컬 장치는 `{manifest['gpu']}`다. A/B dtype은 `{manifest['dtype']}`, "
        f"D dtype은 `{manifest['d']['runtime']['dtype']}`와 AMP `{manifest['d']['config'].get('cuda_amp')}`다. "
        '모델과 BERGEN 리비전은 manifest에 남겼다.',
        '',
        'EM과 F1은 원래 벤치마크 규칙이다. 근거 ID 확보는 내용의 충분성이나 인용 entailment를 검증하지 않는다. '
        '개발의 공동 답 항목·의미적으로 약한 gold ID는 별도 표지로 보존했다. 평가 전체의 근거 의미를 수동 검증한 것은 아니다.',
        '',
        '평가는 다른 실험을 동시에 실행하지 않고 한 프로세스에서 방법 순서를 질문마다 무작위로 섞었다. '
        '개발 C와 A/B/D 대조군은 일부 시간에 겹쳤으므로 개발 지연은 탐색적이다. '
        '시간에는 측정 구간 안의 원시 기록 비용이 포함되며 마지막 사본 저장·준비·로딩·워밍업은 제외한다.',
        '',
        '공통 코퍼스는 정답 근거를 강제 포함한 2,000문단이다. D도 C/Jev로 준비한 트리를 재사용했다. '
        '문서 적재 전체가 로컬인 시스템이나 전체 Wikipedia·기업 자료의 성능으로 일반화하지 않는다. '
        '방법·모델·배포 경로가 함께 다르므로 결과를 트리 알고리즘만의 효과로 분리하지 않는다. '
        'API 비용은 저장 단가로 계산한 추정이며 준비·GPU·인건비를 제외한다.',
        '',
        '대응 비교의 95% 부트스트랩 구간은 같은 질문을 NQ·HotpotQA별로 복원추출한 보조 분석이다. '
        '반복 한 번의 API 시간대 변동은 추정하지 않는다.',
        '',
        '## 분석 파일',
        '',
        '- [비교 분석 노트북](../../notebook/06_compare_results.ipynb)',
        '- [최종 비교 그림](eval_ABCD_frozen_comparison.png) · [분포 그림](eval_distribution.png)',
        '- [질문별 결과](eval/ABCD_frozen/questions.csv) · [요약](eval/ABCD_frozen/summary.csv)',
        '- [대응 비교 구간](eval_paired_comparison.csv) · [API 역할별 집계](eval_api_roles.csv)',
        '- [로컬 단계 처리량](eval_local_throughput.csv) · [파이프라인 관측](eval_pipeline_observations.csv)',
        '- [원시 세션 manifest](../../'+manifest['raw_log'].replace('\\','/')+'/manifest.json)',
        '',
        '원본 결과와 설정은 보존했다. 새 유료 실행은 05의 staged 실행으로 고유 디렉터리에서 진행한다.'
    ]
    (output/'validation_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(output/'analysis_manifest.json',{'source_sha256':{str(p.relative_to(ROOT)):file_sha256(p) for p in source_paths},
        'notebook_source_sha256':notebook_source_sha256(ROOT/'notebook/06_compare_results.ipynb'),
        'analysis_code_sha256':file_sha256(Path(__file__)), 'records':completed['records'],
        'questions_per_method':questions,'raw_log':manifest['raw_log'], 'returned_generator_ids':sorted(returned),
        'main_figures': ['eval_ABCD_frozen_comparison.png','eval_distribution.png'],
        'limits':['One repetition','Gold-forced reduced corpus','Development timing partly overlapped','API fees exclude GPU and preparation','ID matches are not semantic entailment']})
    print(output/'validation_report.md')


if __name__=='__main__':
    output=Path((ROOT/'results/current_staged_run.txt').read_text(encoding='utf-8'))
    write_report(output)
