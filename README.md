# decision_LLM — A/B/C/D 비교 구현

## Git 저장 범위

연구 저장소: [shinyeonjun/break_RAG](https://github.com/shinyeonjun/break_RAG).
코드, 설정, 연구 문서, 노트북 원본, 의존성 버전과 빈 .env.example을 관리한다.
모델 가중치, 데이터, 관계/벡터 색인, 원시 로그, 실험 결과, 가상환경과 vendor checkout은 로컬에 보관하며 Git에서 제외한다.
09의 실행 결론은 [관계 계획 가설 검증](research/plan-routing-2026-10-04.md)에 남겼다.

노트북 실행 출력과 실행 횟수는 커밋할 때만 제거한다. 로컬 노트북의 그래프와 출력은 유지된다.
새 checkout에서도 다음 로컬 Git 필터 설정을 적용한다(Python 3 필요).

~~~powershell
git config --local filter.notebook-source.clean "python scripts/git_notebook_filter.py"
git config --local filter.notebook-source.required true
~~~

BERGEN checkout은 저장소에 포함하지 않는다. 고정 revision이 필요한 새 환경에서는 다음과 같이 준비한다.

~~~powershell
git clone https://github.com/naver/bergen.git vendor/bergen
git -C vendor/bergen checkout eb8ec046eb6cf08be95b035b2d3bcc4b241ff212
~~~

모델·데이터는 아래 재현 절차와 01번 노트북으로 준비한다. .env.example을 .env로 복사하고 로컬에서 키를 입력한다.

측정은 `notebook/*.ipynb`에서 한다. 검색·문서 준비·평가·공통 생성기는 `rag_baselines/`의 함수를 사용한다. A/B RAG, C 원격 Jev 트리 탐색, D 로컬 Laya 트리 탐색을 같은 원문 저장소·질문·평가기·GPT-6 Luna 생성기로 비교한다. C와 D는 Jev로 미리 적재한 같은 트리를 공유하므로 온라인 판단 경로를 비교한다.

| 순서 | 노트북 | 역할 |
|---|---|---|
| 1 | [01_prepare_data.ipynb](notebook/01_prepare_data.ipynb) | 고정 질문·공통 corpus·정답 근거 위치·해시·준비 비용 검증 |
| 2 | [02_measure_retrieval.ipynb](notebook/02_measure_retrieval.ipynb) | 같은 후보를 사용하는 품질 평가 / 실제 검색을 수행하는 지연 측정 |
| 3 | [03_measure_system_api.ipynb](notebook/03_measure_system_api.ipynb) | 같은 GPT-6 Luna로 생성, 실제 전체 시간·EM/F1·usage·비용 기록 |
| 4 | [04_prepare_c_tree.ipynb](notebook/04_prepare_c_tree.ipynb) | Jev로 공통 자료를 자연어 트리에 적재, 준비 비용·전체 원문 접근·재개 검증 |
| 5 | [05_measure_abcd.ipynb](notebook/05_measure_abcd.ipynb) | 입력 검사 → C 개별효과 → D 개발 대조 → 정책 고정 → A/B/C/D 별도 평가. 기존 일반 비교도 선택 가능 |
| 6 | [06_compare_results.ipynb](notebook/06_compare_results.ipynb) | 완료 결과·입력·원시 로그·비용을 재검증하고 개발 효과·전체 비교·질문별 분포·대응 구간과 CSV 저장. API 호출 없음 |
| 7 | [07_mathematical_framework.ipynb](notebook/07_mathematical_framework.ipynb) | A/B 수식·기존 로그 검산 → 2단계 의미 라우팅 개발 측정 → 정책 고정 → 새 문항의 A/B 비교·수식 대조·그림. 기본은 결과 읽기, 실행 스위치를 켜면 유료 API 호출 |
| 8 | [08_compare_frozen_routing.ipynb](notebook/08_compare_frozen_routing.ipynb) | 07의 길이 실험에서 고른 설명·정책을 고정하고, 새 문항의 A/B/C/D 실제 답변·시간·GPT 입력·API 비용 비교. 기본은 완료 결과 검증·시각화 |
| 9 | [09_test_plan_routing.ipynb](notebook/09_test_plan_routing.ipynb) | 원문 인용에 연결된 관계 색인 → 1~2단계 계획 → Jev/Laya 판단 가설의 개발 검증. A/B, C 원문/인용, D 인용, 모델 없는 코드 대조군의 답변·시간·실제 입력/출력·비용·실패 단계 비교 |

09의 문항 수와 후보·선택 예산은 [config/plan_routing.json](config/plan_routing.json)에서 바꾼다. `RUN_PLAN_EXPERIMENT=False`가 기본이며 완료된 결과만 읽고, True이면 사전 관계 추출과 실제 비교를 실행한다. 기본 40문항에서는 본 비교 240회 생성과 동일 근거 인용 품질 재생 40회를 수행하며 각각의 호출 상한을 설정에서 관리한다. `RUN_FIXED_PACKING=True`로 저장된 C 원문 결과만 별도로 재생할 수도 있다. 이 품질 재생은 실제 검색 지연 측정에 합치지 않는다. 색인은 질문·정답 없이 공통 코퍼스만으로 준비하고 `data/plan_index/`에 배치별로 저장한다. 원시 호출은 준비/온라인을 구분해 `results/raw/`에 보존한다. 새 결과는 `results/plans_*`, 그림·CSV는 그 안의 `analysis/`에 저장하며 `results/current_plan_run.txt`가 최근 완료 실행을 가리킨다. 제공사 필터로 중단된 준비 배치는 재요청하지 않고 색인 누락으로 기록한다. 중단된 요청의 청구량이 확인되지 않으면 준비 비용을 정확한 총액으로 표시하지 않는다. 원문 ID 확보율과 인용 내용의 충분성은 별도이며 09는 최종 held-out 평가가 아닌 개발 파일럿이다.

커널은 **decision_LLM (Python 3.14, CUDA)**를 선택한다. 04는 2,000문단의 Jev 트리를 준비했으며 D는 동일한 트리를 재사용한다. D 원본 가중치는 `model/laya-multilingual/`에 있다. 개발 40문항과 별도 A/B/C/D 평가 100문항은 이미 완료했으며 [단계 검증 기록](research/validation-2026-10-04.md)에 보존했다. 입력 검사와 단계 실험을 05, 결과 검증과 시각화를 06으로 통합했다.

Windows Jupyter의 05 설정 셀에서는 Pandas/PyArrow를 PyTorch보다 먼저 import해야 한다. 역순에서 `arrow.dll` access violation이 재현돼 두 05 노트북의 초기화 순서를 수정했다. 같은 오류가 보이면 수정된 파일을 다시 열고 커널을 재시작한 뒤 첫 셀부터 실행한다. 재현 조건과 회귀 검사 범위는 [커널 충돌 진단](research/debug-kernel-2026-10-04.md)에 기록했다.

과거 두 문항 A/B/C 점검은 [audit-2026-10-04.md](research/audit-2026-10-04.md)에 보존되어 있다. 06은 가장 최근에 완료한 일반 ABC/ABCD 시스템 런 또는 단계 검증의 평가를 자동으로 읽는다. 그림과 CSV는 새 `results/comparison_<UTC시간>_<ID>/`에 저장하고 원본을 덮어쓰지 않는다.

## 문항 수를 바꾸고 직접 실행

05의 기본 `WORKFLOW='staged'`는 [config/validation_study.json](config/validation_study.json)을 사용한다. `questions_per_task`는 **NQ와 HotpotQA 각각**의 문항 수다. 기본 개발 각 20개(40문항), 평가 각 50개(100문항)이며 개발 변형·대조군도 실행하므로 GPT 호출 상한은 합계 720회다. 정책 선택은 개발에서만 하고 평가 전에 고정한다. 입력 재생·48개 판단 검사·개발·평가를 순차 실행하며 `RUN_STUDY=False`이면 설정만 확인한다. 단계 검증은 원시 로그와 반복 1회를 요구한다.

기존 직접 비교는 05에서 `WORKFLOW='standard'`를 선택하고 [config/experiment.json](config/experiment.json)을 수정한다. 품질 캐시·독립 검색 지연·공통 생성기를 구분해 실행하는 이전 기능을 유지한다. 아래 설정은 일반 비교에 적용한다.

```json
{
  "split": "dev",
  "questions_per_task": {"dev": 20, "eval": 50},
  "repeats": 1,
  "corpus_size": 2000,
  "methods": ["A", "B", "C", "D"],
  "retrieve_k": 50,
  "evidence_k": 5,
  "run_retrieval": true,
  "run_generation": true,
  "save_raw_logs": true
}
```

`split: dev`는 총 40문항·GPT 생성 최대 160회, `split: eval`은 총 100문항·최대 400회다. 문항 수나 반복 수에 맞춰 상한을 계산한다. 고정 질문은 네 방법에 공통으로 전달하고, 실행 manifest에 설정 스냅샷·해시·선택 질문 ID·실행 methods를 남긴다. 개발은 종류별 최대 100, 평가는 종류별 최대 500문항이며 부족한 문항 수는 실행 전에 거부한다.

05를 처음부터 끝까지 실행한 다음 06을 실행한다. 단계 검증의 `RUN_STUDY=True` 또는 일반 비교의 `run_retrieval`·`run_generation=true`는 GPT/Jev 유료 API 호출을 포함한다. corpus 크기를 바꿀 때는 04에서 공통 트리를 먼저 준비해야 한다. 새 단계 검증은 고유 `results/staged_*` 폴더를 만들고, 평가까지 완료되면 `results/current_staged_run.txt`를 갱신한다. 06의 `SYSTEM_RUN`으로 과거 묶음이나 개별 런을 선택할 수 있다.

```powershell
.\venv\Scripts\python.exe scripts/execute_notebooks.py 05_measure_abcd.ipynb 06_compare_results.ipynb
```

비교만 다시 실행하려면 아래 명령을 사용한다. 유료 API 호출 없이 원문·프롬프트·평가·usage·비용을 재검증하고 그림을 만든다.

```powershell
.\venv\Scripts\python.exe scripts/execute_notebooks.py 06_compare_results.ipynb
```

07부터는 **변수·가정 → 수식 → 개발 측정 → 정책 고정 → 새 평가의 검산** 형식으로 새 알고리즘을 정리한다. 1~9절은 완료한 기존 런을 읽고 `results/math_<UTC시간>_<ID>/`에 질문별 계산·출처 해시·그림을 저장한다. 10~14절은 [2단계 라우팅 실험](research/parallel-routing.md)과 관측 결과의 해석이다. 기본 `RUN_PARALLEL_EXPERIMENT=False`는 마지막 완료 실험만 읽고 검증·시각화하며 유료 호출은 없다.

새 실험은 07의 `RUN_PARALLEL_EXPERIMENT=True`로 실행한다. [config/parallel_routing.json](config/parallel_routing.json)에 그룹 크기·선택 그룹 수·개발/평가 문항 수·호출 한도를 둔다. 최초 설정은 2,000문단, 그룹 20/50개, 선택 1/3/5묶음, 개발 각 20문항, 새 평가 각 10문항이다. 기존 Jev 분류 결과를 재사용하고 공통 GPT-6 Luna로 질문 없는 묶음 설명을 준비한다. 새 준비 비용과 상속한 준비 비용을 따로 기록한다. 개발 판단의 부분집합은 품질만 재생하며, 지연 측정은 정책 고정 뒤 과거 평가 ID를 제외한 문항에서 실제 검색을 수행한다.

실험은 `results/parallel_<UTC시간>_<ID>/`에 설정·준비·개발 원시 판단·정책 고정·질문별 평가·완료 해시를 저장한다. 모델/API 경계는 기존 `results/raw/`에 보존하고, 추가 결정 요청은 실험 폴더의 `decision_calls.jsonl`에 있다. 비교 그림과 측정된 토큰 모형은 `analysis/`에 저장한다. `results/current_parallel_run.txt`는 마지막 완료 실험을 가리킨다. 중단된 실험의 부분 결과와 실패 기록도 보존한다. 05/06 및 기존 C/D 구현은 유지한다.

07의 **15~18절**은 대표 설명 길이 실험이다. [config/representation_length.json](config/representation_length.json)에 고정 g=20, K=5, 길이 목표 180/360/720, 개발 각 20문항을 둔다. 모든 길이 조건에서 동일한 새 준비 프롬프트를 사용하고 실제 설명 길이를 기록한다. 같은 그룹·모델·검색 규칙에서 조건 순서를 섞어 Jev 검색을 새로 실행한다. 평가 문항과 답변 생성은 사용하지 않는다. `RUN_LENGTH_STUDY=False`가 기본이며 True일 때만 유료 설명 준비·판단을 호출한다. 이전 180토큰 설명은 준비 프롬프트가 달라 이번 길이 비교에 섞지 않는다.

완료 결과는 `results/length_<UTC시간>_<ID>/`에, 원시 HTTP 요청/응답·물리 청구 기록은 manifest의 `raw_log` 경로에 보존한다. `retrievals.jsonl`에는 질문·조건별 실제 근거·판단·usage·시간, `analysis/`에는 설명 길이 분포·근거 확보율·입력·요금·시간 그림과 검산 결과가 있다. `results/current_length_run.txt`가 마지막 완료 결과를 가리킨다. 설명 준비는 일치하는 체크포인트를 재사용하되 온라인 검색은 재사용하지 않는다.

08은 [config/frozen_comparison.json](config/frozen_comparison.json)의 명시된 완료 길이 실험에서 720 목표 설명을 그대로 재사용한다. 개발에서 고른 g=20,K=5,최종 근거 최대 5개를 유지한다. D에도 같은 설명과 정책을 적용하는 비교이며 D의 최적 설정은 아니다. `questions_per_task`만 20 또는 50으로 바꾸면 NQ·HotpotQA에 각각 적용된다. 기본 평가 40문항, 반복 1회이며, 기존 평가 ID와 개발 ID를 제외한다. 그 이상의 문항·반복은 호출 상한도 확인한다.

08의 `RUN_FROZEN_COMPARISON=True`만 새 GPT/Jev 유료 호출을 실행한다. 기본 False는 `results/current_frozen_run.txt`가 가리키는 검증 완료 결과만 읽는다. `results/frozen_*/`에 실제 원문·생성 답변·질문별 CSV·준비 비용·동결 정책과 원시 로그 연결을 저장하고, `analysis/`에 비교 그림·자료 유형별 결과·같은 문항에서의 정오 차이를 남긴다. 근거 확보율과 답변 EM을 구분하며 생성기 입력 감소를 전제하지 않는다.

```powershell
.\venv\Scripts\python.exe scripts/execute_notebooks.py 07_mathematical_framework.ipynb
```

## 고정한 기준

- A: RetroMAE CLS embedding → 정규화 없는 dot-product exact search → 원문 선택.
- B: A와 동일한 dense 후보 → 질문/원문 쌍을 DeBERTa-v3로 재정렬 → 원문 선택.
- C: Jev Choice로 사전 적재한 자연어 트리 → Noul로 필요한 여러 분기 탐색 → 원문 Score → 근거 완결성 Noul → 공통 LLM. 임베딩·BM25·A/B 후보를 사용하지 않는다.
- D: C와 같은 Jev 준비 트리·원문·탐색 코드 → pinned Laya Multilingual을 로컬에서 분기 Noul·원문 Score·완결성 Noul에 사용 → 공통 LLM. 별도 로컬 비교군으로 두며 C는 보존한다. D는 독립 트리 준비 성능을 측정하지 않는다.
- 최초 검색 50개·선택 5개는 `config/baselines.json`과 노트북 매개변수에서 변경한다.
- A/B 검색 모델은 각각 최대 256 토큰으로 고정한다. B의 DeBERTa는 질문을 보존하고 문서만 자르며 최종 반환은 원문이다. D의 Laya 입력 한도는 별도 설정에서 관리한다.
- [BERGEN](https://github.com/naver/bergen/tree/eb8ec046eb6cf08be95b035b2d3bcc4b241ff212) `eb8ec046eb6cf08be95b035b2d3bcc4b241ff212`.
- [RetroMAE](https://huggingface.co/Shitao/RetroMAE_MSMARCO_distill) `a60b7d0ba401b1d50ffee16c3a89fad923f7fc28`.
- [DeBERTa](https://huggingface.co/naver/trecdl22-crossencoder-debertav3) `24f6a61d11707432d5780a1d5cf4e3af25cfaddb`.

이 구현은 고정 BERGEN의 모델·pooling·유사도·토큰화 설정을 따른 **네이티브 Windows용 adapter**다. BERGEN 전체 runner를 설치하거나 공식 논문 결과를 재현했다고 쓰면 안 된다. embedding batch는 512에서 32로 줄였고, 동일 점수에는 corpus 순서로 tie를 해소한다. 모델·GPU·토큰화·동기화 등 실제 환경은 실행 manifest에 기록한다. vendor checkout에는 Windows의 대소문자 파일명 충돌 경고가 있었지만 사용한 NQ/HotpotQA 처리 및 모델 설정 파일은 확인했다.

## 공통 데이터

[공식 KILT Wikipedia](https://github.com/facebookresearch/KILT#kilt-knowledge-source) 원본 37,318,876,722바이트와 [facebook/kilt_tasks](https://huggingface.co/datasets/facebook/kilt_tasks) revision `0f7881c6bf693742af91a31b4d8f827db2f41233`을 받았다. 변환 mirror는 사용하지 않는다. 원본은 `data/raw/`, 실제 사용 자료는 `data/prepared/`에 있다.

NQ와 HotpotQA **각각** train에서 개발 100문항, validation에서 평가 500문항을 고정했다. HotpotQA는 한 annotation에 서로 다른 두 개 이상 문서가 함께 필요한 질문을 사용한다. 각 annotation의 모든 span을 확보해야 성공이며, 유효한 다른 annotation은 대안으로 허용한다. BLEU ≥ 0.99의 정렬과 완전한 위치 매핑을 요구한다. 매핑 실패 19문항은 예비 순서로 대체하고 manifest에 남겼다. 질문 파일에는 정답·근거 라벨이 없다.

BERGEN KILT100w 규칙(100단어, 제목 추가, 겹침 없음, BULLET 문단 제외)에 따라 원문과 원래 문단·문자 좌표를 보존했다. 공동 정답 근거 1,901개를 포함한 nested corpus는 **2,000 / 10,000 / 40,000 passage**다. 배경은 전체 원본의 균일 문서 표본 15,000개에서 추출한다. 모든 passage에 대한 전역 균일 표본은 아니다. 정답을 강제로 포함한 축소 corpus이므로 KILT 전체 benchmark 수치와 직접 비교하지 않는다.

## 측정과 저장

04/05는 기본으로 **분석용 원시 로그**를 `results/raw/<UTC시간>_<ID>/`에 저장한다. 질문·방법·반복·단계별로 실제 로컬 입력/출력, 임베딩/검색 점수/재정렬 logits, Jev·GPT 요청/응답을 연결한다. D의 실제 state·typed questions·원시 응답·usage·잘림 정보·단계 시간은 `results/d_requests/`와 질문별 trace에 저장된다. API 비용은 실제 API 시도만 합산하며 캐시 응답을 중복 계산하지 않는다. 저장 범위·파일 구조·노트북 분석 예시는 [원시 로그 안내](research/raw-logs.md)에 있다.

원시 입출력 기록 비용은 관측 시간에 포함되고 `diagnostic_capture_seconds`에 추가 기록 시간을 남긴다. 06의 질문별 CSV에는 원시 로그 연결과 해당 시간이 포함된다. 기록 없는 비교는 `save_raw_logs: false`로 별도 실행하고 조건을 구분한다. API 키·인증 헤더와 `.env`는 저장하지 않는다. D 원본 모델은 다운로드했고 로컬 어댑터·탐색·노트북 초기화의 회귀 테스트를 확인했다. 실제 Jupyter 커널에서 D의 GPU 로딩·짧은 `noul`/`score` 추론·입력 잘림 거부까지 확인했다. 새 A/B/C/D 비교 실험과 품질 평가는 아직 실행하지 않았다.

품질 평가의 dense 후보 캐시는 질문·corpus·실제 모델 파일 해시·검색 개수에 묶인다. A/B는 같은 후보 ID·dense 점수를 공유한다. C/D는 같은 트리 기반 경로를 쓰되 모델·정책·코드·질문에 묶인 별도 판단 캐시를 사용한다. 지연 측정은 검색 결과 캐시를 읽지 않고 각 방법이 처음부터 검색한다. Jev API와 D 로컬 처리량은 각각 별도로 기록한다.

매 실행마다 `results/<UTC시간>_<ID>/`에 `retrieval.jsonl`, `retrieval.csv`, `summary.csv`, 실행 manifest를 저장한다. 선택한 원문·ID·좌표, 50개 후보·점수, 단계 시간, 원래/처리/패딩 토큰과 잘림 수를 기록한다. 모델 로딩·인덱스 준비 시간은 온라인 시간에서 제외한다. 준비 시간·디스크 크기는 `data/prepared/preparation_metrics.json`에 있다. 최초 다운로드의 활성 시간은 측정하지 않아 비어 있다. 로컬 처리 토큰은 API 청구 토큰이 아니며, 로컬 GPU의 금전 비용도 자동으로 0으로 간주하지 않는다.

C/D는 `data/c_trees/<해시>/`의 같은 원문 참조 트리·준비 manifest를 사용한다. 이 트리는 기존 Jev 적재물이며 D의 독립 준비 결과가 아니다. `results/c_requests/`에는 Jev 요청, `results/d_requests/`에는 Laya 로컬 판단 요청을 기록한다. 05는 완료한 질문 결과를 `results/partial/`에도 즉시 저장한다. Manifest는 A/B 로컬 GPU, C 원격 API, D 로컬 모델과 공통 트리 준비를 명시한다.

공통 생성기는 `config/generation.json`의 **gpt-6-luna**, reasoning `none`, temperature 0, 출력 최대 256 토큰으로 고정했다. 원문 근거 2,000 토큰·총 입력 텍스트 4,096 토큰 예산을 넘으면 조용히 버리지 않고 오류를 낸다. o200k_base는 로컬 예산 proxy이고 실제 usage를 청구 토큰으로 기록한다. 응답 모델명·원시 응답·캐시 읽기/쓰기·실제 프롬프트·가격표 날짜를 저장한다. [공식 모델](https://developers.openai.com/api/docs/models/gpt-6-luna), [가격](https://developers.openai.com/api/docs/pricing), [캐시 비용](https://developers.openai.com/api/docs/guides/prompt-caching)을 확인했다. API의 모델 alias는 서버에서 바뀔 수 있어 로컬 모델 revision과 동일한 불변성을 보장하지 않는다.

실제 생성은 프로젝트 루트 `D:\decision_LLM\.env`에 `OPENAI_API_KEY=실제키`를 설정하고 커널을 다시 시작한 뒤 03에서 `RUN_API=True`로 실행한다. `.env.example`을 설정 양식으로 사용할 수 있다. 03의 첫 설정 셀은 [python-dotenv](https://bbc2.github.io/python-dotenv/)로 루트의 `.env`를 읽으며, 이미 설정된 환경변수가 있으면 그 값을 우선한다. `.env`는 Git 제외 대상이다. 초기 호출 한도는 4회다. SDK 자동 재시도는 0회이며 실패 시 노트북이 중단된다. 전체 런 성공 전에 API 오류가 나면 메모리에 남은 성공 기록을 저장해야 하므로 큰 유료 런 전에는 이 파일럿부터 확인한다. API 키·인증 헤더는 결과에 저장하지 않는다.

C는 같은 `.env`에 `TYPESAFE_API_KEY=실제키`가 필요하다. 04의 `RUN_PREPARATION=True`는 새 트리에 유료 적재를 수행하고, 완료된 동일 트리는 재사용한다. 05는 `config/experiment.json`의 `run_retrieval`·`run_generation`을 따른다. 새 설정과 과거 두 문항 결과가 섞이지 않도록 05의 이전 출력은 비웠고, 과거 원시 결과와 그림은 그대로 보관했다. 노트북 전체 실행 명령은 설정에 따라 유료 호출을 포함한다.

Jev는 [공식 버전](https://docs.typesafe.ai/models) `jev-1.13.0`을 고정했다. D는 [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) commit `1720e3e3357cfe1e281542e223f8273b0890ca34`과 `laya==0.3.26`을 고정하고 local eager backend로 직접 로드한다. C/D 임계값은 별도 정책으로 관리하고 개발셋에서 조정·동결해야 한다. 초기 D 설정은 [config/d_laya.json](config/d_laya.json)에 있다.

`laya==0.3.26`의 배포 wheel에는 `laya.backends`가 누락돼 있어 `backend='eager'` 대신 기존 `fast=False, compile=False` 로더 인자를 사용하고 실제 backend가 eager인지 확인한다. 패치·검증 기록은 [디버깅 기록](research/debug-kernel-2026-10-04.md)에 있다. 실제 GPU/Jupyter 회귀 검사는 해당 체크포인트가 Hugging Face 캐시에 준비된 환경에서 `DECISION_LLM_TEST_LAYA_RUNTIME=1`을 설정해 실행한다. 이 검사는 offline이며 Jev/GPT 호출을 하지 않는다.

## 실행 환경 재현

기존 환경의 의존성은 `requirements-notebook.txt`, 전체 버전은 `environment.lock.txt`에 기록했다. PyTorch CUDA wheel은 공식 인덱스를 사용한다.

```powershell
uv pip install --python .\venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
uv pip install --python .\venv\Scripts\python.exe -r requirements-notebook.txt
.\venv\Scripts\python.exe -m ipykernel install --prefix D:\decision_LLM\venv --name decision-llm --display-name 'decision_LLM (Python 3.14, CUDA)'
.\venv\Scripts\python.exe scripts/execute_notebooks.py
.\venv\Scripts\python.exe -m pytest -q
```

모델 파일이 없는 새 환경에서는 `venv/Scripts/hf.exe download`에 위 repo/revision과 `--local-dir`을 지정한다. RetroMAE 저장소는 원본 PyTorch bin을, DeBERTa는 Safetensors를 사용했다. Wikipedia 재다운로드는 `scripts/download_kilt.py`로 재개할 수 있다. QA 원본은 같은 revision의 `nq/`·`hotpotqa/` train/validation parquet를 받아야 한다.

공통 규격은 [research/baseline-contract.md](research/baseline-contract.md), C의 실험 범위와 실행 결과는 [research/c-method.md](research/c-method.md)를 따른다.
