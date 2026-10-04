# A/B/C/D 원시 로그

일반 비교는 `config/experiment.json`의 `save_raw_logs: true`가 기본값이고, 단계 검증은 `config/validation_study.json`의 원시 기록을 요구한다. 04는 공통 Jev 트리 준비를, 05는 A/B/C/D 비교 호출을 기록한다. D는 같은 C 트리를 사용하며 local Laya 요청은 API 청구 ledger와 분리된 decision journal에 남긴다.

입력 수정·C 개별효과·정책 고정 후 별도 평가는 [05_measure_abcd.ipynb](../notebook/05_measure_abcd.ipynb)의 기본 `WORKFLOW='staged'`에서 실행한다. 문항 수는 `config/validation_study.json`에서 바꾼다. [06_compare_results.ipynb](../notebook/06_compare_results.ipynb)는 저장 결과만 읽어 검증·시각화한다. `results/current_staged_run.txt`는 마지막 완료 묶음이며 과거 원시 로그와 결과는 보존한다.

05 → 06 순서로 실행한다. 06의 질문별 CSV에는 원시 이벤트 위치와 추가 기록 시간이 포함된다. 준비만 다시 하려면 04를 사용한다. 02/03은 이전 A/B 연결 확인용 노트북이며 새 원시 기록 연결은 04/05에 있다.

## 저장 위치

매 실행에 새로운 `results/raw/<UTC시간>_<ID>/` 묶음을 만든다. 같은 노트북을 다시 실행해도 앞선 기록을 덮어쓰지 않는다. 품질·독립 검색 지연·전체 시스템이 서로 다른 `phase`를 갖고, 질문 ID·task·방법·반복 번호·scope ID로 연결된다. 개별 모델/HTTP 요청은 `call_id`로 요청과 응답을 연결한다.

| 파일 | 기록 |
|---|---|
| `manifest.json` | 노트북·실험 설정·가격 스냅샷·기록/시간/집계 규칙 |
| `events.jsonl` | 각 scope의 이벤트 파일을 찾는 전체 색인 |
| `api_calls.jsonl` | 실제 HTTP 요청 시도당 한 행: 질문·방법·단계·반복·요청/응답 위치·시간·usage·비용·오류 |
| `<phase>/<method>/<scope ID>/events.jsonl` | 시작·완료·실패, 모델 입출력, API 입출력, 캐시 사용, 평가 라벨·지표 |
| 같은 폴더의 `*_request.json` / `*_response.json` | 실제 HTTP 요청 본문과 디코딩된 응답 본문. SDK 변환 전 본문 |
| 같은 폴더의 `*_input.npz` / `*_output.npz` | 실제 로컬 모델/검색 입력·출력 숫자 배열. pickle 불필요 |
| 같은 폴더의 `retrieval.json` | 후보·점수·선택한 원문과 위치. 전체 시스템에서는 LLM 호출 전에 저장 |
| 같은 폴더의 `system.json` | 생성 성공 시 실제 프롬프트·원시 응답·정답 지표·usage·비용·시간 |
| `sources/` | 사용한 corpus·질문·평가 라벨·설정·코드·인덱스·C 트리와 적재 체크포인트의 해시 포함 복사본 |

`phase`는 `preparation`, `warmup`, `quality_candidates`, `quality`, `latency`, `system`이다. A/B 공통 인덱스 준비와 C/D 공유 트리 재사용은 `shared`, 공통 워밍업은 `AB`에 기록된다. 공유 트리 이벤트의 `shared_by_methods: ["C", "D"]`로 소유 방법을 구분한다. `phase`·`method`는 영문·숫자·밑줄·하이픈만 사용하는 저장 경로 이름이며, 화면 표시 이름 `C+D`를 넣으면 거부된다. 준비·워밍업·품질 실행을 온라인 시스템의 비용에 더하지 않는다.

각 결과의 `raw_log.events_path`와 실행 manifest의 `raw_log.session`에서도 해당 묶음으로 이동할 수 있다. 기존 `results/<런>/`, `results/partial/`, `results/c_requests/` 기록도 유지하며, D 실행은 `results/d_requests/`에 기록한다.

## 방법별로 남기는 내용

- **A**: 질문 원문, 실제 잘림/패딩 후 토큰 ID·attention mask 등 모델 입력, CLS 임베딩, corpus 전체 dot-product 점수와 상위 후보 인덱스, 선택 원문. 준비 시 문서 임베딩 배열과 순서·모델 파일 해시를 저장한다. 새 인덱스를 만들면 문서 배치 입력·출력도 저장하고, 기존 인덱스를 읽으면 캐시임을 기록한다.
- **B**: A의 기록과 질문·후보 원문 쌍, 실제 토큰 입력, 배치별 DeBERTa logits, 잘림 정보, 재정렬 결과.
- **C**: Choice 적재, Noul 분기 선택, Score 원문 평가, Noul 완결성의 실제 state/questions와 응답, 요청별 사용량·시간·비용. 성공 결과의 trace에는 방문 노드·평가한 원문·탈락/복구·중단 이유가 있다. 실패한 C 탐색의 부분 trace와 요청 journal도 현재 묶음에 복사한다.
- **D**: C와 동일한 tree traversal에 넣은 Laya local state/questions, raw answers/usage, actual input tokens, token budget과 truncation diagnostics, local inference time/device. 입력 잘림이 감지되면 결과를 정상 측정에서 거부하고 실패 요청을 journal에 보존한다. 이 D arm은 C가 미리 준비한 같은 트리를 공유한다.

수정된 D는 원래 요청 외에 `effective_inputs`의 대상별 state, 실제 input IDs·SHA-256·지시문/선택지/본문 잘림 진단을 남긴다. `native_responses`에는 대상별 SDK 응답을 보존하고 일시적 CPU fallback도 기록·거부한다. D의 성공/실패 journal은 해당 raw 묶음의 sources에도 복사한다.
- **공통 LLM**: 실제 SDK가 보내는 설정과 최종 근거를 포함한 요청 본문, 받은 원시 응답·API 사용량, 성공 시 프롬프트 및 파싱한 답변·인용. HTTP 오류 응답과 연결 실패도 저장한다.

기록 범위는 모델 경계의 입력·출력이다. 모든 hidden layer의 activation이나 서버 내부 계산을 덤프하지는 않는다. Jev 토크나이저는 공개되어 있지 않아 Jev의 토큰 수는 응답 usage만 사용한다. 바이트 수는 토큰 수가 아니다.

정답 라벨은 오프라인 평가용으로 따로 보관한다. 검색기와 생성기의 실제 요청에는 라벨을 넣지 않는다.

## 호출과 비용을 집계할 때

**실제 API 사용량의 출발점은 `api_calls.jsonl`이다.** 요청 한 번에 여러 Jev 질문이 있어도 HTTP 요청 행은 하나다. 응답 usage가 그 요청의 청구 기준이며 질문 수로 다시 곱하지 않는다. 반환 결과·trace·캐시·준비 checkpoint에 복사된 같은 응답을 또 합산하지 않는다.

`role`은 `placement`, `navigation`, `scoring`, `coverage`, `generation`이다. 준비 호출은 별도 phase이고, 캐시 재사용은 HTTP 호출을 만들지 않는다. C 품질 캐시의 `original_calls`·`original_api_usage`는 과거 실행 기록이며 이번 새 비용이 아니다. D는 local inference이므로 `api_calls.jsonl`에 없으며 D의 call count·tokens·seconds는 local journal/trace에서 읽는다.

HTTP 오류, 연결 실패, 잘못되거나 없는 usage는 비용을 **미확인(`null`)**으로 남긴다. 오류 응답에 정상 usage가 있다면 알려진 사용량은 보존한다. 알려진 비용만 더한 값은 총비용이 아니라 부분 합이다. 각 원시 본문에는 원본/저장본 SHA-256·바이트 수·비밀 치환 여부가 있어 대응을 확인할 수 있다.

API 키와 Authorization/Cookie 헤더는 저장하지 않는다. 키가 오류 본문·예외에 되돌아오면 알려진 키 문자열을 치환하고 치환 여부/해시를 남긴다. `.env`는 스냅샷 대상이 될 수 없다. 질문과 실제 문서 내용은 원시 로그에 포함된다.

## 실패와 시간 해석

요청 본문을 **전송 전에** 저장하고, 받은 응답 본문을 SDK 파싱 전에 저장한다. scope에는 오류 종류·비밀을 제거한 메시지/traceback을 남긴다. LLM 실패 전 완료한 검색 근거는 `retrieval.json`에 남는다. 응답 없이 실패한 시도도 ledger에 남고 비용은 미확인이다.

커널 강제 종료나 전원 차단은 Python 정리 코드를 실행하지 못할 수 있다. 이때 완료 행이 없는 `http_request` 또는 시작만 있는 scope가 미완료의 단서다. 응답·과금 여부를 만들어내지 않는다. 로그 쓰기 자체가 실패하면 실험도 중단한다.

`diagnostic_capture_seconds`는 **이번에 추가한 RawLog**의 배열 CPU 복사·파일 쓰기·해시·이벤트 처리 시간이다. 모델 compute 시간은 별도로 기록한다. 추가 CUDA 동기화나 기존 C journal 쓰기까지 모두 따로 측정한 값은 아니다. 기록 때문에 관측 시간과 처리량이 달라질 수 있고, 측정 구간 안의 기록은 전체 시간에 포함된다. 최종 결과/평가 저장은 타이머 밖이며, 검색과 생성 사이의 근거 보존은 전체 시스템 타이머 안이다.

기록 시간을 전체 시간에서 빼서 무기록 성능으로 추정하지 않는다. 무기록 실행은 `save_raw_logs: false`로 따로 수행하고 조건을 명시한다. 이 옵션은 문항 수·검색 설정·C 정책을 변경하지 않는다.

## 노트북에서 호출별 분석하기

실제 로그 폴더를 지정한 뒤 아래처럼 읽을 수 있다. 원시 파일을 수정할 필요는 없다.

```python
from pathlib import Path
import json
import pandas as pd

session = Path(r'D:\decision_LLM\results\raw\실제_폴더명')
calls = pd.read_json(session/'api_calls.jsonl', lines=True)
if not calls.empty:
    def token_count(usage, key):
        value = usage.get(key) if isinstance(usage, dict) else None
        return value if type(value) is int and value >= 0 else None
    for key in ('input_tokens', 'output_tokens'):
        calls[key] = calls.api_usage.map(lambda u: token_count(u, key))
    online = calls[calls.phase == 'system']
    display(online[['question_id', 'task', 'method', 'repeat', 'role',
                    'http_status', 'http_seconds', 'input_tokens',
                    'output_tokens', 'api_cost_usd']])
    display(online.groupby(['method', 'role']).agg(
        requests=('call_id', 'size'),
        seconds=('http_seconds', 'sum'),
        known_input_tokens=('input_tokens', lambda s: s.sum(min_count=1)),
        known_api_cost_usd=('api_cost_usd', lambda s: s.sum(min_count=1)),
        unknown_cost_requests=('api_cost_usd', lambda s: s.isna().sum()),
    ))
    first = online.iloc[0] if not online.empty else calls.iloc[0]
    sent = json.loads((session/first['request']['path']).read_bytes())
    print(sent)  # Jev: state/questions, LLM: input 등 실제 전달 본문
```

반복 전송 여부는 같은 **phase·question_id·method·repeat** 안에서 요청 body의 `source_sha256`, Jev `state`, 각 `questions`의 내용·원문 ID를 비교한다. 정답에 기여했는지는 trace의 방문/평가/선택 ID와 평가 라벨을 연결해서 확인한다. 질문·문서 수가 다른 요청을 토큰 수 하나로만 판단하지 않는다.

로컬 배열은 scope의 `model_request`/`model_response` 이벤트에 저장 위치·shape·dtype가 있다.

```python
import numpy as np
arrays = np.load(session/'실제_npz_상대경로', allow_pickle=False)
print(arrays.files)
# input_ids, attention_mask / cls_embeddings / all_scores, top_indices / logits
```

## 과거 두 문항은 어디까지 분석할 수 있나

기존 파일럿은 `results/20261004T012602_22979a7e`에 보존한다. 당시 C 요청/응답은 `trace.calls`와 연결된 `results/c_requests/`에 있고 LLM 프롬프트·원시 응답도 있다. 새 로컬 토큰 배열, 전송 전 HTTP 본문, 실패 LLM 기록은 과거에 저장하지 않았으므로 소급해서 만들지 않는다.

이 파일럿의 C 두 문항에서 Jev 입력 usage 합계는 342,699, 분기 판단 합계는 310,372(약 90.6%)였다. 원시 요청의 중복 state·분기 범위·근거 확보를 분석할 출발점이다. 이 비율만으로 반복이나 버그를 확정하지 않는다.
