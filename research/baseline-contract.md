# A/B/C/D 공통 규격

2026-10-03 구현 기준, D arm은 2026-10-04에 추가. 기존 관련 연구 문서의 C Choice/tournament 설계는 과거 제안이다. 현재 C는 Jev, D는 local Laya를 사용해 같은 자연어 설명 트리의 분기를 독립 판단하고 여러 경로의 원문 근거를 모은다. D 비교는 Jev로 준비한 공통 트리를 공유하므로 별도 D 자료 적재/분류 성능을 검증하지 않는다. C 정책은 [c-method.md](c-method.md)를 따른다.

## 입력과 저장소

`DocumentStore(corpus_path)`는 고정 corpus의 원문 passage·ID·문단/문자 좌표와 SHA-256을 제공한다. A/B/C/D는 동일 파일을 사용한다. C/D는 SHA-256이 같은 준비 트리도 공유한다. 공개 질문셋의 `id/question/task`만 온라인 입력에 사용하며 별도의 `labels_*.jsonl`은 평가자가 읽는다.

`RetrievalAdapter.retrieve(question: str, method: str) -> dict`가 반환 경계다. C는 이 인터페이스와 동일한 원문 저장소를 사용한다. C의 트리·판단 정책·노드 방문은 adapter 내부 구현으로 두고 별도 trace에 기록한다. 공통 평가기·프롬프트는 원문 근거를 소비한다.

## 최소 반환 규격

| 필드 | 의미 |
|---|---|
| `method`, `question` | A/B/C/D, 원래 질문 |
| `corpus_sha256`, `corpus_count` | 실제 접근 가능한 공동 자료 집합 |
| `evidence` | 선택한 원문 passage: `id/content/title/wikipedia_id/source_spans` |
| `candidates` | 최종 선택 전 접근한 passage ID와 점수. A/B는 동일 dense 후보 |
| `seconds.retrieval_total` | 실제 온라인 전체 검색 시간. 재사용한 품질 결과는 `None` |
| `model_tokens` | 구성 요소별 nonpad/padded/original/dropped 토큰·잘림 수. 미측정은 `None` |
| `api_usage`, `api_cost_usd` | 해당 adapter API의 원시 usage/추정 비용. 미측정은 `None` |
| `execution_mode` | `online_retrieval` 또는 `cached_quality` |
| `trace` (선택) | C/D의 방문 분기·판단·중단 조건 등 |

`store.validate_evidence(result)`는 원문 내용·제목·원본 문서 ID·위치·corpus 해시·중복 ID를 검사한다. 요약문이나 추측으로 원문을 바꿔 반환하면 실패한다. C는 선택 개수를 고정하지 않으며 같은 입력 예산 안에서 빈 근거 또는 복수 근거를 반환한다. A/B 초기 50/5와 C의 탐색 정책은 각각 기록한다.

## 공유하는 평가와 생성

`retrieval_metrics(result, label)`은 한 유효 annotation의 모든 근거가 선택되었는지를 계산한다. HotpotQA의 두 문서 근거는 하나의 joint group이며 둘 중 하나만 확보하면 불완전하다. NQ short-answer span 기준의 성공과 답변 사실성을 혼동하지 않는다. `complete_candidate_hit`은 passage 후보가 기록되는 경로에서만 해석한다.

`generate_answer(result, client=..., model='gpt-6-luna', ...)`는 방법과 무관하게 같은 프롬프트/예산을 사용한다. `None`인 검색 시간을 가진 품질 결과에도 사용할 수 있지만 전체 지연으로 계산하지 않는다. `measure_system`은 실제 검색 직전부터 API 응답까지 벽시계 시간을 남긴다. `ComparativeRetrieval`은 A/B/C/D를 연결하며 05에서 네 방법을 전달한다. 기존 02/03과 함수 기본 방법 목록은 A/B로 유지한다. C 처리 토큰은 `model_tokens.jev`와 `api_usage`, D의 local tokens·잘림·call timing은 `model_tokens.laya`와 `local_usage`·`trace.calls`에 저장한다. D의 외부 retrieval API cost는 0이며 local GPU 비용은 별도 미측정이다.

`evaluate_comparative_quality`는 A/B의 같은 dense 후보와 C/D의 별도 판단 캐시를 평가한다. `measure_retrieval`과 `measure_system`은 캐시를 재생하지 않는다. `on_result` 콜백으로 완료한 질문별 결과를 즉시 보존한다. 시스템 결과의 `retrieval_api_cost_usd`, `generation_api_cost_usd`, `online_api_cost_usd`는 실제 외부 API 요금이며, local GPU 비용은 포함하지 않는다. 알려지지 않은 API 사용량은 비용을 `None`으로 남긴다.

모든 결과에 실행 조건·원문·usage를 보존한다. C의 전처리 LLM, 트리 설명 생성, Jev 판단 호출 비용도 공통 문서 준비와 별도로 기록해야 한다. 동일 상용 생성기를 쓴다는 이유만으로 준비 비용과 검색 호출 비용이 같아지는 것은 아니다.

2026-10-04 비용 경계 검증: 입력·출력 usage가 비정상 타입·음수이거나 캐시 정보가 불명확하면 비용은 `None`이다. 가격은 유한한 0 이상 숫자여야 하며 실제 요청 전에 검사한다. 알려진 일부 Jev 사용량은 보존하되 한 호출이라도 불명확하면 합계 비용을 확정하지 않는다. `load_comparison`은 과거 `live_system_abc` 및 신규 `live_system_abcd` 원시 결과에서 원문·프롬프트·지표·usage·비용을 재계산해 저장된 값과 대조한다. D는 local input/truncation usage도 원시 call 결과와 비교한다. 06의 시각화는 검증을 통과한 동일 런만 사용한다.
