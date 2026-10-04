# 2026-10-04 · 05 노트북 커널 충돌 진단

`05_measure_abc.ipynb`와 `05_measure_abcd.ipynb`의 초기 설정 셀에서 발생하는 Windows Jupyter 커널 충돌을 재현했다. Pandas/PyArrow를 프로젝트 모듈보다 먼저 불러오도록 순서를 변경하면 이 환경에서 충돌이 사라진다. 모델·검색·탐색·생성·측정 설정은 바꾸지 않았다.

## 확인한 증거

- Windows Application 이벤트 1000: `python.exe`, faulting module `venv/Lib/site-packages/pyarrow/arrow.dll`, exception `c0000005`.
- Jupyter 로그: kernel process died, exit code `3221225477` (`0xC0000005`).
- `-X faulthandler`로 시작한 실제 커널: `pandas.compat.pyarrow` → `pyarrow.__init__` → 네이티브 모듈 로딩에서 access violation.
- Python 3.14.6, PyTorch 2.11.0+cu128, PyArrow 24.0.0, Pandas 3.0.6, ipykernel 7.4.0.

| 동일 환경에서 비교한 입력 | 결과 |
|---|---|
| 실제 Jupyter: PyArrow만 import하고 작은 표 생성 | 통과 |
| 실제 Jupyter: PyTorch만 import하고 작은 텐서 계산 | 통과 |
| 실제 Jupyter: PyTorch → PyArrow | access violation 재현 |
| 실제 Jupyter: PyArrow → PyTorch, 표·텐서 계산 | 통과 |
| 일반 CLI: PyTorch → PyArrow | 통과 |
| 수정 전 ABC·ABCD 초기 설정 셀 | 둘 다 실패 |
| 초기 설정에 Pandas 선행 import 추가 | 통과 |

진행 표시줄의 `IProgress not found` 경고는 성공한 재현에서도 나타났다. 따라서 이 경고를 현재 커널 충돌의 원인으로 판단하지 않는다. 단일 라이브러리의 손상이나 모델 가중치 문제로도 확정하지 않는다. 확인한 것은 **Windows Jupyter에서 특정 로딩 순서가 충돌을 유발하고, 순서 변경이 이를 회피한다**는 사실이다. 내부 DLL의 구체적인 결함까지 추적한 것은 아니다.

## 적용한 변경

두 05 노트북에서 기존 `import pandas as pd`를 PyTorch를 불러오는 프로젝트 모듈 reload 앞에 배치했다. 변경한 설정 셀의 과거 출력만 비웠고, 나머지 셀·출력과 기존 결과 파일은 보존했다. 패키지 버전을 변경하거나 가중치를 재다운로드하지 않았다.

D 실행에 필요한 `laya==0.3.26`이 설치되지 않은 것도 확인해 해당 패키지만 `--no-deps`로 설치했다. 기존 requirements와 lock에 이미 지정된 버전이다. 라이브러리 설치는 실제 D 모델 추론의 검증을 의미하지 않는다.

## 회귀 검사

```powershell
.\venv\Scripts\python.exe -X faulthandler -m pytest tests/test_notebook_imports.py -q
```

기존 `exec` 기반 검사는 수정 전에도 통과해 네이티브 커널 충돌을 발견하지 못했다. 실제 Jupyter 커널을 시작하는 회귀 검사를 추가했고, 수정 전 두 노트북에서 실패하는 것을 확인한 다음 수정을 적용했다.

새 검사는 각 노트북의 설정 셀을 같은 커널에서 두 번 실행하고, PyArrow 표·Pandas DataFrame·PyTorch 텐서 연산과 커널 생존을 확인한다. 네이티브 오류와 Python 오류를 모두 검사한다. API 키를 비우고 Hugging Face를 offline으로 설정하며, 임시 IPython 설정·히스토리 디렉터리를 사용한다. 모델 로딩·벤치마크·Jev/GPT API 요청은 실행하지 않는다.

실제 실험은 커널을 재시작한 뒤 05를 처음부터 실행한다. 유료 API와 전체 A/B/C/D 비교는 사용자가 별도로 실행해야 한다.

## 전체 검사에서 추가로 발견한 호출 상한 문제

최초 전체 검사에서는 47개가 통과하고 `test_c_call_budget_returns_partial_evidence_without_overrunning` 하나가 실패했다. 엔진 생성 이후 `engine.config['search']['max_api_calls']`를 1로 낮췄지만, 생성 시 캐시한 32라는 값이 남아 모의 요청이 7번 수행됐다. 설정 파일의 상한을 엔진 생성 전에 지정하는 경우에는 이 재현 조건이 적용되지 않는다.

C/D가 공유하는 탐색기에서 중복 캐시를 제거하고, 검색 시작 시 현재 정책의 상한을 읽어 검증·고정하도록 수정했다. C는 `max_api_calls`, D는 `max_decision_calls`를 사용한다. 기본 설정 값은 바꾸지 않았다. 실행 중 한 검색에 사용할 상한은 고정하고 다음 검색에 수정한 정책을 적용한다.

기존 C 상한 테스트, 잘못된 런타임 상한(0, 음수, bool, 실수)을 호출 전에 거부하는 테스트, D의 변경한 로컬 상한을 지키는 테스트를 실행했다. 수정 전 6개 실패, 수정 후 6개 통과를 확인했다. 모두 모의 클라이언트를 사용하므로 유료 API 요청과 실제 Laya 추론은 없다.

## 최종 확인

- `python -X faulthandler -m pytest tests -q --tb=short`: **53 passed in 21.38s**.
- 모든 7개 노트북: nbformat 형식 검증과 코드 셀 문법 검사 통과.
- `python -m pip check`: `No broken requirements found`.
- `laya` import, 버전 0.3.26, `load` 함수 존재 확인. 실제 가중치 로딩은 실행하지 않았다.
- 기존 첫 셀을 실제 커널에서 실행한 원래 재현도 수정 후 통과했다. 임시 재현 스크립트는 제거하고 실제 커널 회귀 테스트를 남겼다.
- Jev Review를 두 번 수행해 변경을 보조 검토했다. 점수로 완료를 판단하지 않고 위 재현·테스트 결과를 기준으로 완료했다.

`IProgress not found` 경고는 남아 있다. 성공한 실제 커널 실행에서도 관측됐으며 진행 표시줄 위젯 문제로 구분했다.

## 추가 패치 · ABCD 공유 트리 로그 초기화

이후 실제 실행의 초기화 셀에서 `Raw-log phase/method must be simple directory labels` 오류가 발생했다. 로그 경로 이름으로 쓴 `method='C+D'`가 기존 RawLog의 영문·숫자·밑줄·하이픈 규약을 위반했다. 커널 충돌과는 다른 오류이며, 모델 로딩 전에 중단됐다.

노트북의 준비 로그 경로만 `method='shared'`로 수정했다. 화면에 보이는 C/D 표기와 `shared_by_methods: ['C', 'D']` 메타데이터는 유지했다. 경로 검증 규칙은 완화하지 않았고 기존 로그·결과도 보존했다.

`tests/test_notebook_initialization.py`는 실제 노트북에서 준비 scope 문을 읽어 real RawLog로 실행한다. 수정 전 같은 ValueError로 실패, 수정 후 통과를 확인했다. C/D 공유 메타데이터, 단일 준비 scope, 4개 준비 파일 스냅샷, 완료 이벤트, API 청구 ledger가 비어 있는 것을 검증한다. 전체 모델·검색·유료 API 실행을 검증하는 테스트는 아니다.

추가 패치 후 전체 테스트는 **54 passed in 18.27s**를 기록했다. 7개 노트북의 nbformat 형식·코드 문법·정적 로그 경로 이름 검사도 통과했다. 유료 API와 실제 모델 추론은 실행하지 않았다.

## 추가 패치 · Laya 실제 로더의 backend 모듈 누락

ABCD 초기화 셀의 `LayaDecisionClient`에서 `ModuleNotFoundError: No module named 'laya.backends'`가 발생했다. 설치된 0.3.26뿐 아니라 [공식 PyPI 0.3.26 wheel](https://pypi.org/project/laya/0.3.26/)의 ZIP 파일 목록도 확인했고 backend 모듈이 없는 것을 확인했다. 실제 `laya.load(..., backend='eager')`로 같은 오류를 두 번 재현했다. 동일 버전 재설치로 해결할 수 없는 배포 파일 누락이다.

모델 SHA와 라이브러리 버전은 유지하고, 공개 로더의 기존 `fast=False, compile=False` 인자를 사용한다. 이 경로의 기본 backend는 eager이며 로딩 후 `agent.backend`가 설정과 다르면 오류를 낸다. Runtime 로그에도 실제 backend를 기록한다. 환경이나 site-packages를 직접 패치하지 않았다.

이전 어댑터 단위 테스트는 `laya.load` 자체를 대체해서 이 오류를 잡지 못했다. 새 회귀 검사는 설치된 실제 `load`와 `Agent.__init__`를 실행하고 체크포인트 취득 직전에만 중단한다. 수정 전 사용자의 missing-module 오류로 실패했고, 수정 후 고정 repo/SHA의 체크포인트 취득 단계에 도달했다. 예상과 다른 backend 거부도 검증한다.

실제 GPU 검증도 수행했다. 고정 SHA `1720e3e3357cfe1e281542e223f8273b0890ca34`로 가중치를 로드하고 eager/CUDA/FP16, warmup, 짧은 `noul`과 `score` 응답을 확인했다. CLI 검증의 usage는 input 155, output 0, state 누락 0, 잘린 질문 없음이었다. 최초 로딩에는 Hugging Face 캐시 준비가 포함됐으므로 그 시간을 검색 성능 수치로 사용하지 않는다. `model/laya-multilingual`의 원본 8개 파일은 다운로드 manifest의 SHA-256과 모두 일치했다.

`tests/test_notebook_imports.py`의 opt-in GPU 검사는 실제 ABCD 설정 셀을 Jupyter 커널에서 실행하고 실제 Laya 모델을 로드한다. GPU·backend·SHA·FP16과 `noul`/`score` 반환을 검사하며, 작은 입력 예산에서 실제 tokenizer가 잘라낸 입력을 어댑터가 거부하는 것도 확인한다. API 키를 비우고 Hub/Transformers offline 모드로 실행한다. 검사 후 커널과 임시 파일을 정리한다.

- `DECISION_LLM_TEST_LAYA_RUNTIME=1` 환경에서 전체 테스트: **57 passed in 21.49s**.
- 실제 Jupyter/Laya GPU 검사 단독 실행: **1 passed in 7.17s**.
- 7개 노트북 형식·문법 검사, `pip check` 통과.

위 검사는 로딩·응답 규격·잘림 검증이다. 판단 정확도나 A/B/C/D 비교 결과가 아니며 유료 비교 실험은 실행하지 않았다.

## 추가 패치 · 06 비교 노트북의 구버전 로더 캐시

06에서 최신 완료 결과 `20261004T052947_1e5333aa`를 선택했지만 `Choose a completed live_system_abc run` 오류가 발생했다. 이 결과는 `live_system_abcd`이며 40문항 × A/B/C/D × 1회 = 160개 응답으로 정상 저장돼 있다. 현재 비교 모듈은 같은 결과를 두 번 읽어 원문·라벨·프롬프트·usage·비용·균형 검증을 통과했다.

화면의 오류 문구는 원시 로그에 남은 과거 A/B/C 전용 `comparison.py`와 일치했다. 이 과거 소스를 현재 모듈의 메모리 상태에 재생한 뒤 원래 06 설정 셀을 실행하니 같은 오류가 두 번 재현됐다. 원인은 커널에 남아 있던 구버전 `load_comparison`이며 결과 손상이나 미완료 실행이 아니다.

06의 첫 셀이 `rag_baselines.comparison`을 reload한 뒤 함수를 다시 가져오도록 수정하고 선택한 원본 폴더·mode·방법을 출력한다. 비교 모듈의 검증 규칙, 결과 폴더 선택 기준, 계산식, 그림 코드는 변경하지 않았다. 회귀 검사는 구버전 로더를 캐시한 상태에서 실제 노트북 설정 셀을 실행하고 A/B/C/D 자료를 읽는다. 더 최신인 품질 캐시가 있어도 완료된 시스템 결과를 선택하고, 품질 캐시를 직접 넘기면 계속 거부하는 것도 확인했다. 수정 전 같은 오류로 실패, 수정 후 통과했다.

수정 후 과거 소스 재생도 두 번 모두 160개 응답을 읽는 데 성공했다. API 키를 비우고 실제 Jupyter 커널에서 06을 처음부터 끝까지 실행해 출력도 저장했다. 비교 결과는 `results/comparison_20261004T053505_4fe4688b/`에 저장했고, manifest에 기록된 8개 CSV·PNG·SVG 출력의 SHA-256을 확인했다. 두 PNG를 열어 40문항·네 방법·단위·라벨과 그래프 표시를 확인했다. 기존 실험 원본은 유지했으며 유료 API 호출이나 새 실험은 수행하지 않았다.

- 비교 관련 테스트: **8 passed in 3.52s**.
- 전체 테스트: **57 passed, 1 skipped in 25.50s**. 별도 opt-in Laya GPU 검사는 이번 변경과 무관해 실행하지 않았다.
- 06 형식·코드 문법·전체 셀 실행·그림 검사 통과.
