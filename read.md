# LLM-based BERTopic: 논문 요약과 실행 방법

**Chibok Yang · Yangsok Kim (2025)**<br>
*Enhancing topic coherence and diversity in document embeddings using LLMs: A focus on BERTopic*<br>
Expert Systems with Applications, 281, 127517.<br>
[논문 DOI](https://doi.org/10.1016/j.eswa.2025.127517)

## 1. 연구 개요

이 연구는 **LLM으로 생성한 문서 임베딩이 BERTopic의 토픽 품질을 개선하는지**, 그리고 **텍스트 전처리 수준이 그 성능에 어떤 영향을 주는지** 비교한다. S-BERT, DistilBERT, Falcon, LLaMA2, LLaMA3의 임베딩으로 BERTopic을 구성하고 LDA를 비교 모델로 사용했다.

| 데이터셋 | 도메인 | 논문에 제시된 문서 수 |
|---|---|---:|
| 20 Newsgroups | 뉴스그룹 게시물, 20개 주제 | 11,314 |
| BBC News | 뉴스 기사, 5개 분야 | 2,225 |
| IMDB | 영화 리뷰 | 50,000 |

논문의 BERTopic 흐름은 **전처리 → 문서 임베딩 → UMAP 차원 축소 → HDBSCAN 군집화 → CountVectorizer / c-TF-IDF 토픽 표현 → 평가**다. LDA는 같은 전처리 문서로 단어 사전과 BoW를 만들어 학습한다. 토픽 수는 5부터 50까지 5씩 증가시키며 비교했다.

평가 지표는 토픽 단어 간의 일관성을 보는 **NPMI**, 토픽 단어 목록의 중복을 평가하는 **IRBO**, 균등 분포에서 벗어난 정도를 보는 **KL-Uniform**이다. 논문에서는 세 지표 모두 높은 값을 선호하되 서로 다른 품질 측면을 측정하는 것으로 해석한다.

## 2. 주요 연구 결과

- **LLaMA3는 NG20과 IMDB에서 가장 높은 평균 NPMI**를 기록했다. 논문 Table 5의 값은 각각 0.0872, 0.0869다. BBC에서는 Falcon이 0.1188로 가장 높았다.
- **모든 지표와 데이터셋에서 동시에 최고인 모델은 없었다.** LDA는 세 데이터셋에서 가장 높은 KL-Uniform을, S-BERT는 BBC에서 가장 높은 IRBO를 기록했다.
- **전처리는 LLM 임베딩에서도 중요했다.** LLaMA3는 전처리 단계를 추가하면서 세 데이터셋의 평균 NPMI가 개선되었다. 반면 IMDB의 S-BERT와 LLaMA2는 표제어 추출을 하지 않은 Level 2에서 더 높은 NPMI를 보였다.
- LLM은 임베딩 생성에 더 많은 시간과 GPU 메모리가 필요하다. 생성한 임베딩을 저장하면 이후 하이퍼파라미터 실험에서 재사용할 수 있지만, 토픽 모델 학습과 평가에는 별도의 계산이 필요하다.

논문은 UMAP과 HDBSCAN을 사용한 비교 연구다. 모든 모델에 대한 하이퍼파라미터 최적화나 다른 차원 축소·군집화 방법까지 평가한 결과는 아니다. 위 수치는 출판 논문에 보고된 결과이며 이 공개본을 새 환경에서 실행한 값이 항상 같다는 의미는 아니다.

## 3. 전처리: full과 minimal 두 종류

논문 Fig. 2의 세 단계를 다음 설정으로 선택한다. **‘불용어 유지’는 문서에 the, is, and 등을 남긴다는 뜻이고, ‘불용어 제거’는 이 단어들을 임베딩 생성 전에 제거한다는 뜻이다.**

| 논문 단계 | 설정 파일 | 임베딩/LDA 입력의 불용어 | 표제어 추출 | CountVectorizer 불용어 설정 |
|---|---|---|---|---|
| Level 1: 기본 정리 | `configs/minimal_keep_stopwords.yaml` | 유지 | 없음 | `null` |
| Level 2: 기본 정리 + 불용어 제거 | `configs/minimal_remove_stopwords.yaml` | 제거 | 없음 | `english` |
| Level 3: 전체 전처리 | `configs/full.yaml` | 제거 | 적용 | `english` |

두 minimal은 구두점·이모티콘 제거, 소문자 변환, 공백 정리를 수행하고 숫자와 단어 형태를 유지한다. Level 2는 원본 minimal 코드처럼 NLTK로 토큰화하고 scikit-learn의 `ENGLISH_STOP_WORDS`를 제거한다. 전처리 결과는 모든 임베딩 모델과 LDA에 동일하게 적용한다.

full은 원본 `TextPreprocessor`의 처리를 유지한다. 숫자·기호 제거, NLTK 영어 불용어 제거 후 spaCy로 표제어를 추출하며 NOUN/ADJ/VERB/ADV 품사만 남긴다. 따라서 full은 Level 2에 단순히 표제어 추출 한 줄만 추가한 구현은 아니다. 빈 문서는 전처리 후 제외하고 남은 문서의 순서를 유지한다.

예를 들어 `The bright stars are shining!`은 minimal 불용어 유지에서 `the bright stars are shining`, 불용어 제거에서 `bright stars shining`이 된다.

**문서 전처리와 토픽 단어 추출의 불용어 설정은 별개다.** 원본 일부 BERTopic 셀은 문서/임베딩을 그대로 두고 `CountVectorizer(stop_words='english')`만 변경했다. 그 방식을 실행하려면 다음처럼 지정한다.

```bash
python -m llm_bertopic all --config configs/minimal_keep_stopwords.yaml \
  --set 'experiment.datasets=[bbc]' --set 'experiment.models=[sbert]' \
  --set 'experiment.topic_counts=[5]' --set 'experiment.topk_values=[10]' \
  --set 'datasets.bbc.max_documents=300' --set 'vectorizer.stop_words=english'
```

위 명령은 임베딩 입력의 불용어를 유지하므로 Level 2와 다르다. 기본 세 설정은 이 차이가 섞이지 않도록 구성했다. 전처리 모드를 바꾸면 문서와 임베딩 캐시도 분리된다. full 임베딩을 minimal 입력에 재사용하지 않는다.

## 4. 설치하고 작은 실험 실행

일반 실행은 **Python 3.12** 환경을 사용한다. 압축을 푼 `llm-based-bertopic` 폴더에서 실행한다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[standard,notebooks]'
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt punkt_tab

python -m llm_bertopic all --config configs/quickstart.yaml
```

Windows에서는 `.venv\Scripts\activate`로 활성화한다. 최초 실행에는 데이터셋과 모델 다운로드가 필요하다. Quickstart는 BBC 300개 문서에서 S-BERT BERTopic과 LDA를 각각 한 조건으로 실행한다. 출력 경로는 터미널에 표시되며, 계산 결과는 `runs/<실행 ID>/results.csv`에 저장된다.

`constraints-tested.txt`는 기존 일반 환경에서 사용한 직접 의존성 버전 목록이다. 설치 명령에 `-c constraints-tested.txt`를 추가할 수 있으며, 전체 하위 의존성 lockfile은 아니다.

## 5. 원하는 전처리·데이터·모델 선택

예를 들어 BBC의 S-BERT와 LDA를 **불용어를 제거하는 minimal**로 실행한다.

```bash
python -m llm_bertopic all --config configs/minimal_remove_stopwords.yaml \
  --set 'experiment.datasets=[bbc]' \
  --set 'experiment.models=[sbert,lda]' \
  --set 'experiment.topic_counts=[5]' \
  --set 'experiment.topk_values=[10]' \
  --set 'datasets.bbc.max_documents=300'
```

불용어를 유지하려면 설정 파일을 `configs/minimal_keep_stopwords.yaml`, 전체 전처리는 `configs/full.yaml`로 바꾼다. 전체 데이터를 사용하려면 `max_documents` 옵션을 생략한다. 데이터 키는 `newsgroup20`, `bbc`, `imdb`, 모델 키는 `sbert`, `distilbert`, `falcon`, `llama2`, `llama3`, `lda`다.

```bash
# 데이터셋 하나를 전체 전처리로 실행
python -m llm_bertopic all --config configs/newsgroup20.yaml
python -m llm_bertopic all --config configs/bbc.yaml
python -m llm_bertopic all --config configs/imdb.yaml

# 세 데이터셋·여섯 모델의 전체 설정: LLM 다운로드와 충분한 메모리 필요
python -m llm_bertopic all --config configs/full.yaml
```

전체 설정은 각 전처리 수준에서 480회 학습(BERTopic 450회 + LDA 30회)이다. 같은 전체 실험을 minimal로 실행하려면 `--config`만 해당 minimal 파일로 바꾼다.

일반 데이터 로더는 NG20의 train, BBC의 train 다음 test, IMDB의 unsupervised 분할을 사용한다. IMDB의 원본 BERTopic 노트북은 이미 전처리된 50,000개 문서를 읽었으므로, 이 일반 로더로 새로 처리한 데이터와 원본 배열의 대응이 자동으로 보장되지는 않는다. 원본 임베딩이 있으면 9절의 저장 입력 경로를 사용한다.

## 6. 하이퍼파라미터 변경

YAML을 직접 편집하거나 `--set '경로=값'`을 반복한다. 기본 모델링 설정은 다음과 같다. 이 표는 보존한 코드 설정이며 논문의 Table 4를 그대로 전사한 표는 아니다. BBC의 원본 실행 코드에는 `min_cluster_size=10`이 들어 있다.

| 항목 | NG20 | BBC | IMDB |
|---|---|---|---|
| UMAP n_neighbors | 5 | 5 | 10 |
| UMAP n_components / min_dist / metric | 1 / 0 / cosine | 동일 | 동일 |
| HDBSCAN min_cluster_size | 20 | 10 | 15 |
| HDBSCAN metric / selection | euclidean / eom | 동일 | 동일 |
| CountVectorizer ngram_range / min_df | [1,2] / 2 | 동일 | 동일 |

설정 적용 순서는 **공통 설정 → dataset_overrides → model_overrides**다. 예를 들어 BBC Falcon의 이웃 수는 모델별 설정이 있으므로 가장 구체적인 경로를 바꾼다.

```bash
python -m llm_bertopic all --config configs/bbc.yaml \
  --set 'experiment.models=[falcon]' \
  --set 'experiment.topic_counts=[10,20]' \
  --set 'experiment.topk_values=[10]' \
  --set 'model_overrides.bbc.falcon.umap.n_neighbors=15' \
  --set 'dataset_overrides.bbc.hdbscan.min_cluster_size=20'
```

`seed`, `umap.random_state`, `lda.random_state`는 별도 설정이다. 일반 IMDB의 UMAP 시드는 42(S-BERT는 1042), 원본 저장 입력용 IMDB 설정은 1042다. 실험 조건에 맞는 설정 파일을 선택한다. `nr_topics`는 요청값이며 결과의 `actual_topics`는 노이즈를 제외한 실제 토픽 수다.

## 7. LLM 임베딩을 따로 생성

NVIDIA GPU에서 8bit 로딩을 사용하려면 해당 환경의 CUDA PyTorch를 준비하고 `python -m pip install -e '.[gpu]'`를 추가한다. LLaMA 모델 접근 권한이 필요한 경우 Hugging Face 인증을 먼저 완료한다. 토큰은 코드나 노트북에 저장하지 않는다.

```bash
# BBC의 Falcon/LLaMA2/LLaMA3 임베딩 생성
python -m llm_bertopic embed --config configs/cuda_8bit.yaml \
  --set 'experiment.datasets=[bbc]' \
  --set 'experiment.models=[falcon,llama2,llama3]'

# 같은 설정으로 저장된 임베딩을 읽어 학습
python -m llm_bertopic run --config configs/cuda_8bit.yaml \
  --set 'experiment.datasets=[bbc]' \
  --set 'experiment.models=[falcon,llama2,llama3]'
```

GPU 설정에서 Level 1을 실행하려면 두 명령 모두에 `--set 'preprocessing.mode=minimal_keep_stopwords' --set 'vectorizer.stop_words=null'`을 추가한다. Level 2는 두 명령 모두에 `--set 'preprocessing.mode=minimal_remove_stopwords' --set 'vectorizer.stop_words=english'`를 추가한다. 문서·임베딩 설정을 동일하게 사용해야 캐시를 재사용한다.

LLM 기본 pooling은 원본처럼 패딩을 포함하는 `legacy_mean`이다. 배치 크기, dtype, 모델 revision, 장치 변경은 임베딩을 바꿀 수 있다. `run`은 임베딩이 없으면 오류로 종료하고, `all`은 없는 임베딩을 생성한다.

Jupyter 사용자는 `jupyter lab`을 실행한다. `01_prepare_data.ipynb`, 모델별 `02a_falcon_embeddings.ipynb` / `02b_llama2_embeddings.ipynb` / `02c_llama3_embeddings.ipynb`, `03_run_experiments.ipynb`가 같은 패키지를 사용한다. 임베딩 노트북의 `PREPROCESSING` 값으로 세 전처리 중 하나를 선택할 수 있다.

## 8. 임베딩 생성·JSON·사용자 지정 파일명

**임베딩 배열은 이 저장소에 포함되지 않는다. 먼저 직접 생성하거나 이미 생성한 배열을 가져와야 한다.** 아래 JSON은 원본 실험 입력의 파일명, shape, dtype, 해시를 기록한 메타데이터다. 임베딩 값이나 문서 본문이 들어 있는 파일은 아니다.

| 데이터셋 | 원본 입력 명세 |
|---|---|
| NG20 | [ng20_2024.json](src/llm_bertopic/ng20_2024.json) |
| BBC | [bbc_2024.json](src/llm_bertopic/bbc_2024.json) |
| IMDB | [imdb_2024.json](src/llm_bertopic/imdb_2024.json) |

NG 정보는 기존 `notebook_2024.json`에서 `ng20_2024.json`으로 분리했다. [notebook_2024.json](src/llm_bertopic/notebook_2024.json)은 세 데이터셋이 공유하는 연구 당시 환경 정보만 보관한다.

**JSON의 `filename`과 `source_path`는 연구자가 당시 실험에서 직접 정한 이름이다. 새 임베딩 파일이 반드시 이 이름을 따라야 하는 것은 아니다.** `filename`은 `--embedding-dir` 안에서 찾는 파일명이고, `source_path`는 `--source-dir` 원본 폴더를 기준으로 한 상대 경로다. 파일명만 같게 바꾸어도 원본 임베딩과 동일해지는 것은 아니며, 입력 값·순서·dtype와 해시를 별도로 확인한다.

### 8.1 새 임베딩 생성

예를 들어 NG20의 S-BERT 임베딩을 생성한 후 한 조건을 학습한다.

```bash
python -m llm_bertopic embed --config configs/newsgroup20.yaml \
  --set 'experiment.models=[sbert]'

python -m llm_bertopic run --config configs/newsgroup20.yaml \
  --set 'experiment.models=[sbert]' \
  --set 'experiment.topic_counts=[5]' --set 'experiment.topk_values=[10]'
```

BBC/IMDB는 해당 데이터셋 설정 파일을 사용한다. LLM은 모델 키와 장치 설정을 바꾸거나 7절의 CUDA/노트북 경로를 사용한다. 새 배열은 `artifacts/embeddings/<dataset>/<model>/<cache-id>/embeddings.npy`에, 생성 정보는 같은 폴더의 `metadata.json`에 자동 저장된다. 생성과 학습에서 같은 전처리·임베딩 설정을 사용하면 `run`이 해당 파일을 찾는다. `all`은 생성과 학습을 한 번에 수행한다.

현재 새 생성 경로는 **float32**로 저장한다. 원본 LLM 임베딩은 float16이었으므로, 새 배열을 만들어 원본 파일명만 붙인 뒤 `run-saved`에 넣지 않는다. 새 임베딩은 일반 `embed`/`run` 경로로 사용하며, 원본 재실행용 해시는 원본 배열에만 해당한다. 패키지가 관리하는 캐시 안의 파일명은 변경하지 않는다.

### 8.2 내가 정한 다른 파일명으로 실행

외부에서 생성한 파일은 `inputs/my_ng_llama3.npy`처럼 자유롭게 이름을 정할 수 있다. 이 배열을 만들 때 사용한 **전처리 완료 문서의 정확한 순서**를 `inputs/my_ng_processed_texts.json`처럼 문자열의 JSON 목록으로 함께 준비한다. 원문 목록이나 순서가 다른 문서 목록을 제공하면 가져오기를 거부한다.

[configs/custom_embeddings.yaml](configs/custom_embeddings.yaml)은 NG20/LLaMA3, full 전처리, 한 조건, 별도 가져오기 캐시를 선택한 예시다.

```bash
python -m llm_bertopic import-embeddings --config configs/custom_embeddings.yaml \
  --dataset newsgroup20 --model llama3 \
  --array 'inputs/my_ng_llama3.npy' \
  --texts 'inputs/my_ng_processed_texts.json'

python -m llm_bertopic run --config configs/custom_embeddings.yaml
```

두 경로를 자신이 정한 실제 파일명으로 바꾼다. 신뢰하는 `.pkl`/`.pickle`이면 `--trust-pickle`을 추가한다. 가져오기는 문서 내용·순서와 배열 shape를 확인하고 dtype를 유지한 채 관리용 캐시에 복사한다. 이미 가져온 캐시가 있으면 설정의 `paths.artifact_dir`를 사용하지 않은 새 폴더로 바꾼다. `llama3`는 파일명이 아니라 임베딩 모델 종류를 지정하는 키다.

다른 데이터셋은 설정의 `experiment.datasets`와 명령의 `--dataset`을 함께 바꾼다. 다른 모델은 `experiment.models`와 `--model`을 함께 바꾼다. minimal로 생성한 배열은 `extends`를 해당 minimal 설정 파일로 바꾼다. 생성·가져오기·학습의 전처리와 모델 설정이 일치해야 한다.

**원본 배열의 이름만 바꾼 경우:** 연구 당시 Python 3.8.19 환경을 사용하고, 예시 설정의 `extends: full.yaml`을 `extends: saved_newsgroup20.yaml`로 바꾼 뒤 위의 `import-embeddings` → `run` 명령을 사용한다. BBC/IMDB는 해당 `saved_*.yaml`과 데이터셋 선택을 함께 바꾼다. IMDB는 원본 전처리 문서가 `inputs/imdb_processed.json` 또는 `datasets.imdb.path`에 지정한 위치에 있어야 한다. 이 방식은 사용자 지정 파일명으로 입력을 읽으면서 학습 시 원본 입력 검사를 유지한다.

`run-saved`는 연구자가 사용한 원본 파일명을 자동으로 찾는 편의 명령이다. 다른 파일명을 쓸 때는 JSON의 원본 이름·해시를 고치는 대신 `import-embeddings`와 `run`을 사용한다.

## 9. 기존 연구 임베딩으로 학습

<details>
<summary>원본 임베딩이 있는 경우의 별도 실행 방법 펼치기</summary>

원본 임베딩과 전처리 문서가 있다면 **별도 Python 3.8.19 환경**에서 `saved_*.yaml`과 `run-saved`를 사용한다. 이 명령은 입력과 환경을 확인하고 실제 학습 결과를 저장한다. 과거 결과 CSV를 읽어 비교하는 기능은 공개본에 포함하지 않았다.

```bash
conda create -n llm-bertopic-saved python=3.8.19 pip=24.0 -y
conda activate llm-bertopic-saved
python -m pip install -c constraints-notebook2024.txt -e '.[notebook2024]'
python -m pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl'
python -m nltk.downloader stopwords punkt
python -m llm_bertopic check-environment
```

아래 원본 경로를 자신의 경로로 바꾼다. `--trust-pickle`은 신뢰하는 원본 pickle에만 사용한다. NG20/BBC는 원문을 불러와 full 전처리를 수행하고 원본 배열과 연결한다.

```bash
python -m llm_bertopic run-saved --config configs/saved_newsgroup20.yaml \
  --source-dir '/path/to/Bert_topic' --trust-pickle \
  --set 'experiment.topic_counts=[5]' --set 'experiment.topk_values=[10]'

python -m llm_bertopic run-saved --config configs/saved_bbc.yaml \
  --source-dir '/path/to/Bert_topic' --trust-pickle \
  --set 'experiment.topic_counts=[5]' --set 'experiment.topk_values=[10]'
```

IMDB는 신뢰하는 원본 `data/imdb_preprocessed_data.pkl`의 문자열 목록을 그대로 JSON으로 저장한다. 다시 전처리하지 않는다.

```python
import json
import pickle
from pathlib import Path

with Path('/path/to/Bert_topic/data/imdb_preprocessed_data.pkl').open('rb') as f:
    texts = pickle.load(f)
assert isinstance(texts, list) and len(texts) == 50000
assert all(isinstance(text, str) for text in texts)
Path('inputs').mkdir(exist_ok=True)
Path('inputs/imdb_processed.json').write_text(json.dumps(texts, ensure_ascii=False), encoding='utf-8')
```

```bash
JOBLIB_MULTIPROCESSING=0 python -m llm_bertopic run-saved \
  --config configs/saved_imdb.yaml --source-dir '/path/to/Bert_topic' --trust-pickle \
  --processed-texts-json inputs/imdb_processed.json --set 'experiment.topic_counts=[10]'
```

IMDB의 HDBSCAN `core_dist_n_jobs=4`를 유지한다. 위 환경변수는 Joblib의 프로세스 병렬 실행을 비활성화한다. 원본 배열은 배포본에 포함되지 않으며 파일명·shape·dtype·문서 순서와 배열 해시는 8절의 데이터셋별 입력 JSON에 있다. 숫자를 맞추려고 dtype를 변환하거나 임베딩을 재정규화하지 않는다.

각 명령에서 `--set 'experiment.topic_counts=...'`와 `--set 'experiment.topk_values=...'` 제한을 빼면 해당 설정의 전체 조건을 실행한다. 임의의 minimal 배열은 이 원본 full 입력 전용 설정에 연결하지 않는다. 일반 minimal 실험은 해당 전처리로 임베딩을 새로 생성하거나 `import-embeddings`에 정확히 대응하는 전처리 문서 목록을 함께 제공한다.

</details>

## 10. 결과와 저장소 구성

`runs/<실행 ID>/results.csv`에는 NPMI, IRBO, KL-Uniform과 모델·토픽 수·전처리 모드가 기록된다. `jobs/`에는 토픽 단어와 설정, `manifest.json`에는 입력과 환경 정보가 저장된다. 새 실행 결과는 원본 파일을 덮어쓰지 않는다.

```bash
# 중단된 학습 재시작: 최초 실행과 같은 config 및 --set 옵션을 유지
python -m llm_bertopic run --config configs/full.yaml --resume 'runs/<실행 ID>'
```

```text
README.md                GitHub 첫 화면
read.md                  논문 요약과 실행 방법
CITATION.cff             논문 인용 정보
src/llm_bertopic/         전처리·임베딩·BERTopic/LDA·지표·입출력
configs/                 데이터, 모델, 전처리, 하이퍼파라미터 설정
notebooks/               데이터 준비·모델별 임베딩 생성·학습
pyproject.toml           설치 의존성과 CLI
constraints-*.txt        실행 환경 버전 목록
```

NPMI는 노이즈 토픽을 제외하고, IRBO/KL은 기본적으로 포함한다. 일반 실행은 Gensim과 패키지의 IRBO/KL 구현을 사용하고, 저장 입력용 환경은 원본 OCTIS 계산을 사용한다. LDA는 올바르게 토큰화한 문서와 모든 토픽으로 IRBO를 계산한다. 이 구현·환경 차이와 입력 차이를 포함해 논문의 전체 수치에 대한 일괄적인 동일성을 보장하지 않는다.

## 11. Citation

이 코드를 연구에 사용했다면 아래 BibTeX를 복사하여 논문을 인용한다.

```bibtex
@article{yang2025llmbertopic,
  title   = {Enhancing topic coherence and diversity in document embeddings using {LLMs}: A focus on {BERTopic}},
  author  = {Yang, Chibok and Kim, Yangsok},
  journal = {Expert Systems with Applications},
  volume  = {281},
  pages   = {127517},
  year    = {2025},
  doi     = {10.1016/j.eswa.2025.127517},
  url     = {https://doi.org/10.1016/j.eswa.2025.127517}
}
```

인용 정보는 [CITATION.cff](CITATION.cff)에도 있다. 코드 라이선스는 아직 지정되지 않았다.
