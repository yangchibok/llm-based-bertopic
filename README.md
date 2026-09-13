# LLM-based BERTopic

Research code for **Enhancing topic coherence and diversity in document embeddings using LLMs: A focus on BERTopic**.

**Chibok Yang and Yangsok Kim** · *Expert Systems with Applications* 281 (2025), 127517.<br>
[Paper](https://doi.org/10.1016/j.eswa.2025.127517) · [한국어 논문 요약·실행 방법](read.md) · [Citation](CITATION.cff)

## ABSTRACT

With the rapid growth of digital textual data, the need for systematic organization of large datasets has become critical. Topic modeling stands out as an effective approach for analyzing large volumes of text datasets. Neural Topic Models (NTMs) have been developed to overcome the limitations of traditional methods by using contextual embeddings, such as Bidirectional Encoder Representations from Transformers (BERT), to improve topic coherence. Recent advancements in Natural Language Processing (NLP) have further enhanced document processing capabilities through large language models (LLMs) such as LLaMA and the Generative Pre-trained Transformer (GPT). This research explores whether LLM embeddings within NTMs offer better performance compared to conventional models like Sentence-BERT (S-BERT) and DistilBERT. In particular, we examine the impact of text preprocessing on topic modeling. A comparative analysis is conducted using datasets from three domains, evaluating six topic models, including LLMs such as Falcon and LLaMA3, using three evaluation metrics. Results show that while no single model consistently excelled across all metrics, LLaMA3 demonstrated the best performance in coherence among the LLMs. In addition, overall topic modeling performance improved with the application of all six preprocessing techniques. LLaMA3 showed progressively better performance with additional preprocessing, confirming its stability and effectiveness in topic modeling. These findings suggest that LLMs can be reliable tools for topic identification across diverse datasets.



## Research overview

This study asks whether document embeddings from large language models improve the quality of topics produced by BERTopic, and how text preprocessing changes that quality. It compares five embedding models—S-BERT, DistilBERT, Falcon, LLaMA2, and LLaMA3—with LDA as a non-embedding baseline.

| Dataset | Domain | Documents reported in the paper |
|---|---|---:|
| 20 Newsgroups | Newsgroup discussions | 11,314 |
| BBC News | News articles | 2,225 |
| IMDB | Movie reviews | 50,000 |

The BERTopic pipeline is **preprocessing → document embeddings → UMAP → HDBSCAN → CountVectorizer / c-TF-IDF → evaluation**. LDA uses a dictionary and bag-of-words representation of the processed documents. Requested topic counts range from 5 to 50 in steps of 5. Empty documents are removed after preprocessing, so the final modeling corpus can be smaller than the counts above.

The evaluation covers three complementary aspects: **NPMI** measures coherence, **IRBO** measures diversity through overlap between ranked topic-word lists, and **KL-Uniform** measures divergence from a uniform distribution. The preprocessing comparison separates basic cleanup, cleanup plus stopword removal, and full preprocessing including lemmatization.

## Quickstart

Use Python 3.12. Run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[standard,notebooks]'
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt punkt_tab
python -m llm_bertopic all --config configs/quickstart.yaml
```

On Windows, activate with `.venv\Scripts\activate`. The first run downloads the data and model. Quickstart runs S-BERT BERTopic and LDA on 300 BBC documents and saves results to `runs/<run-id>/results.csv`.

## Preprocessing levels

| Paper level | Configuration | Stopwords in embedding/LDA input | Lemmatization |
|---|---|---|---|
| Level 1 | `configs/minimal_keep_stopwords.yaml` | Kept | No |
| Level 2 | `configs/minimal_remove_stopwords.yaml` | Removed | No |
| Level 3 | `configs/full.yaml` | Removed | Yes |

Both minimal modes remove punctuation/emoticons, lowercase, and normalize whitespace while keeping numbers and word forms. Level 2 uses the original minimal code's NLTK tokenizer and scikit-learn English stopword list. Full preprocessing retains the original NLTK/spaCy pipeline, including number removal and POS filtering. The selected mode applies to all five embedding models and LDA.

```bash
# Small Level 2 experiment; change the config for Level 1 or full preprocessing
python -m llm_bertopic all --config configs/minimal_remove_stopwords.yaml \
  --set 'experiment.datasets=[bbc]' --set 'experiment.models=[sbert,lda]' \
  --set 'experiment.topic_counts=[5]' --set 'experiment.topk_values=[10]' \
  --set 'datasets.bbc.max_documents=300'
```

Text stopword removal happens **before embedding generation**. `vectorizer.stop_words` controls topic-word extraction separately; changing only that setting does not produce Level 2 embeddings. Different preprocessing modes use separate document and embedding caches.

## Embedding files and input JSONs

**Embedding arrays are not included in this repository. Generate them before training, or import arrays you have already generated.** The following JSON files describe the author's original experiment inputs; they contain neither vectors nor document text.

| Dataset | Original-input specification |
|---|---|
| NG20 | [ng20_2024.json](src/llm_bertopic/ng20_2024.json) |
| BBC | [bbc_2024.json](src/llm_bertopic/bbc_2024.json) |
| IMDB | [imdb_2024.json](src/llm_bertopic/imdb_2024.json) |

[notebook_2024.json](src/llm_bertopic/notebook_2024.json) stores the shared historical Python/package environment. NG input metadata, previously stored in that file, now has its own `ng20_2024.json` file.

**`filename` and `source_path` are names chosen by the author during the original experiments. They are not mandatory names for your own embeddings.** `filename` is the historical name looked up under `--embedding-dir`; `source_path` is the historical path relative to `--source-dir`. Shapes, dtypes, and hashes identify the original data; editing a filename does not make a newly generated array equivalent to the original one.

### Generate new embeddings

For example, generate NG20 S-BERT embeddings and train one condition:

```bash
python -m llm_bertopic embed --config configs/newsgroup20.yaml \
  --set 'experiment.models=[sbert]'

python -m llm_bertopic run --config configs/newsgroup20.yaml \
  --set 'experiment.models=[sbert]' \
  --set 'experiment.topic_counts=[5]' --set 'experiment.topk_values=[10]'
```

Use the matching BBC/IMDB config for those datasets. For Falcon/LLaMA2/LLaMA3, select the appropriate model keys and memory/device settings; separate notebooks and `configs/cuda_8bit.yaml` are available. Use the same preprocessing and embedding settings for generation and training.

New arrays are stored automatically at `artifacts/embeddings/<dataset>/<model>/<cache-id>/embeddings.npy`, with a `metadata.json` next to each array. The current generation path saves float32 arrays. The original LLM arrays were float16, and historical hashes apply only to the original inputs. Use `embed`/`run` for new experiments; use the historical saved-input mode only with the matching original data. `all` combines generation and training. Do not rename files inside the managed cache.

### Use your own filenames

An external array can have any filename, such as `inputs/my_ng_llama3.npy`. Provide the **ordered, already-processed document list used to generate that array**, for example `inputs/my_ng_processed_texts.json`, as a JSON list of strings. Raw documents or a list in a different order will be rejected.

The example [custom_embeddings.yaml](configs/custom_embeddings.yaml) selects NG20/LLaMA3, full preprocessing, one topic-count setting, and a separate import cache:

```bash
python -m llm_bertopic import-embeddings --config configs/custom_embeddings.yaml \
  --dataset newsgroup20 --model llama3 \
  --array 'inputs/my_ng_llama3.npy' \
  --texts 'inputs/my_ng_processed_texts.json'

python -m llm_bertopic run --config configs/custom_embeddings.yaml
```

Replace the two paths with your actual filenames. For a trusted `.pkl` or `.pickle`, add `--trust-pickle`. The importer checks document content/order and array shape, preserves dtype, and copies the array into the managed cache. If that cache already exists, choose an unused `paths.artifact_dir` in the config. The name `llama3` is a model key, not a filename; select the actual model used to generate the embeddings.

To use BBC or IMDB, change `experiment.datasets` in the config and `--dataset` in the command together. Change `experiment.models` and `--model` together for another embedding model. For minimal-preprocessed arrays, set `extends` to the matching minimal config. Generation, importing, and training must use the same document preprocessing and compatible model settings.

**If you only renamed an original historical array:** use the separate Python 3.8.19 environment and change the example config's `extends: full.yaml` to `extends: saved_newsgroup20.yaml` (or the corresponding BBC/IMDB saved preset, with matching dataset selection). Then use the same `import-embeddings` → `run` sequence. This accepts your custom input path while preserving the historical input checks during training. For IMDB, the saved preset also requires the original processed corpus at `inputs/imdb_processed.json` or the path configured under `datasets.imdb.path`.

`run-saved` remains a convenience command that looks up the author's original filenames. Use `import-embeddings` plus `run` for custom filenames rather than changing the recorded names or hashes in the packaged JSONs. See [read.md](read.md) for historical environment setup.

## Running the research code

Use `all` for preprocessing, embedding generation, and training; `embed` to generate embeddings separately; and `run` to train with existing embeddings. Dataset/model selections, UMAP, HDBSCAN, vectorizer, and LDA parameters are configurable in YAML or through `--set`.

The five notebooks cover data preparation, separate Falcon/LLaMA2/LLaMA3 embedding generation, and BERTopic/LDA training. LLM generation requires sufficient memory and may require model access approval. Original embedding arrays are not bundled.

See [read.md](read.md) for full-dataset commands, GPU settings, preprocessing choices, saved historical inputs, and output files. The historical input mode keeps its separate Python 3.8.19 environment. Code, notebooks, and configuration files are provided here; historical comparison reports and test suites remain outside this distribution.

## Citation

If you use this code, please cite the associated article. Copy the BibTeX entry below:

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

Machine-readable citation metadata is also available in [CITATION.cff](CITATION.cff). A code license has not yet been designated.
