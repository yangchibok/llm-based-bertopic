"""Independent model fits, durable job checkpoints and explicit result aggregation."""

import csv
import gc
import io
import math
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import effective_config, project_path, public_config, validate
from .data import prepare_dataset
from .embeddings import embedding_location, ensure_embeddings
from .metrics import evaluate_coherence, inverted_rbo, kl_uniform
from .storage import atomic_text, environment, file_hash, fingerprint, lock, read_json, write_json


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def bertopic_components(cfg):
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    vectorizer = dict(cfg["vectorizer"])
    vectorizer["ngram_range"] = tuple(vectorizer["ngram_range"])
    return {"umap_model": UMAP(**cfg["umap"]), "hdbscan_model": HDBSCAN(**cfg["hdbscan"]),
            "vectorizer_model": CountVectorizer(**vectorizer)}


def fit_bertopic(texts, embeddings, cfg, count, topk, components=None):
    from bertopic import BERTopic
    notebook = cfg.get("execution_profile") == "notebook_2024"
    if not notebook:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
    kwargs = dict(cfg.get("bertopic", {}))
    for key in ("nr_topics", "top_n_words", "umap_model", "hdbscan_model", "vectorizer_model", "embedding_model"):
        if key in kwargs:
            raise ValueError(f"Configure {key} in its dedicated section, not bertopic")
    model = BERTopic(**(components if components is not None else bertopic_components(cfg)),
                     nr_topics=count, top_n_words=topk, **kwargs)
    assignments, _ = model.fit_transform(texts, embeddings=embeddings)
    topic_map = model.get_topics()
    ids = sorted(topic_map)
    # BERTopic c_tf_idf_ rows follow sorted topic IDs, including -1 when present.
    if model.c_tf_idf_.shape[0] != len(ids):
        raise RuntimeError("Unexpected BERTopic topic-to-matrix alignment")
    words = {k: [w for w, _ in topic_map[k] if notebook or w] for k in ids}
    if notebook:
        from .metrics import evaluate_notebook
        result = evaluate_notebook(model, texts, topk, cfg["metrics"])
    else:
        coherence, ignored = evaluate_coherence([words[k] for k in ids if k != -1], texts, cfg["metrics"])
        rbo_ids = [k for k in ids if k != -1 or cfg["metrics"]["irbo_include_outliers"]]
        kl_rows = [i for i, k in enumerate(ids) if k != -1 or cfg["metrics"]["kl_include_outliers"]]
        result = {"coherence_score": coherence,
                  "inverted_rbo_score": inverted_rbo([words[k] for k in rbo_ids], topk, cfg["metrics"]["rbo_weight"]),
                  "kl_uniform_score": kl_uniform(model.c_tf_idf_[kl_rows], cfg["metrics"]["kl_epsilon"]),
                  "coherence_ignored_words": ignored}
    result.update(actual_topics=sum(k != -1 for k in ids), outlier_count=assignments.count(-1))
    topics = [{"topic_id": k, "words": words[k], "weights": [float(v) for w, v in topic_map[k] if notebook or w]}
              for k in ids]
    del model
    return result, topics, assignments


def fit_lda(texts, cfg, count, topk):
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    tokens = [text.split() for text in texts]
    dictionary = Dictionary(tokens)
    if not dictionary:
        raise ValueError("LDA vocabulary is empty")
    corpus = [dictionary.doc2bow(t) for t in tokens]
    kwargs = {k: v for k, v in cfg["lda"].items() if k not in {"topk_values", "irbo_num_topics"}}
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=count, **kwargs)
    coherence, ignored = evaluate_coherence([], texts, cfg["metrics"], model, dictionary)
    displayed = model.show_topics(num_topics=cfg["lda"]["irbo_num_topics"], formatted=False, num_words=topk)
    rbo_words = [[w for w, _ in topic] for _, topic in displayed]
    result = {"coherence_score": coherence,
              "inverted_rbo_score": inverted_rbo(rbo_words, topk, cfg["metrics"]["rbo_weight"]),
              "kl_uniform_score": kl_uniform(model.get_topics(), cfg["metrics"]["kl_epsilon"]),
              "actual_topics": count, "outlier_count": 0, "coherence_ignored_words": ignored}
    topics = [{"topic_id": i, "words": [w for w, _ in model.show_topic(i, topn=topk)],
               "weights": [float(v) for _, v in model.show_topic(i, topn=topk)]} for i in range(count)]
    assignments = []
    if cfg["experiment"]["save_assignments"]:
        assignments = [max(model.get_document_topics(doc, minimum_probability=0), key=lambda x: x[1])[0]
                       for doc in corpus]
    del model
    return result, topics, assignments


def aggregate_run(folder):
    folder = Path(folder)
    jobs = sorted((folder / "jobs").glob("*.json"))
    results = [read_json(path)["result"] for path in jobs]
    if not results:
        return None
    results.sort(key=lambda r: (r["dataset"], r["embedding"], r["nr_topics"], r["topk"]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
    atomic_text(folder / "results.csv", buffer.getvalue())
    return folder / "results.csv"


def run_experiments(cfg, generate_embeddings=False, resume=None, conditions=None):
    validate(cfg)
    selected = None if conditions is None else {tuple(key) for key in conditions}
    if selected is not None:
        from .saved_inputs import experiment_keys
        if len(cfg["experiment"]["datasets"]) != 1 or not selected or not selected <= set(experiment_keys(cfg)):
            raise ValueError("Selected conditions must be a nonempty subset of one dataset's grid")
    selection = None if selected is None else [list(key) for key in sorted(selected)]
    notebook = cfg.get("execution_profile") == "notebook_2024"
    if notebook:
        from .saved_inputs import check_environment, check_corpus, check_matrix
        check_environment()
    public = public_config(cfg)
    code_hash = fingerprint({p.name: file_hash(p) for p in Path(__file__).parent.iterdir()
                             if p.suffix in {".py", ".json", ".csv"}})
    config_hash = fingerprint(public)
    run_dir = Path(resume).resolve() if resume else project_path(cfg, cfg["paths"]["run_dir"]) / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    if resume and not (run_dir / "manifest.json").exists():
        raise ValueError("Resume requires an existing run manifest")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    with lock(run_dir / ".running.lock"):
        if resume:
            manifest = read_json(manifest_path)
            if manifest["config_sha256"] != config_hash or manifest["code_sha256"] != code_hash:
                raise ValueError("Resume configuration/code differs; start a new run")
            if manifest["environment"] != environment():
                raise ValueError("Resume environment differs; start a new run")
            if manifest.get("selected_conditions") != selection:
                raise ValueError("Resume condition selection differs; start a new run")
        else:
            manifest = {"config": public, "config_sha256": config_hash, "code_sha256": code_hash,
                        "environment": environment(), "corpora": {}, "embeddings": {}, "status": "running",
                        "selected_conditions": selection}
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        print(f"Run directory: {run_dir}", flush=True)
        try:
            shared = {}
            for dataset in cfg["experiment"]["datasets"]:
                for name in cfg["experiment"]["models"]:
                    if selected is not None and not any(key[0] == name for key in selected):
                        continue
                    effective = effective_config(cfg, dataset, name)
                    validate(effective)
                    corpus = prepare_dataset(cfg, dataset, name)
                    if notebook:
                        check_corpus(corpus, dataset)
                    corpus_id = corpus["metadata"]["corpus_sha256"]
                    corpus_key = f"{dataset}/{name}"
                    if corpus_key in manifest["corpora"] and manifest["corpora"][corpus_key]["corpus_sha256"] != corpus_id:
                        raise ValueError("Corpus changed since the original run")
                    manifest["corpora"][corpus_key] = corpus["metadata"]
                    embeddings = None
                    if name != "lda":
                        embeddings = ensure_embeddings(cfg, dataset, name, corpus, generate_embeddings)
                        if notebook:
                            check_matrix(name, embeddings, dataset)
                        folder, _ = embedding_location(cfg, dataset, name, corpus)
                        metadata = read_json(folder / "metadata.json")
                        old = manifest["embeddings"].get(corpus_key)
                        if old is not None and old["array_sha256"] != metadata["array_sha256"]:
                            raise ValueError("Embedding array changed since the original run")
                        manifest["embeddings"][corpus_key] = metadata
                    write_json(manifest_path, manifest)
                    topks = effective["lda"]["topk_values"] if name == "lda" else effective["experiment"]["topk_values"]
                    components = None
                    if notebook:
                        # Cell 20 shares estimators across the three LLM grids; cell 19 is independent.
                        group = "distilbert" if name == "distilbert" else "llm"
                        component_key = fingerprint([dataset, group, effective["umap"],
                                                     effective["hdbscan"], effective["vectorizer"]])
                        if component_key not in shared:
                            shared[component_key] = bertopic_components(effective)
                        components = shared[component_key]
                    for count in effective["experiment"]["topic_counts"]:
                        for topk in topks:
                            if selected is not None and (name, count, topk) not in selected:
                                continue
                            job_id = fingerprint([dataset, name, count, topk, corpus_id, config_hash])[:24]
                            job_path = run_dir / "jobs" / (job_id + ".json")
                            if job_path.exists():
                                continue
                            print(f"{dataset} / {name}: nr_topics={count}, topk={topk}", flush=True)
                            scores, topics, assignments = (fit_lda(corpus["texts"], effective, count, topk)
                                if name == "lda" else fit_bertopic(corpus["texts"], embeddings, effective, count, topk,
                                                                  components=components))
                            valid = all(math.isfinite(scores[k]) for k in ("coherence_score", "inverted_rbo_score", "kl_uniform_score"))
                            result = {"dataset": dataset, "embedding": "LDA" if name == "lda" else name,
                                      "nr_topics": count, "topk": topk,
                                      "preprocessing": effective["preprocessing"]["mode"],
                                      "coherence_metric": effective["metrics"]["coherence"], **scores,
                                      "document_count": len(corpus["texts"]), "seed": effective["seed"],
                                      "umap_random_state": effective["umap"]["random_state"] if name != "lda" else None,
                                      "lda_random_state": effective["lda"]["random_state"] if name == "lda" else None,
                                      "status": "ok" if valid else "undefined_metric",
                                      "corpus_sha256": corpus_id}
                            payload = {"result": result, "topics": topics, "effective_config": public_config(effective)}
                            if effective["experiment"]["save_assignments"]:
                                payload["assignments"] = [{"document_id": r["id"], "topic_id": int(label)}
                                    for r, label in zip(corpus["records"], assignments)]
                            write_json(job_path, finite_json(payload))
                            aggregate_run(run_dir)
                            gc.collect()
                    del embeddings
            manifest["status"] = "complete"
        except BaseException as exc:
            manifest["status"] = "failed"
            # Keep partial results and the original traceback for the caller.
            manifest["error_type"] = type(exc).__name__
            raise
        finally:
            aggregate_run(run_dir)
            write_json(manifest_path, manifest)
    return run_dir
