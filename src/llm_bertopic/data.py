"""Prepare each corpus once and preserve document identity through filtering."""

import csv
import json
import re

from .config import effective_config, project_path
from .storage import atomic_text, environment, file_hash, fingerprint, lock, read_json, write_json


PREPROCESSING_MODES = {
    "full", "minimal_keep_stopwords", "minimal_remove_stopwords", "minimal", "minimal_stopwords", "none",
}


def preprocess(texts, cfg, execution_profile="standard"):
    mode = cfg["mode"]
    if mode not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    # Older programmatic callers can still use these two unambiguous aliases.
    mode = {"minimal": "minimal_keep_stopwords", "minimal_stopwords": "minimal_remove_stopwords"}.get(mode, mode)
    if mode == "none":
        return [text.strip() for text in texts]
    if mode in {"minimal_keep_stopwords", "minimal_remove_stopwords"}:
        # Match the original MinimalTextPreprocessor's order; retain numbers and word forms.
        cleaned = []
        for text in texts:
            text = re.sub(r"[^\w\s,]", "", text).lower()
            text = re.sub(r"[^\w\s]", "", text)
            cleaned.append(re.sub(r"\s+", " ", text).strip())
        if mode == "minimal_remove_stopwords":
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            from nltk.tokenize import word_tokenize
            try:
                cleaned = [" ".join(t for t in word_tokenize(text) if t not in ENGLISH_STOP_WORDS)
                           for text in cleaned]
            except LookupError as exc:
                raise RuntimeError("Run: python -m nltk.downloader punkt punkt_tab") from exc
        return cleaned
    import spacy
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    try:
        stops = set(stopwords.words("english"))
        word_tokenize("resource check")
    except LookupError as exc:
        raise RuntimeError("Run: python -m nltk.downloader stopwords punkt punkt_tab") from exc
    try:
        nlp = spacy.load(cfg["spacy_model"])
    except OSError as exc:
        raise RuntimeError(f"Run: python -m spacy download {cfg['spacy_model']}") from exc
    cleaned = []
    for text in texts:
        text = re.sub(r"\d+", " ", re.sub(r"\W+", " ", text))
        text = "".join(c.lower() for c in text if c.isalpha() or c.isspace())
        cleaned.append(" ".join(t for t in word_tokenize(text) if t not in stops))
    # The historical preprocessor called nlp(text) separately for every document.
    docs = ((nlp(text) for text in cleaned) if execution_profile == "notebook_2024"
            else nlp.pipe(cleaned, batch_size=cfg["batch_size"]))
    return [" ".join(t.lemma_ for t in doc if t.pos_ in cfg["allowed_postags"]) for doc in docs]


def raw_records(cfg, dataset):
    ds = cfg["datasets"][dataset]
    field = ds["text_column"]
    records, details = [], {}
    if ds["source"] == "huggingface":
        from datasets import load_dataset
        loaded = load_dataset(ds["id"], name=ds.get("name"), revision=ds.get("revision"),
                              cache_dir=str(project_path(cfg, cfg["paths"]["cache_dir"])))
        details["splits"] = {}
        for split in ds["splits"]:
            table = loaded[split]
            if field not in table.column_names:
                raise ValueError(f"Missing {field!r} in {dataset}/{split}")
            details["splits"][split] = {"rows": len(table), "fingerprint": table._fingerprint}
            records.extend((f"{dataset}:{split}:{i}", value) for i, value in enumerate(table[field]))
    else:
        source = project_path(cfg, ds["path"])
        details["file_sha256"] = file_hash(source)
        with source.open(encoding="utf-8-sig") as stream:
            if ds["source"] == "json":
                values = json.load(stream)
                if not isinstance(values, list):
                    raise ValueError("JSON corpus must be an ordered list of strings or objects")
                rows = ({field: value} if isinstance(value, str) else value for value in values)
            else:
                rows = csv.DictReader(stream) if ds["source"] == "csv" else (json.loads(x) for x in stream if x.strip())
            for i, row in enumerate(rows):
                if field not in row:
                    raise ValueError(f"Missing text column {field!r} in row {i}")
                records.append((f"{dataset}:local:{i}", row[field]))
    # Head selection is deterministic and occurs BEFORE empty-document filtering.
    if ds.get("max_documents") is not None:
        records = records[:ds["max_documents"]]
    if any(not isinstance(value, str) for _, value in records):
        raise ValueError("Corpus contains non-string text; clean the source explicitly")
    return records, details


def corpus_hash(records):
    return fingerprint(records)  # Includes order, stable row ID and processed text.


def prepare_dataset(cfg, dataset, model="data"):
    cfg = effective_config(cfg, dataset, model)
    profile = cfg.get("execution_profile", "standard")
    if profile == "notebook_2024":
        from .saved_inputs import check_environment
        check_environment()
    identity = {"dataset": cfg["datasets"][dataset], "preprocessing": cfg["preprocessing"],
                "execution_profile": profile, "format_version": 2,
                "preprocessor_sha256": file_hash(__file__),
                "preprocessing_environment": {k: v for k, v in environment()["packages"].items()
                                              if k in {"spacy", "en-core-web-sm", "nltk", "scikit-learn"}}}
    if identity["dataset"]["source"] != "huggingface":
        identity["local_sha256"] = file_hash(project_path(cfg, identity["dataset"]["path"]))
    folder = project_path(cfg, cfg["paths"]["artifact_dir"]) / "corpora" / dataset / fingerprint(identity)[:20]
    metadata_file = folder / "metadata.json"
    with lock(folder.with_suffix(".lock")):
        if metadata_file.exists():
            return load_corpus(folder)
        raw, source = raw_records(cfg, dataset)
        processed = preprocess([t for _, t in raw], cfg["preprocessing"], profile)
        records = [{"id": row_id, "text": text} for (row_id, _), text in zip(raw, processed) if text.strip()]
        if len(records) < 3:
            raise ValueError("Fewer than three nonempty documents remain")
        folder.mkdir(parents=True, exist_ok=True)
        atomic_text(folder / "documents.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        metadata = {"identity": identity, "source": source, "input_count": len(raw),
                    "raw_texts_sha256": fingerprint([t for _, t in raw]),
                    "processed_texts_sha256": fingerprint([r["text"] for r in records]),
                    "document_count": len(records), "dropped_count": len(raw) - len(records),
                    "corpus_sha256": corpus_hash(records), "environment": environment()}
        write_json(metadata_file, metadata)
    return load_corpus(folder)


def load_corpus(folder):
    metadata = read_json(folder / "metadata.json")
    records = [json.loads(x) for x in (folder / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    if corpus_hash(records) != metadata["corpus_sha256"]:
        raise ValueError("Prepared corpus has changed; use a new cache directory to regenerate")
    return {"folder": folder, "metadata": metadata, "records": records,
            "texts": [r["text"] for r in records]}
