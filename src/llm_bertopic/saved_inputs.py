"""Load original saved embeddings and verify their ordered corpus before training."""

import hashlib
import platform
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .storage import file_hash, fingerprint, read_json, write_json


def input_metadata(dataset="newsgroup20"):
    files = {"newsgroup20": "ng20_2024.json", "bbc": "bbc_2024.json", "imdb": "imdb_2024.json"}
    if dataset not in files:
        raise ValueError("Saved input specifications are available only for newsgroup20, bbc and imdb")
    return read_json(Path(__file__).with_name(files[dataset]))


def check_environment():
    expected = read_json(Path(__file__).with_name("notebook_2024.json"))
    differences = []
    if platform.python_version() != expected["python"]:
        differences.append("Python: expected " + expected["python"] + ", got " + platform.python_version())
    for name, required in expected["packages"].items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = "missing"
        if actual != required:
            differences.append(f"{name}: expected {required}, got {actual}")
    if differences:
        raise RuntimeError("notebook_2024 environment mismatch; see read.md\n" + "\n".join(differences))
    from nltk.corpus import stopwords
    if fingerprint(sorted(stopwords.words("english"))) != expected["stopwords_sha256"]:
        raise RuntimeError("NLTK English stopwords differ from the verified reference")


def check_corpus(corpus, dataset="newsgroup20"):
    expected = input_metadata(dataset)
    metadata = corpus["metadata"]
    if metadata["input_count"] != expected["raw_count"] or metadata["raw_texts_sha256"] != expected["raw_texts_sha256"]:
        label = expected.get("input_kind", "raw")
        raise ValueError(f"{dataset}: {label} text content/order differs from the verified {expected['raw_count']} documents")
    if len(corpus["texts"]) != expected["processed_count"] or fingerprint(corpus["texts"]) != expected["processed_texts_sha256"]:
        raise ValueError(f"{dataset}: processed text content/order differs from the verified {expected['processed_count']} documents")


def check_matrix(name, matrix, dataset="newsgroup20"):
    expected = input_metadata(dataset)["embeddings"][name]
    if list(matrix.shape) != expected["shape"] or str(matrix.dtype) != expected["dtype"]:
        raise ValueError(f"{name}: original embedding shape/dtype required: {expected['shape']}, {expected['dtype']}")
    if hashlib.sha256(matrix.tobytes(order="C")).hexdigest() != expected["matrix_sha256"]:
        raise ValueError(f"{name}: embedding values/order differ from the verified saved array")


def experiment_keys(cfg):
    """Stable ordering for a recorded, local-RNG sample taken before any fits."""
    return [(m, n, k) for m in cfg["experiment"]["models"] for n in cfg["experiment"]["topic_counts"]
            for k in cfg["experiment"]["topk_values"]]


def select_conditions(cfg, sample_size=None, sample_seed=20260911):
    keys = experiment_keys(cfg)
    if sample_size is None:
        return keys
    if type(sample_size) is not int or not 1 <= sample_size <= len(keys):
        raise ValueError(f"sample_size must be between 1 and {len(keys)}")
    return sorted(random.Random(sample_seed).sample(keys, sample_size))


def run_saved(cfg, embedding_dir="inputs", source_dir=None, trust_pickle=False, resume=None,
              sample_size=None, sample_seed=20260911):
    from .config import project_path, validate
    from .data import prepare_dataset
    from .embeddings import embedding_location, ensure_embeddings, import_legacy
    from .experiment import run_experiments
    if cfg.get("execution_profile") != "notebook_2024":
        raise ValueError("run-saved requires a configs/saved_<dataset>.yaml preset")
    validate(cfg)
    dataset = cfg["experiment"]["datasets"][0]
    if dataset == "imdb" and cfg["experiment"]["topk_values"] != [10]:
        raise ValueError("IMDB saved-input preset contains topk=10 only; use run for other topk values")
    if (set(cfg["experiment"]["topic_counts"]) - set(range(5, 51, 5))
            or set(cfg["experiment"]["topk_values"]) - {10, 20, 30}):
        raise ValueError("run-saved supports topic counts 5..50 (step 5) and topk 10,20,30; use run for other grids")
    check_environment()
    keys = select_conditions(cfg, sample_size, sample_seed)
    selected_models = [m for m in cfg["experiment"]["models"] if any(k[0] == m for k in keys)]
    metadata = input_metadata(dataset)
    references = metadata["embeddings"]
    input_paths = {}
    for name in selected_models:
        spec = references[name]
        path = (Path(source_dir) / spec["source_path"] if source_dir is not None
                else project_path(cfg, embedding_dir) / spec["filename"])
        if source_dir is None and not path.exists():
            path = path.with_suffix(".npy")
        if not path.exists():
            raise FileNotFoundError(f"Saved embedding required: {path}; see read.md")
        if path.suffix == ".pkl":
            if not trust_pickle:
                raise ValueError("For your own saved .pkl files, explicitly pass --trust-pickle")
            if file_hash(path) != spec["pickle_sha256"]:
                raise ValueError(f"Saved pickle checksum differs: {path.name}")
        input_paths[name] = path
    corpus = prepare_dataset(cfg, dataset, selected_models[0])
    check_corpus(corpus, dataset)
    print(f"Verified {dataset}: {metadata.get('input_kind', 'raw')} {metadata['raw_count']} -> processed {metadata['processed_count']}; content and order match.", flush=True)
    texts_path = corpus["folder"] / "ordered_texts.json"
    write_json(texts_path, corpus["texts"])
    for name in selected_models:
        model_corpus = prepare_dataset(cfg, dataset, name)
        check_corpus(model_corpus, dataset)
        folder, _ = embedding_location(cfg, dataset, name, model_corpus)
        if not (folder / "metadata.json").exists():
            import_legacy(cfg, dataset, name, model_corpus, input_paths[name], texts_path, trust_pickle)
        matrix = ensure_embeddings(cfg, dataset, name, model_corpus, generate=False)
        check_matrix(name, matrix, dataset)
        print(f"Verified {name}: {matrix.shape}, {matrix.dtype}", flush=True)
        del matrix
    folder = run_experiments(cfg, generate_embeddings=False, resume=resume,
                             conditions=keys if sample_size is not None else None)
    write_json(folder / "selection.json", {"dataset": dataset, "sample_size": sample_size,
               "sample_seed": sample_seed if sample_size is not None else None, "conditions": keys})
    return folder
