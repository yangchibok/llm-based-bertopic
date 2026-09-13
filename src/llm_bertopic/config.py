"""YAML configuration, inheritance, and per-dataset/per-embedding overrides."""

from copy import deepcopy
from pathlib import Path

import yaml

MODELS = ("sbert", "distilbert", "falcon", "llama2", "llama3")


def merge(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read(path, seen):
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    parent = data.pop("extends", None)
    return merge(_read(path.parent / parent, seen | {path}), data) if parent else data


def load_config(path, overrides=(), root=None):
    path = Path(path).resolve()
    cfg = _read(path, set())
    for item in overrides:
        if "=" not in item:
            raise ValueError("Overrides must be dotted.path=YAML_VALUE")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown config mapping: {key}")
            node = node[part]
        # New constructor kwargs (e.g. umap.spread) are intentionally allowed.
        node[parts[-1]] = yaml.safe_load(value)
    if root is None:
        root = next((p for p in path.parents if (p / "pyproject.toml").is_file()), path.parent)
    cfg["_root"] = str(Path(root).resolve())
    validate(cfg)
    return cfg


def public_config(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def effective_config(cfg, dataset, model):
    result = merge(cfg, cfg.get("dataset_overrides", {}).get(dataset, {}))
    return merge(result, cfg.get("model_overrides", {}).get(dataset, {}).get(model, {}))


def project_path(cfg, value):
    return (Path(cfg["_root"]) / Path(value)).resolve()


def validate(cfg):
    required = {"datasets", "experiment", "preprocessing", "embeddings", "umap", "hdbscan",
                "vectorizer", "lda", "metrics", "paths"}
    if required - cfg.keys():
        raise ValueError(f"Missing configuration sections: {sorted(required - cfg.keys())}")
    unknown = set(cfg) - required - {"seed", "dataset_overrides", "model_overrides", "bertopic", "_root",
                                   "execution_profile"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    if cfg.get("execution_profile", "standard") not in {"standard", "notebook_2024"}:
        raise ValueError("execution_profile must be standard or notebook_2024")
    exp = cfg["experiment"]
    for section, allowed in {
        "experiment": {"datasets", "models", "topic_counts", "topk_values", "save_assignments"},
        "preprocessing": {"mode", "spacy_model", "allowed_postags", "batch_size"},
        "metrics": {"coherence", "processes", "window_size", "rbo_weight", "kl_epsilon",
                    "irbo_include_outliers", "kl_include_outliers", "lda_coherence_topn"},
        "paths": {"artifact_dir", "cache_dir", "run_dir"},
    }.items():
        if set(cfg[section]) - allowed:
            raise ValueError(f"Unknown keys in {section}: {sorted(set(cfg[section]) - allowed)}")
    if not exp["datasets"] or not exp["models"]:
        raise ValueError("Select at least one dataset and one model")
    for key in ("datasets", "models", "topic_counts", "topk_values"):
        if len(exp[key]) != len(set(exp[key])):
            raise ValueError(f"Duplicate experiment.{key} values")
    if not set(exp["datasets"]) <= set(cfg["datasets"]):
        raise ValueError("Unknown dataset selection")
    if not set(exp["models"]) <= set(MODELS) | {"lda"}:
        raise ValueError("Unknown embedding/model selection")
    if cfg.get("execution_profile") == "notebook_2024":
        supported = {"newsgroup20": {"llama2", "llama3", "falcon", "distilbert"},
                     "bbc": {"falcon", "llama2", "llama3"}, "imdb": {"falcon", "llama2", "llama3"}}
        if (len(exp["datasets"]) != 1 or exp["datasets"][0] not in supported
                or not set(exp["models"]) <= supported[exp["datasets"][0]]):
            raise ValueError("notebook_2024 supports one dataset: Newsgroup20 (four saved models), BBC or IMDB (three LLMs)")
        required_mode = "none" if exp["datasets"] == ["imdb"] else "full"
        if cfg["preprocessing"]["mode"] != required_mode:
            raise ValueError(f"notebook_2024 requires preprocessing.mode={required_mode}; IMDB uses saved processed texts")
        if exp["datasets"] == ["imdb"] and cfg["datasets"]["imdb"]["source"] != "json":
            raise ValueError("IMDB notebook reproduction requires the original processed text list as JSON")
        if (not cfg["metrics"]["irbo_include_outliers"] or not cfg["metrics"]["kl_include_outliers"]
                or cfg["metrics"]["kl_epsilon"] != 1e-5):
            raise ValueError("notebook_2024 uses OCTIS with outliers included and fixed KL epsilon=1e-5")
    for key in ("topic_counts", "topk_values"):
        if not exp[key] or any(type(x) is not int or x < 1 for x in exp[key]):
            raise ValueError(f"experiment.{key} must contain positive integers")
    from .data import PREPROCESSING_MODES
    if cfg["preprocessing"]["mode"] not in PREPROCESSING_MODES:
        raise ValueError("preprocessing.mode must be full, minimal_keep_stopwords, minimal_remove_stopwords or none")
    if cfg["metrics"]["coherence"] not in {"c_npmi", "c_v", "u_mass", "c_uci"}:
        raise ValueError("Unsupported coherence metric")
    if not 0 < cfg["metrics"]["rbo_weight"] < 1:
        raise ValueError("rbo_weight must be between 0 and 1 exclusively")
    for name in ("irbo_include_outliers", "kl_include_outliers"):
        if type(cfg["metrics"][name]) is not bool:
            raise ValueError(f"metrics.{name} must be a boolean")
    if (cfg["metrics"]["processes"] != -1 and cfg["metrics"]["processes"] < 1
            or cfg["metrics"]["lda_coherence_topn"] < 1):
        raise ValueError("metrics.processes must be positive or -1; lda_coherence_topn must be positive")
    if cfg["metrics"]["kl_epsilon"] < 0:
        raise ValueError("kl_epsilon must be nonnegative")
    lda_topks = cfg["lda"]["topk_values"]
    if not lda_topks or any(type(k) is not int or k < 1 for k in lda_topks) or len(lda_topks) != len(set(lda_topks)):
        raise ValueError("lda.topk_values must contain distinct positive integers")
    if cfg["lda"]["irbo_num_topics"] != -1 and cfg["lda"]["irbo_num_topics"] < 2:
        raise ValueError("lda.irbo_num_topics must be -1 (all topics) or at least 2")
    for name in MODELS:
        emb = cfg["embeddings"][name]
        if emb["batch_size"] < 1 or emb["max_length"] < 1:
            raise ValueError(f"Invalid embedding batch/sequence length: {name}")
        if emb["pooling"] not in {"sentence_transformer", "masked_mean", "legacy_mean"}:
            raise ValueError(f"Unknown pooling strategy: {name}")
        if emb["quantization"] not in {"none", "8bit", "4bit"}:
            raise ValueError(f"Unknown quantization strategy: {name}")
        if emb["padding"] not in {"longest", "max_length"}:
            raise ValueError(f"Unknown padding strategy: {name}")
        if emb["pad_token_strategy"] not in {"eos", "add_new"}:
            raise ValueError(f"Unknown pad token strategy: {name}")
    for name in exp["datasets"]:
        ds = cfg["datasets"][name]
        if ds["source"] not in {"huggingface", "csv", "jsonl", "json"}:
            raise ValueError(f"Unknown source: {ds['source']}")
        limit = ds.get("max_documents")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("max_documents must be a positive integer or null")
