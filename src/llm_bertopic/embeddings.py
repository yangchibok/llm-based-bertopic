"""Standalone embedding generation with document-order-checked .npy caches."""

import gc
import os
import pickle
from pathlib import Path

import numpy as np

from .config import effective_config, project_path
from .storage import environment, file_hash, fingerprint, lock, read_json, write_json


def pool_hidden(hidden, attention_mask, strategy, normalize=False):
    import torch.nn.functional as functional
    # Float32 reduction avoids accumulation overflow in half precision.
    hidden = hidden.float()
    if strategy == "legacy_mean":
        result = hidden.mean(dim=1)
    elif strategy == "masked_mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        result = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        raise ValueError(f"Unsupported transformer pooling: {strategy}")
    return functional.normalize(result, p=2, dim=1) if normalize else result


def resolve_device(requested):
    import torch
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return requested


def transformer_batches(texts, spec, seed):
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, set_seed
    set_seed(seed)
    device = resolve_device(spec["device"])
    dtype_name = spec["dtype"]
    if dtype_name == "auto":
        dtype_name = "float32" if device == "cpu" else "float16"
    if dtype_name not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be auto, float32, float16 or bfloat16")
    dtype = getattr(torch, dtype_name)
    # Never save tokens in configuration or metadata. HF_TOKEN or the hub login cache is used.
    shared = {"revision": spec.get("revision"), "token": os.environ.get("HF_TOKEN"),
              "local_files_only": spec.get("local_files_only", False), "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], **shared)
    kwargs = {**shared, "torch_dtype": dtype}
    quantization = spec["quantization"]
    if quantization != "none":
        if not device.startswith("cuda"):
            raise ValueError("This pipeline supports bitsandbytes quantization on CUDA only; use none on CPU/MPS")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quantization == "8bit", load_in_4bit=quantization == "4bit",
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": device}
    model = AutoModel.from_pretrained(spec["model_id"], **kwargs)
    if tokenizer.pad_token_id is None:
        if spec["pad_token_strategy"] == "add_new":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            # Old notebooks added a new embedding row; initialization is now seeded.
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer lacks pad/eos tokens; use pad_token_strategy: add_new")
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    tokenizer.padding_side = "right"
    if quantization == "none":
        model.to(device)
    model.eval()
    try:
        with torch.inference_mode():
            for start in tqdm(range(0, len(texts), spec["batch_size"]), desc=spec["model_id"]):
                batch = tokenizer(texts[start:start + spec["batch_size"]], return_tensors="pt",
                                  padding=spec["padding"], truncation=True, max_length=spec["max_length"],
                                  return_token_type_ids=False)
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch).last_hidden_state
                yield pool_hidden(output, batch["attention_mask"], spec["pooling"], spec["normalize"]).cpu().numpy()
    finally:
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()


def sentence_batches(texts, spec, seed):
    from sentence_transformers import SentenceTransformer
    from transformers import set_seed
    set_seed(seed)
    model = SentenceTransformer(spec["model_id"], device=resolve_device(spec["device"]),
                                revision=spec.get("revision"), token=os.environ.get("HF_TOKEN"),
                                local_files_only=spec.get("local_files_only", False))
    model.max_seq_length = spec["max_length"]
    try:
        # Bound host memory as well as GPU memory; results are streamed to disk.
        chunk_size = max(spec["batch_size"], 256)
        for start in range(0, len(texts), chunk_size):
            yield model.encode(texts[start:start + chunk_size], batch_size=spec["batch_size"],
                               normalize_embeddings=spec["normalize"], show_progress_bar=True,
                               convert_to_numpy=True).astype(np.float32)
    finally:
        del model
        gc.collect()


def embedding_location(cfg, dataset, model, corpus):
    effective = effective_config(cfg, dataset, model)
    identity = {"corpus_sha256": corpus["metadata"]["corpus_sha256"],
                "spec": effective["embeddings"][model], "seed": effective["seed"], "format_version": 2}
    folder = project_path(cfg, cfg["paths"]["artifact_dir"]) / "embeddings" / dataset / model / fingerprint(identity)[:20]
    return folder, identity


def validate_matrix(matrix, count):
    if matrix.ndim != 2 or matrix.shape[0] != count or matrix.shape[1] < 1:
        raise ValueError(f"Embedding shape {matrix.shape} is incompatible with {count} documents")
    if not np.issubdtype(matrix.dtype, np.floating):
        raise ValueError("Embeddings must be a floating point matrix")
    for start in range(0, count, 4096):
        chunk = matrix[start:start + 4096]
        if not np.isfinite(chunk).all():
            raise ValueError("Embeddings contain non-finite values")
        if np.any(np.linalg.norm(chunk.astype(np.float64), axis=1) == 0):
            raise ValueError("Embeddings contain zero-length vectors")


def read_embeddings(folder, identity, count):
    meta = read_json(folder / "metadata.json")
    if meta["identity"] != identity:
        raise ValueError("Embedding cache identity mismatch")
    if meta["array_sha256"] != file_hash(folder / "embeddings.npy"):
        raise ValueError("Embedding file changed after generation")
    matrix = np.load(folder / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    validate_matrix(matrix, count)
    return matrix


def ensure_embeddings(cfg, dataset, model, corpus, generate=True):
    if generate and cfg.get("execution_profile") == "notebook_2024":
        raise ValueError("notebook_2024 uses verified saved embeddings; use run-saved or import-embeddings")
    folder, identity = embedding_location(cfg, dataset, model, corpus)
    with lock(folder.with_suffix(".lock")):
        if (folder / "metadata.json").exists():
            return read_embeddings(folder, identity, len(corpus["texts"]))
        if not generate:
            raise FileNotFoundError(f"Embeddings missing for {dataset}/{model}; run the embed command first")
        spec = identity["spec"]
        if model == "sbert" and spec["pooling"] != "sentence_transformer":
            raise ValueError("sbert requires sentence_transformer pooling")
        batches = (sentence_batches if model == "sbert" else transformer_batches)(
            corpus["texts"], spec, identity["seed"])
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / "embeddings.partial.npy"
        matrix, offset = None, 0
        try:
            for batch in batches:
                if matrix is None:
                    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                                       shape=(len(corpus["texts"]), batch.shape[1]))
                matrix[offset:offset + len(batch)] = batch
                offset += len(batch)
            if matrix is None or offset != len(corpus["texts"]):
                raise ValueError("Embedding generator returned the wrong number of rows")
            validate_matrix(matrix, offset)
            matrix.flush()
            del matrix
            temporary.replace(folder / "embeddings.npy")
            write_json(folder / "metadata.json", {"identity": identity, "origin": "generated",
                        "array_sha256": file_hash(folder / "embeddings.npy"), "environment": environment()})
        finally:
            temporary.unlink(missing_ok=True)
    return read_embeddings(folder, identity, len(corpus["texts"]))


def import_legacy(cfg, dataset, model, corpus, array_path, texts_path, trust_pickle=False):
    """Import user-owned arrays only after comparing their original processed text order."""
    def read(path):
        path = Path(path)
        if path.suffix in {".pkl", ".pickle"}:
            if not trust_pickle:
                raise ValueError("Pickle can execute code. Only for your own files, pass --trust-pickle")
            with path.open("rb") as stream:
                return pickle.load(stream)
        if path.suffix == ".npy":
            return np.load(path, allow_pickle=False, mmap_mode="r")
        return read_json(path)
    texts = read(texts_path)
    if isinstance(texts, list) and texts and isinstance(texts[0], list):
        texts = [" ".join(tokens) for tokens in texts]
    if texts != corpus["texts"]:
        raise ValueError("Legacy text order/content does not match the prepared corpus; import refused")
    matrix = np.asarray(read(array_path))
    validate_matrix(matrix, len(texts))
    folder, identity = embedding_location(cfg, dataset, model, corpus)
    with lock(folder.with_suffix(".lock")):
        if (folder / "metadata.json").exists():
            raise FileExistsError("A cache already exists; select a different artifact_dir for this import")
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / "embeddings.partial.npy"
        try:
            # Float16 was the input to the original UMAP fits; casting changes its result.
            np.save(temporary, matrix, allow_pickle=False)
            temporary.replace(folder / "embeddings.npy")
            write_json(folder / "metadata.json", {"identity": identity, "origin": "legacy_import",
                        "model_identity_verified": False, "dtype": str(matrix.dtype),
                        "original_array_sha256": file_hash(array_path),
                        "original_texts_sha256": file_hash(texts_path),
                        "array_sha256": file_hash(folder / "embeddings.npy"), "environment": environment()})
        finally:
            temporary.unlink(missing_ok=True)
    return folder
