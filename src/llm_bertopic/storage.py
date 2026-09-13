"""Atomic artifacts, ordered corpus fingerprints and run metadata."""

import hashlib
import json
import os
import platform
import tempfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Artifact is in use: {path}. Remove a stale lock only after its process exits.") from exc
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def environment():
    names = ["llm-based-bertopic", "numpy", "scipy", "pandas", "scikit-learn", "bertopic",
             "umap-learn", "hdbscan", "gensim", "torch", "transformers", "sentence-transformers",
             "datasets", "spacy", "en-core-web-sm", "nltk", "bitsandbytes", "accelerate",
             "octis", "numba", "llvmlite", "pynndescent", "joblib", "threadpoolctl", "huggingface-hub"]
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}
