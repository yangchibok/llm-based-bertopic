"""Small CLI; notebooks call these same functions instead of duplicating logic."""

import argparse
import json

from .config import MODELS, load_config
from .data import prepare_dataset
from .embeddings import ensure_embeddings, import_legacy
from .experiment import aggregate_run, run_experiments


def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM-based BERTopic research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "embed", "run", "all", "import-embeddings", "run-saved"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--root", help="Repository root for relative paths; auto-detected by default")
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Repeatable YAML override")
        if name in {"run", "all", "run-saved"}:
            sub.add_argument("--resume", help="Resume this run with exactly the same config/code/environment")
        if name == "run-saved":
            inputs = sub.add_mutually_exclusive_group()
            inputs.add_argument("--source-dir", help="Original Bert_topic root; input files are only read")
            inputs.add_argument("--embedding-dir", default="inputs", help="Directory containing the dataset's saved arrays")
            texts = sub.add_mutually_exclusive_group()
            texts.add_argument("--raw-texts-json", "--raw-train-json", dest="raw_texts_json",
                               help="Ordered raw texts (NG train; BBC train followed by test)")
            texts.add_argument("--processed-texts-json", help="IMDB only: original ordered processed text list")
            sub.add_argument("--sample-size", type=int, help="Random sample without replacement; default runs full grid")
            sub.add_argument("--sample-seed", type=int, default=20260911)
            sub.add_argument("--trust-pickle", action="store_true")
        if name == "import-embeddings":
            sub.add_argument("--dataset", required=True)
            sub.add_argument("--model", choices=MODELS, required=True)
            sub.add_argument("--array", required=True, help="Existing .npy or trusted .pkl embeddings")
            sub.add_argument("--texts", required=True, help="Original ordered processed text list (.json or .pkl)")
            sub.add_argument("--trust-pickle", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("run_directory")
    commands.add_parser("check-environment")
    args = parser.parse_args(argv)
    if args.command == "check-environment":
        from .saved_inputs import check_environment
        check_environment()
        print("Historical package versions and NLTK English stopwords match.")
        return
    if args.command == "summarize":
        print(aggregate_run(args.run_directory))
        return
    cfg = load_config(args.config, args.set, args.root)
    if args.command == "run-saved":
        from .saved_inputs import run_saved
        dataset = cfg["experiment"]["datasets"][0]
        if args.processed_texts_json and dataset != "imdb":
            parser.error("--processed-texts-json is only for IMDB; NG/BBC require --raw-texts-json")
        if args.raw_texts_json and dataset == "imdb":
            parser.error("IMDB uses saved processed texts; pass --processed-texts-json")
        input_json = args.processed_texts_json or args.raw_texts_json
        if input_json:
            from pathlib import Path
            cfg["datasets"][dataset].update(source="json", path=str(Path(input_json).resolve()))
        print(run_saved(cfg, args.embedding_dir, args.source_dir, args.trust_pickle, args.resume,
                        args.sample_size, args.sample_seed))
        return
    if args.command in {"run", "all"}:
        print(run_experiments(cfg, generate_embeddings=args.command == "all", resume=args.resume))
        return
    if args.command == "import-embeddings":
        corpus = prepare_dataset(cfg, args.dataset, args.model)
        print(import_legacy(cfg, args.dataset, args.model, corpus, args.array, args.texts, args.trust_pickle))
        return
    seen = set()
    for dataset in cfg["experiment"]["datasets"]:
        for model in cfg["experiment"]["models"]:
            if args.command == "embed" and model == "lda":
                continue
            corpus = prepare_dataset(cfg, dataset, model)
            key = corpus["metadata"]["corpus_sha256"]
            if key not in seen:
                print(json.dumps({"dataset": dataset, "documents": len(corpus["texts"]),
                                  "dropped": corpus["metadata"]["dropped_count"],
                                  "corpus_sha256": key}, indent=2))
                seen.add(key)
            if args.command == "embed":
                matrix = ensure_embeddings(cfg, dataset, model, corpus)
                print(f"{dataset}/{model}: {matrix.shape}")


if __name__ == "__main__":
    main()
