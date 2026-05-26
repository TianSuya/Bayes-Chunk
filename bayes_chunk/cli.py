import argparse
import json

from .config import load_config
from .algorithms import list_algorithms
from .evaluation import evaluate_records, load_model_and_tokenizer, run_evaluation
from .segmentation import create_segmenter


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bayes-chunk")
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment_parser = subparsers.add_parser("segment", help="Segment a target with Bayes boundaries.")
    segment_parser.add_argument("--config", required=True)
    segment_parser.add_argument("--prompt")
    segment_parser.add_argument("--target")
    segment_parser.add_argument("--dry-run", action="store_true")

    edit_parser = subparsers.add_parser("edit", help="Run editing from a config.")
    edit_parser.add_argument("--config", required=True)
    edit_parser.add_argument("--dry-run", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Run dataset evaluation from a config.")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("list-algorithms", help="List available editing algorithms.")

    args = parser.parse_args(argv)
    if args.command == "list-algorithms":
        print("\n".join(list_algorithms()))
        return 0
    if args.command == "segment":
        return _segment(args)
    if args.command in {"edit", "evaluate"}:
        config = load_config(args.config)
        if args.dry_run:
            print(json.dumps({"config": args.config, "algorithm": config.algorithm.name}, indent=2))
            return 0
        results = run_evaluation(config)
        print(json.dumps(evaluate_records(results), ensure_ascii=False, indent=2))
        return 0
    return 1


def _segment(args) -> int:
    config = load_config(args.config)
    segmenter = create_segmenter(config.segmentation)
    if args.dry_run:
        print(json.dumps({"config": args.config, "segmentation": config.segmentation.__dict__}, indent=2))
        return 0
    if not args.prompt or not args.target:
        raise SystemExit("--prompt and --target are required unless --dry-run is set")
    model, tokenizer = load_model_and_tokenizer(config)
    result = segmenter.segment(model, tokenizer, args.prompt, args.target)
    print(json.dumps({"boundaries": result.boundaries, "segments": [s.__dict__ for s in result.segments]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
