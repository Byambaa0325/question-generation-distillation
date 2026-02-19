#!/usr/bin/env python3
"""
pipeline.py — CLI wrapper for the topic-QG pipeline.

All subcommands delegate to src.pipeline.Pipeline.

Examples
--------
  python pipeline.py status
  python pipeline.py convert --dataset squad
  python pipeline.py wikify  --dataset squad --tool wikifier
  python pipeline.py topics  --dataset squad
  python pipeline.py dataset --dataset squad --mode mixsquad
  python pipeline.py train   --mode topic
  python pipeline.py evaluate
  python pipeline.py generate --topic "Photosynthesis" --context "Plants convert..."
  python pipeline.py run --dataset squad
"""

import argparse
import sys

# Load .env before any other imports so API keys are available pipeline-wide
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Topic-controlled question generation pipeline",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline YAML config (default: config/pipeline.yaml)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ status
    sub.add_parser("status", help="Show which pipeline outputs exist")

    # ---------------------------------------------------------------- convert
    p_conv = sub.add_parser("convert", help="Stage 1: convert raw datasets to JSON")
    p_conv.add_argument(
        "--dataset", default="all", choices=["squad", "khanq", "all"],
        help="Dataset(s) to convert (default: all)",
    )

    # ------------------------------------------------------------------ wikify
    p_wik = sub.add_parser("wikify", help="Stage 2: annotate texts/questions")
    p_wik.add_argument("--dataset", default="all", choices=["squad", "khanq", "all"])
    p_wik.add_argument("--tool", choices=["wikifier", "wat"],
                       help="Override wikification tool")
    p_wik.add_argument("--target", default="all", choices=["texts", "questions", "all"])

    # ------------------------------------------------------------------ topics
    p_top = sub.add_parser("topics", help="Stage 3: select best topic per QA pair")
    p_top.add_argument("--dataset", default="all", choices=["squad", "khanq", "all"])
    p_top.add_argument("--tool", choices=["wikifier", "wat"])

    # ----------------------------------------------------------------- dataset
    p_ds = sub.add_parser("dataset", help="Stage 4: build train/val/test CSVs")
    p_ds.add_argument("--dataset", default="squad", choices=["squad", "khanq"])
    p_ds.add_argument(
        "--mode", default="mixsquad",
        choices=["baseline", "mixsquad", "mixsquad2x", "mixkhanq"],
    )
    p_ds.add_argument("--tool", choices=["wikifier", "wat"])

    # ------------------------------------------------------------------- train
    p_tr = sub.add_parser("train", help="Stage 5: fine-tune T5-small")
    p_tr.add_argument(
        "--mode", default="topic", choices=["baseline", "topic", "topic2x"],
    )
    p_tr.add_argument("--dataset", default="squad")

    # ---------------------------------------------------------------- evaluate
    p_ev = sub.add_parser("evaluate", help="Stage 6: full evaluation")
    p_ev.add_argument("--models", default="all",
                      help="Comma-separated model keys or 'all'")
    p_ev.add_argument("--dataset", default="all", choices=["squad", "khanq", "all"])
    p_ev.add_argument("--output-dir")
    p_ev.add_argument("--tool", choices=["wikifier", "wat"])

    # ---------------------------------------------------------------- generate
    p_gen = sub.add_parser("generate", help="Stage 7: generate a single question")
    p_gen.add_argument("--topic", required=True, help="Wikipedia concept title")
    p_gen.add_argument("--context", required=True, help="Source passage")
    p_gen.add_argument("--model-path", help="Explicit model directory")
    p_gen.add_argument(
        "--mode", default="topic", choices=["baseline", "topic", "topic2x"],
    )
    p_gen.add_argument("--num-beams", type=int, default=10)
    p_gen.add_argument("--num-return", type=int, default=8)

    # --------------------------------------------------------------------- run
    p_run = sub.add_parser("run", help="Run the full pipeline end-to-end")
    p_run.add_argument("--dataset", default="squad", choices=["squad", "khanq", "all"])
    p_run.add_argument(
        "--skip", nargs="*", default=[],
        choices=["convert", "wikify", "topics", "dataset", "train", "evaluate"],
        help="Stages to skip",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from src.pipeline import Pipeline

    pipe = Pipeline(config_path=args.config)
    cmd = args.command

    if cmd == "status":
        pipe.status()

    elif cmd == "convert":
        pipe.convert(dataset=args.dataset)

    elif cmd == "wikify":
        pipe.wikify(dataset=args.dataset, tool=args.tool, target=args.target)

    elif cmd == "topics":
        pipe.topics(dataset=args.dataset, tool=args.tool)

    elif cmd == "dataset":
        pipe.dataset(dataset=args.dataset, mode=args.mode, tool=args.tool)

    elif cmd == "train":
        result = pipe.train(mode=args.mode, dataset=args.dataset)
        print(f"Model saved to: {result}")

    elif cmd == "evaluate":
        pipe.evaluate(
            models=args.models,
            dataset=args.dataset,
            output_dir=args.output_dir,
            tool=args.tool,
        )

    elif cmd == "generate":
        q = pipe.generate(
            topic=args.topic,
            context=args.context,
            model_path=args.model_path,
            mode=args.mode,
            num_beams=args.num_beams,
            num_return=args.num_return,
        )
        print(f"\nGenerated question: {q}")

    elif cmd == "run":
        pipe.run(dataset=args.dataset, skip=args.skip)


if __name__ == "__main__":
    main()
