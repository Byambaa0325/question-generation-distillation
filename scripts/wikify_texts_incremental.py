"""
Wikify text passages with incremental saves for progress tracking.

Thin wrapper — delegates to src.wikification.Wikifier.
Kept for backwards-compatibility; CLI args are unchanged.
"""

import json
import sys

from src.wikification import Wikifier


def main(input_filepath, output_filepath, wikifier_api_key):
    """Main wikification function."""
    print(f"Loading data from {input_filepath}...", flush=True)
    with open(input_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Wikifying {len(data)} entries...", flush=True)
    wikifier = Wikifier(api_key=wikifier_api_key, top_n=5)
    wikifier.annotate_batch(data, output_path=output_filepath, save_every=50, chunk_texts=True)

    print(f"[DONE] Wikifier annotations saved to {output_filepath}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python wikify_texts_incremental.py <input_json> <output_json> <api_key>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
