"""
Create training dataset from enriched data.

Produces:
  Baseline  (SQuAD+):  plain context, 70/15/15 split
  MixSQuAD  (--mix):   10,000 mixed-context pairs with context1/context2/topic2/question2
  MixSQuAD2X (--mix --double): MixSQuAD doubled by reversing context order (~20,000)
  MixKhanQ  (--mix --num-samples 653 --no-split): evaluation set for KhanQ

Paper reference: creat_dataset.ipynb (Topic-controllable-Question-Generator)
"""
import json
import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def filter_by_token_length(data, tokenizer, max_tokens=510):
    """Filter entries where topic + text <= max_tokens. Used for baseline only."""
    filtered, skipped = [], 0
    print(f"Filtering {len(data)} entries by token length (max {max_tokens})...")
    for entry in data:
        combined = entry.get('topic', '') + " " + entry.get('text', '')
        if count_tokens(combined, tokenizer) <= max_tokens:
            filtered.append(entry)
        else:
            skipped += 1
    print(f"  Kept: {len(filtered)}, Filtered out: {skipped}")
    return filtered


# ---------------------------------------------------------------------------
# MixSQuAD creation  (paper: creat_dataset.ipynb cell-2)
# ---------------------------------------------------------------------------

def create_mixed_dataset(data, tokenizer, num_samples=10000, max_tokens=510, seed=42):
    """
    Create MixSQuAD by randomly pairing entries and concatenating contexts.

    Per the paper (creat_dataset.ipynb cell-2 and cell-4):
      - Each pair produces ONE entry: question/topic from entry_i, combined context.
      - Stores context1, context2, question2, topic2 for evaluation (Table 4).
      - Both (topic_i + combined_text) and (topic_j + combined_text) must be
        within max_tokens — the paper checks both (cell-4).
      - Pairs sharing the same text passage are skipped.
      - Sampling continues until num_samples entries are collected.

    Args:
        data:        List of enriched entries (each with id, text, topic, question).
        tokenizer:   T5Tokenizer for token-length checking.
        num_samples: Target number of mixed entries (default 10,000 for MixSQuAD,
                     ~653 for MixKhanQ).
        max_tokens:  Max tokens for topic + combined_text (default 510).
        seed:        Random seed (paper uses 42).

    Returns:
        List of dicts with keys: id, question, topic, text,
                                 context1, context2, question2, topic2.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(data))
    mixed = []
    max_attempts = num_samples * 50  # safety cap

    print(f"Creating MixSQuAD ({num_samples} entries from {len(data)} source entries)...")

    with tqdm(total=num_samples, desc="MixSQuAD") as pbar:
        attempts = 0
        while len(mixed) < num_samples and attempts < max_attempts:
            attempts += 1

            i, j = rng.choice(indices, size=2, replace=False)
            entry_i = data[int(i)]
            entry_j = data[int(j)]

            # Skip pairs from the same context passage
            if entry_i.get('text') == entry_j.get('text'):
                continue

            combined_text = entry_i['text'] + " " + entry_j['text']

            # Paper (cell-4): filter on BOTH topic_i and topic_j vs combined_text
            tok_i = count_tokens(entry_i['topic'] + " " + combined_text, tokenizer)
            tok_j = count_tokens(entry_j['topic'] + " " + combined_text, tokenizer)

            if tok_i <= max_tokens and tok_j <= max_tokens:
                mixed.append({
                    'id':        f"{entry_i['id']}_mix",
                    'question':  entry_i['question'],
                    'topic':     entry_i['topic'],
                    'text':      combined_text,
                    'context1':  entry_i['text'],    # for doubling & evaluation
                    'context2':  entry_j['text'],    # for doubling & evaluation
                    'question2': entry_j['question'],  # alternative reference
                    'topic2':    entry_j['topic'],     # alternative topic (t')
                })
                pbar.update(1)

    if len(mixed) < num_samples:
        print(f"  Warning: collected {len(mixed)}/{num_samples} entries "
              f"after {attempts} attempts")
    else:
        print(f"  Created {len(mixed)} mixed entries ({attempts} attempts)")

    return mixed


# ---------------------------------------------------------------------------
# MixSQuAD2X doubling  (paper: creat_dataset.ipynb cell-5)
# ---------------------------------------------------------------------------

def double_dataset(data):
    """
    Create MixSQuAD2X by reversing the context concatenation order.

    Per the paper (creat_dataset.ipynb cell-5):
        df_copy['text'] = df_copy['context2'] + ' ' + df_copy['context1']

    The same question/topic/question2/topic2 are kept; only the context
    order changes, creating a new training example that forces the model
    to rely on the topic signal rather than position.

    Requires entries to have context1 and context2 fields (produced by
    create_mixed_dataset).
    """
    if not data or ('context1' not in data[0] or 'context2' not in data[0]):
        raise ValueError(
            "double_dataset requires context1/context2 fields. "
            "Run create_mixed_dataset (--mix) first."
        )

    print(f"Doubling dataset by reversing context order...")
    doubled = list(data)
    for entry in data:
        rev = entry.copy()
        rev['id']   = entry['id'] + '_rev'
        rev['text'] = entry['context2'] + " " + entry['context1']  # reversed
        doubled.append(rev)

    print(f"  Doubled: {len(data)} → {len(doubled)} entries")
    return doubled


# ---------------------------------------------------------------------------
# Train / val / test splits
# ---------------------------------------------------------------------------

def create_splits(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
        "Ratios must sum to 1"
    print(f"Creating splits (train={train_ratio}, val={val_ratio}, test={test_ratio})...")

    train_data, temp = train_test_split(
        data, test_size=(val_ratio + test_ratio), random_state=42
    )
    val_data, test_data = train_test_split(
        temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42
    )
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_to_csv(data, output_path):
    """Save dataset to CSV, preserving all columns that exist."""
    df = pd.DataFrame(data)
    # Always include these; include extra columns only if present
    base_cols  = ['text', 'topic', 'question']
    extra_cols = ['context1', 'context2', 'question2', 'topic2']
    cols = base_cols + [c for c in extra_cols if c in df.columns]
    df[cols].to_csv(output_path, index=False)
    print(f"  Saved {len(df)} rows -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Create training/evaluation datasets from enriched JSON.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Baseline SQuAD+ (context-only input):
  python scripts/create_training_dataset.py \\
      --input data/processed/wat/enriched_squad_filtered.json \\
      --output-dir data/training/squad/baseline

  # MixSQuAD — 10,000 mixed-context entries (TopicQG training):
  python scripts/create_training_dataset.py \\
      --input data/processed/wat/enriched_squad_filtered.json \\
      --output-dir data/training/squad/mixsquad --mix

  # MixSQuAD2X — 20,000 entries (TopicQG2X training):
  python scripts/create_training_dataset.py \\
      --input data/processed/wat/enriched_squad_filtered.json \\
      --output-dir data/training/squad/mixsquad2x --mix --double

  # MixKhanQ — evaluation set (~653 entries, no split):
  python scripts/create_training_dataset.py \\
      --input data/processed/wikifier/enriched_khanq_filtered.json \\
      --output-dir data/training/khanq/mixkhanq \\
      --mix --num-samples 653 --no-split
"""
    )
    parser.add_argument('--input',       required=True,
                        help='Enriched filtered JSON file (enriched_*_filtered.json)')
    parser.add_argument('--output-dir',  default='data/training',
                        help='Output directory')
    parser.add_argument('--max-tokens',  type=int, default=510,
                        help='Max tokens for topic + text (default: 510)')
    parser.add_argument('--mix',         action='store_true',
                        help='Create MixSQuAD (randomly paired, mixed contexts)')
    parser.add_argument('--double',      action='store_true',
                        help='Double by reversing context order (MixSQuAD2X). '
                             'Requires --mix.')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Target mixed entries (default 10000; use 653 for MixKhanQ)')
    parser.add_argument('--no-split',    action='store_true',
                        help='Save as single data.csv instead of train/val/test')
    parser.add_argument('--seed',        type=int, default=42)

    args = parser.parse_args()

    if args.double and not args.mix:
        parser.error("--double requires --mix")

    # Load tokenizer
    print("Loading T5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

    # Load source data
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded: {len(data)} entries")

    # Build dataset
    if args.mix:
        # Token filtering happens inside create_mixed_dataset (post-mixing)
        data = create_mixed_dataset(
            data, tokenizer, args.num_samples, args.max_tokens, args.seed
        )
    else:
        # Baseline: filter by token length before splitting
        data = filter_by_token_length(data, tokenizer, args.max_tokens)

    if args.double:
        data = double_dataset(data)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    if args.no_split:
        save_to_csv(data, os.path.join(args.output_dir, 'data.csv'))
    else:
        train_data, val_data, test_data = create_splits(data)
        save_to_csv(train_data, os.path.join(args.output_dir, 'train.csv'))
        save_to_csv(val_data,   os.path.join(args.output_dir, 'val.csv'))
        save_to_csv(test_data,  os.path.join(args.output_dir, 'test.csv'))

    print(f"\n[DONE] Dataset saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
