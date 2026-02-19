#!/usr/bin/env python3
"""
Prepare training data from filtered WAT datasets.
Creates train/val/test splits following paper's methodology.
"""

import json
import random
import argparse
from pathlib import Path

def load_filtered_data(filepath):
    """Load filtered enriched data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_splits(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data into train/val/test.
    Paper uses 70/15/15 split.
    """
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test

def prepare_baseline_format(items):
    """
    Baseline format (no topic conditioning).
    Input: context
    Output: question
    """
    prepared = []
    for item in items:
        prepared.append({
            'id': item['id'],
            'input': item['text'],
            'target': item['question']
        })
    return prepared

def prepare_topic_format(items):
    """
    Topic-conditioned format (paper's method).
    Input: topic<sep>context
    Output: question
    """
    prepared = []
    for item in items:
        # Use <sep> as separator (paper's format)
        input_text = f"{item['topic']}<sep>{item['text']}"
        prepared.append({
            'id': item['id'],
            'input': input_text,
            'target': item['question'],
            'topic': item['topic']
        })
    return prepared

def save_split(data, filepath):
    """Save data split to JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} items to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Prepare training data from filtered datasets')
    parser.add_argument('input_file', help='Filtered enriched data JSON')
    parser.add_argument('output_dir', help='Output directory for splits')
    parser.add_argument('--dataset-name', required=True, help='Dataset name (squad/khanq)')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Val split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print("="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)

    # Load data
    print(f"\nLoading data from {args.input_file}...")
    data = load_filtered_data(args.input_file)
    print(f"  Loaded {len(data)} items")

    if len(data) == 0:
        print("ERROR: No data to process!")
        return

    # Create splits
    print(f"\nCreating splits ({args.train_ratio}/{args.val_ratio}/{args.test_ratio})...")
    train, val, test = create_splits(
        data,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")

    # Create output directories
    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / 'baseline'
    topic_dir = output_dir / 'topic'

    baseline_dir.mkdir(parents=True, exist_ok=True)
    topic_dir.mkdir(parents=True, exist_ok=True)

    # Prepare baseline format (no topic)
    print("\nPreparing BASELINE format (context only)...")
    train_baseline = prepare_baseline_format(train)
    val_baseline = prepare_baseline_format(val)
    test_baseline = prepare_baseline_format(test)

    save_split(train_baseline, baseline_dir / f'{args.dataset_name}_train.json')
    save_split(val_baseline, baseline_dir / f'{args.dataset_name}_val.json')
    save_split(test_baseline, baseline_dir / f'{args.dataset_name}_test.json')

    # Prepare topic-conditioned format
    print("\nPreparing TOPIC-CONDITIONED format (topic<sep>context)...")
    train_topic = prepare_topic_format(train)
    val_topic = prepare_topic_format(val)
    test_topic = prepare_topic_format(test)

    save_split(train_topic, topic_dir / f'{args.dataset_name}_train.json')
    save_split(val_topic, topic_dir / f'{args.dataset_name}_val.json')
    save_split(test_topic, topic_dir / f'{args.dataset_name}_test.json')

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Dataset: {args.dataset_name}
Total items: {len(data)}

Splits:
  Train: {len(train)} ({100*len(train)/len(data):.1f}%)
  Val:   {len(val)} ({100*len(val)/len(data):.1f}%)
  Test:  {len(test)} ({100*len(test)/len(data):.1f}%)

Output directories:
  Baseline: {baseline_dir}
  Topic:    {topic_dir}

Ready for fine-tuning!
""")

if __name__ == "__main__":
    main()
