#!/usr/bin/env python3
"""
Run complete Wikifier.org wikification pipeline.
This is the paper's ACTUAL implementation (not WAT).
"""

import json
import subprocess
import sys
from pathlib import Path

def run_wikification(dataset_name):
    """Run wikification for a dataset using Wikifier.org API."""

    # Load config
    with open('config/wikifier_config.json', 'r') as f:
        config = json.load(f)

    api_key = config['wikifier_api_key']

    # Define paths
    if dataset_name == 'khanq':
        ready_text = 'data/processed/ready_khanq_text.json'
        ready_question = 'data/processed/ready_khanq_question.json'
        wikified_text = 'data/processed/wikifier/wikified_khanq_text.json'
        wikified_question = 'data/processed/wikifier/wikified_khanq_question.json'
    elif dataset_name == 'squad':
        ready_text = 'data/processed/ready_squad_text.json'
        ready_question = 'data/processed/ready_squad_question.json'
        wikified_text = 'data/processed/wikifier/wikified_squad_text.json'
        wikified_question = 'data/processed/wikifier/wikified_squad_question.json'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print("="*60)
    print(f"Wikifier.org Wikification Pipeline - {dataset_name.upper()}")
    print("="*60)
    print(f"\nUsing Wikifier.org API (paper's actual implementation)")
    print(f"API Key: {api_key[:20]}...")
    print()

    # 1. Wikify texts
    print(f"1. Wikifying {dataset_name} texts with Wikifier.org...")
    subprocess.run([
        sys.executable,
        "scripts/wikify_texts_incremental.py",
        ready_text,
        wikified_text,
        api_key
    ], check=True)

    # 2. Wikify questions
    print(f"\n2. Wikifying {dataset_name} questions with Wikifier.org...")
    subprocess.run([
        sys.executable,
        "scripts/wikify_questions_incremental.py",
        ready_question,
        wikified_question,
        api_key
    ], check=True)

    print(f"\n[SUCCESS] {dataset_name} Wikifier.org wikification complete!")
    print(f"  Text output: {wikified_text}")
    print(f"  Question output: {wikified_question}")
    print()

def main():
    """Main pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Run Wikifier.org wikification pipeline')
    parser.add_argument('--dataset', choices=['khanq', 'squad', 'all'], required=True,
                      help='Dataset to process')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("WIKIFIER.ORG WIKIFICATION PIPELINE")
    print("="*60)
    print("\nThis is the paper's ACTUAL implementation!")
    print("(Not WAT - the paper's code uses Wikifier.org)\n")

    if args.dataset == 'all':
        datasets = ['khanq', 'squad']
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        run_wikification(dataset)

    print("="*60)
    print("Pipeline complete!")
    print("="*60)
    print()

if __name__ == "__main__":
    main()
