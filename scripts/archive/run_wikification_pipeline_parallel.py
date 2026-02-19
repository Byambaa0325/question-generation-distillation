"""
Master script to run the complete WAT wikification pipeline with PARALLEL processing.
Can run in test mode (small sample) or full mode.

Uses parallel workers to speed up processing by ~10x.
"""
import json
import subprocess
import sys
from pathlib import Path

def create_sample(input_file, output_file, sample_size=10):
    """Create a small sample for testing."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sample = data[:sample_size]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2)

    print(f"Created sample: {sample_size} entries -> {output_file}")
    return len(sample)

def run_wikification(dataset_name, test_mode=True, workers=10):
    """Run parallel WAT wikification for a specific dataset."""
    # Load WAT token
    with open('config/wat_config.json', 'r') as f:
        config = json.load(f)
    gcube_token = config['gcube_token']
    workers = config.get('parallel_workers', workers)

    # Define paths
    if test_mode:
        mode_suffix = "_test"
        print(f"\n{'='*60}")
        print(f"Running in TEST MODE - Processing sample of {dataset_name}")
        print(f"Using PARALLEL processing with {workers} workers")
        print(f"{'='*60}\n")
    else:
        mode_suffix = ""
        print(f"\n{'='*60}")
        print(f"Running in FULL MODE - Processing complete {dataset_name}")
        print(f"Using PARALLEL processing with {workers} workers")
        print(f"{'='*60}\n")

    # Input files
    ready_text = f"data/processed/ready_{dataset_name}_text{mode_suffix}.json"
    ready_question = f"data/processed/ready_{dataset_name}_question{mode_suffix}.json"

    # Output files
    wikified_text = f"data/processed/wikified_{dataset_name}_text{mode_suffix}.json"
    wikified_question = f"data/processed/wikified_{dataset_name}_question{mode_suffix}.json"

    # Create test samples if in test mode
    if test_mode:
        original_text = f"data/processed/ready_{dataset_name}_text.json"
        original_question = f"data/processed/ready_{dataset_name}_question.json"
        create_sample(original_text, ready_text, sample_size=10)
        create_sample(original_question, ready_question, sample_size=10)

    # Run PARALLEL WAT wikification for texts
    print(f"\n1. Wikifying {dataset_name} texts with WAT (PARALLEL)...")
    subprocess.run([
        sys.executable,
        "scripts/wikify_texts_wat_parallel.py",
        ready_text,
        wikified_text,
        gcube_token,
        "--workers", str(workers)
    ], check=True)

    # Run PARALLEL WAT wikification for questions
    print(f"\n2. Wikifying {dataset_name} questions with WAT (PARALLEL)...")
    subprocess.run([
        sys.executable,
        "scripts/wikify_wat_parallel.py",
        ready_question,
        wikified_question,
        gcube_token,
        "--workers", str(workers)
    ], check=True)

    print(f"\n[SUCCESS] {dataset_name} WAT wikification complete!")
    print(f"  Text output: {wikified_text}")
    print(f"  Question output: {wikified_question}")
    print(f"  Using WAT API (paper's methodology) with {workers} parallel workers")

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Run parallel wikification pipeline')
    parser.add_argument('--dataset', choices=['squad', 'khanq', 'all'], default='all',
                      help='Which dataset to process')
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                      help='Test mode (small sample) or full mode')
    parser.add_argument('--workers', type=int, default=10,
                      help='Number of parallel workers (default: 10)')

    args = parser.parse_args()

    test_mode = (args.mode == 'test')

    if args.dataset == 'all':
        datasets = ['khanq', 'squad']
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        try:
            run_wikification(dataset, test_mode=test_mode, workers=args.workers)
        except Exception as e:
            print(f"\n[ERROR] Error processing {dataset}: {e}")
            continue

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
