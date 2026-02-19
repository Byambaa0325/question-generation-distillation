"""
Parallel WAT wikification for questions/single texts.

Uses multiple parallel workers to speed up API calls.
Reduces processing time from ~30min to ~3-5min for 1000 items.

Usage:
    python scripts/wikify_wat_parallel.py \
        data/processed/ready_khanq_question.json \
        data/processed/wikified_khanq_question.json \
        YOUR_GCUBE_TOKEN \
        --workers 10
"""
import json
import requests
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# WAT API endpoint
WAT_API_URL = "https://wat.d4science.org/wat/tag/tag"

# Paper's parameters
TOP_N_CONCEPTS = 5
RETRY_DELAY = 2
MAX_RETRIES = 3

def call_wat_api(text, token, lang="en", max_retries=MAX_RETRIES):
    """Call WAT API to get Wikipedia annotations."""
    params = {
        "gcube-token": token,
        "text": text,
        "lang": lang,
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(WAT_API_URL, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                annotations = data.get('annotations', [])

                # Sort by rho score (descending)
                annotations_sorted = sorted(
                    annotations,
                    key=lambda x: x.get('rho', 0),
                    reverse=True
                )

                # Keep top N concepts
                top_annotations = annotations_sorted[:TOP_N_CONCEPTS]

                concepts = [
                    {
                        'title': ann['title'],
                        'wiki_id': ann['id'],
                        'rho': ann['rho'],
                        'spot': ann['spot'],
                        'start': ann.get('start'),
                        'end': ann.get('end')
                    }
                    for ann in top_annotations
                ]

                return {
                    'status': 'success',
                    'annotation_data': concepts,
                    'total_found': len(annotations)
                }

            elif response.status_code == 429:  # Rate limit
                if attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'status': f'error: rate limit after {max_retries} retries',
                        'annotation_data': [],
                        'total_found': 0
                    }
            else:
                return {
                    'status': f'error: HTTP {response.status_code}',
                    'annotation_data': [],
                    'total_found': 0
                }

        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                return {
                    'status': 'error: timeout',
                    'annotation_data': [],
                    'total_found': 0
                }
        except Exception as e:
            return {
                'status': f'error: {str(e)}',
                'annotation_data': [],
                'total_found': 0
            }

    return {
        'status': 'error: max retries exceeded',
        'annotation_data': [],
        'total_found': 0
    }

def wikify_item(idx, item, token):
    """Wikify a single item (for parallel processing)."""
    text = item.get('text') or item.get('question') or item.get('context', '')

    if not text:
        return idx, item, {'status': 'error: no text', 'annotation_data': [], 'total_found': 0}

    annotations = call_wat_api(text, token)
    return idx, item, annotations

def load_existing_progress(output_path):
    """Load existing wikified data if available."""
    if Path(output_path).exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            return []
    return []

def wikify_parallel(input_data, output_path, token, workers=10, save_every=100):
    """
    Wikify data in parallel with resume capability.

    Args:
        input_data: List of items to wikify
        output_path: Path to save results
        token: WAT API token
        workers: Number of parallel workers
        save_every: Save progress every N items
    """
    # Load existing progress
    results = load_existing_progress(output_path)

    # Create a set of already processed indices
    processed_indices = set()
    if results:
        # Assume results are in order
        processed_indices = set(range(len(results)))
        print(f"Loading existing progress from {output_path}")
        print(f"  Found {len(results)} already processed items")

    # Determine items to process
    items_to_process = [
        (i, item) for i, item in enumerate(input_data)
        if i not in processed_indices
    ]

    if not items_to_process:
        print("All items already processed!")
        return results

    print(f"Processing {len(items_to_process)} items with {workers} parallel workers...")
    print(f"Resuming from item {len(results) + 1}/{len(input_data)}")

    # Process items in parallel
    errors = 0
    completed = 0

    # Use ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(wikify_item, idx, item, token): (idx, item)
            for idx, item in items_to_process
        }

        # Process completed tasks
        with tqdm(total=len(items_to_process), desc="Wikifying", initial=0) as pbar:
            pending_results = {}

            for future in as_completed(future_to_item):
                idx, item, annotations = future.result()

                # Store result temporarily
                item_copy = item.copy()
                item_copy['annotations'] = annotations
                pending_results[idx] = item_copy

                # Track errors
                if annotations['status'] != 'success':
                    errors += 1

                completed += 1
                pbar.update(1)

                # Save progress when we have enough sequential results
                if completed % save_every == 0:
                    # Add all sequential pending results to main results
                    next_idx = len(results)
                    while next_idx in pending_results:
                        results.append(pending_results.pop(next_idx))
                        next_idx += 1

                    # Save to disk
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

                    pbar.set_postfix({'saved': len(results), 'errors': errors})

            # Add remaining results in order
            next_idx = len(results)
            while next_idx in pending_results:
                results.append(pending_results.pop(next_idx))
                next_idx += 1

    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nWikification complete!")
    print(f"  Total items: {len(results)}")
    print(f"  Errors: {errors}")
    print(f"  Success rate: {100 * (len(results) - errors) / len(results):.1f}%")

    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Parallel WAT wikification')
    parser.add_argument('input_json', help='Input JSON file')
    parser.add_argument('output_json', help='Output JSON file')
    parser.add_argument('gcube_token', help='WAT API gcube-token')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers (default: 10)')
    parser.add_argument('--save-every', type=int, default=100, help='Save progress every N items (default: 100)')

    args = parser.parse_args()

    # Validate token format
    if len(args.gcube_token) < 20:
        print("Error: Invalid gcube-token format")
        sys.exit(1)

    # Load input data
    print(f"Loading data from {args.input_json}...")
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_json}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in input file")
        sys.exit(1)

    print(f"Loaded {len(input_data)} items")

    # Test API
    print("\nTesting WAT API...")
    test_text = input_data[0].get('text') or input_data[0].get('question', 'test')
    test_result = call_wat_api(test_text[:100], args.gcube_token)

    if test_result['status'] != 'success':
        print(f"Error: WAT API test failed: {test_result['status']}")
        print("Please check your gcube-token")
        sys.exit(1)

    print(f"  API test successful!")

    # Run parallel wikification
    print(f"\nConfiguration:")
    print(f"  Workers: {args.workers}")
    print(f"  Save every: {args.save_every} items")
    print(f"  Top N concepts: {TOP_N_CONCEPTS}")
    print()

    results = wikify_parallel(
        input_data,
        args.output_json,
        args.gcube_token,
        workers=args.workers,
        save_every=args.save_every
    )

    print(f"\n[DONE] Wikified data saved to {args.output_json}")

if __name__ == "__main__":
    main()
