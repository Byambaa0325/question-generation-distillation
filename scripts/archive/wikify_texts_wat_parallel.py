"""
Parallel WAT wikification for text passages with chunking.

Uses multiple parallel workers to speed up API calls.
Reduces processing time from ~60min to ~5-8min for 1000 texts.

Usage:
    python scripts/wikify_texts_wat_parallel.py \
        data/processed/ready_khanq_text.json \
        data/processed/wikified_khanq_text.json \
        YOUR_GCUBE_TOKEN \
        --workers 10
"""
import json
import requests
import time
import sys
import re
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
MAX_CHUNK_SIZE = 650

def segment_sentences(text):
    """Simple sentence segmentation."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def partition_text(text, max_size):
    """Partition text into chunks not exceeding max_size."""
    sentences = segment_sentences(text)
    chunks = []
    temp_sents = []
    temp_len = 0
    sentence_aggregator = " "
    len_aggr = len(sentence_aggregator)

    for sentence in sentences:
        len_sentence = len(sentence)
        expected_len = temp_len + len_aggr + len_sentence

        if expected_len > max_size:
            if len(temp_sents) > 0:
                chunks.append(sentence_aggregator.join(temp_sents))
                temp_sents = []
                temp_len = 0

        temp_sents.append(sentence)
        temp_len += len_sentence

    if len(temp_sents) > 0:
        chunks.append(sentence_aggregator.join(temp_sents))

    return chunks

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

def wikify_chunk(chunk_data):
    """Wikify a single text chunk (for parallel processing)."""
    item_idx, chunk_idx, chunk_text, token = chunk_data
    annotations = call_wat_api(chunk_text, token)
    return (item_idx, chunk_idx, annotations)

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

def wikify_parallel(input_data, output_path, token, workers=10, save_every=50):
    """
    Wikify text data in parallel with resume capability.

    Args:
        input_data: List of items to wikify
        output_path: Path to save results
        token: WAT API token
        workers: Number of parallel workers
        save_every: Save progress every N items
    """
    # Load existing progress
    results = load_existing_progress(output_path)

    # Determine items already processed
    processed_indices = set()
    if results:
        processed_indices = set(range(len(results)))
        print(f"Loading existing progress from {output_path}")
        print(f"  Found {len(results)} already processed items")

    # Prepare chunks for items that need processing
    items_to_process = [
        (i, item) for i, item in enumerate(input_data)
        if i not in processed_indices
    ]

    if not items_to_process:
        print("All items already processed!")
        return results

    print(f"Processing {len(items_to_process)} items with {workers} parallel workers...")
    print(f"Resuming from item {len(results) + 1}/{len(input_data)}")

    # Create chunk tasks
    chunk_tasks = []
    for item_idx, item in items_to_process:
        text = item.get('text', '')
        if not text:
            continue

        # Partition into chunks
        chunks = partition_text(text, MAX_CHUNK_SIZE)

        # Create task for each chunk
        for chunk_idx, chunk in enumerate(chunks):
            chunk_tasks.append((item_idx, chunk_idx, chunk, token))

    print(f"Total chunks to process: {len(chunk_tasks)}")

    # Process chunks in parallel
    errors = 0
    item_annotations = {}  # {item_idx: {chunk_idx: annotations}}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_chunk = {
            executor.submit(wikify_chunk, chunk_data): chunk_data
            for chunk_data in chunk_tasks
        }

        with tqdm(total=len(chunk_tasks), desc="Wikifying chunks") as pbar:
            for future in as_completed(future_to_chunk):
                item_idx, chunk_idx, annotations = future.result()

                # Store chunk annotation
                if item_idx not in item_annotations:
                    item_annotations[item_idx] = {}
                item_annotations[item_idx][chunk_idx] = annotations

                # Track errors
                if annotations['status'] != 'success':
                    errors += 1

                pbar.update(1)

    # Assemble results in order
    print("\nAssembling results...")
    for item_idx, item in items_to_process:
        if item_idx not in item_annotations:
            # No chunks processed (empty text)
            item_copy = item.copy()
            item_copy['annotations'] = []
            results.append(item_copy)
            continue

        # Sort chunks by index and create annotation list
        chunks_dict = item_annotations[item_idx]
        sorted_chunks = sorted(chunks_dict.items())
        chunk_annotations = [ann for _, ann in sorted_chunks]

        item_copy = item.copy()
        item_copy['annotations'] = chunk_annotations
        results.append(item_copy)

        # Save periodically
        if len(results) % save_every == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Progress: {len(results)}/{len(input_data)} items saved")

    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nWikification complete!")
    print(f"  Total items: {len(results)}")
    print(f"  Chunk errors: {errors}")
    total_chunks = sum(len(item.get('annotations', [])) for item in results)
    print(f"  Success rate: {100 * (total_chunks - errors) / total_chunks:.1f}%")

    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Parallel WAT wikification for texts')
    parser.add_argument('input_json', help='Input JSON file')
    parser.add_argument('output_json', help='Output JSON file')
    parser.add_argument('gcube_token', help='WAT API gcube-token')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers (default: 10)')
    parser.add_argument('--save-every', type=int, default=50, help='Save progress every N items (default: 50)')

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
    test_text = input_data[0].get('text', 'test')[:100]
    test_result = call_wat_api(test_text, args.gcube_token)

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
    print(f"  Max chunk size: {MAX_CHUNK_SIZE} chars")
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
