"""
Wikify questions/contexts using WAT (Wikipedia Annotation Tool) API.

This matches the paper's methodology:
- Uses WAT API from https://sobigdata.d4science.org/group/tagme/wat-help
- Retains top 5 concepts based on rho score (confidence/authority)
- Incremental saves with resume capability

Reference [45] from paper: WAT wikification tool

Usage:
    python scripts/wikify_wat_incremental.py \\
        data/processed/ready_khanq_question.json \\
        data/processed/wikified_khanq_question.json \\
        YOUR_GCUBE_TOKEN

To get a gcube-token:
1. Visit https://sobigdata.d4science.org/
2. Register/login
3. Get your token from the WAT help page

To resume after interruption, run the same command.
"""
import json
import requests
import time
import sys
from pathlib import Path

# WAT API endpoint
WAT_API_URL = "https://wat.d4science.org/wat/tag/tag"

# Paper's parameters
TOP_N_CONCEPTS = 5  # Paper uses top 5 concepts
SAVE_EVERY = 100  # Save progress every N items
RETRY_DELAY = 2  # Seconds to wait on error

def call_wat_api(text, token, lang="en", max_retries=3):
    """
    Call WAT API to get Wikipedia annotations.

    Args:
        text: Input text to annotate
        token: gcube-token from sobigdata.d4science.org
        lang: Language code (default: en)
        max_retries: Maximum retry attempts

    Returns:
        dict with 'annotations' and 'status' fields
    """
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

                # Extract and sort annotations by rho (confidence/authority)
                annotations = data.get('annotations', [])

                # Sort by rho score (descending) to get most confident/authoritative concepts
                annotations_sorted = sorted(
                    annotations,
                    key=lambda x: x.get('rho', 0),
                    reverse=True
                )

                # Keep top N concepts as per paper
                top_annotations = annotations_sorted[:TOP_N_CONCEPTS]

                # Extract relevant fields
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
                    print(f"  Rate limit hit, waiting {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'status': f'error: rate limit after {max_retries} retries',
                        'annotation_data': []
                    }

            else:
                return {
                    'status': f'error: HTTP {response.status_code}',
                    'annotation_data': []
                }

        except requests.Timeout:
            if attempt < max_retries - 1:
                print(f"  Timeout, retrying... (attempt {attempt + 1}/{max_retries})", flush=True)
                time.sleep(RETRY_DELAY)
                continue
            else:
                return {
                    'status': 'error: timeout',
                    'annotation_data': []
                }

        except Exception as e:
            return {
                'status': f'error: {str(e)}',
                'annotation_data': []
            }

    return {
        'status': 'error: max retries exceeded',
        'annotation_data': []
    }

def load_existing_progress(output_path):
    """Load existing wikified data if available."""
    if Path(output_path).exists():
        print(f"Loading existing progress from {output_path}")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  Found {len(data)} already processed items")
            return data
        except json.JSONDecodeError:
            print("  Warning: Existing file is corrupted, starting fresh")
            return []
    return []

def wikify_incremental(input_data, output_path, token, start_from=0):
    """
    Wikify data incrementally with resume capability.

    Args:
        input_data: List of items to wikify
        output_path: Path to save results
        token: WAT API token
        start_from: Index to start from (for resuming)

    Returns:
        List of wikified items
    """
    # Load existing progress
    results = load_existing_progress(output_path)

    # Determine where to start
    if len(results) > 0:
        start_idx = len(results)
        print(f"Resuming from item {start_idx + 1}/{len(input_data)}")
    else:
        start_idx = start_from
        print(f"Starting wikification of {len(input_data)} items")

    # Process remaining items
    total = len(input_data)
    errors = 0

    for i in range(start_idx, total):
        item = input_data[i]

        # Get text to wikify
        text = item.get('text') or item.get('question') or item.get('context', '')

        if not text:
            print(f"Warning: Item {i} has no text, skipping")
            continue

        # Call WAT API
        annotations = call_wat_api(text, token)

        # Add annotations to item
        item_copy = item.copy()
        item_copy['annotations'] = annotations
        results.append(item_copy)

        # Track errors
        if annotations['status'] != 'success':
            errors += 1
            if errors % 10 == 0:
                print(f"  Warning: {errors} errors so far", flush=True)

        # Save progress periodically
        if (i + 1) % SAVE_EVERY == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Progress: {i + 1}/{total} items processed and saved", flush=True)

        # Small delay to be nice to API
        time.sleep(0.01)

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
    if len(sys.argv) != 4:
        print("Usage: python wikify_wat_incremental.py <input_json> <output_json> <gcube_token>")
        print("\nExample:")
        print("  python scripts/wikify_wat_incremental.py \\")
        print("    data/processed/ready_khanq_question.json \\")
        print("    data/processed/wikified_khanq_question.json \\")
        print("    YOUR_GCUBE_TOKEN")
        print("\nTo get a gcube-token:")
        print("  1. Visit https://sobigdata.d4science.org/")
        print("  2. Register/login")
        print("  3. Get token from WAT help page")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    token = sys.argv[3]

    # Validate token format (basic check)
    if len(token) < 20:
        print("Error: Invalid gcube-token format")
        print("Expected format: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX-XXXXXXXXX")
        sys.exit(1)

    # Load input data
    print(f"Loading data from {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in input file")
        sys.exit(1)

    print(f"Loaded {len(input_data)} items")

    # Test API with first item
    print("\nTesting WAT API...")
    test_text = input_data[0].get('text') or input_data[0].get('question', 'test')
    test_result = call_wat_api(test_text[:100], token)

    if test_result['status'] != 'success':
        print(f"Error: WAT API test failed: {test_result['status']}")
        print("Please check your gcube-token")
        sys.exit(1)

    print(f"  API test successful! Found {len(test_result['annotation_data'])} concepts")
    if test_result['annotation_data']:
        print(f"  Top concept: {test_result['annotation_data'][0]['title']}")

    # Run wikification
    print(f"\nStarting wikification...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Top N concepts: {TOP_N_CONCEPTS}")
    print(f"  Save every: {SAVE_EVERY} items")
    print()

    results = wikify_incremental(input_data, output_path, token)

    print(f"\n[DONE] Wikified data saved to {output_path}")
    print(f"\nTo resume if interrupted, run the same command again.")

if __name__ == "__main__":
    main()
