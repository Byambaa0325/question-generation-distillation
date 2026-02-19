"""
Select topics by matching Wikipedia entities between questions and texts.
Selects the concept with highest PageRank when multiple matches exist.
"""
import json
import sys
from pathlib import Path

def find_highest_page_rank(common_titles, annotations):
    """Find title with highest PageRank among common titles."""
    max_page_rank = -1
    selected_title = "NA"

    for annotation in annotations:
        if annotation['title'] in common_titles and annotation['pageRank'] > max_page_rank:
            max_page_rank = annotation['pageRank']
            selected_title = annotation['title']

    return selected_title

def process_dataset(question_file, text_file, output_file):
    """Process wikified data to extract topics."""
    print(f"Loading {question_file}...")
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"Loading {text_file}...")
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = json.load(f)

    # Index texts by ID for fast lookup
    text_index = {}
    for text_item in texts:
        # Handle both single IDs and comma-separated IDs
        text_id = text_item['id']
        ids = text_id.split(', ') if ', ' in text_id else [text_id]
        for id_ in ids:
            text_index[id_] = text_item

    enriched_data = []
    skipped = 0

    print(f"Processing {len(questions)} questions...")
    for i, question in enumerate(questions):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{len(questions)}")

        question_id = question.get('id')
        question_text = question.get('text')
        question_annotations = question.get('annotations', {})

        # Check if there's a text_id field (for datasets like KhanQ)
        text_lookup_id = question.get('text_id', question_id)

        # Skip if no annotations
        if not question_annotations or 'annotation_data' not in question_annotations:
            skipped += 1
            continue

        question_annotation_data = question_annotations['annotation_data']
        if not question_annotation_data:
            skipped += 1
            continue

        # Find matching text using text_id if available, otherwise question_id
        text_item = text_index.get(text_lookup_id)
        if not text_item:
            skipped += 1
            continue

        text_content = text_item.get('text')
        text_annotations = text_item.get('annotations', [])

        if not text_annotations:
            skipped += 1
            continue

        # Extract titles from question and text
        question_titles = {ann['title']: ann['pageRank'] for ann in question_annotation_data}

        # Find common titles
        common_titles = set()
        for text_ann_group in text_annotations:
            for text_ann in text_ann_group.get('annotation_data', []):
                if text_ann['title'] in question_titles:
                    common_titles.add(text_ann['title'])

        # Select topic
        if not common_titles:
            topic = "NA"
        elif len(common_titles) == 1:
            topic = common_titles.pop()
        else:
            # Select title with highest PageRank from question annotations
            topic = max(common_titles, key=lambda t: question_titles[t])

        enriched_entry = {
            "id": question_id,
            "question": question_text,
            "text": text_content,
            "topic": topic
        }
        enriched_data.append(enriched_entry)

    print(f"\nProcessed: {len(enriched_data)} entries")
    print(f"Skipped: {skipped} entries (no annotations or no match)")

    # Save enriched data
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    # Also save filtered version (without "NA" topics)
    filtered_data = [entry for entry in enriched_data if entry['topic'] != "NA"]
    filtered_file = output_file.replace('.json', '_filtered.json')

    print(f"Saving filtered to {filtered_file}...")
    with open(filtered_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults:")
    print(f"  Total entries: {len(enriched_data)}")
    print(f"  With valid topics: {len(filtered_data)}")
    print(f"  Without topics (NA): {len(enriched_data) - len(filtered_data)}")

    return len(enriched_data), len(filtered_data)

def main():
    """Main function."""
    if len(sys.argv) != 4:
        print("Usage: python select_topics.py <question_json> <text_json> <output_json>")
        sys.exit(1)

    question_file = sys.argv[1]
    text_file = sys.argv[2]
    output_file = sys.argv[3]

    process_dataset(question_file, text_file, output_file)
    print("\n[DONE] Topic selection complete!")

if __name__ == "__main__":
    main()
