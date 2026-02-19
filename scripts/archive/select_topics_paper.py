#!/usr/bin/env python3
"""
Topic selection following paper's exact methodology.

Finds matching Wikipedia concepts between question and text.
Assigns "NA" if no overlap, or selects highest pageRank/rho if overlap exists.
"""

import json
import argparse
from pathlib import Path

def find_best_matching_concept(question_annotations, text_annotations, score_field='pageRank'):
    """
    Find best matching concept between question and text.

    Args:
        question_annotations: Annotations from question
        text_annotations: Annotations from text (may be list of chunks)
        score_field: 'pageRank' for Wikifier, 'rho' for WAT

    Returns:
        (topic_title, score) or ("NA", 0) if no overlap
    """
    # Extract question titles
    question_titles = {}
    if isinstance(question_annotations, dict):
        ann_data = question_annotations.get('annotation_data', [])
        for ann in ann_data:
            title = ann.get('title')
            score = ann.get(score_field, 0)
            if title:
                question_titles[title] = score

    # Extract text titles (handle chunked format)
    text_title_set = set()
    if isinstance(text_annotations, list):
        # Chunked format (for texts)
        for chunk in text_annotations:
            if isinstance(chunk, dict):
                ann_data = chunk.get('annotation_data', [])
                for ann in ann_data:
                    title = ann.get('title')
                    if title:
                        text_title_set.add(title)
    elif isinstance(text_annotations, dict):
        # Single annotation (for questions)
        ann_data = text_annotations.get('annotation_data', [])
        for ann in ann_data:
            title = ann.get('title')
            if title:
                text_title_set.add(title)

    # Find overlapping titles
    common_titles = []
    for title, q_score in question_titles.items():
        if title in text_title_set:
            common_titles.append((title, q_score))

    if not common_titles:
        return ("NA", 0)

    # Select highest scoring concept from question
    best_title, best_score = max(common_titles, key=lambda x: x[1])
    return (best_title, best_score)

def select_topics(question_file, text_file, output_file, score_field='pageRank'):
    """
    Select topics from wikified data.

    Paper's methodology:
    1. For each question-text pair
    2. Find concepts that appear in BOTH question and text
    3. If no overlap: topic = "NA"
    4. If overlap: topic = highest scoring concept from question
    """
    print(f"\nLoading data...")
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    with open(text_file, 'r', encoding='utf-8') as f:
        texts = json.load(f)

    print(f"  Questions: {len(questions)}")
    print(f"  Texts: {len(texts)}")

    # Create text index by ID
    text_index = {}
    for text_item in texts:
        text_id = text_item.get('id')
        if text_id:
            text_index[text_id] = text_item

    # Process each question
    enriched_data = []
    na_count = 0

    for i, question in enumerate(questions):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{len(questions)}")

        question_id = question.get('id')
        question_text = question.get('text')
        text_id = question.get('text_id')

        # Get corresponding text
        text_item = text_index.get(text_id)
        if not text_item:
            # Skip if no matching text
            continue

        context_text = text_item.get('text')

        # Get annotations
        question_anns = question.get('annotations', {})
        text_anns = text_item.get('annotations', [])

        # Find best matching topic
        topic, score = find_best_matching_concept(
            question_anns,
            text_anns,
            score_field=score_field
        )

        if topic == "NA":
            na_count += 1

        # Create enriched entry
        enriched_entry = {
            "id": question_id,
            "text_id": text_id,
            "question": question_text,
            "text": context_text,
            "topic": topic,
            f"topic_{score_field}": score
        }
        enriched_data.append(enriched_entry)

    # Save enriched data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults:")
    print(f"  Total items: {len(enriched_data)}")
    print(f"  With topics: {len(enriched_data) - na_count} ({100*(len(enriched_data)-na_count)/len(enriched_data):.1f}%)")
    print(f"  Without topics (NA): {na_count} ({100*na_count/len(enriched_data):.1f}%)")
    print(f"\nSaved to: {output_file}")

def filter_na_topics(input_file, output_file):
    """
    Filter out items with topic = "NA".
    Paper's methodology: Only keep items with at least 1 overlapping concept.
    """
    print(f"\nFiltering out NA topics...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    before_count = len(data)
    filtered_data = [item for item in data if item.get('topic') != "NA"]
    after_count = len(filtered_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"  Before: {before_count}")
    print(f"  After: {after_count}")
    print(f"  Removed: {before_count - after_count} ({100*(before_count-after_count)/before_count:.1f}%)")
    print(f"\nSaved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Select topics from wikified data (paper methodology)'
    )
    parser.add_argument('question_file', help='Wikified questions JSON')
    parser.add_argument('text_file', help='Wikified texts JSON')
    parser.add_argument('output_file', help='Output enriched data JSON')
    parser.add_argument('--score-field', default='pageRank',
                       choices=['pageRank', 'rho'],
                       help='Score field: pageRank (Wikifier) or rho (WAT)')
    parser.add_argument('--filter', action='store_true',
                       help='Also create filtered version (removes NA topics)')

    args = parser.parse_args()

    print("="*60)
    print("TOPIC SELECTION (Paper Methodology)")
    print("="*60)
    print(f"\nScore field: {args.score_field}")
    print(f"  (pageRank = Wikifier.org, rho = WAT)")

    # Select topics
    select_topics(
        args.question_file,
        args.text_file,
        args.output_file,
        score_field=args.score_field
    )

    # Optionally filter out NA
    if args.filter:
        filtered_output = args.output_file.replace('.json', '_filtered.json')
        filter_na_topics(args.output_file, filtered_output)

    print("\n" + "="*60)
    print("Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
