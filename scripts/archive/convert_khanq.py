"""
Convert KhanQ format to the format needed for wikification.
Creates two files:
- ready_khanq_text.json: List of contexts with unique IDs
- ready_khanq_question.json: List of questions with unique IDs
"""
import json
import sys
from pathlib import Path

def convert_khanq_to_wikifier_format(khanq_file, output_text, output_question):
    """
    Convert KhanQ dataset to wikifier-ready format.

    Args:
        khanq_file: Path to KhanQ.json file
        output_text: Output path for contexts
        output_question: Output path for questions
    """
    with open(khanq_file, 'r', encoding='utf-8') as f:
        khanq_data = json.load(f)

    text_entries = []
    question_entries = []

    # Process each entry
    for i, entry in enumerate(khanq_data):
        source = entry.get('Source', 'Unknown')
        context = entry.get('Context', '')
        question = entry.get('Question', '')

        # Add context entry
        text_entry = {
            "id": f"khanq_text_{i}",
            "source": source,
            "text": context
        }
        text_entries.append(text_entry)

        # Add question entry
        question_entry = {
            "id": f"khanq_q_{i}",
            "text_id": text_entry["id"],  # Link to the context
            "source": source,
            "text": question
        }
        question_entries.append(question_entry)

    # Save to files
    with open(output_text, 'w', encoding='utf-8') as f:
        json.dump(text_entries, f, indent=2)

    with open(output_question, 'w', encoding='utf-8') as f:
        json.dump(question_entries, f, indent=2)

    print(f"Conversion complete!")
    print(f"  Contexts: {len(text_entries)} entries -> {output_text}")
    print(f"  Questions: {len(question_entries)} entries -> {output_question}")

    return len(text_entries), len(question_entries)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_khanq.py <khanq_json> <output_text> <output_question>")
        sys.exit(1)

    khanq_file = sys.argv[1]
    output_text = sys.argv[2]
    output_question = sys.argv[3]

    convert_khanq_to_wikifier_format(khanq_file, output_text, output_question)
