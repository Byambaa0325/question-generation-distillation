"""
Convert SQuAD 1.1 format to the format needed for wikification.
Creates two files:
- ready_squad_text.json: List of contexts with unique IDs
- ready_squad_question.json: List of questions with unique IDs
"""
import json
import sys
from pathlib import Path

def convert_squad_to_wikifier_format(squad_file, output_text, output_question):
    """
    Convert SQuAD dataset to wikifier-ready format.

    Args:
        squad_file: Path to SQuAD JSON file (train-v1.1.json or dev-v1.1.json)
        output_text: Output path for contexts
        output_question: Output path for questions
    """
    with open(squad_file, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)

    text_entries = []
    question_entries = []

    text_id_counter = 0
    question_id_counter = 0

    # Process each article
    for article in squad_data['data']:
        title = article['title']

        # Process each paragraph
        for paragraph in article['paragraphs']:
            context = paragraph['context']

            # Add context entry (one per unique context)
            text_entry = {
                "id": f"text_{text_id_counter}",
                "title": title,
                "text": context
            }
            text_entries.append(text_entry)
            text_id = text_entry["id"]
            text_id_counter += 1

            # Add question entries
            for qa in paragraph['qas']:
                question_entry = {
                    "id": qa['id'],
                    "text_id": text_id,  # Link to the context
                    "title": title,
                    "text": qa['question']
                }
                question_entries.append(question_entry)
                question_id_counter += 1

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
        print("Usage: python convert_squad.py <squad_json> <output_text> <output_question>")
        sys.exit(1)

    squad_file = sys.argv[1]
    output_text = sys.argv[2]
    output_question = sys.argv[3]

    convert_squad_to_wikifier_format(squad_file, output_text, output_question)
