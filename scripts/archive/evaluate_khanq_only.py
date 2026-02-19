#!/usr/bin/env python3
"""
KhanQ evaluation with concise zero-shot prompt.
"""

import json
import torch
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import time

# Concise prompt for zero-shot
ZERO_SHOT_PROMPT_TEMPLATE = """Generate ONE scientific question about the topic from the paragraph.

Paragraph: {context}

Topic: {topic}

CRITICAL Requirements:
- Length: 8-12 words MAXIMUM
- Start with: "How", "Why", "Does", "Do", "What", "Is", "Would"
- Use direct, simple scientific language
- Focus on the SPECIFIC topic given
- Ask about relationships, comparisons, or mechanisms

Question Patterns to Follow:
1. "Does X [verb] Y?" - Compare or question relationships
2. "How does X [affect/relate to] Y?" - Ask about mechanisms
3. "Why does X [happen/occur]?" - Ask for explanations
4. "What is [concept]?" - Ask definitions
5. "How [verb] X?" - Ask about processes

EXAMPLES (notice the brevity and directness):

Context: "Electronegativity is how strongly the element hogs electrons once bonded."
Topic: "Electronegativity"
Question: "Do electronegativity and electropotential both describe electron attraction?"

Context: "Lithium has high ionization potential making it a strong reducing agent."
Topic: "reducing agent"
Question: "How does lithium behave as a strong reducing agent?"

Context: "The viscosity of liquids generally decreases with temperature."
Topic: "Viscosity"
Question: "How will viscosity be affected by temperature increase?"

Context: "Water molecules can be classified by their state of matter."
Topic: "Water"
Question: "Does one classify H2O molecules as solid or liquid?"

Output ONLY the question (8-12 words):"""

def load_model(model_path, device):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"[OK] Model loaded on {device}")
    return model, tokenizer

def prepare_zero_shot_input(topic, context):
    """Prepare input for zero-shot with concise prompt."""
    return ZERO_SHOT_PROMPT_TEMPLATE.format(context=context.strip(), topic=topic.strip())

def generate_questions(model, tokenizer, test_data, device, is_zero_shot=False,
                      max_input_length=512, max_output_length=64):
    """Generate questions for test data."""
    predictions = []
    references = []

    print(f"\nGenerating questions for {len(test_data)} examples...")
    print(f"  Zero-shot: {is_zero_shot}")
    print(f"  Max output length: {max_output_length}")

    with torch.no_grad():
        for item in tqdm(test_data):
            # Prepare input
            if is_zero_shot:
                input_text = prepare_zero_shot_input(item['topic'], item['text'])
            else:
                # Fine-tuned models use topic<sep>context
                input_text = f"{item['topic']}<sep>{item['text']}"

            # Tokenize input
            input_ids = tokenizer(
                input_text,
                max_length=max_input_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids.to(device)

            # Generate
            outputs = model.generate(
                input_ids,
                max_length=max_output_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )

            # Decode
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ref = item['question']

            predictions.append(pred)
            references.append(ref)

    return predictions, references

def calculate_bleu(predictions, references):
    """Calculate BLEU score."""
    # Tokenize for BLEU
    predictions_tokenized = [pred.split() for pred in predictions]
    references_tokenized = [[ref.split()] for ref in references]

    # Calculate BLEU-4
    bleu_score = corpus_bleu(references_tokenized, predictions_tokenized)
    return bleu_score * 100  # Convert to percentage

def evaluate_model(model_path, test_data, device, model_name, is_zero_shot=False):
    """Evaluate a model on test data."""
    print("\n" + "="*70)
    print(f"EVALUATING: {model_name}")
    print("="*70)

    model, tokenizer = load_model(model_path, device)

    predictions, references = generate_questions(
        model, tokenizer, test_data, device, is_zero_shot=is_zero_shot
    )

    bleu = calculate_bleu(predictions, references)

    return {
        'model': model_name,
        'bleu': bleu,
        'predictions': predictions,
        'references': references,
        'num_examples': len(test_data)
    }

def save_results(results, output_dir):
    """Save evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results for each model
    for result in results:
        model_slug = result['model'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')

        # Save predictions
        predictions_file = output_dir / f'khanq_{model_slug}_predictions.json'
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump([
                {
                    'reference': result['references'][i],
                    'prediction': result['predictions'][i]
                }
                for i in range(len(result['predictions']))
            ], f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved predictions: {predictions_file.name}")

    # Save summary
    summary_file = output_dir / 'khanq_evaluation_summary.json'

    paper_baselines = {
        'T5-base (zero-shot)': 9.32,
        'T5-small (baseline)': 11.45,
        'T5-small (+ topic)': 13.78
    }

    summary = {
        'dataset': 'khanq',
        'num_examples': results[0]['num_examples'],
        'results': [
            {
                'model': r['model'],
                'bleu': round(r['bleu'], 2),
                'num_examples': r['num_examples']
            }
            for r in results
        ],
        'paper_baselines': paper_baselines
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved summary: {summary_file.name}")

    # Create markdown report
    create_report(results, output_dir, paper_baselines)

def create_report(results, output_dir, paper_baselines):
    """Create markdown report."""
    report_file = output_dir / 'KHANQ_RESULTS.md'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# KhanQ Evaluation Results - Concise Prompt\n\n")
        f.write(f"**Date**: {time.strftime('%B %d, %Y')}\n\n")

        f.write("## Evaluation Setup\n\n")
        f.write("**Zero-shot prompt:** Concise format (8-12 words max)\n")
        f.write("- Emphasizes brevity and directness\n")
        f.write("- Specific question patterns provided\n")
        f.write("- Multiple examples given\n\n")

        f.write("**Fine-tuned models:** topic<sep>context format\n\n")

        f.write("## Results\n\n")
        f.write("| Model | Our BLEU | Paper BLEU | Difference | Status |\n")
        f.write("|-------|----------|------------|------------|--------|\n")

        for result in results:
            model = result['model']
            our_bleu = result['bleu']

            # Match with paper baseline
            if 'zero-shot' in model.lower():
                paper_key = 'T5-base (zero-shot)'
            elif 'topic' in model.lower() or 'plus' in model.lower():
                paper_key = 'T5-small (+ topic)'
            else:
                paper_key = 'T5-small (baseline)'

            paper_bleu = paper_baselines.get(paper_key, 0)
            diff = our_bleu - paper_bleu
            diff_str = f"{diff:+.2f}"

            # Status
            if abs(diff) <= 2.0:
                status = "[OK] Match"
            elif abs(diff) <= 5.0:
                status = "[~] Close"
            else:
                status = "[X] Different"

            f.write(f"| {model} | **{our_bleu:.2f}** | {paper_bleu:.2f} | {diff_str} | {status} |\n")

        f.write("\n## Analysis\n\n")

        # Best model
        best_result = max(results, key=lambda x: x['bleu'])
        f.write(f"### Best Model\n")
        f.write(f"- **{best_result['model']}**: {best_result['bleu']:.2f} BLEU\n\n")

        # Topic improvement
        baseline_result = next((r for r in results if 'baseline' in r['model'].lower() and 'zero' not in r['model'].lower()), None)
        topic_result = next((r for r in results if 'topic' in r['model'].lower() or 'plus' in r['model'].lower()), None)

        if baseline_result and topic_result:
            improvement = topic_result['bleu'] - baseline_result['bleu']
            if baseline_result['bleu'] > 0:
                rel_improvement = (improvement / baseline_result['bleu']) * 100
            else:
                rel_improvement = 0

            f.write(f"### Topic Conditioning Impact\n")
            f.write(f"- **Our improvement**: {improvement:.2f} BLEU (+{rel_improvement:.1f}%)\n")
            f.write(f"- **Paper improvement**: 2.33 BLEU (+20.3%)\n\n")

        # Zero-shot analysis
        zero_shot = next((r for r in results if 'zero-shot' in r['model'].lower()), None)
        if zero_shot:
            f.write(f"### Zero-Shot Performance\n")
            f.write(f"- **Our score**: {zero_shot['bleu']:.2f} BLEU\n")
            f.write(f"- **Paper score**: 9.32 BLEU\n")
            f.write(f"- **Difference**: {zero_shot['bleu'] - 9.32:+.2f}\n\n")

        # Sample predictions
        f.write("## Sample Predictions\n\n")

        for result in results:
            f.write(f"### {result['model']}\n\n")

            import random
            sample_indices = random.sample(range(len(result['predictions'])),
                                          min(5, len(result['predictions'])))

            for i, idx in enumerate(sample_indices, 1):
                f.write(f"**Example {i}:**\n")
                f.write(f"- Reference: {result['references'][idx]}\n")
                f.write(f"- Predicted: {result['predictions'][idx]}\n\n")

    print(f"[OK] Saved report: {report_file.name}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("KHANQ EVALUATION - CONCISE PROMPT")
    print("="*70)
    print(f"\nDevice: {device}")

    # Load KhanQ data
    khanq_file = 'data/processed/wikifier/enriched_khanq_filtered.json'
    print(f"\nLoading KhanQ data from: {khanq_file}")

    with open(khanq_file, 'r', encoding='utf-8') as f:
        khanq_data = json.load(f)

    print(f"Loaded {len(khanq_data)} examples")

    results = []

    # 1. Zero-shot with concise prompt
    print("\n[1/3] Zero-shot T5-base with concise prompt...")
    zero_shot_result = evaluate_model(
        'google-t5/t5-base',
        khanq_data,
        device,
        'T5-base (zero-shot + concise prompt)',
        is_zero_shot=True
    )
    results.append(zero_shot_result)
    print(f"BLEU: {zero_shot_result['bleu']:.2f}")

    # 2. Fine-tuned baseline (cross-domain from SQuAD)
    print("\n[2/3] Fine-tuned T5-small baseline...")
    baseline_result = evaluate_model(
        'models/squad_baseline_t5small/best_model',
        khanq_data,
        device,
        'T5-small (fine-tuned baseline)',
        is_zero_shot=False
    )
    results.append(baseline_result)
    print(f"BLEU: {baseline_result['bleu']:.2f}")

    # 3. Fine-tuned topic-conditioned (cross-domain from SQuAD)
    print("\n[3/3] Fine-tuned T5-small topic-conditioned...")
    topic_result = evaluate_model(
        'models/squad_topic_t5small/best_model',
        khanq_data,
        device,
        'T5-small (fine-tuned + topic)',
        is_zero_shot=False
    )
    results.append(topic_result)
    print(f"BLEU: {topic_result['bleu']:.2f}")

    # Save results
    output_dir = 'results/khanq_concise_eval'
    save_results(results, output_dir)

    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nKhanQ Results:")
    for r in results:
        print(f"  {r['model']:45s} {r['bleu']:6.2f} BLEU")

    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
