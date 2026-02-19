#!/usr/bin/env python3
"""
Comprehensive evaluation with proper zero-shot prompting and topic conditioning.
Evaluates on both SQuAD and KhanQ test sets.
"""

import json
import torch
import argparse
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import time

# Specialized prompt for zero-shot (from paper's evaluation)
ZERO_SHOT_PROMPT_TEMPLATE = """Generate a scientific question about the given topic based on the paragraph.

Paragraph: {context}

Topic: {topic}

Requirements:
- Generate ONE clear, direct question about the topic
- Use proper scientific terminology from the paragraph
- Question should require understanding the concept (not just recall)
- Length: 10-15 words
- Use question starters: "How", "Why", "Does", "Do", "What", "Is", "Would"

Style Guidelines:
- Be direct and technical (use scientific terms naturally)
- Ask about mechanisms, relationships, or comparisons
- Can be comparative: "Does X imply Y?", "How does X affect Y?"
- Can be definitional: "What is the meaning of...", "How to..."

Output only the question text (10-15 words):"""

def load_model(model_path, device):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"[OK] Model loaded on {device}")
    return model, tokenizer

def prepare_zero_shot_input(item, use_specialized_prompt=True):
    """Prepare input for zero-shot with topic conditioning."""
    # Extract topic and context
    if '<sep>' in item['input']:
        topic, context = item['input'].split('<sep>', 1)
    else:
        # If no topic separator, use empty topic
        topic = ""
        context = item['input']

    if use_specialized_prompt and topic:
        # Use specialized prompt from paper's evaluation
        return ZERO_SHOT_PROMPT_TEMPLATE.format(context=context.strip(), topic=topic.strip())
    else:
        # Use simple topic-conditioned format
        if topic:
            return f"{topic.strip()}<sep>{context.strip()}"
        else:
            return f"generate question: {context.strip()}"

def generate_questions(model, tokenizer, test_data, device,
                      is_zero_shot=False, use_specialized_prompt=True,
                      max_input_length=512, max_output_length=64):
    """Generate questions for test data."""
    predictions = []
    references = []

    print(f"\nGenerating questions for {len(test_data)} examples...")
    print(f"  Zero-shot: {is_zero_shot}")
    print(f"  Specialized prompt: {use_specialized_prompt}")

    with torch.no_grad():
        for item in tqdm(test_data):
            # Prepare input based on model type
            if is_zero_shot:
                input_text = prepare_zero_shot_input(item, use_specialized_prompt)
            else:
                # Fine-tuned models use the data as-is
                input_text = item['input']

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
                early_stopping=True
            )

            # Decode
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ref = item['target']

            predictions.append(pred)
            references.append(ref)

    return predictions, references

def calculate_bleu(predictions, references):
    """Calculate BLEU score."""
    # Tokenize for BLEU
    predictions_tokenized = [pred.split() for pred in predictions]
    references_tokenized = [[ref.split()] for ref in references]

    # Calculate BLEU
    bleu_score = corpus_bleu(references_tokenized, predictions_tokenized)
    return bleu_score * 100  # Convert to percentage

def evaluate_model(model_path, test_file, device, model_name, is_zero_shot=False,
                  use_specialized_prompt=True):
    """Evaluate a model on test data."""
    print("\n" + "="*70)
    print(f"EVALUATING: {model_name}")
    print("="*70)

    model, tokenizer = load_model(model_path, device)

    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"Test examples: {len(test_data)}")

    predictions, references = generate_questions(
        model, tokenizer, test_data, device,
        is_zero_shot=is_zero_shot,
        use_specialized_prompt=use_specialized_prompt
    )

    bleu = calculate_bleu(predictions, references)

    return {
        'model': model_name,
        'bleu': bleu,
        'predictions': predictions,
        'references': references,
        'num_examples': len(test_data)
    }

def save_results(results, output_dir, dataset_name):
    """Save evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results for each model
    for result in results:
        model_slug = result['model'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('+', 'plus')

        # Save predictions
        predictions_file = output_dir / f'{dataset_name}_{model_slug}_predictions.json'
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump([
                {
                    'reference': result['references'][i],
                    'prediction': result['predictions'][i]
                }
                for i in range(len(result['predictions']))
            ], f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved {dataset_name} predictions: {predictions_file.name}")

    # Save summary
    summary_file = output_dir / f'{dataset_name}_evaluation_summary.json'

    # Paper baselines
    if dataset_name == 'squad':
        paper_baselines = {
            'T5-base (zero-shot)': 16.51,
            'T5-small (baseline)': 18.23,
            'T5-small (+ topic)': 19.45
        }
    else:  # khanq
        paper_baselines = {
            'T5-base (zero-shot)': 9.32,
            'T5-small (baseline)': 11.45,
            'T5-small (+ topic)': 13.78
        }

    summary = {
        'dataset': dataset_name,
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

    print(f"[OK] Saved {dataset_name} summary: {summary_file.name}")

    # Create markdown report
    create_report(results, output_dir, dataset_name, paper_baselines)

def create_report(results, output_dir, dataset_name, paper_baselines):
    """Create markdown report."""
    report_file = output_dir / f'{dataset_name.upper()}_RESULTS.md'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# {dataset_name.upper()} Evaluation Results\n\n")
        f.write(f"**Date**: {time.strftime('%B %d, %Y')}\n\n")

        f.write("## Summary\n\n")
        f.write(f"Evaluated all models on {dataset_name.upper()} test set with proper topic conditioning.\n\n")

        f.write("### Key Setup\n")
        f.write("- **Zero-shot**: Uses specialized prompt + topic conditioning\n")
        f.write("- **Fine-tuned**: Uses topic<sep>context format\n")
        f.write("- **All models**: Include topic information\n\n")

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
            paper_baseline_bleu = paper_baselines.get('T5-small (baseline)', 0)
            paper_topic_bleu = paper_baselines.get('T5-small (+ topic)', 0)
            paper_improvement = paper_topic_bleu - paper_baseline_bleu

            f.write(f"### Topic Conditioning Impact\n")
            f.write(f"- **Our improvement**: {improvement:.2f} BLEU")
            if baseline_result['bleu'] > 0:
                f.write(f" (+{improvement/baseline_result['bleu']*100:.1f}%)")
            f.write("\n")
            f.write(f"- **Paper improvement**: {paper_improvement:.2f} BLEU")
            if paper_baseline_bleu > 0:
                f.write(f" (+{paper_improvement/paper_baseline_bleu*100:.1f}%)")
            f.write("\n\n")

        # Sample predictions
        f.write("## Sample Predictions\n\n")
        f.write(f"### {best_result['model']}\n\n")

        import random
        sample_indices = random.sample(range(len(best_result['predictions'])),
                                      min(5, len(best_result['predictions'])))

        for i, idx in enumerate(sample_indices, 1):
            f.write(f"**Example {i}:**\n")
            f.write(f"- Reference: {best_result['references'][idx]}\n")
            f.write(f"- Predicted: {best_result['predictions'][idx]}\n\n")

    print(f"[OK] Saved {dataset_name} report: {report_file.name}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation with topic conditioning')

    # Model paths
    parser.add_argument('--baseline-model', default='models/squad_baseline_t5small/best_model',
                        help='Path to baseline model')
    parser.add_argument('--topic-model', default='models/squad_topic_t5small/best_model',
                        help='Path to topic model')
    parser.add_argument('--zero-shot-model', default='google-t5/t5-base',
                        help='Model for zero-shot evaluation')

    # Test files
    parser.add_argument('--squad-test', default='data/training/squad/topic/squad_test.json',
                        help='SQuAD test file (with topics)')
    parser.add_argument('--khanq-test', default='data/processed/wikifier/enriched_khanq_filtered.json',
                        help='KhanQ test file (Wikifier version)')

    # Options
    parser.add_argument('--output-dir', default='results/comprehensive_evaluation',
                        help='Output directory')
    parser.add_argument('--skip-squad', action='store_true',
                        help='Skip SQuAD evaluation')
    parser.add_argument('--skip-khanq', action='store_true',
                        help='Skip KhanQ evaluation')
    parser.add_argument('--use-simple-prompt', action='store_true',
                        help='Use simple prompt for zero-shot instead of specialized')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("COMPREHENSIVE EVALUATION WITH TOPIC CONDITIONING")
    print("="*70)
    print(f"\nDevice: {device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Zero-shot prompt: {'Simple' if args.use_simple_prompt else 'Specialized'}")

    # === SQuAD EVALUATION ===
    if not args.skip_squad:
        print("\n" + "="*70)
        print("SQUAD TEST SET EVALUATION")
        print("="*70)

        results_squad = []

        # 1. Zero-shot with topic
        print("\n[1/3] Zero-shot T5-base with topic conditioning...")
        zero_shot_result = evaluate_model(
            args.zero_shot_model,
            args.squad_test,
            device,
            'T5-base (zero-shot + topic)',
            is_zero_shot=True,
            use_specialized_prompt=not args.use_simple_prompt
        )
        results_squad.append(zero_shot_result)
        print(f"BLEU: {zero_shot_result['bleu']:.2f}")

        # 2. Fine-tuned baseline
        print("\n[2/3] Fine-tuned T5-small baseline...")
        baseline_result = evaluate_model(
            args.baseline_model,
            args.squad_test,
            device,
            'T5-small (fine-tuned baseline)',
            is_zero_shot=False
        )
        results_squad.append(baseline_result)
        print(f"BLEU: {baseline_result['bleu']:.2f}")

        # 3. Fine-tuned topic-conditioned
        print("\n[3/3] Fine-tuned T5-small topic-conditioned...")
        topic_result = evaluate_model(
            args.topic_model,
            args.squad_test,
            device,
            'T5-small (fine-tuned + topic)',
            is_zero_shot=False
        )
        results_squad.append(topic_result)
        print(f"BLEU: {topic_result['bleu']:.2f}")

        # Save SQuAD results
        save_results(results_squad, args.output_dir, 'squad')

        print("\n" + "="*70)
        print("SQUAD RESULTS")
        print("="*70)
        for r in results_squad:
            print(f"  {r['model']:40s} {r['bleu']:6.2f} BLEU")

    # === KHANQ EVALUATION ===
    if not args.skip_khanq:
        print("\n" + "="*70)
        print("KHANQ TEST SET EVALUATION")
        print("="*70)

        # Prepare KhanQ data in the same format as SQuAD
        print("\nPreparing KhanQ test data...")
        with open(args.khanq_test, 'r', encoding='utf-8') as f:
            khanq_data = json.load(f)

        # Convert to test format
        khanq_test = []
        for item in khanq_data:
            khanq_test.append({
                'id': item.get('id', ''),
                'input': f"{item['topic']}<sep>{item['text']}",
                'target': item['question']
            })

        print(f"KhanQ test examples: {len(khanq_test)}")

        # Save temporary test file
        khanq_test_file = Path(args.output_dir) / 'khanq_test_formatted.json'
        khanq_test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(khanq_test_file, 'w', encoding='utf-8') as f:
            json.dump(khanq_test, f, indent=2)

        results_khanq = []

        # 1. Zero-shot with topic
        print("\n[1/3] Zero-shot T5-base with topic conditioning...")
        zero_shot_result = evaluate_model(
            args.zero_shot_model,
            str(khanq_test_file),
            device,
            'T5-base (zero-shot + topic)',
            is_zero_shot=True,
            use_specialized_prompt=not args.use_simple_prompt
        )
        results_khanq.append(zero_shot_result)
        print(f"BLEU: {zero_shot_result['bleu']:.2f}")

        # 2. Fine-tuned baseline (trained on SQuAD, tested on KhanQ)
        print("\n[2/3] Fine-tuned T5-small baseline (cross-domain)...")
        baseline_result = evaluate_model(
            args.baseline_model,
            str(khanq_test_file),
            device,
            'T5-small (fine-tuned baseline)',
            is_zero_shot=False
        )
        results_khanq.append(baseline_result)
        print(f"BLEU: {baseline_result['bleu']:.2f}")

        # 3. Fine-tuned topic-conditioned (trained on SQuAD, tested on KhanQ)
        print("\n[3/3] Fine-tuned T5-small topic-conditioned (cross-domain)...")
        topic_result = evaluate_model(
            args.topic_model,
            str(khanq_test_file),
            device,
            'T5-small (fine-tuned + topic)',
            is_zero_shot=False
        )
        results_khanq.append(topic_result)
        print(f"BLEU: {topic_result['bleu']:.2f}")

        # Save KhanQ results
        save_results(results_khanq, args.output_dir, 'khanq')

        print("\n" + "="*70)
        print("KHANQ RESULTS")
        print("="*70)
        for r in results_khanq:
            print(f"  {r['model']:40s} {r['bleu']:6.2f} BLEU")

    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print("  - squad_RESULTS.md / khanq_RESULTS.md")
    print("  - squad_evaluation_summary.json / khanq_evaluation_summary.json")
    print("  - *_predictions.json (detailed predictions)")

if __name__ == "__main__":
    main()
