"""
Incremental evaluation with resume capability.

Usage:
    python scripts/evaluate_incremental.py \\
        --test-data data/processed/enriched_khanq_test.json \\
        --model-type t5 \\
        --model-path models/t5-qg-khanq-paper-format \\
        --output-dir experiments/my_eval \\
        --save-every 10

To resume after interruption, just run the same command again.
"""
import pandas as pd
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_full_metrics import (
    T5Model, GeminiBaseline, OllamaBaseline,
    calculate_bleu_ngrams, calculate_bleu_ngrams_paper_method,
    calculate_f1
)

class IncrementalEvaluator:
    def __init__(self, output_dir, save_every=10):
        """
        Initialize incremental evaluator.

        Args:
            output_dir: Directory to save results
            save_every: Save progress every N examples
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every

        # Files
        self.progress_file = self.output_dir / "progress.json"
        self.results_file = self.output_dir / "results.json"
        self.questions_file = self.output_dir / "questions.csv"

        # Load existing progress
        self.progress = self._load_progress()

    def _load_progress(self):
        """Load existing progress if available."""
        if self.progress_file.exists():
            print(f"Loading existing progress from {self.progress_file}")
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            print(f"  Resuming from example {progress['last_completed_idx'] + 1}/{progress['total_examples']}")
            return progress
        else:
            return {
                'last_completed_idx': -1,  # -1 means no examples completed yet
                'total_examples': 0,
                'predictions': [],
                'references': [],
                'bleu_scores': [],
                'bleu_scores_paper': [],
                'f1_scores': [],
                'perplexities': []
            }

    def _save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2)

    def _save_questions(self):
        """Save question pairs to CSV."""
        df = pd.DataFrame({
            'reference': self.progress['references'],
            'prediction': self.progress['predictions']
        })
        df.to_csv(self.questions_file, index=False, encoding='utf-8')

    def evaluate(self, model, test_data, model_name, compute_perplexity=False):
        """
        Evaluate model incrementally with resume capability.

        Args:
            model: Model instance
            test_data: DataFrame with 'text', 'topic', 'question' columns
            model_name: Name of the model
            compute_perplexity: Whether to compute perplexity (T5 only)
        """
        # Update total if first run
        if self.progress['total_examples'] == 0:
            self.progress['total_examples'] = len(test_data)

        # Start from where we left off
        start_idx = self.progress['last_completed_idx'] + 1

        if start_idx >= len(test_data):
            print(f"Evaluation already complete ({start_idx}/{len(test_data)} examples)")
            return self._finalize_results(model_name)

        print(f"\nEvaluating {model_name} on {len(test_data)} examples...")
        print(f"Starting from example {start_idx + 1}")

        # Process remaining examples
        for idx in tqdm(range(start_idx, len(test_data)),
                       initial=start_idx,
                       total=len(test_data),
                       desc=f"{model_name} - Generating"):
            row = test_data.iloc[idx]
            topic = str(row['topic'])
            context = str(row['text'])
            reference = str(row['question'])

            # Generate prediction
            try:
                generated = model.generate_question(topic, context)

                if generated in ["[ERROR]", "[TIMEOUT]"]:
                    print(f"\nSkipping example {idx} due to error")
                    continue

                # Store prediction and reference
                self.progress['predictions'].append(generated)
                self.progress['references'].append(reference)

                # Calculate BLEU - correct word-level
                bleu1, bleu2, bleu3, bleu4 = calculate_bleu_ngrams(reference, generated)
                self.progress['bleu_scores'].append({
                    'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4
                })

                # Calculate BLEU - paper's buggy character-level
                bleu1_p, bleu2_p, bleu3_p, bleu4_p = calculate_bleu_ngrams_paper_method(reference, generated)
                self.progress['bleu_scores_paper'].append({
                    'bleu1': bleu1_p, 'bleu2': bleu2_p, 'bleu3': bleu3_p, 'bleu4': bleu4_p
                })

                # Calculate F1
                f1 = calculate_f1(reference, generated)
                self.progress['f1_scores'].append(f1)

                # Calculate perplexity if needed
                if compute_perplexity and hasattr(model, 'get_perplexity'):
                    try:
                        perp = model.get_perplexity(topic, context, reference)
                        self.progress['perplexities'].append(perp)
                    except:
                        pass

                # Update last completed index
                self.progress['last_completed_idx'] = idx

                # Save progress periodically
                if (idx + 1) % self.save_every == 0:
                    self._save_progress()
                    self._save_questions()
                    print(f"\n  Progress saved: {idx + 1}/{len(test_data)} examples")

            except Exception as e:
                print(f"\nError on example {idx}: {e}")
                # Save progress even on error
                self._save_progress()
                continue

        # Final save
        self._save_progress()
        self._save_questions()

        return self._finalize_results(model_name)

    def _finalize_results(self, model_name):
        """Compute final metrics and save results."""
        import numpy as np
        import evaluate

        predictions = self.progress['predictions']
        references = self.progress['references']

        if len(predictions) == 0:
            print("No predictions to evaluate!")
            return None

        print(f"\n{model_name} - Computing ROUGE and METEOR...")
        rouge_metric = evaluate.load('rouge')
        meteor_metric = evaluate.load('meteor')

        rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
        meteor_score_result = meteor_metric.compute(predictions=predictions, references=references)

        # Calculate averages
        bleu_scores = self.progress['bleu_scores']
        bleu_scores_paper = self.progress['bleu_scores_paper']

        avg_metrics = {
            'model': model_name,
            'num_samples': len(predictions),
            # Correct word-level BLEU
            'bleu1': np.mean([b['bleu1'] for b in bleu_scores]),
            'bleu2': np.mean([b['bleu2'] for b in bleu_scores]),
            'bleu3': np.mean([b['bleu3'] for b in bleu_scores]),
            'bleu4': np.mean([b['bleu4'] for b in bleu_scores]),
            # Paper's buggy character-level BLEU
            'bleu1_paper_method': np.mean([b['bleu1'] for b in bleu_scores_paper]),
            'bleu2_paper_method': np.mean([b['bleu2'] for b in bleu_scores_paper]),
            'bleu3_paper_method': np.mean([b['bleu3'] for b in bleu_scores_paper]),
            'bleu4_paper_method': np.mean([b['bleu4'] for b in bleu_scores_paper]),
            # Other metrics
            'f1': np.mean(self.progress['f1_scores']),
            'meteor': meteor_score_result['meteor'],
            'rouge_l': rouge_scores['rougeL'],
        }

        if self.progress['perplexities']:
            avg_metrics['perplexity'] = np.mean(self.progress['perplexities'])

        # Save final results
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(avg_metrics, f, indent=2)

        print(f"\nResults saved to: {self.results_file}")
        print(f"Questions saved to: {self.questions_file}")

        # Print summary
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Model: {model_name}")
        print(f"Samples: {avg_metrics['num_samples']}")
        print(f"\nCorrect Word-Level BLEU:")
        print(f"  BLEU-1: {avg_metrics['bleu1']:.3f}")
        print(f"  BLEU-4: {avg_metrics['bleu4']:.3f}")
        print(f"\nPaper's Character-Level BLEU:")
        print(f"  BLEU-1: {avg_metrics['bleu1_paper_method']:.3f}")
        print(f"  BLEU-4: {avg_metrics['bleu4_paper_method']:.3f}")
        print(f"\nOther Metrics:")
        print(f"  F1:     {avg_metrics['f1']:.3f}")
        print(f"  METEOR: {avg_metrics['meteor']:.3f}")
        print(f"  ROUGE:  {avg_metrics['rouge_l']:.3f}")
        if 'perplexity' in avg_metrics:
            print(f"  Perplexity: {avg_metrics['perplexity']:.2f}")
        print("="*80)

        return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Incremental evaluation with resume capability')
    parser.add_argument('--test-data', required=True, help='Test data file (CSV or JSON)')
    parser.add_argument('--model-type', required=True, choices=['t5', 'gemini', 'ollama'], help='Model type')
    parser.add_argument('--model-path', help='Model path (for T5) or model name (for Gemini/Ollama)')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--save-every', type=int, default=10, help='Save progress every N examples')
    parser.add_argument('--max-samples', type=int, help='Maximum samples to evaluate')

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    if args.test_data.endswith('.json'):
        with open(args.test_data, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        test_data = pd.DataFrame([
            {
                'text': item['text'],
                'topic': item['topic'],
                'question': item['question']
            }
            for item in json_data
        ])
    else:
        test_data = pd.read_csv(args.test_data)

    if args.max_samples:
        test_data = test_data.head(args.max_samples)

    print(f"Loaded {len(test_data)} test examples")

    # Initialize model
    print(f"Initializing {args.model_type} model...")
    if args.model_type == 't5':
        if not args.model_path:
            parser.error("--model-path required for T5")
        model = T5Model(args.model_path)
        model_name = f"T5-{Path(args.model_path).name}"
        compute_perplexity = True
    elif args.model_type == 'gemini':
        model_name = args.model_path or "gemini-2.0-flash-001"
        model = GeminiBaseline(model_name)
        model_name = f"Gemini-{model_name}"
        compute_perplexity = False
    elif args.model_type == 'ollama':
        model_name = args.model_path or "llama3.1:8b"
        model = OllamaBaseline(model_name)
        model_name = f"Ollama-{model_name}"
        compute_perplexity = False

    # Create evaluator
    evaluator = IncrementalEvaluator(args.output_dir, save_every=args.save_every)

    # Run evaluation
    results = evaluator.evaluate(model, test_data, model_name, compute_perplexity=compute_perplexity)

    print(f"\n✓ Evaluation complete! Results in {args.output_dir}")

if __name__ == "__main__":
    main()
