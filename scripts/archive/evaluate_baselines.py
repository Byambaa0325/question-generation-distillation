"""
Evaluate question generation models including Ollama baselines and fine-tuned T5.
"""
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import time
from tqdm import tqdm
import argparse
from typing import List, Dict
import subprocess

# Metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: NLTK/ROUGE not available. Install with: pip install nltk rouge-score")
    METRICS_AVAILABLE = False

class OllamaBaseline:
    """Baseline using Ollama models."""

    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        print(f"Using Ollama model: {model_name}")

    def generate_question(self, topic: str, context: str) -> str:
        """Generate question using Ollama."""
        prompt = f"""Given the following topic and context, generate a clear, specific question that can be answered using the context.

Topic: {topic}
Context: {context}

Generate only the question, nothing else:"""

        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                question = result.stdout.strip()
                # Clean up the response
                if question.startswith("Question:"):
                    question = question[9:].strip()
                return question
            else:
                return "[ERROR]"
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]"
        except Exception as e:
            return f"[ERROR: {str(e)}]"

class T5Model:
    """Fine-tuned T5 model."""

    def __init__(self, model_dir: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading T5 model from {model_dir}...")
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
        self.model.to(self.device)
        self.model.eval()

    def generate_question(self, topic: str, context: str) -> str:
        """Generate question using T5."""
        input_text = f"{topic}<sep>{context}"

        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=45,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def calculate_bleu(reference: str, hypothesis: str) -> float:
    """Calculate BLEU score."""
    if not METRICS_AVAILABLE:
        return 0.0

    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()

    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing)

def calculate_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    if not METRICS_AVAILABLE:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def evaluate_model(model, test_data: pd.DataFrame, model_name: str, max_samples: int = None) -> Dict:
    """Evaluate a model on test data."""
    results = []

    if max_samples:
        test_data = test_data.head(max_samples)

    print(f"\nEvaluating {model_name} on {len(test_data)} examples...")

    start_time = time.time()

    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc=model_name):
        topic = str(row['topic'])
        context = str(row['text'])
        reference = str(row['question'])

        # Generate question
        gen_start = time.time()
        generated = model.generate_question(topic, context)
        gen_time = time.time() - gen_start

        # Calculate metrics
        bleu = calculate_bleu(reference, generated)
        rouge = calculate_rouge(reference, generated)

        results.append({
            'topic': topic,
            'context': context[:100] + '...',
            'reference': reference,
            'generated': generated,
            'bleu': bleu,
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
            'gen_time': gen_time
        })

    total_time = time.time() - start_time

    # Calculate average metrics
    avg_metrics = {
        'model': model_name,
        'num_samples': len(results),
        'avg_bleu': sum(r['bleu'] for r in results) / len(results),
        'avg_rouge1': sum(r['rouge1'] for r in results) / len(results),
        'avg_rouge2': sum(r['rouge2'] for r in results) / len(results),
        'avg_rougeL': sum(r['rougeL'] for r in results) / len(results),
        'avg_gen_time': sum(r['gen_time'] for r in results) / len(results),
        'total_time': total_time
    }

    return {
        'metrics': avg_metrics,
        'predictions': results
    }

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate question generation models')
    parser.add_argument('--test-data', required=True, help='Test CSV file')
    parser.add_argument('--t5-model', help='Path to fine-tuned T5 model')
    parser.add_argument('--ollama-models', nargs='+', default=['llama3.2:3b', 'mistral:7b'],
                       help='Ollama models to use as baselines')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max number of samples to evaluate (for quick testing)')
    parser.add_argument('--output', default='evaluation_results.json',
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = pd.read_csv(args.test_data)
    print(f"  Loaded {len(test_data)} test examples")

    all_results = {}

    # Evaluate Ollama baselines
    for ollama_model in args.ollama_models:
        try:
            model = OllamaBaseline(ollama_model)
            results = evaluate_model(model, test_data, f"Ollama-{ollama_model}", args.max_samples)
            all_results[f"ollama_{ollama_model}"] = results
        except Exception as e:
            print(f"Error with {ollama_model}: {e}")

    # Evaluate T5 model
    if args.t5_model:
        try:
            model = T5Model(args.t5_model)
            results = evaluate_model(model, test_data, "T5-Fine-tuned", args.max_samples)
            all_results['t5_finetuned'] = results
        except Exception as e:
            print(f"Error with T5 model: {e}")

    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'BLEU':<10} {'ROUGE-1':<10} {'ROUGE-L':<10} {'Time/Q (s)':<12}")
    print("-"*80)

    for model_key, result in all_results.items():
        metrics = result['metrics']
        print(f"{metrics['model']:<25} "
              f"{metrics['avg_bleu']:<10.4f} "
              f"{metrics['avg_rouge1']:<10.4f} "
              f"{metrics['avg_rougeL']:<10.4f} "
              f"{metrics['avg_gen_time']:<12.3f}")

    print("="*80)
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
