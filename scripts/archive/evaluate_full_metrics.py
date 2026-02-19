"""
Comprehensive evaluation matching paper metrics:
BLEU1-4, F1, METEOR, Perplexity, ROUGE-L
Updated to match paper's exact methodology from eval_main.ipynb
"""
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import time
from tqdm import tqdm
import argparse
import subprocess
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Load environment variables
load_dotenv()

# Install missing packages if needed
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import sent_tokenize, word_tokenize
    import evaluate
    from sentence_transformers import SentenceTransformer, util
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except ImportError:
    print("Installing required packages...")
    subprocess.run(["pip", "install", "nltk", "evaluate", "sentence-transformers", "-q"])
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import sent_tokenize, word_tokenize
    import evaluate
    from sentence_transformers import SentenceTransformer, util
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Try to import Google Gen AI (new unified SDK)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class OllamaBaseline:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate_question(self, topic, context):
        prompt = f"""Generate a scientific question about the given topic based on the paragraph.

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

EXAMPLES:

Context: "Electronegativity is how strongly the element hogs the electron once the covalent bond is made."
Topic: "Electronegativity"
Question: "Do electronegativity and electropotential both describe to what extent an atom can attract electrons?"

Context: "Lithium has a very high ionization potential making it a strong reducing agent."
Topic: "reducing agent"
Question: "How does lithium behave as a strong reducing agent?"

Context: "The viscosity of liquids generally decreases with temperature."
Topic: "Viscosity"
Question: "How will the viscosity of liquid be affected by increase in temperature?"

Context: "Water molecules can be classified by their state of matter."
Topic: "Water"
Question: "Does one classify the H2O molecules as solid or liquid?"

Output only the question text (10-15 words):"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                question = result.stdout.strip()
                if question.startswith("Question:"):
                    question = question[9:].strip()
                return question
            return "[ERROR]"
        except:
            return "[ERROR]"

class GeminiBaseline:
    def __init__(self, model_name="gemini-2.0-flash-exp"):
        """
        Initialize Gemini model using the new Google Gen AI SDK.
        Works with gcloud authentication or API key.
        """
        self.model_name = model_name
        self.client = None

        # Use the new Google Gen AI SDK
        if GENAI_AVAILABLE:
            try:
                # Initialize client (uses Application Default Credentials from gcloud)
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID")
                location = os.getenv("GCP_LOCATION", "us-central1")

                if project_id:
                    # Use Vertex AI endpoint with gcloud auth
                    self.client = genai.Client(
                        vertexai=True,
                        project=project_id,
                        location=location
                    )
                    print(f"Initialized Gemini via Google Gen AI SDK (Vertex AI): {model_name}")
                    return
                else:
                    # Fallback to API key if available
                    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GCP_API_KEY")
                    if api_key:
                        self.client = genai.Client(api_key=api_key)
                        print(f"Initialized Gemini via Google Gen AI SDK (API Key): {model_name}")
                        return
            except Exception as e:
                print(f"Warning: Google Gen AI SDK initialization failed: {e}")

        # If we get here, no initialization succeeded
        raise ValueError(
            "Gemini initialization failed. Please:\n"
            "  1. Run 'gcloud auth application-default login', or\n"
            "  2. Set GOOGLE_CLOUD_PROJECT environment variable, or\n"
            "  3. Set GEMINI_API_KEY environment variable\n"
            "Install required package: pip install google-genai"
        )

    def generate_question(self, topic, context):
        if not self.client:
            return "[ERROR]"

        prompt = f"""Generate ONE scientific question about the topic from the paragraph.

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

Context: "Liquid viscosity generally decreases with temperature."
Topic: "Viscosity"
Question: "How will viscosity be affected by temperature increase?"

Context: "Water molecules exist in different states of matter."
Topic: "Water"
Question: "Does one classify H2O molecules as solid or liquid?"

Context: "Osmosis moves water across membranes without energy."
Topic: "Osmosis"
Question: "How can osmosis be a form of passive transport?"

Generate ONLY the question (8-12 words, be concise):"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,  # Balanced for reasoning questions
                    max_output_tokens=512,  # High limit to prevent cutoffs
                )
            )
            # Handle potential None response
            if response and hasattr(response, 'text') and response.text:
                question = response.text.strip()
                if question.startswith("Question:"):
                    question = question[9:].strip()
                # Ensure question ends with ?
                if question and not question.endswith('?'):
                    question = question + '?'
                return question if question else "[ERROR]"
            else:
                return "[ERROR]"
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return "[ERROR]"

class SentenceEmbeddings:
    """Sentence embedding model for selecting best question from beam search"""
    def __init__(self):
        self.embedder = SentenceTransformer('sentence-transformers/sentence-t5-base')

    def encode(self, text):
        return self.embedder.encode(text, convert_to_tensor=True)

    def get_most_similar(self, context, qa_list, text_weight=0.2, topic_weight=0.8):
        """Select best question based on similarity to context and topic"""
        text_embeddings = self.encode(context)
        top1 = {'idx': None, 'score': float('-inf')}

        for i in range(len(qa_list)):
            topic_embeddings = self.encode(qa_list[i]['topic'])
            question_embeddings = self.encode(qa_list[i]['question'])
            text_sim = util.pytorch_cos_sim(text_embeddings, question_embeddings)
            topic_sim = util.pytorch_cos_sim(topic_embeddings, question_embeddings)
            combined_score = (text_sim[0][0].item() * text_weight) + (topic_sim[0][0].item() * topic_weight)

            if combined_score > top1['score']:
                top1['score'] = combined_score
                top1['idx'] = i

        if top1['idx'] is not None:
            return qa_list[top1['idx']]['question']
        else:
            return qa_list[0]['question'] if qa_list else "[ERROR]"


class T5Model:
    def __init__(self, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
        self.model.to(self.device)
        self.model.eval()
        self.sentence_embedder = SentenceEmbeddings()

    def process_context(self, topic, context):
        """Extract sentences around topic mentions (paper's method)"""
        sentences = sent_tokenize(context)
        indices = [i for i, sentence in enumerate(sentences) if topic in sentence]

        if len(indices) == 0:
            return context

        start_index = max(0, indices[0] - 2)
        end_index = min(len(sentences), indices[-1] + 3)
        return " ".join(sentences[start_index:end_index])

    def generate_question(self, topic, context):
        """Generate question using paper's exact methodology"""
        processed_context = self.process_context(topic, context)
        # Paper's input format: '<topic> {} <context> {} '
        input_text = '<topic> {} <context> {} '.format(topic, processed_context)

        encoding = self.tokenizer.encode_plus(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            # Paper uses num_beams=10, num_return_sequences=8
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=10,
                num_return_sequences=8,
                max_length=45,
                early_stopping=True
            )

        # Create list of candidate questions
        qa_list = []
        for output in outputs:
            question = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            qa_list.append({'question': question, 'topic': topic, 'context': processed_context})

        # Select best question using sentence embeddings
        best_question = self.sentence_embedder.get_most_similar(processed_context, qa_list)
        return best_question

    def get_perplexity(self, topic, context, reference):
        """Calculate perplexity on reference question"""
        processed_context = self.process_context(topic, context)
        input_text = '<topic> {} <context> {} '.format(topic, processed_context)
        inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
        labels = self.tokenizer(reference, return_tensors='pt', max_length=45, truncation=True).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

        return torch.exp(loss).item()

def calculate_bleu_ngrams(reference, hypothesis):
    """Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 using CORRECT word-level method"""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    # Paper uses smooth.method2 (not method1)
    smoothing = SmoothingFunction().method2

    # Paper uses exact weights for each n-gram
    bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0, 1, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0, 0, 1, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0, 0, 0, 1), smoothing_function=smoothing)

    return bleu1, bleu2, bleu3, bleu4

def calculate_bleu_ngrams_paper_method(reference, hypothesis):
    """Calculate BLEU using paper's BUGGY method (character-level by passing strings)

    This replicates the bug in the paper's code where they pass strings directly
    to sentence_bleu, which treats each CHARACTER as a token instead of each WORD.

    This is INCORRECT but allows comparison with paper's reported scores.
    """
    # Paper's buggy code: sentence_bleu([references[i]], predictions[i], ...)
    # Passing strings makes it treat characters as tokens!
    smoothing = SmoothingFunction().method2

    bleu1 = sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu([reference], hypothesis, weights=(0, 1, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu([reference], hypothesis, weights=(0, 0, 1, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu([reference], hypothesis, weights=(0, 0, 0, 1), smoothing_function=smoothing)

    return bleu1, bleu2, bleu3, bleu4

def calculate_f1(reference, hypothesis):
    """Calculate token-level F1 score (paper's method)"""
    ref_tokens = set(word_tokenize(reference.lower()))
    hyp_tokens = set(word_tokenize(hypothesis.lower()))

    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    common = ref_tokens & hyp_tokens
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def evaluate_model(model, test_data, model_name, max_samples=None, compute_perplexity=False):
    """Evaluate with full paper metrics using BOTH correct and paper's buggy methodology"""
    predictions = []
    references = []
    bleu_scores = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []}
    bleu_scores_paper = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []}  # Paper's buggy char-level
    f1_scores = []
    perplexities = []

    if max_samples:
        test_data = test_data.head(max_samples)

    print(f"\nEvaluating {model_name} on {len(test_data)} examples...")

    # Generate predictions
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc=f"{model_name} - Generating"):
        topic = str(row['topic'])
        context = str(row['text'])
        reference = str(row['question'])

        generated = model.generate_question(topic, context)

        if generated in ["[ERROR]", "[TIMEOUT]"]:
            continue

        predictions.append(generated)
        references.append(reference)

        # Calculate BLEU scores - CORRECT word-level method
        bleu1, bleu2, bleu3, bleu4 = calculate_bleu_ngrams(reference, generated)
        bleu_scores['bleu1'].append(bleu1)
        bleu_scores['bleu2'].append(bleu2)
        bleu_scores['bleu3'].append(bleu3)
        bleu_scores['bleu4'].append(bleu4)

        # Calculate BLEU scores - PAPER'S BUGGY character-level method (for comparison)
        bleu1_p, bleu2_p, bleu3_p, bleu4_p = calculate_bleu_ngrams_paper_method(reference, generated)
        bleu_scores_paper['bleu1'].append(bleu1_p)
        bleu_scores_paper['bleu2'].append(bleu2_p)
        bleu_scores_paper['bleu3'].append(bleu3_p)
        bleu_scores_paper['bleu4'].append(bleu4_p)

        # Calculate F1
        f1 = calculate_f1(reference, generated)
        f1_scores.append(f1)

        # Calculate perplexity if needed
        if compute_perplexity and hasattr(model, 'get_perplexity'):
            try:
                perp = model.get_perplexity(topic, context, reference)
                perplexities.append(perp)
            except:
                pass

    # Use Hugging Face's evaluate library for ROUGE and METEOR (paper's method)
    print(f"{model_name} - Computing ROUGE and METEOR...")
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')

    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
    meteor_score_result = meteor_metric.compute(predictions=predictions, references=references)

    # Calculate averages
    avg_metrics = {
        'model': model_name,
        'num_samples': len(predictions),
        # Correct word-level BLEU
        'bleu1': np.mean(bleu_scores['bleu1']),
        'bleu2': np.mean(bleu_scores['bleu2']),
        'bleu3': np.mean(bleu_scores['bleu3']),
        'bleu4': np.mean(bleu_scores['bleu4']),
        # Paper's buggy character-level BLEU (for comparison)
        'bleu1_paper_method': np.mean(bleu_scores_paper['bleu1']),
        'bleu2_paper_method': np.mean(bleu_scores_paper['bleu2']),
        'bleu3_paper_method': np.mean(bleu_scores_paper['bleu3']),
        'bleu4_paper_method': np.mean(bleu_scores_paper['bleu4']),
        # Other metrics
        'f1': np.mean(f1_scores),
        'meteor': meteor_score_result['meteor'],
        'rouge_l': rouge_scores['rougeL'],
    }

    if perplexities:
        avg_metrics['perplexity'] = np.mean(perplexities)

    # Store predictions and references for inspection
    avg_metrics['predictions'] = predictions
    avg_metrics['references'] = references

    return avg_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', required=True)
    parser.add_argument('--t5-model', help='T5 model directory')
    parser.add_argument('--ollama-models', nargs='+', help='Ollama models')
    parser.add_argument('--gemini-models', nargs='+', help='Gemini models (e.g., gemini-2.0-flash-exp)')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output', default='full_evaluation.json')
    parser.add_argument('--experiment-name', help='Optional experiment name (default: auto-generated)')

    args = parser.parse_args()

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment_name if args.experiment_name else f"eval_{timestamp}"
    exp_dir = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")

    # Load test data (support both CSV and JSON formats)
    if args.test_data.endswith('.json'):
        import json
        with open(args.test_data, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        # Convert to DataFrame with expected columns
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

    print(f"Loaded {len(test_data)} test examples from {args.test_data}")

    all_results = {}

    # Evaluate Ollama models
    if args.ollama_models:
        for model_name in args.ollama_models:
            try:
                model = OllamaBaseline(model_name)
                metrics = evaluate_model(model, test_data, f"Ollama-{model_name}", args.max_samples, False)
                all_results[f"ollama_{model_name}"] = metrics
            except Exception as e:
                print(f"Error with {model_name}: {e}")

    # Evaluate Gemini models
    if args.gemini_models:
        for model_name in args.gemini_models:
            try:
                model = GeminiBaseline(model_name)
                metrics = evaluate_model(model, test_data, f"Gemini-{model_name}", args.max_samples, False)
                all_results[f"gemini_{model_name}"] = metrics
            except Exception as e:
                print(f"Error with {model_name}: {e}")

    # Evaluate T5
    if args.t5_model:
        try:
            model = T5Model(args.t5_model)
            metrics = evaluate_model(model, test_data, "T5-Fine-tuned", args.max_samples, True)
            all_results['t5_finetuned'] = metrics
        except Exception as e:
            print(f"Error with T5: {e}")

    # Save results to experiment directory
    output_path = exp_dir / args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Also save predictions to CSV for easy inspection
    output_base = args.output.replace('.json', '')
    for model_key, metrics in all_results.items():
        if 'predictions' in metrics and 'references' in metrics:
            df = pd.DataFrame({
                'reference': metrics['references'],
                'prediction': metrics['predictions']
            })
            # Sanitize filename (replace : with _ for Windows compatibility)
            safe_model_key = model_key.replace(':', '_')
            csv_path = exp_dir / f"{output_base}_{safe_model_key}_questions.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"Saved {len(df)} question pairs to: {csv_path}")

    # Print formatted table
    print("\n" + "="*120)
    print("EVALUATION RESULTS (Paper Metrics)")
    print("="*120)
    print(f"{'Model':<25} {'BLEU1':<10} {'BLEU2':<10} {'BLEU3':<10} {'BLEU4':<10} {'F1':<10} {'METEOR':<10} {'Perplexity':<12} {'ROUGE-L':<10}")
    print("-"*120)

    for key, metrics in all_results.items():
        perp_str = f"{metrics['perplexity']:.3f}" if 'perplexity' in metrics else "N/A"
        print(f"{metrics['model']:<25} "
              f"{metrics['bleu1']:<10.3f} "
              f"{metrics['bleu2']:<10.3f} "
              f"{metrics['bleu3']:<10.3f} "
              f"{metrics['bleu4']:<10.3f} "
              f"{metrics['f1']:<10.3f} "
              f"{metrics['meteor']:<10.3f} "
              f"{perp_str:<12} "
              f"{metrics['rouge_l']:<10.3f}")

    print("="*120)
    print("\nPaper Baseline:")
    print("BLEU1: 0.519, BLEU2: 0.316, BLEU3: 0.216, BLEU4: 0.175, F1: 0.319, METEOR: 0.216, Perplexity: 1.303, ROUGE-L: 0.207")
    print("\nResults saved to:", args.output)

if __name__ == "__main__":
    main()
