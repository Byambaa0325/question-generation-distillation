"""
Generate questions using the fine-tuned T5 model.
Updated to match paper's methodology with beam selection.
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import argparse

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
            return qa_list[top1['idx']]
        return qa_list[0] if qa_list else None

def process_context(topic, context):
    """Extract sentences around topic mentions (paper's method)"""
    sentences = sent_tokenize(context)
    indices = [i for i, sentence in enumerate(sentences) if topic in sentence]

    if len(indices) == 0:
        return context

    start_index = max(0, indices[0] - 2)
    end_index = min(len(sentences), indices[-1] + 3)
    return " ".join(sentences[start_index:end_index])

def generate_questions(model, tokenizer, contexts, topics, max_length=45, num_beams=10, num_return=8, use_beam_selection=True):
    """Generate questions for given contexts and topics using paper's methodology."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    if use_beam_selection:
        sentence_embedder = SentenceEmbeddings()

    results = []

    for context, topic in zip(contexts, topics):
        # Process context (extract relevant sentences)
        processed_context = process_context(topic, context)

        # Paper's input format: '<topic> {} <context> {} '
        input_text = '<topic> {} <context> {} '.format(topic, processed_context)

        # Tokenize (use __call__; encode_plus is deprecated/removed in newer transformers)
        encoding = tokenizer(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(device)

        # Generate with beam search
        with torch.no_grad():
            if use_beam_selection:
                # Paper's method: generate multiple candidates and select best
                outputs = model.generate(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask'],
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return,
                    early_stopping=True
                )

                # Create candidate list
                qa_list = []
                for output in outputs:
                    question = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    qa_list.append({'question': question, 'topic': topic, 'context': processed_context})

                # Select best question
                best_qa = sentence_embedder.get_most_similar(processed_context, qa_list)
                question = best_qa['question']
            else:
                # Simple beam search
                outputs = model.generate(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask'],
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
                question = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            'topic': topic,
            'context': context,
            'processed_context': processed_context,
            'generated_question': question
        })

    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate questions with T5 model (paper methodology)')
    parser.add_argument('--model-dir', required=True, help='Path to fine-tuned model')
    parser.add_argument('--topic', required=True, help='Topic for question generation')
    parser.add_argument('--context', required=True, help='Context passage')
    parser.add_argument('--max-length', type=int, default=45, help='Max question length')
    parser.add_argument('--num-beams', type=int, default=10, help='Number of beams (paper uses 10)')
    parser.add_argument('--num-return', type=int, default=8, help='Number of sequences to return (paper uses 8)')
    parser.add_argument('--no-beam-selection', action='store_true', help='Disable beam selection with sentence embeddings')

    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model_dir}...")
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir, legacy=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not args.no_beam_selection:
        print("Using beam selection with sentence embeddings (paper's method)")

    # Generate question
    results = generate_questions(
        model,
        tokenizer,
        [args.context],
        [args.topic],
        max_length=args.max_length,
        num_beams=args.num_beams,
        num_return=args.num_return,
        use_beam_selection=not args.no_beam_selection
    )

    # Print result
    print("\n" + "="*70)
    print("GENERATED QUESTION (Paper Methodology)")
    print("="*70)
    print(f"Topic: {results[0]['topic']}")
    print(f"\nOriginal Context: {results[0]['context'][:200]}...")
    print(f"\nProcessed Context: {results[0]['processed_context'][:200]}...")
    print(f"\nGenerated Question: {results[0]['generated_question']}")
    print("="*70)

if __name__ == "__main__":
    main()
