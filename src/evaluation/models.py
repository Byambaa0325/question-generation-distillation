"""
Model wrappers for question generation evaluation.

All heavy imports (torch, transformers, sentence_transformers, google.genai)
are deferred to object construction time so that ``from src.evaluation import *``
stays fast.

Classes
-------
SentenceEmbeddings  Sentence-transformer reranker (paper's beam selection method).
T5Model             Fine-tuned T5 with beam search + reranking + perplexity.
OllamaBaseline      Zero-shot via locally running Ollama (subprocess).
GeminiBaseline      Zero-shot via Google Gen AI SDK (Vertex AI or API key).

Factory
-------
load_model(config, model_key)
    Construct a model from a ``"type:name"`` string.
    Supported prefixes: ``t5:``, ``ollama:``, ``gemini:``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Sentence embedding reranker
# ---------------------------------------------------------------------------


class SentenceEmbeddings:
    """Sentence-embedding reranker used by T5Model for beam selection."""

    def __init__(self, model_name: str = "sentence-transformers/sentence-t5-base"):
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(model_name)

    def encode(self, text: str):
        return self.embedder.encode(text, convert_to_tensor=True)

    def get_most_similar(
        self,
        context: str,
        qa_list: list[dict],
        text_weight: float = 0.2,
        topic_weight: float = 0.8,
    ) -> str:
        """Return the candidate question with the highest weighted similarity score."""
        from sentence_transformers import util

        context_emb = self.encode(context)
        best = {"idx": None, "score": float("-inf")}

        for i, item in enumerate(qa_list):
            topic_emb = self.encode(item["topic"])
            q_emb = self.encode(item["question"])
            text_sim  = util.pytorch_cos_sim(context_emb, q_emb)[0][0].item()
            topic_sim = util.pytorch_cos_sim(topic_emb,   q_emb)[0][0].item()
            score = text_weight * text_sim + topic_weight * topic_sim
            if score > best["score"]:
                best = {"idx": i, "score": score}

        idx = best["idx"]
        return qa_list[idx]["question"] if idx is not None else (
            qa_list[0]["question"] if qa_list else "[ERROR]"
        )


# ---------------------------------------------------------------------------
# T5 fine-tuned model
# ---------------------------------------------------------------------------


class T5Model:
    """
    Fine-tuned T5 model wrapper.

    Implements paper's exact generation:
      - ``<topic> {topic} <context> {context}`` input format
      - 10 beams, 8 return sequences
      - Sentence-embedding reranking to select the best candidate
      - ``get_perplexity`` for computing model perplexity on references
    """

    def __init__(self, model_dir: str | Path):
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
        self.tokenizer = T5Tokenizer.from_pretrained(str(model_dir), legacy=False)
        self.model.to(self.device)
        self.model.eval()
        self._sentence_embedder: SentenceEmbeddings | None = None

    @property
    def sentence_embedder(self) -> SentenceEmbeddings:
        if self._sentence_embedder is None:
            self._sentence_embedder = SentenceEmbeddings()
        return self._sentence_embedder

    def process_context(self, topic: str, context: str) -> str:
        """Extract sentences around topic mention (paper's method)."""
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(context)
        indices = [i for i, s in enumerate(sentences) if topic in s]
        if not indices:
            return context
        start = max(0, indices[0] - 2)
        end = min(len(sentences), indices[-1] + 3)
        return " ".join(sentences[start:end])

    def generate_question(self, topic: str, context: str) -> str:
        """Generate a question using paper's methodology (beam search + reranking)."""
        import torch

        processed = self.process_context(topic, context)
        input_text = f"<topic> {topic} <context> {processed} "

        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                num_beams=10,
                num_return_sequences=8,
                max_length=45,
                early_stopping=True,
            )

        qa_list = [
            {
                "question": self.tokenizer.decode(
                    o, skip_special_tokens=True, clean_up_tokenization_spaces=True
                ),
                "topic":   topic,
                "context": processed,
            }
            for o in outputs
        ]
        return self.sentence_embedder.get_most_similar(processed, qa_list)

    def get_perplexity(self, topic: str, context: str, reference: str) -> float:
        """Calculate perplexity on a reference question."""
        import torch

        processed = self.process_context(topic, context)
        input_text = f"<topic> {topic} <context> {processed} "

        inputs = self.tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        labels = self.tokenizer(
            reference, return_tensors="pt", max_length=45, truncation=True
        ).input_ids.to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, labels=labels)

        return __import__("torch").exp(out.loss).item()


# ---------------------------------------------------------------------------
# T5 zero-shot baseline (base pretrained model, no fine-tuning)
# ---------------------------------------------------------------------------

_T5_ZERO_SHOT_PROMPT = (
    "generate question: {context} generate question about {topic}:"
)


class T5ZeroShotModel:
    """
    Zero-shot T5 baseline using the base pretrained model (no fine-tuning).

    Sandwich pattern: task prefix at the start (never truncated) then context,
    then the task + topic repeated at the end so the model has both signals
    immediately before generation regardless of context length.

    Prompt: ``generate question: {context} generate question about {topic}:``
    """

    def __init__(self, model_name: str = "google-t5/t5-small"):
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model.to(self.device)
        self.model.eval()

    def generate_question(self, topic: str, context: str) -> str:
        import torch

        input_text = _T5_ZERO_SHOT_PROMPT.format(topic=topic, context=context)
        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                num_beams=4,
                max_length=45,
                early_stopping=True,
            )

        return self.tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def get_perplexity(self, topic: str, context: str, reference: str) -> float:
        import torch

        input_text = _T5_ZERO_SHOT_PROMPT.format(topic=topic, context=context)
        inputs = self.tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        labels = self.tokenizer(
            reference, return_tensors="pt", max_length=45, truncation=True
        ).input_ids.to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, labels=labels)

        return __import__("torch").exp(out.loss).item()


# ---------------------------------------------------------------------------
# Ollama zero-shot baseline
# ---------------------------------------------------------------------------

_OLLAMA_PROMPT = """\
Generate a scientific question about the given topic based on the paragraph.

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

Output only the question text (10-15 words):\
"""


class OllamaBaseline:
    """Zero-shot question generation via a locally running Ollama model."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_question(self, topic: str, context: str) -> str:
        prompt = _OLLAMA_PROMPT.format(topic=topic, context=context)
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=60,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                q = result.stdout.strip()
                return q[len("Question:"):].strip() if q.startswith("Question:") else q
            return "[ERROR]"
        except Exception:
            return "[ERROR]"


# ---------------------------------------------------------------------------
# Gemini zero-shot baseline
# ---------------------------------------------------------------------------

_GEMINI_PROMPT = """\
Generate ONE scientific question about the topic from the paragraph.

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

Generate ONLY the question (8-12 words, be concise):\
"""


class GeminiBaseline:
    """Zero-shot question generation via Google Gemini (Gen AI SDK)."""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            from google import genai

            project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID")
            location = os.getenv("GCP_LOCATION", "us-central1")

            if project_id:
                self.client = genai.Client(
                    vertexai=True, project=project_id, location=location
                )
                return

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
                return

        except Exception as exc:
            print(f"Warning: Gemini init failed: {exc}")

        if self.client is None:
            raise ValueError(
                "Gemini init failed. Set GOOGLE_CLOUD_PROJECT, GOOGLE_API_KEY, "
                "or GEMINI_API_KEY, or run 'gcloud auth application-default login'."
            )

    def generate_question(self, topic: str, context: str) -> str:
        if not self.client:
            return "[ERROR]"
        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=_GEMINI_PROMPT.format(topic=topic, context=context),
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    max_output_tokens=512,
                ),
            )
            if response and hasattr(response, "text") and response.text:
                q = response.text.strip()
                if q.startswith("Question:"):
                    q = q[len("Question:"):].strip()
                if q and not q.endswith("?"):
                    q += "?"
                return q or "[ERROR]"
            return "[ERROR]"
        except Exception as exc:
            print(f"Gemini error: {exc}")
            return "[ERROR]"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_model(config: "PipelineConfig", model_key: str):
    """
    Construct a model from a ``type:name`` key.

    Parameters
    ----------
    config:    Pipeline configuration (used to resolve model directories).
    model_key: A string like ``"t5:topic"``, ``"ollama:llama3.1:8b"``,
               or ``"gemini:gemini-2.0-flash-exp"``.

    Returns
    -------
    Object with a ``generate_question(topic, context) -> str`` method.

    Raises
    ------
    FileNotFoundError  If the T5 model directory does not exist.
    ValueError         If the model key prefix is unrecognised.
    """
    if model_key.startswith("t5:"):
        mode = model_key[3:]
        if mode == "zero":
            return T5ZeroShotModel(config.training.model_name)
        model_dir = config.model_dir(mode) / "best_model"
        if not model_dir.exists():
            raise FileNotFoundError(f"T5 model not found: {model_dir}")
        return T5Model(model_dir)

    if model_key.startswith("ollama:"):
        return OllamaBaseline(model_key[7:])

    if model_key.startswith("gemini:"):
        return GeminiBaseline(model_key[7:])

    raise ValueError(
        f"Unknown model key: {model_key!r}. "
        "Expected prefix: 't5:', 'ollama:', or 'gemini:'."
    )
