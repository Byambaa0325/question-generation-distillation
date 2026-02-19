"""Concept storage, retrieval, and deduplication."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Concept:
    """A single teachable concept extracted from context."""

    id: str
    text: str
    source_context: str
    embedding: Optional[np.ndarray] = None
    difficulty_tags: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)  # concept IDs

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "source_context": self.source_context,
            "difficulty_tags": self.difficulty_tags,
            "prerequisites": self.prerequisites,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Concept":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            source_context=data["source_context"],
            difficulty_tags=data.get("difficulty_tags", []),
            prerequisites=data.get("prerequisites", []),
        )


class ConceptBank:
    """Storage and retrieval for concepts with deduplication."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
    ):
        self.concepts: dict[str, Concept] = {}
        self.similarity_threshold = similarity_threshold
        self._encoder: Optional[SentenceTransformer] = None
        self._embedding_model = embedding_model
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._id_to_idx: dict[str, int] = {}

    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._embedding_model)
        return self._encoder

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        return self.encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def _rebuild_embeddings_matrix(self):
        """Rebuild the embeddings matrix from all concepts."""
        if not self.concepts:
            self._embeddings_matrix = None
            self._id_to_idx = {}
            return

        self._id_to_idx = {cid: idx for idx, cid in enumerate(self.concepts.keys())}
        embeddings = []
        for concept in self.concepts.values():
            if concept.embedding is None:
                concept.embedding = self._compute_embedding(concept.text)
            embeddings.append(concept.embedding)
        self._embeddings_matrix = np.stack(embeddings)

    def find_similar(self, text: str, top_k: int = 5) -> list[tuple[Concept, float]]:
        """Find similar concepts by text."""
        if not self.concepts or self._embeddings_matrix is None:
            return []

        query_emb = self._compute_embedding(text)
        similarities = self._embeddings_matrix @ query_emb

        top_indices = np.argsort(similarities)[::-1][:top_k]
        idx_to_id = {v: k for k, v in self._id_to_idx.items()}

        results = []
        for idx in top_indices:
            cid = idx_to_id[idx]
            results.append((self.concepts[cid], float(similarities[idx])))
        return results

    def is_duplicate(self, text: str) -> tuple[bool, Optional[Concept]]:
        """Check if concept text is duplicate of existing concept."""
        if not self.concepts:
            return False, None

        similar = self.find_similar(text, top_k=1)
        if similar and similar[0][1] >= self.similarity_threshold:
            return True, similar[0][0]
        return False, None

    def add(
        self,
        text: str,
        source_context: str,
        difficulty_tags: list[str] | None = None,
        prerequisites: list[str] | None = None,
        deduplicate: bool = True,
    ) -> tuple[Concept, bool]:
        """
        Add a concept to the bank.

        Returns:
            Tuple of (concept, is_new). If deduplicate=True and concept exists,
            returns the existing concept with is_new=False.
        """
        if deduplicate:
            is_dup, existing = self.is_duplicate(text)
            if is_dup and existing:
                return existing, False

        concept = Concept(
            id=str(uuid.uuid4()),
            text=text,
            source_context=source_context,
            embedding=self._compute_embedding(text),
            difficulty_tags=difficulty_tags or [],
            prerequisites=prerequisites or [],
        )
        self.concepts[concept.id] = concept
        self._rebuild_embeddings_matrix()
        return concept, True

    def get(self, concept_id: str) -> Optional[Concept]:
        """Get concept by ID."""
        return self.concepts.get(concept_id)

    def remove(self, concept_id: str) -> bool:
        """Remove concept by ID."""
        if concept_id in self.concepts:
            del self.concepts[concept_id]
            self._rebuild_embeddings_matrix()
            return True
        return False

    def save(self, path: str | Path):
        """Save concept bank to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self._embedding_model,
            "concepts": [c.to_dict() for c in self.concepts.values()],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ConceptBank":
        """Load concept bank from JSON file."""
        with open(path) as f:
            data = json.load(f)

        bank = cls(
            embedding_model=data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            similarity_threshold=data.get("similarity_threshold", 0.85),
        )

        for concept_data in data["concepts"]:
            concept = Concept.from_dict(concept_data)
            bank.concepts[concept.id] = concept

        bank._rebuild_embeddings_matrix()
        return bank

    def __len__(self) -> int:
        return len(self.concepts)

    def __iter__(self):
        return iter(self.concepts.values())
