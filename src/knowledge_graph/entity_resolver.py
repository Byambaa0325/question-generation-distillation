"""Entity resolution and deduplication for knowledge graph construction."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .ontology import Entity, EntityType


@dataclass
class ResolutionResult:
    """Result of entity resolution."""

    is_duplicate: bool
    matched_entity: Optional[Entity] = None
    similarity_score: float = 0.0
    canonical_name: Optional[str] = None


class EntityResolver:
    """
    Entity resolution component for deduplicating and merging entities.

    Handles:
    - Synonym detection (ML vs Machine Learning)
    - Abbreviation expansion (CNN vs Convolutional Neural Network)
    - Fuzzy matching for typos and variations
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        use_llm_verification: bool = False,
        llm=None,
    ):
        """
        Initialize entity resolver.

        Args:
            embedding_model: Sentence transformer model for semantic similarity
            similarity_threshold: Threshold for considering entities as duplicates
            use_llm_verification: Use LLM to verify ambiguous matches
            llm: Language model for verification (required if use_llm_verification=True)
        """
        self.similarity_threshold = similarity_threshold
        self.use_llm_verification = use_llm_verification
        self.llm = llm
        self._encoder = None
        self._embedding_model = embedding_model

        # Cache for embeddings
        self._embedding_cache: dict[str, np.ndarray] = {}

        # Known synonyms/abbreviations
        self.synonym_map: dict[str, str] = {
            "ml": "machine learning",
            "dl": "deep learning",
            "nn": "neural network",
            "cnn": "convolutional neural network",
            "rnn": "recurrent neural network",
            "nlp": "natural language processing",
            "cv": "computer vision",
            "ai": "artificial intelligence",
            "rl": "reinforcement learning",
        }

    @property
    def encoder(self):
        """Lazy-load sentence transformer."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self._embedding_model)
        return self._encoder

    def get_embedding(self, text: str) -> np.ndarray:
        """Get or compute embedding for text."""
        text_lower = text.lower().strip()

        if text_lower not in self._embedding_cache:
            embedding = self.encoder.encode(text_lower, convert_to_numpy=True)
            self._embedding_cache[text_lower] = embedding

        return self._embedding_cache[text_lower]

    def normalize_name(self, name: str) -> str:
        """
        Normalize an entity name.

        - Lowercase
        - Expand known abbreviations
        - Remove extra whitespace
        """
        normalized = name.lower().strip()

        # Check synonym map
        if normalized in self.synonym_map:
            normalized = self.synonym_map[normalized]

        # Remove extra whitespace
        normalized = " ".join(normalized.split())

        return normalized

    def compute_similarity(self, name1: str, name2: str) -> float:
        """Compute semantic similarity between two entity names."""
        emb1 = self.get_embedding(name1)
        emb2 = self.get_embedding(name2)

        # Cosine similarity (embeddings are normalized)
        similarity = float(np.dot(emb1, emb2))

        return similarity

    def find_similar(
        self,
        entity_name: str,
        existing_entities: list[Entity],
        entity_type: Optional[EntityType] = None,
    ) -> Optional[Entity]:
        """
        Find the most similar existing entity.

        Args:
            entity_name: Name of the new entity
            existing_entities: List of existing entities to compare against
            entity_type: Optional type filter

        Returns:
            Most similar entity if above threshold, None otherwise
        """
        if not existing_entities:
            return None

        # Normalize the query name
        normalized_name = self.normalize_name(entity_name)

        best_match = None
        best_score = 0.0

        for entity in existing_entities:
            # Filter by type if specified
            if entity_type and entity.entity_type != entity_type:
                continue

            # Exact match after normalization
            if self.normalize_name(entity.name) == normalized_name:
                return entity

            # Semantic similarity
            similarity = self.compute_similarity(normalized_name, entity.name)

            if similarity > best_score:
                best_score = similarity
                best_match = entity

        if best_score >= self.similarity_threshold:
            # Optional LLM verification for edge cases
            if self.use_llm_verification and self.llm and 0.85 <= best_score < 0.95:
                if self._verify_with_llm(entity_name, best_match.name):
                    return best_match
                else:
                    return None

            return best_match

        return None

    def resolve(
        self,
        entity: Entity,
        existing_entities: list[Entity],
    ) -> ResolutionResult:
        """
        Resolve an entity against existing entities.

        Args:
            entity: New entity to resolve
            existing_entities: Existing entities in the graph

        Returns:
            ResolutionResult indicating if duplicate and matched entity
        """
        matched = self.find_similar(
            entity.name,
            existing_entities,
            entity.entity_type,
        )

        if matched:
            similarity = self.compute_similarity(entity.name, matched.name)
            return ResolutionResult(
                is_duplicate=True,
                matched_entity=matched,
                similarity_score=similarity,
                canonical_name=matched.name,
            )

        return ResolutionResult(
            is_duplicate=False,
            canonical_name=self.normalize_name(entity.name).title(),
        )

    def _verify_with_llm(self, name1: str, name2: str) -> bool:
        """Use LLM to verify if two names refer to the same concept."""
        prompt = f"""Do these two terms refer to the same educational concept?

Term 1: {name1}
Term 2: {name2}

Answer only "yes" or "no".
"""
        response = self.llm.generate(prompt).lower().strip()
        return "yes" in response

    def merge_entities(self, entity1: Entity, entity2: Entity) -> Entity:
        """
        Merge two entities into one.

        Takes the longer/more descriptive name and combines properties.
        """
        # Use the longer name (usually more descriptive)
        if len(entity1.name) >= len(entity2.name):
            primary, secondary = entity1, entity2
        else:
            primary, secondary = entity2, entity1

        # Merge properties
        merged_properties = {**secondary.properties, **primary.properties}

        # Use description from whichever has one
        description = primary.description or secondary.description

        return Entity(
            id=primary.id,
            name=primary.name,
            entity_type=primary.entity_type,
            description=description,
            properties=merged_properties,
            embedding=primary.embedding,
            source_chunk=primary.source_chunk,
        )

    def add_synonym(self, abbreviation: str, full_form: str):
        """Add a custom synonym mapping."""
        self.synonym_map[abbreviation.lower()] = full_form.lower()

    def batch_resolve(
        self,
        entities: list[Entity],
        existing_entities: list[Entity],
    ) -> tuple[list[Entity], list[tuple[Entity, Entity]]]:
        """
        Resolve a batch of entities.

        Returns:
            Tuple of (new_entities, merged_pairs)
        """
        new_entities = []
        merged_pairs = []
        resolved_names = set()

        for entity in entities:
            normalized = self.normalize_name(entity.name)

            # Skip if we already resolved this name in this batch
            if normalized in resolved_names:
                continue

            result = self.resolve(entity, existing_entities + new_entities)

            if result.is_duplicate:
                merged_pairs.append((entity, result.matched_entity))
            else:
                # Update name to canonical form
                entity.name = result.canonical_name or entity.name
                new_entities.append(entity)
                resolved_names.add(normalized)

        return new_entities, merged_pairs
