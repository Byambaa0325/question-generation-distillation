"""Educational ontology definitions for the knowledge graph."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EntityType(Enum):
    """Types of entities in the educational knowledge graph."""

    CONCEPT = "Concept"           # Core knowledge unit (e.g., "Photosynthesis")
    TOPIC = "Topic"               # Broader category (e.g., "Plant Biology")
    LESSON = "Lesson"             # Learning unit
    RESOURCE = "Resource"         # Learning material (PDF, video, etc.)
    QUIZ_QUESTION = "QuizQuestion"
    LEARNING_OBJECTIVE = "LearningObjective"
    SKILL = "Skill"               # Practical ability
    EXAMPLE = "Example"           # Concrete example of a concept


class RelationType(Enum):
    """Types of relationships between entities."""

    # Prerequisite relationships (crucial for learning paths)
    PREREQUISITE_OF = "PREREQUISITE_OF"      # A must be learned before B
    BUILDS_ON = "BUILDS_ON"                  # A extends/deepens B

    # Hierarchical relationships
    CONTAINS = "CONTAINS"                    # Topic contains Concept
    PART_OF = "PART_OF"                      # Concept is part of Topic
    SUBTOPIC_OF = "SUBTOPIC_OF"              # Hierarchy within topics

    # Content relationships
    EXPLAINS = "EXPLAINS"                    # Resource explains Concept
    DEMONSTRATES = "DEMONSTRATES"            # Example demonstrates Concept
    ASSESSES = "ASSESSES"                    # QuizQuestion assesses Concept

    # Semantic relationships
    RELATED_TO = "RELATED_TO"                # General semantic relation
    SIMILAR_TO = "SIMILAR_TO"                # Concepts are similar
    CONTRASTS_WITH = "CONTRASTS_WITH"        # Concepts are opposites/contrasts
    APPLIES_TO = "APPLIES_TO"                # Concept applies to domain/context

    # Learning relationships
    TEACHES = "TEACHES"                      # Lesson teaches Concept
    ACHIEVES = "ACHIEVES"                    # Learning Concept achieves Objective
    REQUIRES_SKILL = "REQUIRES_SKILL"        # Concept requires Skill


@dataclass
class Entity:
    """An entity in the knowledge graph."""

    id: str
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    properties: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    source_chunk: Optional[str] = None  # Original text this was extracted from

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type.value,
            "description": self.description,
            "properties": self.properties,
        }


@dataclass
class Relationship:
    """A relationship between two entities."""

    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relation_type.value,
            "properties": self.properties,
            "confidence": self.confidence,
        }


class EducationalOntology:
    """
    Educational ontology defining valid entity types and relationships.

    This ontology is specifically designed for educational content,
    with a focus on prerequisite relationships and learning paths.
    """

    # Valid relationships for each entity type pair
    VALID_RELATIONS = {
        (EntityType.CONCEPT, EntityType.CONCEPT): [
            RelationType.PREREQUISITE_OF,
            RelationType.BUILDS_ON,
            RelationType.RELATED_TO,
            RelationType.SIMILAR_TO,
            RelationType.CONTRASTS_WITH,
        ],
        (EntityType.TOPIC, EntityType.CONCEPT): [
            RelationType.CONTAINS,
        ],
        (EntityType.CONCEPT, EntityType.TOPIC): [
            RelationType.PART_OF,
        ],
        (EntityType.TOPIC, EntityType.TOPIC): [
            RelationType.SUBTOPIC_OF,
            RelationType.RELATED_TO,
        ],
        (EntityType.RESOURCE, EntityType.CONCEPT): [
            RelationType.EXPLAINS,
        ],
        (EntityType.EXAMPLE, EntityType.CONCEPT): [
            RelationType.DEMONSTRATES,
        ],
        (EntityType.QUIZ_QUESTION, EntityType.CONCEPT): [
            RelationType.ASSESSES,
        ],
        (EntityType.LESSON, EntityType.CONCEPT): [
            RelationType.TEACHES,
        ],
        (EntityType.CONCEPT, EntityType.LEARNING_OBJECTIVE): [
            RelationType.ACHIEVES,
        ],
        (EntityType.CONCEPT, EntityType.SKILL): [
            RelationType.REQUIRES_SKILL,
        ],
    }

    @classmethod
    def get_valid_relations(
        cls,
        source_type: EntityType,
        target_type: EntityType,
    ) -> list[RelationType]:
        """Get valid relationship types between two entity types."""
        return cls.VALID_RELATIONS.get((source_type, target_type), [RelationType.RELATED_TO])

    @classmethod
    def is_valid_triple(
        cls,
        source_type: EntityType,
        relation: RelationType,
        target_type: EntityType,
    ) -> bool:
        """Check if a triple is valid according to the ontology."""
        valid_relations = cls.get_valid_relations(source_type, target_type)
        return relation in valid_relations or relation == RelationType.RELATED_TO

    @classmethod
    def get_schema_description(cls) -> str:
        """Get a natural language description of the ontology for LLM prompts."""
        return """Educational Knowledge Graph Ontology:

ENTITY TYPES:
- Concept: Core knowledge units (e.g., "Photosynthesis", "Derivatives", "Neural Networks")
- Topic: Broader categories containing concepts (e.g., "Plant Biology", "Calculus")
- Lesson: Teaching units that cover concepts
- Resource: Learning materials (PDFs, videos, articles)
- QuizQuestion: Assessment items testing concepts
- LearningObjective: Goals that concepts help achieve
- Skill: Practical abilities required or taught
- Example: Concrete demonstrations of concepts

RELATIONSHIP TYPES:
- PREREQUISITE_OF: Concept A must be understood before Concept B
- BUILDS_ON: Concept A extends or deepens Concept B
- CONTAINS: Topic contains Concept (hierarchical)
- PART_OF: Concept belongs to Topic
- EXPLAINS: Resource explains Concept
- DEMONSTRATES: Example demonstrates Concept
- ASSESSES: QuizQuestion tests understanding of Concept
- RELATED_TO: General semantic relationship
- SIMILAR_TO: Concepts are closely related
- CONTRASTS_WITH: Concepts are opposites or contrasting"""
