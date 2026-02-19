"""Knowledge graph extraction utilities."""

from dataclasses import dataclass
from typing import Optional
import re
import json

from .ontology import EntityType, RelationType, Entity, Relationship, EducationalOntology


@dataclass
class ExtractedTriple:
    """A single extracted triple (subject, predicate, object)."""

    subject: str
    subject_type: EntityType
    predicate: RelationType
    object: str
    object_type: EntityType
    confidence: float = 1.0
    source_text: Optional[str] = None


class KGExtractor:
    """
    Direct knowledge graph extraction without agent loop.

    Use this for simpler extraction tasks or when you need
    more control over the extraction process.
    """

    def __init__(self, llm, ontology: Optional[EducationalOntology] = None):
        """
        Initialize extractor.

        Args:
            llm: Language model with generate() method
            ontology: Optional custom ontology (uses default if None)
        """
        self.llm = llm
        self.ontology = ontology or EducationalOntology()

    def extract_entities(self, text: str, entity_types: list[EntityType] | None = None) -> list[Entity]:
        """
        Extract entities of specified types from text.

        Args:
            text: Educational text
            entity_types: Types to extract (default: Concept, Topic)

        Returns:
            List of extracted entities
        """
        if entity_types is None:
            entity_types = [EntityType.CONCEPT, EntityType.TOPIC]

        type_names = [t.value for t in entity_types]

        prompt = f"""Extract educational entities from this text.

Entity types to extract: {type_names}

Text:
{text}

Instructions:
- Extract specific, well-defined educational concepts
- Each entity should be distinct and meaningful
- Avoid generic terms like "learning" or "understanding"
- Focus on domain-specific terminology

Return as JSON array:
[
  {{"name": "Entity Name", "type": "Concept", "description": "Brief description"}}
]

JSON:"""

        response = self.llm.generate(prompt)
        return self._parse_entities(response, text)

    def extract_relations(
        self,
        text: str,
        entities: list[Entity],
        relation_types: list[RelationType] | None = None,
    ) -> list[Relationship]:
        """
        Extract relationships between given entities.

        Args:
            text: Original text for context
            entities: List of entities to find relationships between
            relation_types: Types of relations to extract

        Returns:
            List of extracted relationships
        """
        if not entities:
            return []

        if relation_types is None:
            relation_types = [
                RelationType.PREREQUISITE_OF,
                RelationType.BUILDS_ON,
                RelationType.RELATED_TO,
                RelationType.CONTAINS,
            ]

        entity_names = [e.name for e in entities]
        relation_names = [r.value for r in relation_types]

        prompt = f"""Identify relationships between these educational entities.

Entities: {entity_names}

Valid relationship types:
{self._format_relation_descriptions(relation_types)}

Original text for context:
{text[:1000]}

Instructions:
- Only identify relationships that are clearly supported by the text
- Focus on prerequisite and hierarchical relationships
- Be specific about the direction of relationships

Return as JSON array:
[
  {{"source": "Entity A", "target": "Entity B", "relation": "PREREQUISITE_OF", "confidence": 0.9}}
]

JSON:"""

        response = self.llm.generate(prompt)
        return self._parse_relations(response, entities)

    def extract_triples(self, text: str) -> list[ExtractedTriple]:
        """
        Extract complete triples (subject-predicate-object) from text.

        This is a single-pass extraction that extracts both entities
        and relationships together.
        """
        prompt = f"""Extract knowledge graph triples from this educational text.

{EducationalOntology.get_schema_description()}

Text:
{text}

Extract triples in the format (Subject, Predicate, Object).
Focus on:
1. Prerequisite relationships between concepts
2. Hierarchical relationships (topic contains concept)
3. Semantic relationships (concepts that are related/similar)

Return as JSON array:
[
  {{
    "subject": "Concept A",
    "subject_type": "Concept",
    "predicate": "PREREQUISITE_OF",
    "object": "Concept B",
    "object_type": "Concept",
    "confidence": 0.9
  }}
]

JSON:"""

        response = self.llm.generate(prompt)
        return self._parse_triples(response, text)

    def _format_relation_descriptions(self, relation_types: list[RelationType]) -> str:
        """Format relation type descriptions for prompt."""
        descriptions = {
            RelationType.PREREQUISITE_OF: "A must be understood before B",
            RelationType.BUILDS_ON: "A extends or deepens B",
            RelationType.RELATED_TO: "A is semantically related to B",
            RelationType.CONTAINS: "A (topic) contains B (concept)",
            RelationType.SIMILAR_TO: "A and B are similar concepts",
            RelationType.CONTRASTS_WITH: "A and B are contrasting concepts",
        }

        lines = []
        for rt in relation_types:
            desc = descriptions.get(rt, "General relationship")
            lines.append(f"- {rt.value}: {desc}")

        return "\n".join(lines)

    def _parse_entities(self, response: str, source_text: str) -> list[Entity]:
        """Parse entities from LLM response."""
        entities = []

        try:
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                data = json.loads(json_match.group())

                for i, item in enumerate(data):
                    try:
                        entity_type = EntityType[item.get("type", "CONCEPT").upper()]
                    except KeyError:
                        entity_type = EntityType.CONCEPT

                    entity = Entity(
                        id=f"e_{i}_{hash(item.get('name', '')) % 10000}",
                        name=item.get("name", ""),
                        entity_type=entity_type,
                        description=item.get("description"),
                        source_chunk=source_text[:200],
                    )
                    entities.append(entity)

        except json.JSONDecodeError:
            pass

        return entities

    def _parse_relations(self, response: str, entities: list[Entity]) -> list[Relationship]:
        """Parse relationships from LLM response."""
        relationships = []

        # Build name to ID mapping
        name_to_id = {e.name.lower(): e.id for e in entities}

        try:
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                data = json.loads(json_match.group())

                for item in data:
                    source_name = item.get("source", "").lower()
                    target_name = item.get("target", "").lower()

                    source_id = name_to_id.get(source_name)
                    target_id = name_to_id.get(target_name)

                    if source_id and target_id:
                        try:
                            relation_type = RelationType[item.get("relation", "RELATED_TO").upper()]
                        except KeyError:
                            relation_type = RelationType.RELATED_TO

                        relationship = Relationship(
                            source_id=source_id,
                            target_id=target_id,
                            relation_type=relation_type,
                            confidence=float(item.get("confidence", 1.0)),
                        )
                        relationships.append(relationship)

        except json.JSONDecodeError:
            pass

        return relationships

    def _parse_triples(self, response: str, source_text: str) -> list[ExtractedTriple]:
        """Parse triples from LLM response."""
        triples = []

        try:
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                data = json.loads(json_match.group())

                for item in data:
                    try:
                        subject_type = EntityType[item.get("subject_type", "CONCEPT").upper()]
                        object_type = EntityType[item.get("object_type", "CONCEPT").upper()]
                        predicate = RelationType[item.get("predicate", "RELATED_TO").upper()]
                    except KeyError:
                        continue

                    triple = ExtractedTriple(
                        subject=item.get("subject", ""),
                        subject_type=subject_type,
                        predicate=predicate,
                        object=item.get("object", ""),
                        object_type=object_type,
                        confidence=float(item.get("confidence", 1.0)),
                        source_text=source_text[:200],
                    )
                    triples.append(triple)

        except json.JSONDecodeError:
            pass

        return triples
