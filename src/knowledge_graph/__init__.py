"""Knowledge Graph construction and management for educational content."""

from .ontology import EducationalOntology, EntityType, RelationType
from .react_agent import KGReActAgent, AgentAction, AgentObservation
from .extractor import KGExtractor, ExtractedTriple
from .entity_resolver import EntityResolver
from .graph_store import GraphStore, Neo4jStore, InMemoryGraphStore

__all__ = [
    "EducationalOntology",
    "EntityType",
    "RelationType",
    "KGReActAgent",
    "AgentAction",
    "AgentObservation",
    "KGExtractor",
    "ExtractedTriple",
    "EntityResolver",
    "GraphStore",
    "Neo4jStore",
    "InMemoryGraphStore",
]
