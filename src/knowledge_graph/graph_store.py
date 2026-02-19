"""Graph storage backends for the educational knowledge graph."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path

from .ontology import Entity, Relationship, EntityType, RelationType


class GraphStore(ABC):
    """Abstract base class for graph storage."""

    @abstractmethod
    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the graph."""
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship to the graph."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        pass

    @abstractmethod
    def get_all_entities(self) -> list[Entity]:
        """Get all entities in the graph."""
        pass

    @abstractmethod
    def find_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Find entities by type."""
        pass

    @abstractmethod
    def get_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[RelationType] = None,
    ) -> list[Relationship]:
        """Get relationships matching criteria."""
        pass

    @abstractmethod
    def get_prerequisites(self, entity_id: str) -> list[Entity]:
        """Get prerequisite concepts for an entity."""
        pass

    @abstractmethod
    def get_dependents(self, entity_id: str) -> list[Entity]:
        """Get concepts that depend on this entity."""
        pass


class InMemoryGraphStore(GraphStore):
    """
    Simple in-memory graph store for development and testing.

    For production, use Neo4jStore.
    """

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []

    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the graph."""
        if entity.id in self.entities:
            return False
        self.entities[entity.id] = entity
        return True

    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship to the graph."""
        # Verify both entities exist
        if relationship.source_id not in self.entities:
            return False
        if relationship.target_id not in self.entities:
            return False

        # Check for duplicates
        for r in self.relationships:
            if (r.source_id == relationship.source_id and
                r.target_id == relationship.target_id and
                r.relation_type == relationship.relation_type):
                return False

        self.relationships.append(relationship)
        return True

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_all_entities(self) -> list[Entity]:
        """Get all entities."""
        return list(self.entities.values())

    def find_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Find entities by type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def find_entity_by_name(self, name: str) -> Optional[Entity]:
        """Find entity by name (case-insensitive)."""
        name_lower = name.lower()
        for entity in self.entities.values():
            if entity.name.lower() == name_lower:
                return entity
        return None

    def get_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[RelationType] = None,
    ) -> list[Relationship]:
        """Get relationships matching criteria."""
        results = []
        for r in self.relationships:
            if source_id and r.source_id != source_id:
                continue
            if target_id and r.target_id != target_id:
                continue
            if relation_type and r.relation_type != relation_type:
                continue
            results.append(r)
        return results

    def get_prerequisites(self, entity_id: str) -> list[Entity]:
        """Get prerequisite concepts for an entity."""
        prereq_rels = self.get_relationships(
            target_id=entity_id,
            relation_type=RelationType.PREREQUISITE_OF,
        )
        return [self.entities[r.source_id] for r in prereq_rels if r.source_id in self.entities]

    def get_dependents(self, entity_id: str) -> list[Entity]:
        """Get concepts that depend on this entity."""
        dependent_rels = self.get_relationships(
            source_id=entity_id,
            relation_type=RelationType.PREREQUISITE_OF,
        )
        return [self.entities[r.target_id] for r in dependent_rels if r.target_id in self.entities]

    def get_learning_path(self, target_entity_id: str) -> list[Entity]:
        """
        Get the learning path (ordered prerequisites) for a concept.

        Uses topological sort on prerequisites.
        """
        visited = set()
        path = []

        def dfs(entity_id: str):
            if entity_id in visited:
                return
            visited.add(entity_id)

            for prereq in self.get_prerequisites(entity_id):
                dfs(prereq.id)

            if entity_id in self.entities:
                path.append(self.entities[entity_id])

        dfs(target_entity_id)
        return path

    def save(self, path: str | Path):
        """Save graph to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relationships": [r.to_dict() for r in self.relationships],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryGraphStore":
        """Load graph from JSON file."""
        with open(path) as f:
            data = json.load(f)

        store = cls()

        for e_data in data["entities"]:
            entity = Entity(
                id=e_data["id"],
                name=e_data["name"],
                entity_type=EntityType[e_data["type"]],
                description=e_data.get("description"),
                properties=e_data.get("properties", {}),
            )
            store.entities[entity.id] = entity

        for r_data in data["relationships"]:
            relationship = Relationship(
                source_id=r_data["source"],
                target_id=r_data["target"],
                relation_type=RelationType[r_data["type"]],
                properties=r_data.get("properties", {}),
                confidence=r_data.get("confidence", 1.0),
            )
            store.relationships.append(relationship)

        return store

    def __len__(self) -> int:
        return len(self.entities)


class Neo4jStore(GraphStore):
    """
    Neo4j graph database backend.

    Provides full graph database capabilities including:
    - Vector indexes for semantic search
    - Cypher queries
    - Transaction support
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j bolt URI
            user: Username
            password: Password
            database: Database name
        """
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.database = database
        except ImportError:
            raise ImportError("neo4j package required. Install with: pip install neo4j")

    def close(self):
        """Close the database connection."""
        self.driver.close()

    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to Neo4j."""
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.description = $description
        RETURN e
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                id=entity.id,
                name=entity.name,
                type=entity.entity_type.value,
                description=entity.description,
            )
            return result.single() is not None

    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship to Neo4j."""
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{relationship.relation_type.value}]->(target)
        SET r.confidence = $confidence
        RETURN r
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                confidence=relationship.confidence,
            )
            return result.single() is not None

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID from Neo4j."""
        query = """
        MATCH (e:Entity {id: $id})
        RETURN e
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=entity_id)
            record = result.single()

            if record:
                node = record["e"]
                return Entity(
                    id=node["id"],
                    name=node["name"],
                    entity_type=EntityType[node["type"]],
                    description=node.get("description"),
                )
            return None

    def get_all_entities(self) -> list[Entity]:
        """Get all entities from Neo4j."""
        query = "MATCH (e:Entity) RETURN e"

        entities = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                node = record["e"]
                entity = Entity(
                    id=node["id"],
                    name=node["name"],
                    entity_type=EntityType[node["type"]],
                    description=node.get("description"),
                )
                entities.append(entity)

        return entities

    def find_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Find entities by type in Neo4j."""
        query = """
        MATCH (e:Entity {type: $type})
        RETURN e
        """

        entities = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, type=entity_type.value)
            for record in result:
                node = record["e"]
                entity = Entity(
                    id=node["id"],
                    name=node["name"],
                    entity_type=entity_type,
                    description=node.get("description"),
                )
                entities.append(entity)

        return entities

    def get_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[RelationType] = None,
    ) -> list[Relationship]:
        """Get relationships from Neo4j."""
        conditions = []
        params = {}

        if source_id:
            conditions.append("source.id = $source_id")
            params["source_id"] = source_id

        if target_id:
            conditions.append("target.id = $target_id")
            params["target_id"] = target_id

        rel_pattern = "[r]" if not relation_type else f"[r:{relation_type.value}]"

        where_clause = " AND ".join(conditions) if conditions else "true"

        query = f"""
        MATCH (source:Entity)-{rel_pattern}->(target:Entity)
        WHERE {where_clause}
        RETURN source.id as source_id, target.id as target_id, type(r) as rel_type, r.confidence as confidence
        """

        relationships = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            for record in result:
                rel = Relationship(
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    relation_type=RelationType[record["rel_type"]],
                    confidence=record.get("confidence", 1.0),
                )
                relationships.append(rel)

        return relationships

    def get_prerequisites(self, entity_id: str) -> list[Entity]:
        """Get prerequisites from Neo4j."""
        query = """
        MATCH (prereq:Entity)-[:PREREQUISITE_OF]->(e:Entity {id: $id})
        RETURN prereq
        """

        entities = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=entity_id)
            for record in result:
                node = record["prereq"]
                entity = Entity(
                    id=node["id"],
                    name=node["name"],
                    entity_type=EntityType[node["type"]],
                    description=node.get("description"),
                )
                entities.append(entity)

        return entities

    def get_dependents(self, entity_id: str) -> list[Entity]:
        """Get dependent concepts from Neo4j."""
        query = """
        MATCH (e:Entity {id: $id})-[:PREREQUISITE_OF]->(dependent:Entity)
        RETURN dependent
        """

        entities = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=entity_id)
            for record in result:
                node = record["dependent"]
                entity = Entity(
                    id=node["id"],
                    name=node["name"],
                    entity_type=EntityType[node["type"]],
                    description=node.get("description"),
                )
                entities.append(entity)

        return entities

    def get_learning_path(self, target_entity_id: str) -> list[Entity]:
        """Get learning path using Cypher."""
        query = """
        MATCH path = (prereq:Entity)-[:PREREQUISITE_OF*]->(target:Entity {id: $id})
        UNWIND nodes(path) as n
        RETURN DISTINCT n
        ORDER BY length(path) DESC
        """

        entities = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=target_entity_id)
            for record in result:
                node = record["n"]
                entity = Entity(
                    id=node["id"],
                    name=node["name"],
                    entity_type=EntityType[node["type"]],
                    description=node.get("description"),
                )
                entities.append(entity)

        return entities

    def run_cypher(self, query: str, **params) -> list[dict]:
        """Run a raw Cypher query."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
