"""ReAct Agent for Knowledge Graph extraction from educational content."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
import json
import re

from .ontology import EducationalOntology, EntityType, RelationType, Entity, Relationship


class ActionType(Enum):
    """Types of actions the ReAct agent can take."""

    EXTRACT_ENTITIES = "extract_entities"
    EXTRACT_RELATIONS = "extract_relations"
    RESOLVE_ENTITY = "resolve_entity"
    VALIDATE_TRIPLE = "validate_triple"
    SEARCH_GRAPH = "search_graph"
    ADD_TO_GRAPH = "add_to_graph"
    FINISH = "finish"


@dataclass
class AgentAction:
    """An action taken by the agent."""

    action_type: ActionType
    action_input: dict
    thought: str  # The reasoning behind this action


@dataclass
class AgentObservation:
    """The result of an action."""

    result: dict
    success: bool
    error: Optional[str] = None


class KGReActAgent:
    """
    ReAct (Reasoning + Acting) Agent for Knowledge Graph construction.

    This agent follows the ReAct paradigm:
    1. Thought: Reason about the current state and what to do
    2. Action: Take an action (extract entities, relations, etc.)
    3. Observation: Observe the result
    4. Repeat until task is complete

    The agent is specifically designed for educational content extraction.
    """

    def __init__(
        self,
        llm,  # Teacher model with generate() method
        graph_store=None,
        entity_resolver=None,
        max_iterations: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize the ReAct agent.

        Args:
            llm: Language model for reasoning and extraction
            graph_store: Graph storage backend
            entity_resolver: Entity resolution component
            max_iterations: Maximum reasoning iterations
            verbose: Print agent's thoughts
        """
        self.llm = llm
        self.graph_store = graph_store
        self.entity_resolver = entity_resolver
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Track extracted data
        self.extracted_entities: list[Entity] = []
        self.extracted_relations: list[Relationship] = []
        self.trace: list[tuple[AgentAction, AgentObservation]] = []

    def process_chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Process a text chunk to extract knowledge graph elements.

        Args:
            text: Educational text to process
            metadata: Optional metadata (chapter, section, etc.)

        Returns:
            Dictionary with extracted entities and relationships
        """
        self.extracted_entities = []
        self.extracted_relations = []
        self.trace = []

        # Initial prompt with context
        context = self._build_initial_context(text, metadata)

        for iteration in range(self.max_iterations):
            # Get agent's thought and action
            action = self._get_next_action(context, iteration)

            if self.verbose:
                print(f"\n[Iteration {iteration + 1}]")
                print(f"Thought: {action.thought}")
                print(f"Action: {action.action_type.value}")

            # Execute the action
            observation = self._execute_action(action, text)

            if self.verbose:
                print(f"Observation: {observation.result if observation.success else observation.error}")

            # Record trace
            self.trace.append((action, observation))

            # Update context with observation
            context = self._update_context(context, action, observation)

            # Check if done
            if action.action_type == ActionType.FINISH:
                break

        return {
            "entities": [e.to_dict() for e in self.extracted_entities],
            "relationships": [r.to_dict() for r in self.extracted_relations],
            "trace": [(a.thought, a.action_type.value) for a, o in self.trace],
        }

    def _build_initial_context(self, text: str, metadata: Optional[dict]) -> str:
        """Build the initial context for the agent."""
        ontology_desc = EducationalOntology.get_schema_description()

        metadata_str = ""
        if metadata:
            metadata_str = f"\nDocument Metadata: {json.dumps(metadata)}"

        return f"""You are an educational knowledge graph extraction agent.

{ontology_desc}

Your task is to extract entities and relationships from the following educational text.
Follow the ReAct framework: Think about what to do, take an action, observe the result.
{metadata_str}

TEXT TO PROCESS:
{text}

Available Actions:
1. extract_entities - Extract entities from the text
2. extract_relations - Extract relationships between identified entities
3. resolve_entity - Check if an entity already exists in the graph
4. validate_triple - Validate a relationship against the ontology
5. add_to_graph - Add validated entities/relations to the graph
6. finish - Complete the extraction process

Respond with your thought process and chosen action in this format:
Thought: [Your reasoning]
Action: [action_name]
Action Input: [JSON input for the action]
"""

    def _get_next_action(self, context: str, iteration: int) -> AgentAction:
        """Get the next action from the LLM."""
        # Add iteration-specific guidance
        if iteration == 0:
            prompt = context + "\n\nStart by extracting entities from the text."
        elif iteration < 3:
            prompt = context + "\n\nContinue with the next logical step."
        else:
            prompt = context + "\n\nWrap up extraction if you have identified the main concepts."

        response = self.llm.generate(prompt)

        # Parse the response
        return self._parse_action(response)

    def _parse_action(self, response: str) -> AgentAction:
        """Parse the LLM response into an action."""
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "Continuing extraction"

        # Extract action
        action_match = re.search(r"Action:\s*(\w+)", response)
        action_str = action_match.group(1).lower() if action_match else "finish"

        # Map to ActionType
        action_map = {
            "extract_entities": ActionType.EXTRACT_ENTITIES,
            "extract_relations": ActionType.EXTRACT_RELATIONS,
            "resolve_entity": ActionType.RESOLVE_ENTITY,
            "validate_triple": ActionType.VALIDATE_TRIPLE,
            "search_graph": ActionType.SEARCH_GRAPH,
            "add_to_graph": ActionType.ADD_TO_GRAPH,
            "finish": ActionType.FINISH,
        }
        action_type = action_map.get(action_str, ActionType.FINISH)

        # Extract action input
        input_match = re.search(r"Action Input:\s*(\{.+?\}|\[.+?\])", response, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                action_input = {}
        else:
            action_input = {}

        return AgentAction(
            action_type=action_type,
            action_input=action_input,
            thought=thought,
        )

    def _execute_action(self, action: AgentAction, text: str) -> AgentObservation:
        """Execute an action and return the observation."""
        try:
            if action.action_type == ActionType.EXTRACT_ENTITIES:
                return self._action_extract_entities(text)

            elif action.action_type == ActionType.EXTRACT_RELATIONS:
                return self._action_extract_relations(text)

            elif action.action_type == ActionType.RESOLVE_ENTITY:
                return self._action_resolve_entity(action.action_input)

            elif action.action_type == ActionType.VALIDATE_TRIPLE:
                return self._action_validate_triple(action.action_input)

            elif action.action_type == ActionType.ADD_TO_GRAPH:
                return self._action_add_to_graph()

            elif action.action_type == ActionType.FINISH:
                return AgentObservation(
                    result={"status": "completed", "entities": len(self.extracted_entities)},
                    success=True,
                )

            else:
                return AgentObservation(
                    result={},
                    success=False,
                    error=f"Unknown action: {action.action_type}",
                )

        except Exception as e:
            return AgentObservation(result={}, success=False, error=str(e))

    def _action_extract_entities(self, text: str) -> AgentObservation:
        """Extract entities from text using LLM."""
        prompt = f"""Extract educational entities from this text.

{EducationalOntology.get_schema_description()}

Text: {text}

Return a JSON array of entities with format:
[
  {{"name": "Entity Name", "type": "Concept|Topic|...", "description": "Brief description"}}
]

Only extract clear, distinct educational concepts. Be specific, not generic.
JSON:"""

        response = self.llm.generate(prompt)

        # Parse JSON from response
        try:
            # Find JSON array in response
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                entities_data = json.loads(json_match.group())
            else:
                entities_data = []

            # Convert to Entity objects
            for i, e in enumerate(entities_data):
                entity = Entity(
                    id=f"entity_{len(self.extracted_entities) + i}",
                    name=e.get("name", ""),
                    entity_type=EntityType[e.get("type", "CONCEPT").upper()],
                    description=e.get("description"),
                    source_chunk=text[:200],
                )
                self.extracted_entities.append(entity)

            return AgentObservation(
                result={"extracted": len(entities_data), "entities": [e["name"] for e in entities_data]},
                success=True,
            )

        except (json.JSONDecodeError, KeyError) as e:
            return AgentObservation(result={}, success=False, error=f"Parse error: {e}")

    def _action_extract_relations(self, text: str) -> AgentObservation:
        """Extract relationships between entities."""
        if not self.extracted_entities:
            return AgentObservation(
                result={},
                success=False,
                error="No entities extracted yet. Extract entities first.",
            )

        entity_names = [e.name for e in self.extracted_entities]

        prompt = f"""Extract relationships between these educational entities.

Entities: {entity_names}

Valid relationship types:
- PREREQUISITE_OF: A must be learned before B
- BUILDS_ON: A extends B
- RELATED_TO: A is related to B
- CONTAINS: Topic contains Concept
- SIMILAR_TO: A is similar to B
- CONTRASTS_WITH: A contrasts with B

Original text for context: {text[:500]}

Return a JSON array of relationships:
[
  {{"source": "Entity A", "target": "Entity B", "relation": "PREREQUISITE_OF"}}
]

JSON:"""

        response = self.llm.generate(prompt)

        try:
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                relations_data = json.loads(json_match.group())
            else:
                relations_data = []

            # Create entity name to ID mapping
            name_to_id = {e.name.lower(): e.id for e in self.extracted_entities}

            for r in relations_data:
                source_name = r.get("source", "").lower()
                target_name = r.get("target", "").lower()

                source_id = name_to_id.get(source_name)
                target_id = name_to_id.get(target_name)

                if source_id and target_id:
                    try:
                        relation_type = RelationType[r.get("relation", "RELATED_TO").upper()]
                    except KeyError:
                        relation_type = RelationType.RELATED_TO

                    relationship = Relationship(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=relation_type,
                    )
                    self.extracted_relations.append(relationship)

            return AgentObservation(
                result={"extracted": len(self.extracted_relations)},
                success=True,
            )

        except (json.JSONDecodeError, KeyError) as e:
            return AgentObservation(result={}, success=False, error=f"Parse error: {e}")

    def _action_resolve_entity(self, action_input: dict) -> AgentObservation:
        """Check if an entity already exists in the graph."""
        entity_name = action_input.get("name", "")

        if self.entity_resolver and self.graph_store:
            existing = self.entity_resolver.find_similar(
                entity_name,
                self.graph_store.get_all_entities(),
            )
            if existing:
                return AgentObservation(
                    result={"existing": existing.name, "id": existing.id},
                    success=True,
                )

        return AgentObservation(
            result={"existing": None, "is_new": True},
            success=True,
        )

    def _action_validate_triple(self, action_input: dict) -> AgentObservation:
        """Validate a triple against the ontology."""
        try:
            source_type = EntityType[action_input.get("source_type", "CONCEPT").upper()]
            relation = RelationType[action_input.get("relation", "RELATED_TO").upper()]
            target_type = EntityType[action_input.get("target_type", "CONCEPT").upper()]

            is_valid = EducationalOntology.is_valid_triple(source_type, relation, target_type)

            return AgentObservation(
                result={"valid": is_valid},
                success=True,
            )
        except KeyError as e:
            return AgentObservation(result={}, success=False, error=f"Invalid type: {e}")

    def _action_add_to_graph(self) -> AgentObservation:
        """Add extracted entities and relations to the graph store."""
        if not self.graph_store:
            return AgentObservation(
                result={"status": "no_graph_store"},
                success=True,
            )

        added_entities = 0
        added_relations = 0

        for entity in self.extracted_entities:
            if self.graph_store.add_entity(entity):
                added_entities += 1

        for relation in self.extracted_relations:
            if self.graph_store.add_relationship(relation):
                added_relations += 1

        return AgentObservation(
            result={"added_entities": added_entities, "added_relations": added_relations},
            success=True,
        )

    def _update_context(
        self,
        context: str,
        action: AgentAction,
        observation: AgentObservation,
    ) -> str:
        """Update context with the latest action and observation."""
        update = f"""
Previous Action: {action.action_type.value}
Observation: {json.dumps(observation.result) if observation.success else observation.error}

Current extracted entities: {[e.name for e in self.extracted_entities]}
Current extracted relations: {len(self.extracted_relations)}

What should be the next step?
"""
        return context + update
