#!/usr/bin/env python3
"""Build educational knowledge graph from documents using ReAct agent."""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from config import get_settings
from src.models.teacher import get_teacher
from src.knowledge_graph import (
    KGReActAgent,
    KGExtractor,
    EntityResolver,
    InMemoryGraphStore,
    EducationalOntology,
)
from src.knowledge_graph.chunking import chunk_for_kg_extraction


def main():
    parser = argparse.ArgumentParser(description="Build knowledge graph from documents")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input file (JSON with 'text' field) or directory of .txt files"
    )
    parser.add_argument(
        "--output", type=str, default="data/knowledge_graph/graph.json",
        help="Output path for the knowledge graph"
    )
    parser.add_argument(
        "--mode", type=str, choices=["react", "direct"], default="react",
        help="Extraction mode: 'react' (agent-based) or 'direct' (simple extraction)"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="Maximum number of chunks to process (for testing)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1500,
        help="Maximum chunk size in characters"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print agent thoughts and actions"
    )
    args = parser.parse_args()

    # Load settings
    settings = get_settings(args.config)

    # Initialize teacher LLM
    print("Initializing LLM...")
    if settings.teacher.backend == "colab":
        teacher = get_teacher("colab")
    elif settings.teacher.backend == "gemini":
        teacher = get_teacher(
            "gemini",
            model_name=settings.teacher.gemini.model_name,
            temperature=0.3,  # Lower temperature for extraction
            max_tokens=1024,
        )
    else:
        teacher = get_teacher(
            "ollama",
            model_name=settings.teacher.ollama.model_name,
            base_url=settings.teacher.ollama.base_url,
            temperature=0.3,
            max_tokens=1024,
        )

    # Initialize components
    graph_store = InMemoryGraphStore()
    entity_resolver = EntityResolver(
        embedding_model=settings.concepts.embedding_model,
        similarity_threshold=settings.concepts.similarity_threshold,
    )

    # Load documents
    input_path = Path(args.input)
    documents = load_documents(input_path)
    print(f"Loaded {len(documents)} documents")

    # Chunk documents
    all_chunks = []
    for doc in documents:
        chunks = chunk_for_kg_extraction(
            doc["text"],
            max_chunk_size=args.chunk_size,
        )
        for chunk in chunks:
            chunk["metadata"]["source"] = doc.get("source", "unknown")
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks")

    # Limit chunks if specified
    if args.max_chunks:
        all_chunks = all_chunks[:args.max_chunks]
        print(f"Processing first {len(all_chunks)} chunks")

    # Process chunks
    if args.mode == "react":
        process_with_react_agent(
            all_chunks, teacher, graph_store, entity_resolver, args.verbose
        )
    else:
        process_with_direct_extraction(
            all_chunks, teacher, graph_store, entity_resolver
        )

    # Save graph
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph_store.save(output_path)

    # Print summary
    print("\n" + "=" * 50)
    print("Knowledge Graph Summary")
    print("=" * 50)
    print(f"Entities: {len(graph_store.entities)}")
    print(f"Relationships: {len(graph_store.relationships)}")
    print(f"Saved to: {output_path}")

    # Print entity type breakdown
    from collections import Counter
    type_counts = Counter(e.entity_type.value for e in graph_store.entities.values())
    print("\nEntity types:")
    for etype, count in type_counts.most_common():
        print(f"  {etype}: {count}")


def load_documents(path: Path) -> list[dict]:
    """Load documents from file or directory."""
    documents = []

    if path.is_file():
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents = data
                else:
                    documents = [data]
        elif path.suffix == ".txt":
            documents = [{"text": path.read_text(), "source": path.name}]
    elif path.is_dir():
        for txt_file in path.glob("*.txt"):
            documents.append({
                "text": txt_file.read_text(),
                "source": txt_file.name,
            })
        for json_file in path.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)

    return documents


def process_with_react_agent(chunks, teacher, graph_store, entity_resolver, verbose):
    """Process chunks using ReAct agent."""
    agent = KGReActAgent(
        llm=teacher,
        graph_store=graph_store,
        entity_resolver=entity_resolver,
        max_iterations=5,
        verbose=verbose,
    )

    for chunk in tqdm(chunks, desc="Processing chunks (ReAct)"):
        try:
            result = agent.process_chunk(
                text=chunk["text"],
                metadata=chunk.get("metadata", {}),
            )

            # Resolve and add entities
            for entity in agent.extracted_entities:
                resolution = entity_resolver.resolve(
                    entity, graph_store.get_all_entities()
                )
                if not resolution.is_duplicate:
                    graph_store.add_entity(entity)

            # Add relationships
            for rel in agent.extracted_relations:
                graph_store.add_relationship(rel)

        except Exception as e:
            if verbose:
                print(f"Error processing chunk: {e}")


def process_with_direct_extraction(chunks, teacher, graph_store, entity_resolver):
    """Process chunks using direct extraction (no agent loop)."""
    extractor = KGExtractor(teacher)

    for chunk in tqdm(chunks, desc="Processing chunks (Direct)"):
        try:
            # Extract entities
            entities = extractor.extract_entities(chunk["text"])

            # Resolve and add entities
            resolved_entities = []
            for entity in entities:
                resolution = entity_resolver.resolve(
                    entity, graph_store.get_all_entities()
                )
                if not resolution.is_duplicate:
                    entity.source_chunk = chunk["text"][:200]
                    graph_store.add_entity(entity)
                    resolved_entities.append(entity)
                else:
                    resolved_entities.append(resolution.matched_entity)

            # Extract relationships
            if len(resolved_entities) >= 2:
                relationships = extractor.extract_relations(
                    chunk["text"], resolved_entities
                )
                for rel in relationships:
                    graph_store.add_relationship(rel)

        except Exception as e:
            print(f"Error processing chunk: {e}")


if __name__ == "__main__":
    main()
