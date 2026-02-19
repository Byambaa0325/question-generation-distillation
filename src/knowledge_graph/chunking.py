"""Document chunking utilities for knowledge graph extraction."""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class DocumentChunk:
    """A chunk of document with metadata."""

    text: str
    chunk_id: str
    metadata: dict

    # Hierarchical context
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None

    # Position info
    start_char: int = 0
    end_char: int = 0


class LayoutAwareChunker:
    """
    Layout-aware document chunker that respects document structure.

    Preserves hierarchical information (chapters, sections) and
    avoids cutting mid-sentence.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 100,
    ):
        """
        Initialize chunker.

        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            overlap: Character overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

        # Patterns for detecting structure
        self.chapter_pattern = re.compile(
            r"^(Chapter\s+\d+|CHAPTER\s+\d+|\d+\.)\s*[:\.]?\s*(.+?)$",
            re.MULTILINE
        )
        self.section_pattern = re.compile(
            r"^(\d+\.\d+|\d+\.)\s+(.+?)$",
            re.MULTILINE
        )
        self.subsection_pattern = re.compile(
            r"^(\d+\.\d+\.\d+)\s+(.+?)$",
            re.MULTILINE
        )

    def chunk_document(
        self,
        text: str,
        doc_id: str = "doc",
        base_metadata: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        """
        Chunk a document while preserving structure.

        Args:
            text: Full document text
            doc_id: Document identifier
            base_metadata: Base metadata to include in all chunks

        Returns:
            List of DocumentChunk objects
        """
        base_metadata = base_metadata or {}
        chunks = []

        # First, identify structural elements
        structure = self._identify_structure(text)

        # Then chunk within each section
        current_chapter = None
        current_section = None
        current_subsection = None
        current_pos = 0

        for element in structure:
            if element["type"] == "chapter":
                current_chapter = element["title"]
                current_section = None
                current_subsection = None
            elif element["type"] == "section":
                current_section = element["title"]
                current_subsection = None
            elif element["type"] == "subsection":
                current_subsection = element["title"]

            # Get text for this section
            section_start = element["start"]
            section_end = element.get("end", len(text))
            section_text = text[section_start:section_end]

            # Chunk the section text
            section_chunks = self._chunk_text(
                section_text,
                start_offset=section_start,
            )

            for i, (chunk_text, start, end) in enumerate(section_chunks):
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    metadata={
                        **base_metadata,
                        "element_type": element["type"],
                    },
                    chapter=current_chapter,
                    section=current_section,
                    subsection=current_subsection,
                    start_char=start,
                    end_char=end,
                )
                chunks.append(chunk)

        # If no structure found, just chunk the whole text
        if not chunks:
            simple_chunks = self._chunk_text(text)
            for i, (chunk_text, start, end) in enumerate(simple_chunks):
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_chunk_{i}",
                    metadata=base_metadata,
                    start_char=start,
                    end_char=end,
                )
                chunks.append(chunk)

        return chunks

    def _identify_structure(self, text: str) -> list[dict]:
        """Identify document structure elements."""
        elements = []

        # Find chapters
        for match in self.chapter_pattern.finditer(text):
            elements.append({
                "type": "chapter",
                "number": match.group(1),
                "title": match.group(2).strip(),
                "start": match.start(),
            })

        # Find sections
        for match in self.section_pattern.finditer(text):
            elements.append({
                "type": "section",
                "number": match.group(1),
                "title": match.group(2).strip(),
                "start": match.start(),
            })

        # Find subsections
        for match in self.subsection_pattern.finditer(text):
            elements.append({
                "type": "subsection",
                "number": match.group(1),
                "title": match.group(2).strip(),
                "start": match.start(),
            })

        # Sort by position
        elements.sort(key=lambda x: x["start"])

        # Calculate end positions
        for i, element in enumerate(elements):
            if i + 1 < len(elements):
                element["end"] = elements[i + 1]["start"]
            else:
                element["end"] = len(text)

        return elements

    def _chunk_text(
        self,
        text: str,
        start_offset: int = 0,
    ) -> list[tuple[str, int, int]]:
        """
        Chunk text into smaller pieces.

        Returns list of (text, start, end) tuples.
        """
        if len(text) <= self.max_chunk_size:
            return [(text.strip(), start_offset, start_offset + len(text))]

        chunks = []
        current_start = 0

        while current_start < len(text):
            # Determine end position
            current_end = min(current_start + self.max_chunk_size, len(text))

            # Try to break at sentence boundary
            if current_end < len(text):
                # Look for sentence ending in last 20% of chunk
                search_start = current_end - int(self.max_chunk_size * 0.2)
                best_break = current_end

                for pattern in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                    pos = text.rfind(pattern, search_start, current_end)
                    if pos > search_start:
                        best_break = pos + len(pattern)
                        break

                current_end = best_break

            chunk_text = text[current_start:current_end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append((
                    chunk_text,
                    start_offset + current_start,
                    start_offset + current_end,
                ))

            # Move to next chunk with overlap
            current_start = current_end - self.overlap
            if current_start >= len(text) - self.min_chunk_size:
                break

        return chunks


def chunk_for_kg_extraction(
    text: str,
    max_chunk_size: int = 1500,
    overlap: int = 200,
) -> list[dict]:
    """
    Simple chunking function for KG extraction.

    Returns list of dicts with 'text' and 'metadata' keys.
    """
    chunker = LayoutAwareChunker(
        max_chunk_size=max_chunk_size,
        overlap=overlap,
    )

    chunks = chunker.chunk_document(text)

    return [
        {
            "text": c.text,
            "metadata": {
                "chunk_id": c.chunk_id,
                "chapter": c.chapter,
                "section": c.section,
                **c.metadata,
            }
        }
        for c in chunks
    ]
