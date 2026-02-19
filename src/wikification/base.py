"""Abstract base class for wikification tools."""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path


class BaseWikifier(ABC):
    """
    Abstract base for Wikifier.org and WAT clients.

    Subclasses must implement :meth:`annotate`.
    Shared chunking and batch-processing logic lives here.
    """

    def __init__(self, top_n: int = 5, chunk_size: int = 650):
        self.top_n = top_n
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def annotate(self, text: str) -> list[dict]:
        """
        Annotate a single text with Wikipedia entities.

        Returns a list of concept dicts sorted by the tool's relevance score
        (pageRank for Wikifier, rho for WAT), truncated to ``self.top_n``.

        Raises ``RuntimeError`` on unrecoverable API errors.
        """

    # ------------------------------------------------------------------
    # Shared batch processing
    # ------------------------------------------------------------------

    def annotate_batch(
        self,
        items: list[dict],
        output_path,
        save_every: int = 50,
        chunk_texts: bool = False,
    ) -> list[dict]:
        """
        Annotate a list of items with incremental saves and resume support.

        Each item must have a ``"text"`` key.  The method adds an
        ``"annotations"`` key to a copy of each item:

        * ``chunk_texts=False`` (questions): stores a single
          ``{"status": ..., "annotation_data": [...]}`` dict.
        * ``chunk_texts=True`` (passages): stores a list of such dicts,
          one per text chunk.

        Existing output is loaded and used to resume automatically.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume from existing output
        results: list[dict] = []
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                print(f"Resuming from {len(results)}/{len(items)} items")
            except json.JSONDecodeError:
                print("Warning: corrupted progress file, starting fresh")

        start_idx = len(results)
        total = len(items)

        if start_idx >= total:
            print(f"[SKIP] Already complete: {output_path}")
            return results

        print(f"Annotating items {start_idx + 1}–{total} …")

        for i in range(start_idx, total):
            item = items[i]
            item_copy = item.copy()
            text = item.get("text", "")

            if not text:
                item_copy["annotations"] = (
                    [] if chunk_texts else {"status": "skipped", "annotation_data": []}
                )
                results.append(item_copy)
                continue

            if chunk_texts:
                chunks = self.partition_text(text, self.chunk_size)
                chunk_results = []
                for chunk in chunks:
                    chunk_results.append(self._safe_annotate(chunk))
                item_copy["annotations"] = chunk_results
            else:
                item_copy["annotations"] = self._safe_annotate(text)

            results.append(item_copy)

            if (i + 1) % save_every == 0:
                self._save(results, output_path)
                print(f"Progress: {i + 1}/{total}", flush=True)

        self._save(results, output_path)
        print(f"[DONE] {len(results)} items saved to {output_path}")
        return results

    # ------------------------------------------------------------------
    # Shared text chunking
    # ------------------------------------------------------------------

    def partition_text(self, text: str, max_size: int) -> list[str]:
        """
        Split *text* into chunks whose character length ≤ *max_size*.

        Splits on sentence boundaries (.  !  ?) and reassembles greedily.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            # +1 for the space separator between sentences
            addition = (1 if current else 0) + len(sent)
            if current_len + addition > max_size and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(sent)
            current_len += addition

        if current:
            chunks.append(" ".join(current))

        return chunks or [text]  # never return empty list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_annotate(self, text: str) -> dict:
        """Wrap :meth:`annotate` so errors are stored as status strings."""
        try:
            concepts = self.annotate(text)
            return {"status": "success", "annotation_data": concepts}
        except Exception as exc:
            return {"status": f"error: {exc}", "annotation_data": []}

    @staticmethod
    def _save(data: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
