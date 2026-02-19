"""Wikification utilities — Wikifier.org and WAT clients."""

from .base import BaseWikifier
from .wikifier import Wikifier
from .wat import WAT

__all__ = ["BaseWikifier", "Wikifier", "WAT", "get_wikifier"]


def get_wikifier(tool: str, config) -> BaseWikifier:
    """
    Construct a wikifier from *config* based on *tool* name.

    Parameters
    ----------
    tool:
        ``"wikifier"`` or ``"wat"``.
    config:
        A :class:`~src.pipeline.config.PipelineConfig` instance
        (or any object with a ``wikification`` attribute that exposes
        ``top_n``, ``chunk_size``, ``df_ignore``, ``words_ignore``).

    Raises ``ValueError`` for unknown tool names.
    """
    import os

    wc = config.wikification
    tool = tool.lower().strip()

    if tool == "wikifier":
        api_key = os.environ.get("WIKIFIER_API_KEY", "")
        return Wikifier(
            api_key=api_key,
            top_n=wc.top_n,
            chunk_size=wc.chunk_size,
            df_ignore=wc.df_ignore,
            words_ignore=wc.words_ignore,
        )

    if tool == "wat":
        token = os.environ.get("WAT_TOKEN", "")
        return WAT(
            token=token,
            top_n=wc.top_n,
            chunk_size=wc.chunk_size,
            lang=getattr(wc, "wat_lang", "en"),
        )

    raise ValueError(f"Unknown wikification tool: {tool!r}. Expected 'wikifier' or 'wat'.")
