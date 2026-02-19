"""Wikifier.org client — eliminates duplication across 4 wikify_*.py scripts."""

import json
import time

import requests

from .base import BaseWikifier

_WIKIFIER_URL = "https://www.wikifier.org/annotate-article"


class Wikifier(BaseWikifier):
    """
    Wikifier.org annotation client.

    Parameters
    ----------
    api_key:
        Wikifier API key (obtain from wikifier.org).
    top_n:
        Number of top concepts to return (ranked by pageRank).
    chunk_size:
        Max character length per text chunk before splitting.
    df_ignore:
        ``nTopDfValuesToIgnore`` passed to the API (paper default: 200).
    words_ignore:
        ``nWordsToIgnoreFromList`` passed to the API (paper default: 200).
    """

    def __init__(
        self,
        api_key: str,
        top_n: int = 5,
        chunk_size: int = 650,
        df_ignore: int = 200,
        words_ignore: int = 200,
    ):
        super().__init__(top_n=top_n, chunk_size=chunk_size)
        self.api_key = api_key
        self.df_ignore = df_ignore
        self.words_ignore = words_ignore

    # ------------------------------------------------------------------
    # BaseWikifier interface
    # ------------------------------------------------------------------

    def annotate(self, text: str) -> list[dict]:
        """
        Call Wikifier.org and return top concepts sorted by pageRank.

        Raises ``RuntimeError`` on API failure after retries.
        """
        return self._wikify_with_retry(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, text: str) -> dict:
        params = {
            "text": text,
            "userKey": self.api_key,
            "nTopDfValuesToIgnore": self.df_ignore,
            "nWordsToIgnoreFromList": self.words_ignore,
        }
        r = requests.post(_WIKIFIER_URL, params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        resp = json.loads(r.content)
        if "error" in resp:
            raise RuntimeError(f"API error: {resp['error']}")
        return resp

    def _parse_response(self, resp: dict) -> list[dict]:
        annotations = sorted(
            resp.get("annotations", []),
            key=lambda a: a.get("pageRank", 0),
            reverse=True,
        )
        concepts = [
            {
                "title": a["title"],
                "url": a.get("url", ""),
                "cosine": a.get("cosine", 0),
                "pageRank": a.get("pageRank", 0),
                "wikiDataItemId": a.get("wikiDataItemId"),
            }
            for a in annotations[: self.top_n]
        ]
        return concepts

    def _wikify_with_retry(
        self, text: str, max_retries: int = 5
    ) -> list[dict]:
        for attempt in range(max_retries + 1):
            try:
                resp = self._call_api(text)
                time.sleep(0.1)  # minimal rate-limiting
                return self._parse_response(resp)
            except RuntimeError as exc:
                msg = str(exc)
                if ("rate limit" in msg.lower() or "429" in msg) and attempt < max_retries:
                    wait = 2 ** attempt
                    print(
                        f"  Rate limit, waiting {wait}s "
                        f"(retry {attempt + 1}/{max_retries})…",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Wikifier failed: {exc}") from exc
        raise RuntimeError(f"Wikifier: max retries exceeded for text[:50]={text[:50]!r}")
