"""WAT (Wikipedia Annotation Tool) client — eliminates 4x duplicate WAT code."""

import time

import requests

from .base import BaseWikifier

_WAT_URL = "https://wat.d4science.org/wat/tag/tag"

# Paper's full method string (from eval_relatedness.ipynb)
_WAT_METHOD = (
    "tagme"
    "&token_nerf=1"
    "&token_nerf_method=0"
    "&epsilon=0.3"
)


class WAT(BaseWikifier):
    """
    WAT annotation client (sobigdata.d4science.org).

    Parameters
    ----------
    token:
        gcube-token obtained from sobigdata.d4science.org.
    top_n:
        Number of top concepts to return (ranked by rho).
    chunk_size:
        Max character length per text chunk before splitting.
    lang:
        Language code (default: ``"en"``).
    """

    def __init__(
        self,
        token: str,
        top_n: int = 5,
        chunk_size: int = 650,
        lang: str = "en",
    ):
        super().__init__(top_n=top_n, chunk_size=chunk_size)
        self.token = token
        self.lang = lang

    # ------------------------------------------------------------------
    # BaseWikifier interface
    # ------------------------------------------------------------------

    def annotate(self, text: str) -> list[dict]:
        """
        Call WAT and return top concepts sorted by rho.

        Raises ``RuntimeError`` on API failure after retries.
        """
        return self._call_with_retry(text)

    # ------------------------------------------------------------------
    # Relatedness (WikiSemRel metric)
    # ------------------------------------------------------------------

    def relatedness(self, wiki_ids: list[int], rel_type: str = "jaccard") -> float:
        """
        Compute semantic relatedness between a set of Wikipedia entity IDs.

        Uses WAT's ``/wat/relatedness/relatedness`` endpoint.
        *rel_type* ∈ ``{"jaccard", "cosine", "mw_norm", "lm"}``.

        Returns the mean pairwise relatedness score (0–1).
        """
        if len(wiki_ids) < 2:
            return 0.0

        pairs = [
            (wiki_ids[i], wiki_ids[j])
            for i in range(len(wiki_ids))
            for j in range(i + 1, len(wiki_ids))
        ]

        params = {
            "gcube-token": self.token,
            "ids": " ".join(f"{a} {b}" for a, b in pairs),
            "rel_type": rel_type,
        }

        try:
            r = requests.get(
                "https://wat.d4science.org/wat/relatedness/relatedness",
                params=params,
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                values = [item.get("rel", 0) for item in data.get("pairs", [])]
                return sum(values) / len(values) if values else 0.0
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, text: str) -> dict:
        params = {
            "gcube-token": self.token,
            "text": text,
            "lang": self.lang,
        }
        r = requests.get(_WAT_URL, params=params, timeout=30)
        if r.status_code == 429:
            raise RuntimeError("rate limit: 429")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        return r.json()

    def _parse_response(self, data: dict) -> list[dict]:
        annotations = sorted(
            data.get("annotations", []),
            key=lambda a: a.get("rho", 0),
            reverse=True,
        )
        concepts = [
            {
                "title": a["title"],
                "wiki_id": a.get("id"),
                "rho": a.get("rho", 0),
                "spot": a.get("spot", ""),
                "start": a.get("start"),
                "end": a.get("end"),
            }
            for a in annotations[: self.top_n]
        ]
        return concepts

    def _call_with_retry(self, text: str, max_retries: int = 3) -> list[dict]:
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                data = self._call_api(text)
                time.sleep(0.01)
                return self._parse_response(data)
            except RuntimeError as exc:
                msg = str(exc)
                if "rate limit" in msg and attempt < max_retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    print(f"  Rate limit, waiting {wait}s…", flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"WAT failed: {exc}") from exc
            except requests.Timeout:
                if attempt < max_retries - 1:
                    print(
                        f"  Timeout, retrying ({attempt + 1}/{max_retries})…",
                        flush=True,
                    )
                    time.sleep(retry_delay)
                    continue
                raise RuntimeError("WAT: timeout after retries")
        raise RuntimeError("WAT: max retries exceeded")
