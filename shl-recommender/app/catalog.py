"""
catalog.py — Load and search the SHL assessment catalog using FAISS + sentence-transformers.

The index is built once at startup from data/shl_catalog.json.
All searches are purely in-memory — no external DB needed.
"""

import json
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional
from functools import lru_cache

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


CATALOG_PATH = Path(__file__).parent.parent / "data" / "shl_catalog.json"
INDEX_CACHE  = Path(__file__).parent.parent / "data" / "faiss_index.pkl"
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# ── Fallback keyword index (used when FAISS/sentence-transformers unavailable) ──

def _keyword_score(item: dict, query: str) -> float:
    """Simple TF-style keyword scorer."""
    q_tokens = set(query.lower().split())
    doc = (
        item.get("name", "") + " " +
        item.get("description", "") + " " +
        " ".join(item.get("job_levels", [])) + " " +
        " ".join(item.get("test_type_labels", []))
    ).lower()
    hits = sum(1 for tok in q_tokens if tok in doc)
    return hits / max(len(q_tokens), 1)


# ── Catalog loader ───────────────────────────────────────────────────────────

class CatalogEngine:
    """Holds all assessments and provides semantic + keyword search."""

    def __init__(self):
        self.items: list[dict] = []
        self.model = None
        self.index = None   # faiss.Index
        self._ready = False

    # ── Build ────────────────────────────────────────────────────────────────

    def load(self):
        """Load catalog and build (or restore) vector index."""
        if not CATALOG_PATH.exists():
            raise FileNotFoundError(
                f"Catalog not found at {CATALOG_PATH}. "
                "Run: python scrape_catalog.py"
            )

        with open(CATALOG_PATH, encoding="utf-8") as f:
            self.items = json.load(f)

        print(f"[catalog] Loaded {len(self.items)} assessments.")

        if ST_AVAILABLE and FAISS_AVAILABLE:
            self._build_or_load_index()
        else:
            print("[catalog] sentence-transformers or faiss unavailable; using keyword search.")

        self._ready = True

    def _text_for_embedding(self, item: dict) -> str:
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("test_type_labels", [])),
            " ".join(item.get("job_levels", [])),
        ]
        return " ".join(p for p in parts if p).strip()

    def _build_or_load_index(self):
        if INDEX_CACHE.exists():
            try:
                with open(INDEX_CACHE, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("n") == len(self.items):
                    self.index = faiss.deserialize_index(cached["index_bytes"])
                    self.model = SentenceTransformer(MODEL_NAME)
                    print("[catalog] FAISS index restored from cache.")
                    return
            except Exception as e:
                print(f"[catalog] Cache load failed ({e}); rebuilding.")

        print("[catalog] Building FAISS index (first run, may take ~60s)...")
        self.model = SentenceTransformer(MODEL_NAME)
        texts = [self._text_for_embedding(it) for it in self.items]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=64)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Persist
        INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump(
                {"n": len(self.items), "index_bytes": faiss.serialize_index(self.index)},
                f,
            )
        print(f"[catalog] FAISS index built with {self.index.ntotal} vectors.")

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        test_types: Optional[list[str]] = None,
        job_levels: Optional[list[str]] = None,
        remote_only: bool = False,
    ) -> list[dict]:
        """
        Return up to top_k assessments ranked by relevance to query.
        Optional filters: test_types (list of letters), job_levels, remote_only.
        """
        candidates = self.items

        # Hard filters first
        if remote_only:
            candidates = [c for c in candidates if c.get("remote_testing")]
        if test_types:
            tt_upper = [t.upper() for t in test_types]
            candidates = [
                c for c in candidates
                if any(t in c.get("test_types", []) for t in tt_upper)
            ]
        if job_levels:
            jl_lower = [j.lower() for j in job_levels]
            candidates = [
                c for c in candidates
                if any(
                    any(jl in lvl.lower() for lvl in c.get("job_levels", []))
                    for jl in jl_lower
                )
            ]

        if not candidates:
            return []

        if self.index is not None and self.model is not None:
            return self._semantic_search(query, candidates, top_k)
        else:
            return self._keyword_search(query, candidates, top_k)

    def _semantic_search(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        # Build a mapping from item url → position in self.items
        url_to_idx = {it["url"]: i for i, it in enumerate(self.items)}
        candidate_indices = [url_to_idx[c["url"]] for c in candidates if c["url"] in url_to_idx]

        if not candidate_indices:
            return []

        # Encode query
        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        # Search full index then filter to candidates
        k_search = min(len(self.items), max(top_k * 5, 50))
        scores, indices = self.index.search(q_vec, k_search)
        scores = scores[0]
        indices = indices[0]

        candidate_set = set(candidate_indices)
        results = []
        for idx, score in zip(indices, scores):
            if idx in candidate_set:
                item = dict(self.items[idx])
                item["_score"] = float(score)
                results.append(item)
            if len(results) >= top_k:
                break

        # If semantic didn't fill top_k, pad with keyword results
        if len(results) < top_k:
            seen = {r["url"] for r in results}
            kw = self._keyword_search(query, [c for c in candidates if c["url"] not in seen], top_k - len(results))
            results.extend(kw)

        return results[:top_k]

    def _keyword_search(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        scored = [(it, _keyword_score(it, query)) for it in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:top_k]]

    # ── Lookup ───────────────────────────────────────────────────────────────

    def get_by_name(self, name: str) -> Optional[dict]:
        name_lower = name.lower()
        for item in self.items:
            if item["name"].lower() == name_lower:
                return item
        # Partial match
        for item in self.items:
            if name_lower in item["name"].lower():
                return item
        return None

    def get_by_url(self, url: str) -> Optional[dict]:
        for item in self.items:
            if item["url"] == url:
                return item
        return None

    def all_names(self) -> list[str]:
        return [it["name"] for it in self.items]


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[CatalogEngine] = None


def get_engine() -> CatalogEngine:
    global _engine
    if _engine is None:
        _engine = CatalogEngine()
        _engine.load()
    return _engine
