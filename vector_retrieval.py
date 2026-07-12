"""
Vector retrieval -- retrieval MODE 1 (the standard, similarity-based approach).

Given a new incident summary, find the most similar past incidents in the
knowledge base by embedding everything into vectors and comparing them.

  build:    embed every KB entry's summary -> FAISS index
  retrieve: embed the query summary -> return the top-k nearest KB entries

Similarity is cosine similarity, implemented as inner product over
L2-normalized vectors (a normalized dot product IS cosine). This is the
baseline that the vectorless (structure-aware) mode is compared against.

The embedder is swappable: by default it loads the all-MiniLM-L6-v2
sentence-transformer; a custom embed_fn can be injected (used for testing
without downloading the model).
"""
from __future__ import annotations

import json
import os

import numpy as np

from knowledge_base import KBEntry, load_kb


def _default_embedder():
    """Lazy-load all-MiniLM-L6-v2 only when actually needed."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts: list[str]) -> np.ndarray:
        return np.asarray(
            model.encode(texts, show_progress_bar=False), dtype="float32"
        )
    return embed


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize each row so inner product == cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype("float32")


class VectorRetriever:
    def __init__(self, embed_fn=None):
        self.embed_fn = embed_fn          # None -> real model, lazy-loaded
        self.index = None                 # faiss index
        self.entries: list[KBEntry] = []  # position i in index == entries[i]

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.embed_fn is None:
            self.embed_fn = _default_embedder()
        return self.embed_fn(texts)

    def build(self, entries: list[KBEntry]) -> "VectorRetriever":
        import faiss
        self.entries = list(entries)
        vecs = _normalize(self._embed([e.summary for e in self.entries]))
        self.index = faiss.IndexFlatIP(vecs.shape[1])   # inner product = cosine
        self.index.add(vecs)
        return self

    def retrieve(self, query_summary: str, k: int = 5) -> list[tuple[KBEntry, float]]:
        """Return the top-k most similar KB entries as (entry, similarity)."""
        if self.index is None:
            raise RuntimeError("call build() first")
        q = _normalize(self._embed([query_summary]))
        k = min(k, len(self.entries))
        scores, idx = self.index.search(q, k)
        return [(self.entries[i], float(s)) for i, s in zip(idx[0], scores[0])]

    def save(self, directory: str = "vector_index") -> None:
        import faiss
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "entries.json"), "w") as f:
            json.dump([e.__dict__ for e in self.entries], f)

    def load(self, directory: str = "vector_index") -> "VectorRetriever":
        import faiss
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "entries.json")) as f:
            self.entries = [KBEntry(**d) for d in json.load(f)]
        return self


if __name__ == "__main__":
    entries = load_kb("knowledge_base.json")
    print(f"Loaded {len(entries)} KB entries. Building vector index "
          f"(all-MiniLM-L6-v2)...")
    retriever = VectorRetriever().build(entries)
    retriever.save("vector_index")

    # sanity demo: use the first entry's own summary as a query.
    query = entries[0].summary
    print(f"\nQuery = summary of {entries[0].entry_id} "
          f"(true label: {entries[0].root_cause_service}, {entries[0].fault_type})\n")
    print("Top-5 retrieved:")
    for entry, score in retriever.retrieve(query, k=5):
        print(f"  {score:.3f}  {entry.entry_id:<24} "
              f"({entry.root_cause_service}, {entry.fault_type})")