import json
import re
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FaissRetriever:
    def __init__(
        self,
        index_path: str = "corpus/index.faiss",
        meta_path: str = "corpus/meta.jsonl",
        emb_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(emb_model)

        self.meta: List[Dict[str, Any]] = []
        with Path(meta_path).open("r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

        if self.index.ntotal != len(self.meta):
            raise ValueError(f"Index size ({self.index.ntotal}) != meta size ({len(self.meta)})")

        self._qcache: Dict[str, np.ndarray] = {}

    def clean_query(self, query: str) -> str:
        q = (query or "").strip().lower()

        # einfache Normalisierung typischer Kurzformen / Noise
        replacements = {
            r"\binst\b": "instagram",
            r"\bwhats\b": "what is",
            r"\bwho's\b": "who is",
            r"\bwhere's\b": "where is",
            r"\bdidnt\b": "did not",
            r"\bdoesnt\b": "does not",
            r"\bcant\b": "cannot",
        }

        for pattern, repl in replacements.items():
            q = re.sub(pattern, repl, q)

        # Sonderzeichen bereinigen
        q = re.sub(r"[^a-z0-9\s\-']", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def keyword_query(self, query: str) -> str:
        q = self.clean_query(query)

        stopwords = {
            "what", "when", "where", "who", "why", "how",
            "is", "are", "was", "were", "do", "does", "did",
            "the", "a", "an", "of", "in", "on", "for", "to",
            "come", "out"
        }

        tokens = [t for t in q.split() if t not in stopwords and len(t) > 2]
        return " ".join(tokens)

    def build_query_variants(self, query: str) -> List[str]:
        variants = []

        raw = (query or "").strip()
        cleaned = self.clean_query(query)
        keyword = self.keyword_query(query)

        for v in [raw, cleaned, keyword]:
            if v and v not in variants:
                variants.append(v)

        return variants

    def _encode_query(self, query: str) -> np.ndarray:
        if query in self._qcache:
            return self._qcache[query]

        q_emb = (
            self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            .astype("float32")
        )
        self._qcache[query] = q_emb
        return q_emb

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        variants = self.build_query_variants(query)

        # Kandidaten aus mehreren Suchläufen sammeln
        candidate_hits: Dict[int, Dict[str, Any]] = {}

        # pro Variante etwas mehr holen, dann zusammenführen
        per_variant_k = max(k, 8)

        for variant in variants:
            q_emb = self._encode_query(variant)
            scores, idxs = self.index.search(q_emb, per_variant_k)

            for score, idx in zip(scores[0], idxs[0]):
                idx = int(idx)
                if idx < 0:
                    continue

                # denselben Chunk nur einmal behalten, mit bestem Score
                if idx not in candidate_hits or float(score) > candidate_hits[idx]["score"]:
                    m = self.meta[idx]
                    candidate_hits[idx] = {
                        "score": float(score),
                        "doc_id": m["doc_id"],
                        "title": m.get("title", ""),
                        "text": m["text"],
                        "matched_query": variant,
                    }

        # nach Score sortieren
        ranked = sorted(candidate_hits.values(), key=lambda x: x["score"], reverse=True)[:k]

        hits = []
        for i, h in enumerate(ranked, start=1):
            hits.append(
                {
                    "rank": i,
                    "score": h["score"],
                    "doc_id": h["doc_id"],
                    "title": h["title"],
                    "text": h["text"],
                    "matched_query": h["matched_query"],
                }
            )

        return hits
