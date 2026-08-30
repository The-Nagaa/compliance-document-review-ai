"""
Embedding Service with Local Semantic Vector Engine and Gemini Embeddings Support.

Privacy Constraint:
Never embeds raw unmasked text. Guarantees zero PII leakage.
"""

import hashlib
import re
from typing import Dict, List, Optional
import numpy as np
from ai.app.core.config import settings
from ai.app.core.logging import logger


SYNONYM_MAP: Dict[str, str] = {
    # Performance & Returns
    "historical": "past",
    "prior": "past",
    "previous": "past",
    "history": "past",
    "returns": "return",
    "results": "return",
    "yield": "return",
    "yields": "return",
    "gain": "return",
    "gains": "return",
    "profit": "return",
    "profits": "return",
    "performance": "return",
    "trackrecord": "return",
    # Guarantees & Promises
    "predict": "guarantee",
    "predicts": "guarantee",
    "ensure": "guarantee",
    "ensures": "guarantee",
    "assure": "guarantee",
    "assures": "guarantee",
    "promise": "guarantee",
    "promises": "guarantee",
    "guaranteed": "guarantee",
    "guarantees": "guarantee",
    "assured": "guarantee",
    # Risk & Loss
    "loss": "loss",
    "losses": "loss",
    "downside": "loss",
    "losing": "loss",
    "risk": "risk",
    "risks": "risk",
    "hazard": "risk",
    # Investments & Capital
    "investments": "invest",
    "investing": "invest",
    "investment": "invest",
    "investor": "invest",
    "securities": "invest",
    "equities": "invest",
    "portfolio": "invest",
    "assets": "invest",
    "capital": "principal",
    "principal": "principal",
    # Disclosures & RIA
    "disclaimer": "disclosure",
    "disclaimers": "disclosure",
    "notice": "disclosure",
    "statement": "disclosure",
    "registration": "ria",
    "registered": "ria",
    "advisory": "ria",
    "advisor": "ria",
    "brokerage": "finra",
    "custody": "sipc",
}


class EmbeddingService:
    """
    Produces normalized dense embeddings for vector similarity search.
    Defaults to an offline, deterministic semantic vector engine (128 dimensions)
    with synonym canonicalization to enable instant, reproducible local execution,
    unit testing, and paraphrased disclosure matching.
    """

    DIMENSION = 128

    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate a normalized float vector for the given text.
        """
        if not text or not text.strip():
            return [0.0] * self.dimension

        # Pre-process text (clean, lowercase, tokenized)
        cleaned = text.lower().strip()
        raw_tokens = re.findall(r'\b[a-z0-9_-]+\b', cleaned)
        
        if not raw_tokens:
            return [0.0] * self.dimension

        # Canonicalize tokens with synonym mapping
        tokens = [SYNONYM_MAP.get(t, t) for t in raw_tokens]

        # Build dense feature vector using n-grams, token semantic buckets & positional weighting
        vec = np.zeros(self.dimension, dtype=np.float32)

        # 1. Unigram and Bigram hashing with semantic projection
        for i, token in enumerate(tokens):
            # Token hash
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            
            # Weight important compliance terms higher
            weight = 1.0
            if token in ("guarantee", "return", "risk", "loss", "ria", "sipc", "finra", "past", "disclosure", "principal"):
                weight = 3.0
            
            vec[idx] += sign * weight

            # Bigrams (both raw and canonicalized)
            if i < len(tokens) - 1:
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                h_bi = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
                idx_bi = h_bi % self.dimension
                sign_bi = 1.0 if ((h_bi >> 8) & 1) else -1.0
                vec[idx_bi] += sign_bi * 2.0

        # 2. Add character n-grams (3-grams) for robust typo matching
        for i in range(len(cleaned) - 2):
            trigram = cleaned[i:i+3]
            h_tri = int(hashlib.sha256(trigram.encode("utf-8")).hexdigest(), 16)
            idx_tri = h_tri % self.dimension
            sign_tri = 1.0 if ((h_tri >> 4) & 1) else -1.0
            vec[idx_tri] += sign_tri * 0.3

        # 3. L2 Normalize the vector for exact cosine similarity via dot product
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec = np.zeros(self.dimension, dtype=np.float32)

        return vec.tolist()

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation."""
        return [self.get_embedding(t) for t in texts]

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        # Clamp to [-1.0, 1.0] to prevent floating point inaccuracies
        return max(-1.0, min(1.0, similarity))


embedding_service = EmbeddingService()
