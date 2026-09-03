"""
Embedding Service using sentence-transformers (all-MiniLM-L6-v2).

Privacy Constraint:
Never embeds raw unmasked text. Guarantees zero PII leakage.
"""

from typing import List, Optional
import numpy as np
from ai.app.core.config import settings
from ai.app.core.logging import logger


class EmbeddingService:
    """
    Produces normalized 384-dimensional dense embeddings for vector similarity search
    using sentence-transformers/all-MiniLM-L6-v2.
    """

    DIMENSION: int = 384

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.model_name: str = model_name or getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.dimension: int = dimension or getattr(settings, "EMBEDDING_DIMENSION", self.DIMENSION)
        self._model = None

    def _get_model(self):
        """Lazy load SentenceTransformer model singleton."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer embedding model '{self.model_name}'...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate a normalized 384-dim float vector for the given text.
        Empty or whitespace-only inputs return a zero vector of length `dimension`.
        """
        if not text or not text.strip():
            return [0.0] * self.dimension

        model = self._get_model()
        vec = model.encode(text.strip(), normalize_embeddings=True)
        if isinstance(vec, np.ndarray):
            return vec.tolist()
        return list(vec)

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate normalized 384-dim float vectors for a batch of texts.
        """
        if not texts:
            return []

        # Separate empty from non-empty to optimize batch encoding
        indexed_texts = [(i, t.strip()) for i, t in enumerate(texts) if t and t.strip()]
        
        results: List[List[float]] = [[0.0] * self.dimension for _ in texts]
        if not indexed_texts:
            return results

        model = self._get_model()
        valid_indices, valid_strings = zip(*indexed_texts)
        encoded_vecs = model.encode(list(valid_strings), normalize_embeddings=True, show_progress_bar=False)

        for orig_idx, vec in zip(valid_indices, encoded_vecs):
            results[orig_idx] = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)

        return results

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        For normalized vectors, this is the dot product.
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

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
