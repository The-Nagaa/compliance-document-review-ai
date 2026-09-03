"""
Embedding Service Unit & Integration Tests (Sentence-Transformers all-MiniLM-L6-v2, 384-dim).
"""

import pytest
from ai.app.services.embedding_service import EmbeddingService, embedding_service


@pytest.fixture
def svc():
    return embedding_service


def test_embedding_dimension_is_384(svc):
    text = "Guaranteed 20% return on client investment."
    emb = svc.get_embedding(text)
    assert isinstance(emb, list)
    assert len(emb) == 384
    assert all(isinstance(x, float) for x in emb)
    assert svc.dimension == 384


def test_empty_and_whitespace_text_returns_zero_vector(svc):
    for empty_input in ["", "   ", "\n\t  ", None]:
        emb = svc.get_embedding(empty_input)
        assert len(emb) == 384
        assert all(x == 0.0 for x in emb)


def test_batch_embeddings(svc):
    texts = [
        "First compliance rule text.",
        "",
        "Second disclosure disclaimer text.",
        "   ",
        "Third document commentary."
    ]
    batch_embs = svc.get_embeddings_batch(texts)
    assert len(batch_embs) == len(texts)
    for i, emb in enumerate(batch_embs):
        assert len(emb) == 384
        if i in (1, 3):
            assert all(x == 0.0 for x in emb)
        else:
            assert any(x != 0.0 for x in emb)


def test_cosine_similarity_384_dimensions(svc):
    vec_a = svc.get_embedding("Investments involve market risk and loss of principal.")
    vec_b = svc.get_embedding("Trading securities involves high risk of losing investment capital.")
    vec_c = svc.get_embedding("The recipe calls for two cups of organic whole milk.")

    sim_ab = svc.cosine_similarity(vec_a, vec_b)
    sim_ac = svc.cosine_similarity(vec_a, vec_c)

    # Similar compliance topics should have higher similarity than cooking recipes
    assert sim_ab > sim_ac
    assert 0.5 < sim_ab <= 1.0
    assert -1.0 <= sim_ac < 0.4

    # Identical vector similarity should be 1.0
    assert pytest.approx(svc.cosine_similarity(vec_a, vec_a), abs=1e-4) == 1.0

    # Zero vector handling
    zero_vec = [0.0] * 384
    assert svc.cosine_similarity(vec_a, zero_vec) == 0.0


def test_custom_model_configuration():
    custom_svc = EmbeddingService(model_name="all-MiniLM-L6-v2", dimension=384)
    assert custom_svc.dimension == 384
    assert custom_svc.model_name == "all-MiniLM-L6-v2"
    emb = custom_svc.get_embedding("Sample test")
    assert len(emb) == 384
