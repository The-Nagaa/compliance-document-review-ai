"""
API Routes and Cache Persistence Integration Tests.
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest
from ai.app.main import app
from ai.app.models.entities import AIAnalysis, AnalysisStatus
from ai.app.repositories.analysis_cache import AnalysisCache
from ai.app.repositories.vector_store import VectorStore
from ai.app.schemas.analysis import LLMComplianceOutput
from ai.app.schemas.flags import FlagCreate


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Compliance Document Review App" in resp.json()["service"]


def test_health_endpoint(client):
    resp = client.get("/ai/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "corpus_loaded" in data


def test_privacy_inspect_endpoint(client):
    payload = {
        "document_id": "doc_inspect_api",
        "extracted_text": "Client: Jane Doe with email jane.doe@example.com and Account #123456.",
    }
    resp = client.post("/ai/privacy-inspect", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["detected_entities_count"] >= 2
    assert data["has_raw_pii_in_outbound_payload"] is False
    assert "[CLIENT_1]" in data["masked_text_preview"]


def test_analyze_and_get_analysis_routes(client):
    doc_id = "doc_api_flow_1"
    raw_text = "Client: Alice Smith portfolio review. Past performance is no guarantee of future results."

    mock_llm_out = LLMComplianceOutput(
        summary="Portfolio review for client.",
        flags=[
            FlagCreate(
                passage="review for Alice Smith",
                matched_rule_id="RULE-020",
                matched_rule="SEC RIA disclosure",
                explanation="Check RIA disclaimer",
                severity="low",
            )
        ],
    )

    with patch("ai.app.services.gemini_service.GeminiService.analyze_compliance", return_value=mock_llm_out):
        # 1. Analyze
        post_resp = client.post(
            f"/ai/analyze/{doc_id}",
            json={"document_id": doc_id, "extracted_text": raw_text},
        )
        assert post_resp.status_code == 200
        post_data = post_resp.json()
        assert post_data["document_id"] == doc_id
        assert post_data["cached"] is False

        # 2. Get Cached
        get_resp = client.get(f"/ai/analysis/{doc_id}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["document_id"] == doc_id
        assert get_data["cached"] is True
        assert get_data["summary"] == post_data["summary"]

        # 3. Retry
        retry_resp = client.post(
            f"/ai/analyze/{doc_id}/retry",
            json={"document_id": doc_id, "extracted_text": raw_text},
        )
        assert retry_resp.status_code == 200
        assert retry_resp.json()["cached"] is False


def test_get_nonexistent_analysis_returns_404(client):
    resp = client.get("/ai/analysis/nonexistent_doc_id_9999")
    assert resp.status_code == 404


def test_cache_disk_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_cache.json"
        cache = AnalysisCache(storage_path=tmp_path)
        
        sample_analysis = AIAnalysis(
            document_id="doc_disk_1",
            summary="Disk persisted summary",
            status=AnalysisStatus.COMPLETED,
        )
        cache.set(sample_analysis)
        cache.save_to_disk()

        # Load into new cache instance
        cache2 = AnalysisCache(storage_path=tmp_path)
        loaded = cache2.load_from_disk()
        assert loaded is True
        assert cache2.get("doc_disk_1") is not None
        assert cache2.get("doc_disk_1").summary == "Disk persisted summary"


def test_vector_store_disk_persistence_and_pgvector_sql():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_vector_store.json"
        store = VectorStore(storage_path=tmp_path)
        
        from scripts.seed_corpus import seed_all
        # Populate store
        seed_all(persist=False)
        
        # Test pgvector SQL export
        sql = store.export_pgvector_sql()
        assert "CREATE EXTENSION IF NOT EXISTS vector;" in sql
        assert "compliance_rules" in sql
        assert "compliance_disclosures" in sql
        assert "compliance_precedents" in sql
