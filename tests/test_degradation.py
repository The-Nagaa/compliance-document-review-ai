"""
Graceful Degradation Tests (Requirement 13 & 20).

Validates that AI failures (missing key, rate limits, timeouts) NEVER block
document review or human officer decision making.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest
from ai.app.main import app
from ai.app.models.entities import AnalysisStatus
from ai.app.services.gemini_service import AIRateLimited, AITimeout, AIUnavailable
from scripts.seed_corpus import seed_all


@pytest.fixture(scope="module", autouse=True)
def init_app():
    seed_all(persist=False)


@pytest.fixture
def client():
    return TestClient(app)


def test_missing_api_key_degrades_gracefully(client):
    # Simulate missing/empty GEMINI_API_KEY
    with patch("ai.app.core.config.settings.GEMINI_API_KEY", ""):
        with patch("ai.app.services.gemini_service.gemini_service.api_key", ""):
            payload = {
                "document_id": "doc_degrade_no_key",
                "extracted_text": "Proposal letter for Client: John Doe. We aim to invest in conservative index funds.",
            }

            response = client.post("/ai/analyze/doc_degrade_no_key", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert data["document_id"] == "doc_degrade_no_key"
            # AI panel degrades to UNAVAILABLE but API call succeeds
            assert data["status"] == AnalysisStatus.UNAVAILABLE.value
            assert "AI assist unavailable" in data["message"]
            assert data["retryable"] is True
            # Local vector search assists are STILL present
            assert len(data["precedents"]) == 3
            assert len(data["disclosures_checked"]) > 0


def test_rate_limit_degradation(client):
    # Simulate Gemini 429 rate limit
    with patch(
        "ai.app.services.gemini_service.GeminiService.analyze_compliance",
        side_effect=AIRateLimited("Rate limit hit"),
    ):
        payload = {
            "document_id": "doc_degrade_429",
            "extracted_text": "Market newsletter for Client: Jane Smith.",
        }

        response = client.post("/ai/analyze/doc_degrade_429", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == AnalysisStatus.RATE_LIMITED.value
        assert data["retryable"] is True
        assert "rate-limited" in data["message"]


def test_timeout_degradation(client):
    # Simulate network timeout
    with patch(
        "ai.app.services.gemini_service.GeminiService.analyze_compliance",
        side_effect=AITimeout("Timeout after 30s"),
    ):
        payload = {
            "document_id": "doc_degrade_timeout",
            "extracted_text": "Brochure for Client: Bob Miller.",
        }

        response = client.post("/ai/analyze/doc_degrade_timeout", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == AnalysisStatus.UNAVAILABLE.value
        assert data["retryable"] is True


def test_status_endpoint_before_and_after_analysis(client):
    doc_id = "doc_status_check_1"
    
    # 1. Before analysis: status is PENDING
    status_resp1 = client.get(f"/ai/status/{doc_id}")
    assert status_resp1.status_code == 200
    assert status_resp1.json()["status"] == AnalysisStatus.PENDING.value
    assert status_resp1.json()["has_analysis"] is False

    # 2. Analyze
    with patch("ai.app.services.gemini_service.GeminiService.analyze_compliance") as mock_gemini:
        from ai.app.schemas.analysis import LLMComplianceOutput
        mock_gemini.return_value = LLMComplianceOutput(
            summary="Clean document summary.", flags=[]
        )
        client.post(
            f"/ai/analyze/{doc_id}",
            json={"document_id": doc_id, "extracted_text": "Clean client material."},
        )

    # 3. After analysis: status is COMPLETED
    status_resp2 = client.get(f"/ai/status/{doc_id}")
    assert status_resp2.status_code == 200
    assert status_resp2.json()["status"] == AnalysisStatus.COMPLETED.value
    assert status_resp2.json()["has_analysis"] is True


def test_invalid_api_key_403_degrades_with_retryable(client):
    from ai.app.services.gemini_service import AIConfigurationError
    with patch(
        "ai.app.services.gemini_service.GeminiService.analyze_compliance",
        side_effect=AIConfigurationError("Gemini API request rejected: HTTP 403"),
    ):
        response = client.post(
            "/ai/analyze/doc_403",
            json={"document_id": "doc_403", "extracted_text": "Sample text"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AnalysisStatus.UNAVAILABLE.value
        assert data["retryable"] is True
        assert "HTTP 403" in data["message"]


def test_invalid_model_404_degrades_with_retryable(client):
    from ai.app.services.gemini_service import AIConfigurationError
    with patch(
        "ai.app.services.gemini_service.GeminiService.analyze_compliance",
        side_effect=AIConfigurationError("Gemini model 'invalid-model' is invalid or not found (HTTP 404)."),
    ):
        response = client.post(
            "/ai/analyze/doc_404",
            json={"document_id": "doc_404", "extracted_text": "Sample text"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AnalysisStatus.UNAVAILABLE.value
        assert data["retryable"] is True
        assert "HTTP 404" in data["message"]

