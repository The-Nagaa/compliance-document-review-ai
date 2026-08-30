"""
End-to-End Analysis Pipeline and Caching Tests (Requirement 8, 9, 12, 14).
"""

from unittest.mock import MagicMock
import pytest
from ai.app.models.entities import AnalysisStatus, SeverityLevel
from ai.app.repositories.analysis_cache import AnalysisCache
from ai.app.schemas.analysis import LLMComplianceOutput
from ai.app.schemas.flags import FlagCreate
from ai.app.services.analysis_service import AnalysisService
from ai.app.services.retrieval_service import RetrievalService
from scripts.seed_corpus import seed_all


@pytest.fixture(scope="module", autouse=True)
def setup_corpus():
    seed_all(persist=False)


@pytest.fixture
def mock_gemini():
    mock = MagicMock()
    mock.analyze_compliance.return_value = LLMComplianceOutput(
        summary="Marketing brochure for [CLIENT_1] highlighting portfolio allocation and yield expectations.",
        flags=[
            FlagCreate(
                passage="Guaranteed 15% return for [CLIENT_1] without risk.",
                matched_rule_id="RULE-001",
                matched_rule="Prohibition of Guaranteed Returns",
                explanation="Guaranteed return claims are strictly prohibited by compliance standards.",
                severity=SeverityLevel.HIGH,
            )
        ],
    )
    return mock


@pytest.fixture
def test_analysis_service(mock_gemini):
    cache = AnalysisCache()
    retriever = RetrievalService()
    return AnalysisService(retriever=retriever, llm=mock_gemini, cache=cache)


def test_full_pipeline_execution(test_analysis_service, mock_gemini):
    raw_doc = (
        "Marketing piece for Client: Jessica Alba.\n"
        "Guaranteed 15% return for Jessica Alba without risk.\n"
        "Account #98765432.\n"
    )

    response = test_analysis_service.process_document(
        document_id="doc_pipe_1",
        extracted_text=raw_doc,
    )

    assert response.document_id == "doc_pipe_1"
    assert response.status == AnalysisStatus.COMPLETED
    assert response.cached is False
    assert len(response.flags) >= 1
    assert len(response.precedents) == 3

    # Check LLM call received masked text
    mock_gemini.analyze_compliance.assert_called_once()
    called_masked_text = mock_gemini.analyze_compliance.call_args[1]["masked_text"]
    assert "Jessica Alba" not in called_masked_text
    assert "[CLIENT_1]" in called_masked_text

    # Check flag display unmasking
    high_flag = next((f for f in response.flags if f.matched_rule_id == "RULE-001"), None)
    assert high_flag is not None
    assert "Jessica Alba" in high_flag.passage_excerpt
    assert high_flag.severity == SeverityLevel.HIGH


def test_analysis_caching_avoids_duplicate_llm_calls(test_analysis_service, mock_gemini):
    mock_gemini.reset_mock()
    raw_doc = "Market update for Client: Mark Zuckerberg on Account #11223344."

    # First call: executes pipeline & calls LLM
    resp1 = test_analysis_service.process_document("doc_cache_test", raw_doc)
    assert resp1.cached is False
    assert mock_gemini.analyze_compliance.call_count == 1

    # Second call for the same document ID: returns cached analysis without LLM call
    resp2 = test_analysis_service.process_document("doc_cache_test", raw_doc)
    assert resp2.cached is True
    assert resp2.summary == resp1.summary
    assert mock_gemini.analyze_compliance.call_count == 1  # Still 1, NOT 2!


def test_retry_forces_regeneration(test_analysis_service, mock_gemini):
    mock_gemini.reset_mock()
    raw_doc = "Draft proposal for Client: Sarah Connor."

    # Process first time
    test_analysis_service.process_document("doc_retry_test", raw_doc)
    assert mock_gemini.analyze_compliance.call_count == 1

    # Explicit retry forces re-execution
    retry_resp = test_analysis_service.retry_analysis("doc_retry_test", raw_doc)
    assert retry_resp.cached is False
    assert mock_gemini.analyze_compliance.call_count == 2


def test_ai_never_makes_compliance_decision(test_analysis_service, mock_gemini):
    raw_doc = "Draft proposal with claims."
    resp = test_analysis_service.process_document("doc_boundary_test", raw_doc)

    # 1. Assert response status is only AI analysis lifecycle status (e.g. COMPLETED)
    assert resp.status in (AnalysisStatus.COMPLETED, AnalysisStatus.UNAVAILABLE, AnalysisStatus.RATE_LIMITED, AnalysisStatus.PENDING)
    
    # 2. Strict assertion: verify no review verdict fields exist in analysis response
    resp_dict = resp.model_dump()
    assert "decision" not in resp_dict
    assert "verdict" not in resp_dict
    assert "approved" not in str(resp.status).lower()
    assert "rejected" not in str(resp.status).lower()
    assert "needs_revision" not in str(resp.status).lower()

