"""
Data Engineering Integration & Fallback Tests.

Validates:
1. Data Engineering client HTTP requests and response mapping
2. /rule-lookup, /disclosure-check, /precedent-search contracts
3. Top 3 precedents constraint
4. Privacy boundary: Data Engineering receives masked text/chunks only
5. Timeout, connection failure, and HTTP error handling (graceful degradation)
6. Local vector store fallback
"""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from ai.app.core.config import Settings
from ai.app.models.entities import DisclosureCheckResult, PrecedentMatch, Rule
from ai.app.services.data_engineering_client import (
    DataEngineeringClient,
    DataEngineeringConnectionError,
    DataEngineeringError,
    DataEngineeringTimeout,
)
from ai.app.services.retrieval_service import RetrievalService
from scripts.seed_corpus import seed_all


@pytest.fixture(scope="module", autouse=True)
def init_corpus():
    seed_all(persist=False)


@pytest.fixture
def de_client():
    return DataEngineeringClient(base_url="http://test-de-service:5000", timeout=5.0)


# --- 1. Base URL Configuration & Init ---

def test_data_engineering_client_config():
    client = DataEngineeringClient(base_url="http://custom-de:8080/", timeout=12.5)
    assert client.base_url == "http://custom-de:8080"
    assert client.timeout == 12.5


# --- 2. /rule-lookup Mapping ---

def test_rule_lookup_mapping(de_client):
    mock_response_data = [
        {"rule_id": "RULE-001", "rule_text": "No guaranteed returns", "similarity_score": 0.89},
        {"rule_id": "RULE-004", "rule_text": "Elimination of Downside Risk", "similarity_score": 0.76},
    ]

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response_data
        mock_post.return_value = mock_resp

        rules = de_client.lookup_rules("Invest with guaranteed 20% return for [CLIENT_1]")

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        call_payload = mock_post.call_args[1]["json"]

        assert call_url == "http://test-de-service:5000/rule-lookup"
        assert "[CLIENT_1]" in call_payload["text"]
        assert len(rules) == 2
        assert rules[0][0].id == "RULE-001"
        assert rules[0][0].text == "No guaranteed returns"
        assert rules[0][1] == 0.89


# --- 3. /disclosure-check Mapping ---

def test_disclosure_check_mapping(de_client):
    mock_response_data = {
        "disclosure_id": "DISC-002",
        "disclosure_type": "PAST_PERFORMANCE",
        "present": True,
        "similarity_score": 0.8542,
        "matched_chunk_id": "Past performance is no guarantee of future results."
    }

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response_data
        mock_post.return_value = mock_resp

        chunks = ["Market commentary.", "Past performance is no guarantee of future results."]
        res = de_client.check_disclosure(
            document_chunks=chunks,
            disclosure_id="DISC-002",
            disclosure_text="Past performance is no guarantee of future results.",
            disclosure_type="PAST_PERFORMANCE",
        )

        mock_post.assert_called_once()
        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["document_chunks"] == chunks
        assert call_payload["disclosure_id"] == "DISC-002"

        assert isinstance(res, DisclosureCheckResult)
        assert res.disclosure_id == "DISC-002"
        assert res.disclosure_type == "PAST_PERFORMANCE"
        assert res.is_present is True
        assert res.similarity_score == 0.8542
        assert res.best_matching_passage == "Past performance is no guarantee of future results."


# --- 4. /precedent-search Mapping & Exactly 3 Precedents ---

def test_precedent_search_mapping_and_top_3(de_client):
    mock_response_data = [
        {"document_id": "PREC-DOC-001", "chunk_id": "chunk_1", "similarity_score": 0.92, "chunk_text": "Sample reviewed text 1"},
        {"document_id": "PREC-DOC-002", "chunk_id": "chunk_2", "similarity_score": 0.88, "chunk_text": "Sample reviewed text 2"},
        {"document_id": "PREC-DOC-003", "chunk_id": "chunk_3", "similarity_score": 0.81, "chunk_text": "Sample reviewed text 3"},
        {"document_id": "PREC-DOC-004", "chunk_id": "chunk_4", "similarity_score": 0.70, "chunk_text": "Sample reviewed text 4"},
    ]

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response_data
        mock_post.return_value = mock_resp

        precedents = de_client.search_precedents("Masked document text for [CLIENT_1]", top_k=3)

        assert len(precedents) == 3
        assert precedents[0].document_id == "PREC-DOC-001"
        assert precedents[0].similarity == 0.92
        assert precedents[2].document_id == "PREC-DOC-003"
        assert precedents[2].similarity == 0.81


# --- 5. Privacy Boundary: Masked Text Only Transmitted to DE ---

def test_data_engineering_receives_masked_text_only(de_client):
    raw_text = "Client: John Smith, Email: john@acme.com, Account: 12345678"
    from ai.app.services.privacy_wall import privacy_wall
    masked_text, mapping = privacy_wall.verify_and_sanitize_outbound(raw_text, document_id="doc_privacy_de")

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_post.return_value = mock_resp

        de_client.lookup_rules(masked_text)
        de_client.search_precedents(masked_text)

        for call in mock_post.call_args_list:
            sent_payload = str(call[1]["json"])
            # Assert 0 raw PII leaked in payload to Data Engineering
            assert "John Smith" not in sent_payload
            assert "john@acme.com" not in sent_payload
            assert "12345678" not in sent_payload
            # Placeholders must be present
            assert "[CLIENT_1]" in sent_payload or "[EMAIL_1]" in sent_payload or "[ACCOUNT_1]" in sent_payload


# --- 6. Timeout & Connection Error Handling with Graceful Fallback ---

def test_data_engineering_timeout_fallback():
    mock_de_client = MagicMock()
    mock_de_client.lookup_rules.side_effect = DataEngineeringTimeout("Request timed out")
    mock_de_client.check_disclosure.side_effect = DataEngineeringTimeout("Request timed out")
    mock_de_client.search_precedents.side_effect = DataEngineeringTimeout("Request timed out")

    service = RetrievalService(de_client=mock_de_client, use_de_service=True)
    masked_text = "Quarterly update. We guarantee a 20% annual return."

    # When DE times out, service falls back to local vector store without crashing
    rules = service.retrieve_relevant_rules(masked_text)
    assert len(rules) > 0

    disclosures = service.check_disclosures(masked_text)
    assert len(disclosures) > 0

    precedents = service.retrieve_precedents(masked_text, top_k=3)
    assert len(precedents) == 3


def test_data_engineering_connection_error_fallback():
    mock_de_client = MagicMock()
    mock_de_client.lookup_rules.side_effect = DataEngineeringConnectionError("Failed to connect")

    service = RetrievalService(de_client=mock_de_client, use_de_service=True)
    rules = service.retrieve_relevant_rules("Sample text")
    assert len(rules) > 0


def test_data_engineering_http_500_error_handling(de_client):
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("500 Server Error", request=MagicMock(), response=mock_resp)
        mock_post.return_value = mock_resp

        with pytest.raises(DataEngineeringError):
            de_client.lookup_rules("Sample text")
