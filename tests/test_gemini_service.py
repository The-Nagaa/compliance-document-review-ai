"""
Gemini Service Unit Tests (Requirement 7, 8, 9, 13).
"""

import json
from unittest.mock import MagicMock, patch
import httpx
import pytest

from ai.app.models.entities import Rule, SeverityLevel
from ai.app.schemas.analysis import LLMComplianceOutput
from ai.app.services.gemini_service import (
    AIConfigurationError,
    AIInvalidResponse,
    AIRateLimited,
    AITimeout,
    AIUnavailable,
    GeminiService,
)


@pytest.fixture
def sample_rules():
    return [
        (Rule(id="RULE-001", text="Prohibition of Guaranteed Returns", category="PROHIBITED_CLAIMS"), 0.85),
        (Rule(id="RULE-020", text="SEC Registered Investment Advisor Disclosure", category="REQUIRED_DISCLOSURES"), 0.72),
    ]


def test_missing_api_key_raises_configuration_error(sample_rules):
    svc = GeminiService(api_key="")
    with pytest.raises(AIConfigurationError):
        svc.analyze_compliance("Sample text", sample_rules, document_id="doc_1")


def test_valid_gemini_response_parsing(sample_rules):
    svc = GeminiService(api_key="fake-valid-key")

    mock_llm_json = {
        "summary": "This document is a marketing proposal promising high returns with incomplete disclaimers.",
        "flags": [
            {
                "passage": "We guarantee a 20% annual return on all investments.",
                "matched_rule_id": "RULE-001",
                "matched_rule": "Prohibition of Guaranteed Returns",
                "explanation": "The passage makes an explicit guarantee of return which violates regulatory standards.",
                "severity": "high"
            }
        ]
    }

    mock_gemini_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(mock_llm_json)}]
                }
            }
        ]
    }

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_gemini_payload
        mock_post.return_value = mock_resp

        result = svc.analyze_compliance("We guarantee a 20% annual return on all investments.", sample_rules, "doc_valid")

        assert isinstance(result, LLMComplianceOutput)
        assert len(result.flags) == 1
        assert result.flags[0].matched_rule_id == "RULE-001"
        assert result.flags[0].severity == SeverityLevel.HIGH
        assert "guarantee" in result.flags[0].passage.lower()


def test_gemini_rate_limit_429(sample_rules):
    svc = GeminiService(api_key="fake-key", backoff_schedule=(0.001, 0.001))

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit exceeded"
        mock_post.return_value = mock_resp

        with pytest.raises(AIRateLimited):
            svc.analyze_compliance("Sample text", sample_rules, "doc_ratelimit")
        assert mock_post.call_count == 3  # Initial + 2 retries


def test_gemini_timeout(sample_rules):
    svc = GeminiService(api_key="fake-key", backoff_schedule=(0.001, 0.001))

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")) as mock_post:
        with pytest.raises(AITimeout):
            svc.analyze_compliance("Sample text", sample_rules, "doc_timeout")
        assert mock_post.call_count == 3  # Initial + 2 retries


def test_gemini_malformed_json_fallback(sample_rules):
    svc = GeminiService(api_key="fake-key")

    mock_gemini_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": '{"summary": "Quarterly market review document for retail investors.", "flags": INVALID_JSON}'}]
                }
            }
        ]
    }

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_gemini_payload
        mock_post.return_value = mock_resp

        result = svc.analyze_compliance("Sample text", sample_rules, "doc_malformed")
        assert result.summary == "Quarterly market review document for retail investors."
        assert isinstance(result.flags, list)


def test_gemini_invalid_api_key_403(sample_rules):
    svc = GeminiService(api_key="invalid-api-key", backoff_schedule=(0.001, 0.001))

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "API_KEY_INVALID"
        mock_post.return_value = mock_resp

        with pytest.raises(AIConfigurationError):
            svc.analyze_compliance("Sample text", sample_rules, "doc_invalid_key")
        assert mock_post.call_count == 1  # Fails fast without retrying


def test_gemini_transient_503_retry_success(sample_rules):
    svc = GeminiService(api_key="fake-key", backoff_schedule=(0.001, 0.001))

    mock_llm_json = {
        "summary": "Recovered after 503 retry.",
        "flags": []
    }
    mock_payload = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(mock_llm_json)}]}}]
    }

    resp_503 = MagicMock(status_code=503, text="Service Unavailable")
    resp_200 = MagicMock(status_code=200, json=MagicMock(return_value=mock_payload))

    with patch("httpx.Client.post", side_effect=[resp_503, resp_200]) as mock_post:
        result = svc.analyze_compliance("Sample text", sample_rules, "doc_503_recovery")
        assert result.summary == "Recovered after 503 retry."
        assert mock_post.call_count == 2


def test_gemini_transient_503_exhausted_raises_unavailable(sample_rules):
    svc = GeminiService(api_key="fake-key", backoff_schedule=(0.001, 0.001))
    resp_503 = MagicMock(status_code=503, text="Service Unavailable")

    with patch("httpx.Client.post", return_value=resp_503) as mock_post:
        with pytest.raises(AIUnavailable) as exc_info:
            svc.analyze_compliance("Sample text", sample_rules, "doc_503_exhausted")
        assert "503" in str(exc_info.value)
        assert mock_post.call_count == 3  # Initial + 2 retries


def test_gemini_non_retryable_404_fails_fast(sample_rules):
    svc = GeminiService(api_key="fake-key", backoff_schedule=(0.001, 0.001))
    resp_404 = MagicMock(status_code=404, text="Model Not Found")

    with patch("httpx.Client.post", return_value=resp_404) as mock_post:
        with pytest.raises(AIConfigurationError):
            svc.analyze_compliance("Sample text", sample_rules, "doc_404_fast")
        assert mock_post.call_count == 1  # Exactly 1, no retries


def test_gemini_service_session_reuse():
    svc = GeminiService(api_key="test-key")
    c1 = svc._get_client()
    c2 = svc._get_client()
    assert c1 is c2
    assert not c1.is_closed

    svc.close()
    assert c1.is_closed

    c3 = svc._get_client()
    assert c3 is not c1
    assert not c3.is_closed
    svc.close()
