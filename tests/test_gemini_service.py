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
    svc = GeminiService(api_key="fake-key")

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit exceeded"
        mock_post.return_value = mock_resp

        with pytest.raises(AIRateLimited):
            svc.analyze_compliance("Sample text", sample_rules, "doc_ratelimit")


def test_gemini_timeout(sample_rules):
    svc = GeminiService(api_key="fake-key")

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")):
        with pytest.raises(AITimeout):
            svc.analyze_compliance("Sample text", sample_rules, "doc_timeout")


def test_gemini_malformed_json_fallback(sample_rules):
    svc = GeminiService(api_key="fake-key")

    # LLM returns text with summary in markdown format but unparseable JSON array
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

        # Should recover summary gracefully without crashing
        result = svc.analyze_compliance("Sample text", sample_rules, "doc_malformed")
        assert result.summary == "Quarterly market review document for retail investors."
        assert isinstance(result.flags, list)


def test_gemini_invalid_api_key_403(sample_rules):
    svc = GeminiService(api_key="invalid-api-key")

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "API_KEY_INVALID"
        mock_post.return_value = mock_resp

        with pytest.raises(AIConfigurationError):
            svc.analyze_compliance("Sample text", sample_rules, "doc_invalid_key")

