"""
Privacy Wall Boundary Verification Tests (Requirement 4 & 11).
"""

import pytest
from ai.app.services.privacy_wall import PrivacyViolationError, PrivacyWall


@pytest.fixture
def privacy_wall():
    return PrivacyWall()


def test_outbound_payload_sanitization(privacy_wall):
    raw_document = """
    Marketing Proposal for Client: Jonathan Reynolds
    Contact Email: jonathan.reynolds@financecorp.org
    Custody Account: Acct #8844220011
    SSN on File: 987-65-4321
    Residential Address: 742 Evergreen Terrace, Springfield, OR 97477
    Guaranteed return of 15% annual yield with zero risk.
    """

    masked_text, mapping = privacy_wall.verify_and_sanitize_outbound(
        raw_text=raw_document,
        document_id="doc_test_privacy",
    )

    # 1. Assert NO raw PII exists in outbound text
    assert "Jonathan Reynolds" not in masked_text
    assert "jonathan.reynolds@financecorp.org" not in masked_text
    assert "8844220011" not in masked_text
    assert "987-65-4321" not in masked_text
    assert "742 Evergreen Terrace" not in masked_text

    # 2. Assert placeholders are properly present
    assert "[CLIENT_1]" in masked_text
    assert "[EMAIL_1]" in masked_text
    assert "[ACCOUNT_1]" in masked_text
    assert "[SSN_1]" in masked_text
    assert "[ADDRESS_1]" in masked_text

    # 3. Assert non-sensitive compliance claim text is preserved
    assert "Guaranteed return of 15% annual yield with zero risk" in masked_text

    # 4. Assert mapping contains server-side pairs
    assert mapping["[CLIENT_1]"] == "Jonathan Reynolds"
    assert mapping["[EMAIL_1]"] == "jonathan.reynolds@financecorp.org"


def test_privacy_wall_catches_unmasked_sensitive_token(privacy_wall):
    # If a known sensitive token is present in the outbound text (e.g. unmasked proprietary identifier), PrivacyViolationError is raised
    with pytest.raises(PrivacyViolationError):
        privacy_wall.verify_and_sanitize_outbound(
            raw_text="Direct unmasked text containing CustomSecretClientID_9921",
            document_id="doc_fail",
            known_sensitive_values=["CustomSecretClientID_9921"],
        )


def test_inspect_sanitization_output(privacy_wall):
    raw = "Client: Sarah Connor with account Acct #55443322 and email sarah@skynet.com"
    inspection = privacy_wall.inspect_sanitization(raw, "inspect_doc_1")

    assert inspection["document_id"] == "inspect_doc_1"
    assert inspection["detected_entities_count"] >= 3
    assert inspection["has_raw_pii_in_outbound_payload"] is False
    assert "outbound_payload_preview" in inspection
    assert "sarah@skynet.com" not in inspection["masked_text_preview"]


def test_gemini_outbound_payload_has_zero_raw_pii():
    """Verify intercepted outbound Gemini call contains placeholders only, never raw PII."""
    from unittest.mock import MagicMock, patch
    from ai.app.services.analysis_service import AnalysisService
    from ai.app.services.gemini_service import GeminiService
    from ai.app.repositories.analysis_cache import AnalysisCache
    from ai.app.services.retrieval_service import RetrievalService

    test_doc = """Client Name: John Smith
Email: john.smith@example.com
Phone: 9876543210
Account Number: 123456789012
Address: 12 MG Road, Hyderabad
Investment Amount: $50,000

John Smith contacted us again.
john.smith@example.com
Account Number: 123456789012"""

    raw_pii_tokens = ["John Smith", "john.smith@example.com", "9876543210", "123456789012", "12 MG Road, Hyderabad", "$50,000"]
    intercepted_requests = []

    gemini_svc = GeminiService(api_key="mock_key_for_test")
    analysis_svc = AnalysisService(retriever=RetrievalService(), llm=gemini_svc, cache=AnalysisCache())

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": '{"summary": "Summary for [CLIENT_1]", "flags": []}'}]
                }
            }]
        }
        
        def capture_post(url, json=None, **kwargs):
            intercepted_requests.append(json)
            return mock_resp

        mock_post.side_effect = capture_post

        analysis_svc.process_document("doc_privacy_audit", test_doc)

    assert len(intercepted_requests) == 1
    sent_payload_str = str(intercepted_requests[0])

    # 1. Assert all placeholders exist in outbound Gemini payload
    assert "[CLIENT_1]" in sent_payload_str
    assert "[EMAIL_1]" in sent_payload_str
    assert "[PHONE_1]" in sent_payload_str
    assert "[ACCOUNT_1]" in sent_payload_str
    assert "[ADDRESS_1]" in sent_payload_str
    assert "[AMOUNT_1]" in sent_payload_str

    # 2. Strict assertion: Prove raw PII tokens are completely absent from outbound payload
    for pii in raw_pii_tokens:
        assert pii not in sent_payload_str, f"Privacy Breach! Raw PII token '{pii}' was sent to Gemini!"


def test_embedding_pipeline_receives_masked_text_only():
    """Verify that vector retrieval & embedding calls receive only masked text."""
    from unittest.mock import MagicMock, patch
    from ai.app.services.analysis_service import AnalysisService
    from ai.app.repositories.analysis_cache import AnalysisCache
    from ai.app.services.retrieval_service import RetrievalService

    test_doc = "Client Name: John Smith with Account Number: 123456789012"
    embedded_texts = []

    retriever = RetrievalService()
    mock_llm = MagicMock()
    mock_llm.analyze_compliance.return_value = MagicMock(summary="test", flags=[])

    analysis_svc = AnalysisService(retriever=retriever, llm=mock_llm, cache=AnalysisCache())

    with patch("ai.app.services.embedding_service.embedding_service.get_embedding") as mock_embed:
        def capture_embed(text):
            embedded_texts.append(text)
            return [0.0] * 128

        mock_embed.side_effect = capture_embed

        analysis_svc.process_document("doc_embed_privacy", test_doc)

    assert len(embedded_texts) > 0
    for text in embedded_texts:
        assert "John Smith" not in text, "Raw name leaked into embedding engine!"
        assert "123456789012" not in text, "Raw account number leaked into embedding engine!"

