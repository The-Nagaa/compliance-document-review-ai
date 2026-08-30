"""
Edge Cases Test Suite for PII Masker and AI Pipeline (Requirement 11).
"""

from unittest.mock import MagicMock, patch
import pytest

from ai.app.models.entities import AnalysisStatus
from ai.app.repositories.analysis_cache import AnalysisCache
from ai.app.schemas.analysis import LLMComplianceOutput
from ai.app.services.analysis_service import AnalysisService
from ai.app.services.pii_masker import PIIMasker
from ai.app.services.retrieval_service import RetrievalService
from scripts.seed_corpus import seed_all


@pytest.fixture(scope="module", autouse=True)
def init_corpus():
    seed_all(persist=False)


@pytest.fixture
def masker():
    return PIIMasker()


# --- PII Edge Cases ---

def test_edge_case_no_pii(masker):
    text = "General market updates on S&P 500 index movements and treasury yields."
    masked, mapping, records = masker.mask_text(text)
    assert masked == text
    assert len(mapping) == 0


def test_edge_case_single_pii_item(masker):
    text = "Send confirmation to test.user@sample.org immediately."
    masked, mapping, records = masker.mask_text(text)
    assert "[EMAIL_1]" in masked
    assert "test.user@sample.org" not in masked
    assert len(mapping) == 1


def test_edge_case_multiple_phone_formats(masker):
    formats = [
        "1. +1-555-123-4567",
        "2. (555) 987-6543",
        "3. 555.444.3333",
        "4. Phone: 9876543210",
        "5. Mobile: +91 9876543210",
    ]
    text = "\n".join(formats)
    masked, mapping, records = masker.mask_text(text)
    assert "123-4567" not in masked
    assert "987-6543" not in masked
    assert "444.3333" not in masked
    assert "9876543210" not in masked


def test_edge_case_overlapping_pii(masker):
    # Overlapping client name and email with same base string
    text = "Client: Alice Smith and email alicesmith@alicesmith.com"
    masked, mapping, records = masker.mask_text(text)
    assert "Alice Smith" not in masked
    assert "alicesmith@alicesmith.com" not in masked
    assert "[CLIENT_1]" in masked
    assert "[EMAIL_1]" in masked


def test_edge_case_ssn_variations(masker):
    text = "SSN: 123-45-6789 and Tax ID: 987654321 on custody file."
    masked, mapping, records = masker.mask_text(text)
    assert "123-45-6789" not in masked
    assert "987654321" not in masked


# --- AI Edge Cases ---

def test_edge_case_empty_and_whitespace_document():
    cache = AnalysisCache()
    retriever = RetrievalService()
    mock_llm = MagicMock()
    mock_llm.analyze_compliance.return_value = LLMComplianceOutput(summary="Empty document", flags=[])
    
    svc = AnalysisService(retriever=retriever, llm=mock_llm, cache=cache)

    resp = svc.process_document("doc_empty", "   \n\t  ")
    assert resp.document_id == "doc_empty"
    assert resp.status == AnalysisStatus.COMPLETED


def test_edge_case_very_short_document():
    cache = AnalysisCache()
    retriever = RetrievalService()
    mock_llm = MagicMock()
    mock_llm.analyze_compliance.return_value = LLMComplianceOutput(summary="Short note", flags=[])

    svc = AnalysisService(retriever=retriever, llm=mock_llm, cache=cache)

    resp = svc.process_document("doc_short", "Hello.")
    assert resp.document_id == "doc_short"
    assert len(resp.precedents) == 3


def test_edge_case_long_document():
    cache = AnalysisCache()
    retriever = RetrievalService()
    mock_llm = MagicMock()
    mock_llm.analyze_compliance.return_value = LLMComplianceOutput(summary="Long disclosure review", flags=[])

    svc = AnalysisService(retriever=retriever, llm=mock_llm, cache=cache)

    # 10,000 character document
    long_text = ("Section overview on risk management and asset allocations. " * 150) + "Client: John Doe."
    resp = svc.process_document("doc_long", long_text)
    assert resp.document_id == "doc_long"
    assert len(resp.precedents) == 3


def test_edge_case_irrelevant_document():
    cache = AnalysisCache()
    retriever = RetrievalService()
    mock_llm = MagicMock()
    mock_llm.analyze_compliance.return_value = LLMComplianceOutput(summary="Recipe for chocolate chip cookies.", flags=[])

    svc = AnalysisService(retriever=retriever, llm=mock_llm, cache=cache)

    resp = svc.process_document("doc_irrelevant", "Preheat oven to 350 degrees. Mix flour, sugar, and chocolate chips.")
    assert resp.document_id == "doc_irrelevant"
    assert len(resp.precedents) == 3
