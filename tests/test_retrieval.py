"""
Vector Retrieval Unit Tests (Requirement 5 & 6).
"""

import pytest
from ai.app.models.entities import Disclosure, Precedent, Rule
from ai.app.repositories.vector_store import VectorStore
from ai.app.services.retrieval_service import RetrievalService
from scripts.seed_corpus import seed_all


@pytest.fixture(scope="module", autouse=True)
def setup_vector_store():
    # Ensure vector store is populated
    seed_all(persist=False)


@pytest.fixture
def retrieval():
    return RetrievalService()


def test_rule_retrieval_guaranteed_claims(retrieval):
    query_text = "We assure our clients a guaranteed 25% annual return with zero risk of loss."
    rules = retrieval.retrieve_relevant_rules(query_text, top_k_per_chunk=3)

    assert len(rules) > 0
    rule_ids = [r.id for r, sim in rules]
    # Should retrieve RULE-001 (Prohibition of Guaranteed Returns) or related prohibited claim rules
    assert "RULE-001" in rule_ids or "RULE-002" in rule_ids or "RULE-004" in rule_ids
    # Verify rule contains ID and text
    first_rule, sim = rules[0]
    assert first_rule.id.startswith("RULE-")
    assert len(first_rule.text) > 10
    assert sim > 0.0


def test_rule_retrieval_past_performance(retrieval):
    query_text = "Our historical track record shows 15% CAGR over the last 5 years."
    rules = retrieval.retrieve_relevant_rules(query_text, top_k_per_chunk=3)

    rule_ids = [r.id for r, sim in rules]
    assert "RULE-003" in rule_ids or "RULE-010" in rule_ids or "RULE-011" in rule_ids or "RULE-016" in rule_ids


def test_disclosure_presence_detection(retrieval):
    # Document with past performance disclaimer included
    doc_with_disclosure = (
        "Market update for investors. Past performance is no guarantee of future results. "
        "Investments are subject to market risk, including possible loss of principal."
    )

    results = retrieval.check_disclosures(doc_with_disclosure, threshold=0.70)
    past_perf_res = next((r for r in results if r.disclosure_type == "PAST_PERFORMANCE"), None)

    assert past_perf_res is not None
    assert past_perf_res.is_present is True
    assert past_perf_res.similarity_score >= 0.70


def test_disclosure_absence_detection(retrieval):
    # Short document with NO disclosures at all
    doc_without_disclosures = "Check out our newest fund strategy launching next week."

    results = retrieval.check_disclosures(doc_without_disclosures, threshold=0.70)
    
    # Most standard disclosures should be detected as missing (absent)
    missing_count = sum(1 for r in results if not r.is_present)
    assert missing_count > len(results) // 2

    # Check generated flags
    flags = retrieval.generate_missing_disclosure_flags(results)
    assert len(flags) > 0
    assert any("Required Disclosure" in f.matched_rule for f in flags)


def test_precedent_top_3_search(retrieval):
    doc_query = "Quarterly Market Commentary prepared for client portfolio. Covered equities and bond allocation with SEC RIA disclosures."
    precedents = retrieval.retrieve_precedents(doc_query, top_k=3)

    # Exactly 3 precedents must be returned per specification
    assert len(precedents) == 3
    for p in precedents:
        assert p.document_id.startswith("PREC-DOC-")
        assert p.previous_decision in ("approved", "rejected", "needs_revision")
        assert len(p.officer_comment) > 0
        assert 0.0 <= p.similarity <= 1.0

    # Top match should have higher similarity than the 3rd match
    assert precedents[0].similarity >= precedents[2].similarity


def test_paraphrased_disclosure_detection(retrieval):
    # Paraphrased version of past performance disclaimer
    paraphrased_doc = (
        "Client quarterly report. Please note that historical returns do not predict or ensure future investment performance. "
        "Trading securities involves risk of loss of your initial investment."
    )

    results = retrieval.check_disclosures(paraphrased_doc, threshold=0.55)
    past_perf_res = next((r for r in results if r.disclosure_type == "PAST_PERFORMANCE"), None)

    assert past_perf_res is not None
    assert past_perf_res.is_present is True
    assert past_perf_res.similarity_score >= 0.55

