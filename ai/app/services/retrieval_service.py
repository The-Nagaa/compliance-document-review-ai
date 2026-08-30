"""
Retrieval Service for Grounded Compliance Analysis.

Executes 3 specialized retrieval jobs:
1. Rule Retrieval (semantic search for applicable compliance rules)
2. Missing Disclosure Detection (detection by absence using similarity thresholds)
3. Precedent Search (finds top 3 historically reviewed documents)
"""

import re
from typing import List, Optional, Tuple
from ai.app.core.config import settings
from ai.app.core.logging import logger, SafeAuditLogger
from ai.app.models.entities import (
    Disclosure,
    DisclosureCheckResult,
    Flag,
    Precedent,
    PrecedentMatch,
    Rule,
    SeverityLevel,
)
from ai.app.repositories.vector_store import vector_store
from ai.app.services.embedding_service import embedding_service


class RetrievalService:
    """
    Orchestrates vector-grounded retrieval across Rules, Disclosures, and Precedents.
    """

    def __init__(self, store=vector_store):
        self.store = store

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping paragraph/sentence-aware chunks for granular retrieval."""
        if not text or len(text) <= chunk_size:
            return [text] if text else []

        # Split on paragraph or sentence boundaries
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\.\s+', text) if p.strip()]
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len > chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk))
                # Keep last part for overlap
                current_chunk = current_chunk[-1:]
                current_len = sum(len(c) for c in current_chunk)
            current_chunk.append(p)
            current_len += p_len

        if current_chunk:
            chunks.append(". ".join(current_chunk))

        return chunks if chunks else [text]

    # --- Job 1: Rule Retrieval ---
    def retrieve_relevant_rules(
        self, masked_text: str, top_k_per_chunk: int = 3, max_total_rules: int = 8
    ) -> List[Tuple[Rule, float]]:
        """
        Retrieves compliance rules relevant to the sections of the masked document.
        """
        chunks = self.chunk_text(masked_text)
        rule_scores: dict[str, Tuple[Rule, float]] = {}

        for chunk in chunks:
            chunk_vec = embedding_service.get_embedding(chunk)
            matches = self.store.search_rules(chunk_vec, top_k=top_k_per_chunk)
            for rule, sim in matches:
                if sim > 0.15:  # Relevance cutoff
                    if rule.id not in rule_scores or sim > rule_scores[rule.id][1]:
                        rule_scores[rule.id] = (rule, sim)

        # Also search whole document vector to catch holistic themes
        doc_vec = embedding_service.get_embedding(masked_text)
        holistic_matches = self.store.search_rules(doc_vec, top_k=settings.RULE_TOP_K)
        for rule, sim in holistic_matches:
            if rule.id not in rule_scores or sim > rule_scores[rule.id][1]:
                rule_scores[rule.id] = (rule, sim)

        # Sort descending by similarity
        sorted_rules = sorted(rule_scores.values(), key=lambda x: x[1], reverse=True)
        return sorted_rules[:max_total_rules]

    # --- Job 2: Missing Disclosure Detection by Absence ---
    def check_disclosures(
        self,
        masked_text: str,
        threshold: Optional[float] = None,
    ) -> List[DisclosureCheckResult]:
        """
        Checks each standard required disclosure against the document passages.
        If the maximum similarity is below threshold, it is flagged as missing.
        """
        similarity_threshold = threshold if threshold is not None else settings.DISCLOSURE_SIMILARITY_THRESHOLD
        chunks = self.chunk_text(masked_text)
        disclosures = self.store.get_all_disclosures()
        results: List[DisclosureCheckResult] = []

        # Embed all document chunks once
        chunk_embeddings = [embedding_service.get_embedding(c) for c in chunks]

        for disc in disclosures:
            disc_vec = disc.embedding or embedding_service.get_embedding(disc.text)
            best_sim = -1.0
            best_chunk = None

            for idx, c_vec in enumerate(chunk_embeddings):
                sim = embedding_service.cosine_similarity(disc_vec, c_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_chunk = chunks[idx]

            # Also check against full text for short disclosures
            full_doc_vec = embedding_service.get_embedding(masked_text)
            full_sim = embedding_service.cosine_similarity(disc_vec, full_doc_vec)
            if full_sim > best_sim:
                best_sim = full_sim
                best_chunk = masked_text[:200]

            is_present = best_sim >= similarity_threshold

            results.append(
                DisclosureCheckResult(
                    disclosure_id=disc.id,
                    disclosure_type=disc.type,
                    disclosure_text=disc.text,
                    is_present=is_present,
                    similarity_score=round(max(0.0, best_sim), 4),
                    best_matching_passage=best_chunk if is_present else None,
                )
            )

        return results

    # --- Job 3: Precedent Retrieval ---
    def retrieve_precedents(
        self, masked_text: str, top_k: int = 3
    ) -> List[PrecedentMatch]:
        """
        Finds the top 3 most similar historical reviewed documents.
        """
        doc_vec = embedding_service.get_embedding(masked_text)
        matches = self.store.search_precedents(doc_vec, top_k=top_k)

        results: List[PrecedentMatch] = []
        for prec, sim in matches:
            results.append(
                PrecedentMatch(
                    document_id=prec.document_id,
                    similarity=round(sim, 4),
                    previous_decision=prec.decision,
                    officer_comment=prec.comment,
                    excerpt=prec.masked_text[:200] + ("..." if len(prec.masked_text) > 200 else ""),
                )
            )
        return results

    def generate_missing_disclosure_flags(
        self, disclosure_results: List[DisclosureCheckResult]
    ) -> List[Flag]:
        """
        Converts detected missing mandatory disclosures into structured compliance flags.
        """
        flags: List[Flag] = []
        for res in disclosure_results:
            if not res.is_present:
                flags.append(
                    Flag(
                        passage_excerpt="[ENTIRE_DOCUMENT: DISCLOSURE_ABSENT]",
                        matched_rule_id=f"DISC-{res.disclosure_id}",
                        matched_rule=f"Required Disclosure: {res.disclosure_type}",
                        explanation=f"Required disclosure '{res.disclosure_type}' was not detected in document (max match: {res.similarity_score * 100:.1f}%).",
                        severity=SeverityLevel.HIGH if "RISK" in res.disclosure_type or "PERFORMANCE" in res.disclosure_type else SeverityLevel.MEDIUM,
                    )
                )
        return flags


retrieval_service = RetrievalService()
