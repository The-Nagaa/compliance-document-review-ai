"""
Retrieval Service for Grounded Compliance Analysis.

Executes 3 specialized retrieval jobs:
1. Rule Retrieval (semantic search for applicable compliance rules)
2. Missing Disclosure Detection (detection by absence using similarity thresholds)
3. Precedent Search (finds top 3 historically reviewed documents)

Performance & Reliability:
- Uses persistent connection pool to Data Engineering service.
- Features parallelized disclosure checks with deterministic result ordering.
- Comprehensive non-spamming profiling of retrieval operations.
- Graceful local vector store fallback for local development or transient outages.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
from typing import Dict, List, Optional, Tuple
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
from ai.app.services.data_engineering_client import (
    DataEngineeringClient,
    DataEngineeringError,
    data_engineering_client,
)
from ai.app.services.embedding_service import embedding_service


class RetrievalService:
    """
    Orchestrates vector-grounded retrieval across Rules, Disclosures, and Precedents.
    """

    def __init__(
        self,
        store=vector_store,
        de_client: Optional[DataEngineeringClient] = None,
        use_de_service: Optional[bool] = None,
        max_workers: int = 5,
    ):
        self.store = store
        self.de_client = de_client or data_engineering_client
        self.use_de_service = (
            use_de_service
            if use_de_service is not None
            else settings.USE_DATA_ENGINEERING_SERVICE
        )
        self.max_workers = max_workers

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping paragraph/sentence-aware chunks for granular retrieval."""
        if not text or len(text) <= chunk_size:
            return [text] if text else []

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\.\s+', text) if p.strip()]
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len > chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk))
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
        Uses Data Engineering /rule-lookup in production, falling back to local vector store.
        """
        start_t = time.perf_counter()
        if self.use_de_service:
            try:
                rules = self.de_client.lookup_rules(masked_text)
                elapsed_ms = (time.perf_counter() - start_t) * 1000
                logger.info(f"Retrieved {len(rules)} rules via Data Engineering ({elapsed_ms:.2f} ms).")
                if rules:
                    return rules[:max_total_rules]
            except DataEngineeringError as e:
                logger.warning(f"Data Engineering rule lookup failed ({e}). Falling back to local vector store.")

        return self._local_retrieve_relevant_rules(masked_text, top_k_per_chunk, max_total_rules)

    def _local_retrieve_relevant_rules(
        self, masked_text: str, top_k_per_chunk: int = 3, max_total_rules: int = 8
    ) -> List[Tuple[Rule, float]]:
        """Local vector store rule search for offline/testing development."""
        chunks = self.chunk_text(masked_text)
        rule_scores: dict[str, Tuple[Rule, float]] = {}

        for chunk in chunks:
            chunk_vec = embedding_service.get_embedding(chunk)
            matches = self.store.search_rules(chunk_vec, top_k=top_k_per_chunk)
            for rule, sim in matches:
                if sim > 0.15:
                    if rule.id not in rule_scores or sim > rule_scores[rule.id][1]:
                        rule_scores[rule.id] = (rule, sim)

        doc_vec = embedding_service.get_embedding(masked_text)
        holistic_matches = self.store.search_rules(doc_vec, top_k=settings.RULE_TOP_K)
        for rule, sim in holistic_matches:
            if rule.id not in rule_scores or sim > rule_scores[rule.id][1]:
                rule_scores[rule.id] = (rule, sim)

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
        Uses Data Engineering /disclosure-check in production with connection reuse and profiling.
        """
        start_t = time.perf_counter()
        disclosures = self.store.get_all_disclosures()
        total_disclosures = len(disclosures)

        if self.use_de_service:
            try:
                chunks = self.chunk_text(masked_text)
                results_by_index: Dict[int, DisclosureCheckResult] = {}
                disc_timings: List[Tuple[str, float]] = []

                similarity_threshold = threshold if threshold is not None else settings.DISCLOSURE_SIMILARITY_THRESHOLD

                def _check_single(idx: int, disc: Disclosure) -> Tuple[int, str, float, DisclosureCheckResult]:
                    d_start = time.perf_counter()
                    res = self.de_client.check_disclosure(
                        document_chunks=chunks,
                        disclosure_id=disc.id,
                        disclosure_text=disc.text,
                        disclosure_type=disc.type,
                        threshold=similarity_threshold,
                    )
                    d_elapsed = (time.perf_counter() - d_start) * 1000
                    return idx, disc.id, d_elapsed, res

                # Parallel execution using connection pool
                workers = min(self.max_workers, max(1, total_disclosures))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(_check_single, idx, disc)
                        for idx, disc in enumerate(disclosures)
                    ]
                    for future in as_completed(futures):
                        idx, disc_id, d_elapsed, res = future.result()
                        results_by_index[idx] = res
                        disc_timings.append((disc_id, d_elapsed))

                # Guarantee exact deterministic index ordering
                ordered_results = [results_by_index[i] for i in range(total_disclosures)]
                total_elapsed_ms = (time.perf_counter() - start_t) * 1000
                request_avg_ms = (
                    sum(duration for _, duration in disc_timings) / len(disc_timings)
                    if disc_timings
                    else 0.0
                )
                slowest_id, slowest_ms = max(disc_timings, key=lambda x: x[1]) if disc_timings else ("N/A", 0.0)

                # Log consolidated single profiling summary
                logger.info(
                    f"DISCLOSURE_PROFILING | total_ms={total_elapsed_ms:.2f} | avg_request_ms={request_avg_ms:.2f} | "
                    f"requests={total_disclosures} | slowest={slowest_id} ({slowest_ms:.2f} ms)"
                )

                return ordered_results

            except DataEngineeringError as e:
                logger.warning(f"Data Engineering disclosure check failed ({e}). Falling back to local vector store.")

        # Local fallback execution
        local_results = self._local_check_disclosures(masked_text, threshold)
        local_elapsed_ms = (time.perf_counter() - start_t) * 1000
        logger.info(f"Local disclosure check completed: {len(local_results)} disclosures in {local_elapsed_ms:.2f} ms.")
        return local_results

    def _local_check_disclosures(
        self,
        masked_text: str,
        threshold: Optional[float] = None,
    ) -> List[DisclosureCheckResult]:
        """Local vector store disclosure check for offline/testing development."""
        similarity_threshold = threshold if threshold is not None else settings.DISCLOSURE_SIMILARITY_THRESHOLD
        chunks = self.chunk_text(masked_text)
        disclosures = self.store.get_all_disclosures()
        results: List[DisclosureCheckResult] = []

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
        Uses Data Engineering /precedent-search in production, falling back to local vector store.
        """
        start_t = time.perf_counter()
        if self.use_de_service:
            try:
                precedents = self.de_client.search_precedents(masked_text, top_k=top_k)
                elapsed_ms = (time.perf_counter() - start_t) * 1000
                logger.info(f"Retrieved {len(precedents)} precedents via Data Engineering ({elapsed_ms:.2f} ms).")
                if len(precedents) == top_k:
                    return precedents
                elif len(precedents) > 0:
                    return precedents[:top_k]
            except DataEngineeringError as e:
                logger.warning(f"Data Engineering precedent search failed ({e}). Falling back to local vector store.")

        return self._local_retrieve_precedents(masked_text, top_k)

    def _local_retrieve_precedents(
        self, masked_text: str, top_k: int = 3
    ) -> List[PrecedentMatch]:
        """Local vector store precedent search for offline/testing development."""
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
