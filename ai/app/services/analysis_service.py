"""
Compliance Analysis Orchestration Service.

Coordinates:
1. Privacy-Wall Masking (never sends raw text to external endpoints)
2. Vector Retrieval (Rules, Missing Disclosures, Top-3 Precedents)
3. Gemini Grounded Compliance Flagging & Summarization
4. Graceful Degradation & Non-Blocking Error States
5. Caching & Retry Management
"""

from datetime import datetime, timezone
import time
from typing import List, Optional
import uuid

from ai.app.core.logging import logger, SafeAuditLogger
from ai.app.models.entities import (
    AIAnalysis,
    AnalysisStatus,
    Flag,
    PrecedentMatch,
    SeverityLevel,
)
from ai.app.repositories.analysis_cache import analysis_cache
from ai.app.schemas.analysis import AnalysisResponse, LLMComplianceOutput
from ai.app.schemas.flags import FlagResponse
from ai.app.schemas.retrieval import DisclosureCheck, PrecedentMatchResponse
from ai.app.services.gemini_service import (
    AIConfigurationError,
    AIException,
    AIInvalidResponse,
    AIRateLimited,
    AITimeout,
    AIUnavailable,
    gemini_service,
)
from ai.app.services.privacy_wall import privacy_wall
from ai.app.services.retrieval_service import retrieval_service


class AnalysisService:
    """
    Main orchestrator for AI compliance review assistance.
    """

    def __init__(
        self,
        retriever=retrieval_service,
        llm=gemini_service,
        cache=analysis_cache,
    ):
        self.retriever = retriever
        self.llm = llm
        self.cache = cache

    def process_document(
        self,
        document_id: str,
        extracted_text: str,
        force_refresh: bool = False,
    ) -> AnalysisResponse:
        """
        Processes a document through the AI analysis pipeline with caching.
        """
        start_time = time.time()
        SafeAuditLogger.log_event("DOCUMENT_ANALYSIS_STARTED", document_id)

        # 1. Check Cache
        cached_entry = self.cache.get(document_id)
        if (
            cached_entry
            and not force_refresh
            and cached_entry.status == AnalysisStatus.COMPLETED
        ):
            SafeAuditLogger.log_event("ANALYSIS_CACHE_HIT", document_id)
            return self._build_response_from_entity(cached_entry, cached=True)

        # 2. Privacy Wall: Server-Side PII Masking
        masked_text, pii_mapping = privacy_wall.verify_and_sanitize_outbound(
            raw_text=extracted_text, document_id=document_id
        )

        # 3. Vector Retrieval (Local Vector Engine)
        # A. Grounding Rules
        retrieved_rules = self.retriever.retrieve_relevant_rules(masked_text)

        # B. Missing Disclosure Detection by Absence
        disclosure_results = self.retriever.check_disclosures(masked_text)
        missing_disclosure_flags = self.retriever.generate_missing_disclosure_flags(
            disclosure_results
        )

        # C. Precedent Retrieval (Top 3)
        precedents = self.retriever.retrieve_precedents(masked_text, top_k=3)

        SafeAuditLogger.log_event(
            "RETRIEVAL_COMPLETED",
            document_id,
            extra={
                "rules_retrieved": len(retrieved_rules),
                "disclosures_checked": len(disclosure_results),
                "precedents_matched": len(precedents),
            },
        )

        # 4. LLM Analysis via Gemini (with graceful degradation fallback)
        llm_output: Optional[LLMComplianceOutput] = None
        error_msg: Optional[str] = None
        analysis_status = AnalysisStatus.COMPLETED
        retryable = False

        try:
            llm_output = self.llm.analyze_compliance(
                masked_text=masked_text,
                retrieved_rules=retrieved_rules,
                document_id=document_id,
            )
        except AIConfigurationError as e:
            logger.warning(f"AI Service configuration issue for doc {document_id}: {e.message}")
            analysis_status = AnalysisStatus.UNAVAILABLE
            error_msg = f"AI assist unavailable: {e.message} The document is still accessible for manual review."
            retryable = True
        except AIRateLimited as e:
            logger.warning(f"AI Service rate limited for doc {document_id}: {e.message}")
            analysis_status = AnalysisStatus.RATE_LIMITED
            error_msg = "AI assist is temporarily rate-limited. Please retry shortly."
            retryable = True
        except (AITimeout, AIUnavailable, AIInvalidResponse) as e:
            logger.warning(f"AI Service degraded for doc {document_id}: {e.message}")
            analysis_status = AnalysisStatus.UNAVAILABLE
            error_msg = f"AI assist encountered a temporary issue ({e.message}). Review page remains functional."
            retryable = True
        except Exception as e:
            logger.error(f"Unexpected AI error for doc {document_id}: {e}")
            analysis_status = AnalysisStatus.FAILED
            error_msg = "An unexpected error occurred during AI analysis."
            retryable = True

        # 5. Compile Traceable Flags
        all_flags: List[Flag] = []
        analysis_id = str(uuid.uuid4())

        if llm_output and llm_output.flags:
            for f in llm_output.flags:
                # Unmask passages for authorized UI display only
                display_passage = pii_masker_unmask(f.passage, pii_mapping)
                all_flags.append(
                    Flag(
                        analysis_id=analysis_id,
                        passage_excerpt=display_passage,
                        matched_rule_id=f.matched_rule_id,
                        matched_rule=f.matched_rule,
                        explanation=f.explanation,
                        severity=f.severity,
                    )
                )

        # Merge in missing disclosure flags (from vector retrieval by absence)
        for disc_flag in missing_disclosure_flags:
            disc_flag.analysis_id = analysis_id
            all_flags.append(disc_flag)

        summary_text = (
            llm_output.summary
            if (llm_output and llm_output.summary)
            else ("AI summary unavailable." if analysis_status != AnalysisStatus.COMPLETED else "Document under compliance review.")
        )

        # 6. Build AIAnalysis Entity
        analysis_entity = AIAnalysis(
            id=analysis_id,
            document_id=document_id,
            summary=summary_text,
            status=analysis_status,
            generated_at=datetime.now(timezone.utc),
            flags=all_flags,
            precedents=precedents,
            disclosures_checked=disclosure_results,
            masked_text=masked_text,
            error_message=error_msg,
            retryable=retryable,
        )

        # 7. Store in Cache (even degraded states are recorded so we know document state, but retryable)
        self.cache.set(analysis_entity)

        duration_ms = (time.time() - start_time) * 1000
        SafeAuditLogger.log_event(
            "DOCUMENT_ANALYSIS_COMPLETED",
            document_id,
            duration_ms=duration_ms,
            extra={
                "status": analysis_status.value,
                "flags_count": len(all_flags),
                "cached": False,
            },
        )

        return self._build_response_from_entity(analysis_entity, cached=False)

    def get_analysis(self, document_id: str) -> Optional[AnalysisResponse]:
        """Fetch cached analysis for a document."""
        entity = self.cache.get(document_id)
        if not entity:
            return None
        return self._build_response_from_entity(entity, cached=True)

    def retry_analysis(
        self, document_id: str, extracted_text: str
    ) -> AnalysisResponse:
        """Explicit retry: bypasses cache and forces re-evaluation."""
        SafeAuditLogger.log_event("ANALYSIS_RETRY_REQUESTED", document_id)
        return self.process_document(
            document_id=document_id,
            extracted_text=extracted_text,
            force_refresh=True,
        )

    def _build_response_from_entity(
        self, entity: AIAnalysis, cached: bool = False
    ) -> AnalysisResponse:
        """Convert internal domain entity to clean API response model."""
        flags_resp = [
            FlagResponse(
                id=f.id,
                passage_excerpt=f.passage_excerpt,
                matched_rule_id=f.matched_rule_id,
                matched_rule=f.matched_rule,
                explanation=f.explanation,
                severity=f.severity,
            )
            for f in entity.flags
        ]

        precedents_resp = [
            PrecedentMatchResponse(
                document_id=p.document_id,
                similarity=p.similarity,
                previous_decision=p.previous_decision,
                officer_comment=p.officer_comment,
                excerpt=p.excerpt,
            )
            for p in entity.precedents
        ]

        disclosures_resp = [
            DisclosureCheck(
                disclosure_id=d.disclosure_id,
                disclosure_type=d.disclosure_type,
                disclosure_text=d.disclosure_text,
                is_present=d.is_present,
                similarity_score=d.similarity_score,
                best_matching_passage=d.best_matching_passage,
            )
            for d in entity.disclosures_checked
        ]

        return AnalysisResponse(
            document_id=entity.document_id,
            status=entity.status,
            summary=entity.summary,
            flags=flags_resp,
            precedents=precedents_resp,
            disclosures_checked=disclosures_resp,
            generated_at=entity.generated_at,
            cached=cached,
            message=entity.error_message,
            retryable=entity.retryable,
        )


def pii_masker_unmask(text: str, mapping: dict) -> str:
    """Helper to unmask text."""
    from ai.app.services.pii_masker import pii_masker
    return pii_masker.unmask_text(text, mapping)


analysis_service = AnalysisService()
