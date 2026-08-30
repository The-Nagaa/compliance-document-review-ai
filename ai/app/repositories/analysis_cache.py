"""
Analysis Cache Repository.

Caches AI analysis per document to avoid duplicate API calls on page reloads.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from ai.app.core.config import settings
from ai.app.core.logging import logger
from ai.app.models.entities import (
    AIAnalysis,
    AnalysisStatus,
    DisclosureCheckResult,
    Flag,
    PrecedentMatch,
    SeverityLevel,
)


class AnalysisCache:
    """In-memory and JSON-persisted analysis cache."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or settings.ANALYSIS_CACHE_PATH
        self._cache: Dict[str, AIAnalysis] = {}

    def get(self, document_id: str) -> Optional[AIAnalysis]:
        """Retrieve cached analysis for a document."""
        return self._cache.get(document_id)

    def set(self, analysis: AIAnalysis) -> None:
        """Store or update analysis in cache."""
        self._cache[analysis.document_id] = analysis

    def delete(self, document_id: str) -> bool:
        """Remove a cached analysis (for explicit retry invalidation)."""
        if document_id in self._cache:
            del self._cache[document_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def count(self) -> int:
        return len(self._cache)

    def save_to_disk(self, filepath: Optional[Path] = None) -> None:
        """Persist cache to disk."""
        target = filepath or self.storage_path
        target.parent.mkdir(parents=True, exist_ok=True)

        serialized: Dict[str, dict] = {}
        for doc_id, a in self._cache.items():
            serialized[doc_id] = {
                "id": a.id,
                "document_id": a.document_id,
                "summary": a.summary,
                "status": a.status.value,
                "generated_at": a.generated_at.isoformat() if a.generated_at else None,
                "masked_text": a.masked_text,
                "error_message": a.error_message,
                "retryable": a.retryable,
                "flags": [
                    {
                        "id": f.id,
                        "analysis_id": f.analysis_id,
                        "passage_excerpt": f.passage_excerpt,
                        "matched_rule_id": f.matched_rule_id,
                        "matched_rule": f.matched_rule,
                        "explanation": f.explanation,
                        "severity": f.severity.value,
                    }
                    for f in a.flags
                ],
                "precedents": [
                    {
                        "document_id": p.document_id,
                        "similarity": p.similarity,
                        "previous_decision": p.previous_decision,
                        "officer_comment": p.officer_comment,
                        "excerpt": p.excerpt,
                    }
                    for p in a.precedents
                ],
                "disclosures_checked": [
                    {
                        "disclosure_id": d.disclosure_id,
                        "disclosure_type": d.disclosure_type,
                        "disclosure_text": d.disclosure_text,
                        "is_present": d.is_present,
                        "similarity_score": d.similarity_score,
                        "best_matching_passage": d.best_matching_passage,
                    }
                    for d in a.disclosures_checked
                ],
            }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    def load_from_disk(self, filepath: Optional[Path] = None) -> bool:
        """Load cache from disk."""
        target = filepath or self.storage_path
        if not target.exists():
            return False

        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.clear()
            for doc_id, a_data in data.items():
                gen_at = (
                    datetime.fromisoformat(a_data["generated_at"])
                    if a_data.get("generated_at")
                    else datetime.now(timezone.utc)
                )

                flags = [
                    Flag(
                        id=f["id"],
                        analysis_id=f.get("analysis_id"),
                        passage_excerpt=f["passage_excerpt"],
                        matched_rule_id=f["matched_rule_id"],
                        matched_rule=f["matched_rule"],
                        explanation=f["explanation"],
                        severity=SeverityLevel(f["severity"]),
                    )
                    for f in a_data.get("flags", [])
                ]

                precedents = [
                    PrecedentMatch(
                        document_id=p["document_id"],
                        similarity=p["similarity"],
                        previous_decision=p["previous_decision"],
                        officer_comment=p["officer_comment"],
                        excerpt=p.get("excerpt", ""),
                    )
                    for p in a_data.get("precedents", [])
                ]

                disclosures = [
                    DisclosureCheckResult(
                        disclosure_id=d["disclosure_id"],
                        disclosure_type=d["disclosure_type"],
                        disclosure_text=d["disclosure_text"],
                        is_present=d["is_present"],
                        similarity_score=d["similarity_score"],
                        best_matching_passage=d.get("best_matching_passage"),
                    )
                    for d in a_data.get("disclosures_checked", [])
                ]

                self._cache[doc_id] = AIAnalysis(
                    id=a_data["id"],
                    document_id=a_data["document_id"],
                    summary=a_data.get("summary", ""),
                    status=AnalysisStatus(a_data.get("status", "completed")),
                    generated_at=gen_at,
                    flags=flags,
                    precedents=precedents,
                    disclosures_checked=disclosures,
                    masked_text=a_data.get("masked_text", ""),
                    error_message=a_data.get("error_message"),
                    retryable=a_data.get("retryable", False),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to load analysis cache from {target}: {e}")
            return False


analysis_cache = AnalysisCache()
