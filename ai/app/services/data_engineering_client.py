"""
Data Engineering HTTP Client.

Integrates with Data Engineering's retrieval service for:
1. Rule Lookup (/rule-lookup)
2. Disclosure-by-Absence Check (/disclosure-check)
3. Precedent Search (/precedent-search)

Guarantees:
- Only pre-masked text and chunks are transmitted across the boundary.
- Non-blocking error handling with graceful fallback.
"""

from typing import Any, Dict, List, Optional, Tuple
import httpx
from ai.app.core.config import settings
from ai.app.core.logging import logger
from ai.app.models.entities import DisclosureCheckResult, PrecedentMatch, Rule


class DataEngineeringError(Exception):
    """Base exception for Data Engineering API errors."""
    pass


class DataEngineeringConnectionError(DataEngineeringError):
    """Raised when Data Engineering service cannot be reached."""
    pass


class DataEngineeringTimeout(DataEngineeringError):
    """Raised when Data Engineering service times out."""
    pass


class DataEngineeringClient:
    """
    HTTP client for the Data Engineering retrieval service.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.DATA_ENGINEERING_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.DATA_ENGINEERING_TIMEOUT_SECONDS

    def lookup_rules(self, masked_text: str) -> List[Tuple[Rule, float]]:
        """
        Calls POST /rule-lookup on Data Engineering service.
        Request: {"text": "..."}
        Response: [{"rule_id": "string", "rule_text": "string", "similarity_score": float}]
        """
        url = f"{self.base_url}/rule-lookup"
        payload = {"text": masked_text}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as e:
            logger.warning(f"Data Engineering /rule-lookup timed out: {e}")
            raise DataEngineeringTimeout(f"Data Engineering /rule-lookup timed out: {e}") from e
        except httpx.RequestError as e:
            logger.warning(f"Data Engineering /rule-lookup connection failed: {e}")
            raise DataEngineeringConnectionError(f"Data Engineering /rule-lookup connection failed: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.warning(f"Data Engineering /rule-lookup returned HTTP {e.response.status_code}: {e}")
            raise DataEngineeringError(f"Data Engineering /rule-lookup HTTP {e.response.status_code}") from e
        except Exception as e:
            logger.warning(f"Data Engineering /rule-lookup unexpected error: {e}")
            raise DataEngineeringError(f"Data Engineering /rule-lookup error: {e}") from e

        results: List[Tuple[Rule, float]] = []
        if isinstance(data, list):
            for item in data:
                rule_id = item.get("rule_id", "")
                rule_text = item.get("rule_text", "")
                score = float(item.get("similarity_score", 0.0))
                category = item.get("category", "COMPLIANCE_RULE")
                rule = Rule(id=rule_id, text=rule_text, category=category)
                results.append((rule, score))

        # Sort descending by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def check_disclosure(
        self,
        document_chunks: List[str],
        disclosure_id: str,
        disclosure_text: str,
        disclosure_type: Optional[str] = None,
    ) -> DisclosureCheckResult:
        """
        Calls POST /disclosure-check on Data Engineering service.
        Request:
        {
            "document_chunks": ["..."],
            "disclosure_id": "...",
            "disclosure_text": "..."
        }
        Response:
        {
            "disclosure_id": "string",
            "disclosure_type": "string",
            "present": boolean,
            "similarity_score": float,
            "matched_chunk_id": "string | null"
        }
        """
        url = f"{self.base_url}/disclosure-check"
        payload = {
            "document_chunks": document_chunks,
            "disclosure_id": disclosure_id,
            "disclosure_text": disclosure_text,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as e:
            logger.warning(f"Data Engineering /disclosure-check timed out for {disclosure_id}: {e}")
            raise DataEngineeringTimeout(f"Data Engineering /disclosure-check timed out: {e}") from e
        except httpx.RequestError as e:
            logger.warning(f"Data Engineering /disclosure-check connection failed for {disclosure_id}: {e}")
            raise DataEngineeringConnectionError(f"Data Engineering /disclosure-check connection failed: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.warning(f"Data Engineering /disclosure-check HTTP {e.response.status_code} for {disclosure_id}: {e}")
            raise DataEngineeringError(f"Data Engineering /disclosure-check HTTP {e.response.status_code}") from e
        except Exception as e:
            logger.warning(f"Data Engineering /disclosure-check unexpected error for {disclosure_id}: {e}")
            raise DataEngineeringError(f"Data Engineering /disclosure-check error: {e}") from e

        return DisclosureCheckResult(
            disclosure_id=data.get("disclosure_id", disclosure_id),
            disclosure_type=data.get("disclosure_type", disclosure_type or disclosure_id),
            disclosure_text=disclosure_text,
            is_present=bool(data.get("present", False)),
            similarity_score=round(float(data.get("similarity_score", 0.0)), 4),
            best_matching_passage=data.get("matched_chunk_id"),
        )

    def search_precedents(
        self,
        masked_text: str,
        top_k: int = 3,
    ) -> List[PrecedentMatch]:
        """
        Calls POST /precedent-search on Data Engineering service.
        Request: {"text": "..."}
        Response:
        [
            {
                "document_id": "string",
                "chunk_id": "string",
                "similarity_score": float,
                "chunk_text": "string"
            }
        ]
        """
        url = f"{self.base_url}/precedent-search"
        payload = {"text": masked_text}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as e:
            logger.warning(f"Data Engineering /precedent-search timed out: {e}")
            raise DataEngineeringTimeout(f"Data Engineering /precedent-search timed out: {e}") from e
        except httpx.RequestError as e:
            logger.warning(f"Data Engineering /precedent-search connection failed: {e}")
            raise DataEngineeringConnectionError(f"Data Engineering /precedent-search connection failed: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.warning(f"Data Engineering /precedent-search HTTP {e.response.status_code}: {e}")
            raise DataEngineeringError(f"Data Engineering /precedent-search HTTP {e.response.status_code}") from e
        except Exception as e:
            logger.warning(f"Data Engineering /precedent-search unexpected error: {e}")
            raise DataEngineeringError(f"Data Engineering /precedent-search error: {e}") from e

        results: List[PrecedentMatch] = []
        if isinstance(data, list):
            for item in data:
                doc_id = item.get("document_id", "")
                chunk_id = item.get("chunk_id", "")
                sim = round(float(item.get("similarity_score", 0.0)), 4)
                chunk_text = item.get("chunk_text", "")
                decision = item.get("previous_decision") or item.get("decision", "needs_revision")
                comment = item.get("officer_comment") or item.get("comment", f"Historical reviewed match ({doc_id})")

                excerpt = chunk_text[:200] + ("..." if len(chunk_text) > 200 else "")
                results.append(
                    PrecedentMatch(
                        document_id=doc_id,
                        similarity=sim,
                        previous_decision=decision,
                        officer_comment=comment,
                        excerpt=excerpt,
                    )
                )

        # Sort descending by similarity and clamp to exactly top_k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]


data_engineering_client = DataEngineeringClient()
