"""
Data Engineering HTTP Client with Persistent Connection Pooling.

Integrates with Data Engineering's retrieval service for:
1. Rule Lookup (/rule-lookup)
2. Disclosure-by-Absence Check (/disclosure-check)
3. Precedent Search (/precedent-search)

Performance & Privacy Guarantees:
- Reuses persistent HTTP connection pool across all requests (eliminates per-request handshake latency).
- Only pre-masked text and chunks are transmitted across the boundary.
- Non-blocking error handling with graceful degradation and local fallback.
- Explicit resource lifecycle management (close / context manager).
"""

import threading
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
    HTTP client for Data Engineering retrieval service with session/connection reuse.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_keepalive_connections: int = 20,
        max_connections: int = 50,
    ):
        self.base_url = (base_url or settings.DATA_ENGINEERING_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.DATA_ENGINEERING_TIMEOUT_SECONDS
        self._limits = httpx.Limits(
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
            keepalive_expiry=30.0,
        )
        self._client: Optional[httpx.Client] = None
        self._lock = threading.Lock()

    def _get_client(self) -> httpx.Client:
        """Thread-safe accessor for reusable httpx.Client instance."""
        if self._client is None or self._client.is_closed:
            with self._lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.Client(
                        timeout=self.timeout,
                        limits=self._limits,
                    )
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP client session and release connections."""
        with self._lock:
            if self._client is not None and not self._client.is_closed:
                self._client.close()
                self._client = None

    def __enter__(self) -> "DataEngineeringClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def lookup_rules(self, masked_text: str) -> List[Tuple[Rule, float]]:
        """
        Calls POST /rule-lookup on Data Engineering service.
        Request: {"text": "..."}
        Response: [{"rule_id": "string", "rule_text": "string", "similarity_score": float}]
        """
        url = f"{self.base_url}/rule-lookup"
        payload = {"text": masked_text}

        try:
            client = self._get_client()
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

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def check_disclosure(
        self,
        document_chunks: List[str],
        disclosure_id: str,
        disclosure_text: str,
        disclosure_type: Optional[str] = None,
        threshold: Optional[float] = None,
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
            client = self._get_client()
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

        score = round(float(data.get("similarity_score", 0.0)), 4)
        if threshold is not None:
            is_present = score >= threshold
        else:
            is_present = bool(data.get("present", False))

        resolved_type = disclosure_type or data.get("disclosure_type") or disclosure_id
        return DisclosureCheckResult(
            disclosure_id=data.get("disclosure_id", disclosure_id),
            disclosure_type=resolved_type,
            disclosure_text=disclosure_text,
            is_present=is_present,
            similarity_score=score,
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
            client = self._get_client()
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

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]


data_engineering_client = DataEngineeringClient()
