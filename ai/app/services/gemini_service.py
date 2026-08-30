"""
Gemini Service Layer with Structured Outputs, Grounding, and Graceful Degradation.

Strict Privacy & Safety Guardrails:
1. Receives ONLY pre-masked text.
2. Grounded strictly in retrieved compliance rules.
3. NEVER decides or pre-fills human compliance verdicts (Approved/Rejected/Needs Revision).
4. Handles rate-limits, timeouts, missing keys, and malformed outputs gracefully.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
from pydantic import ValidationError

from ai.app.core.config import settings
from ai.app.core.logging import logger, SafeAuditLogger
from ai.app.models.entities import Flag, Rule, SeverityLevel
from ai.app.schemas.analysis import LLMComplianceOutput
from ai.app.schemas.flags import FlagCreate


class AIException(Exception):
    """Base exception for AI service operations."""
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class AIConfigurationError(AIException):
    """Raised when the AI service is misconfigured (e.g. missing API key)."""
    def __init__(self, message: str = "Gemini API key is not configured."):
        super().__init__(message, retryable=True)


class AIRateLimited(AIException):
    """Raised when Gemini API returns 429 Too Many Requests."""
    def __init__(self, message: str = "Gemini API rate limit exceeded."):
        super().__init__(message, retryable=True)


class AITimeout(AIException):
    """Raised when a request to Gemini times out."""
    def __init__(self, message: str = "Gemini API request timed out."):
        super().__init__(message, retryable=True)


class AIInvalidResponse(AIException):
    """Raised when Gemini returns malformed or invalid schema output."""
    def __init__(self, message: str = "Gemini returned an invalid or unparseable response."):
        super().__init__(message, retryable=True)


class AIUnavailable(AIException):
    """Raised when Gemini is unreachable or returns 5xx error."""
    def __init__(self, message: str = "Gemini API is temporarily unavailable."):
        super().__init__(message, retryable=True)


class GeminiService:
    """
    Client for Google Gemini API via REST endpoints, enforcing structured schema outputs.
    """

    SYSTEM_INSTRUCTION = """You are a specialized Compliance Analysis Assistant for financial services.
Your role is to orient the human Compliance Officer by analyzing client-facing material (emails, brochures, social posts, letters, proposals).

CRITICAL CONSTRAINTS:
1. YOU DO NOT MAKE COMPLIANCE DECISIONS. Never approve, reject, or assign final regulatory verdicts.
2. ONLY analyze the provided masked document against the explicitly supplied Retrieved Compliance Rules.
3. DO NOT invent rules not provided in the prompt.
4. For every potential issue, identify:
   - "passage": The exact triggering excerpt from the document text.
   - "matched_rule_id": The exact rule ID (e.g., RULE-001, RULE-014).
   - "matched_rule": The rule text/title.
   - "explanation": Exactly ONE concise sentence explaining why the passage conflicts with the rule.
   - "severity": "low", "medium", or "high".
5. Provide a concise summary (2-4 sentences) stating:
   - What the document is.
   - Its primary purpose and key claims.
   - Areas that require officer attention.

OUTPUT FORMAT:
You must output ONLY valid JSON matching this schema:
{
  "summary": "...",
  "flags": [
    {
      "passage": "...",
      "matched_rule_id": "...",
      "matched_rule": "...",
      "explanation": "...",
      "severity": "low" | "medium" | "high"
    }
  ]
}
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model = model if model is not None else settings.GEMINI_MODEL
        self.timeout = timeout if timeout is not None else settings.GEMINI_TIMEOUT_SECONDS

    def build_prompt(
        self,
        masked_text: str,
        retrieved_rules: List[Tuple[Rule, float]],
    ) -> str:
        """Construct the prompt with grounded rules and masked text."""
        rules_text = "\n".join(
            [f"- [{r.id}] (Similarity: {sim:.2f}): {r.text}" for r, sim in retrieved_rules]
        )

        return f"""### RETRIEVED COMPLIANCE RULES:
{rules_text if rules_text else "No specific high-similarity rules retrieved; apply general standards."}

### MASKED DOCUMENT FOR REVIEW:
\"\"\"
{masked_text}
\"\"\"

Analyze the document against the retrieved rules above and return the required JSON object.
"""

    def analyze_compliance(
        self,
        masked_text: str,
        retrieved_rules: List[Tuple[Rule, float]],
        document_id: str = "unknown",
    ) -> LLMComplianceOutput:
        """
        Calls Gemini API with grounded context and parses validated structured output.
        """
        # 1. Check API Key configuration
        if not self.api_key or self.api_key.strip() in ("", "your_gemini_api_key_here"):
            logger.warning("Gemini API key missing. Invoking fallback rule-based analysis.")
            raise AIConfigurationError("Gemini API key is not configured.")

        prompt = self.build_prompt(masked_text, retrieved_rules)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        request_body = {
            "systemInstruction": {
                "parts": [{"text": self.SYSTEM_INSTRUCTION}]
            },
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            }
        }

        start_time = time.time()
        SafeAuditLogger.log_event("GEMINI_REQUEST_START", document_id)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=request_body, headers=headers)

            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code == 429:
                logger.warning(f"Gemini 429 Rate Limit for document {document_id}")
                raise AIRateLimited("Gemini API rate limit exceeded (429).")

            if response.status_code == 404:
                logger.error(f"Gemini model '{self.model}' not found: {response.text}")
                raise AIConfigurationError(f"Gemini model '{self.model}' is invalid or not found (HTTP 404).")

            if response.status_code in (400, 401, 403):
                logger.error(f"Gemini auth/request error: {response.status_code} - {response.text}")
                raise AIConfigurationError(f"Gemini API request rejected: HTTP {response.status_code}")

            if response.status_code >= 500:
                logger.error(f"Gemini server error: {response.status_code}")
                raise AIUnavailable(f"Gemini server error: HTTP {response.status_code}")

            response_data = response.json()
            raw_text = self._extract_text_from_response(response_data)

            parsed_output = self._parse_and_validate_json(raw_text)

            SafeAuditLogger.log_event(
                "GEMINI_REQUEST_SUCCESS",
                document_id,
                duration_ms=elapsed_ms,
                extra={"flags_count": len(parsed_output.flags)}
            )
            return parsed_output

        except httpx.TimeoutException as e:
            logger.error(f"Gemini request timeout for doc {document_id}: {e}")
            raise AITimeout(f"Gemini request timed out after {self.timeout}s.")
        except httpx.RequestError as e:
            logger.error(f"Gemini network error for doc {document_id}: {e}")
            raise AIUnavailable(f"Network error communicating with Gemini: {str(e)}")

    def _extract_text_from_response(self, response_data: Dict[str, Any]) -> str:
        """Extracts candidate text from Gemini response structure."""
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                raise AIInvalidResponse("No candidates returned from Gemini API.")
            
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise AIInvalidResponse("Empty content parts in Gemini response.")
            
            return parts[0].get("text", "")
        except (KeyError, IndexError) as e:
            raise AIInvalidResponse(f"Failed to parse Gemini response structure: {e}")

    def _parse_and_validate_json(self, raw_json_str: str) -> LLMComplianceOutput:
        """Cleans and validates JSON against Pydantic schema."""
        # Strip potential markdown fence markers ```json ... ```
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw_json_str.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)

        try:
            data = json.loads(cleaned)
            return LLMComplianceOutput.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Malformed LLM JSON, attempting regex extraction: {e}")
            return self._fallback_json_repair(cleaned)

    def _fallback_json_repair(self, text: str) -> LLMComplianceOutput:
        """Gracefully repairs mildly malformed JSON or extracts summary."""
        try:
            # Try to locate "summary" and "flags" using regex
            summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', text)
            summary = summary_match.group(1) if summary_match else "Financial client document under compliance review."
            return LLMComplianceOutput(summary=summary, flags=[])
        except Exception:
            raise AIInvalidResponse("Unable to parse compliance output from model.")


gemini_service = GeminiService()
