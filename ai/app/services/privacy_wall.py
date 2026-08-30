"""
Privacy Wall Boundary & Outbound Sanitization Service.

Guarantees that raw, unmasked document text never escapes the application perimeter
to Gemini, embedding APIs, or third-party hosted models.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from ai.app.core.logging import logger, SafeAuditLogger
from ai.app.services.pii_masker import pii_masker


class PrivacyViolationError(Exception):
    """Raised when an outbound payload is detected to contain raw, unmasked PII."""
    pass


class PrivacyWall:
    """
    Enforces privacy boundaries prior to any external vendor call.
    """

    @staticmethod
    def verify_and_sanitize_outbound(
        raw_text: str,
        document_id: str,
        known_sensitive_values: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Runs the PII Masker, verifies that no raw sensitive patterns remain in the payload,
        and produces the sanitized masked text and server-side mapping.

        Raises:
            PrivacyViolationError: If unmasked sensitive information is detected in outbound payload.
        """
        SafeAuditLogger.log_event("PRIVACY_WALL_INGRESS", document_id)

        masked_text, pii_mapping, _ = pii_masker.mask_text(raw_text, document_id=document_id)

        # 1. Verification: Ensure all known original values are NOT present in masked text
        if known_sensitive_values:
            for val in known_sensitive_values:
                if val.strip() and val.strip() in masked_text:
                    logger.error(
                        f"PRIVACY VIOLATION: Found known sensitive token in outbound text for doc {document_id}"
                    )
                    raise PrivacyViolationError(
                        f"Outbound payload for document {document_id} failed privacy audit."
                    )

        # 2. Verification: Verify no raw unmasked emails or SSNs exist in masked_text
        email_matches = pii_masker.EMAIL_PATTERN.findall(masked_text)
        if email_matches:
            logger.error(f"PRIVACY VIOLATION: Unmasked email detected in outbound text: {len(email_matches)}")
            raise PrivacyViolationError("Unmasked email detected in outbound payload.")

        ssn_matches = pii_masker.SSN_PATTERN.findall(masked_text)
        if ssn_matches:
            logger.error(f"PRIVACY VIOLATION: Unmasked SSN detected in outbound text: {len(ssn_matches)}")
            raise PrivacyViolationError("Unmasked SSN detected in outbound payload.")

        SafeAuditLogger.log_event(
            "PRIVACY_WALL_EGRESS_VERIFIED",
            document_id,
            extra={"placeholders_count": len(pii_mapping)},
        )

        return masked_text, pii_mapping

    @staticmethod
    def inspect_sanitization(
        raw_text: str, document_id: str = "demo_inspect"
    ) -> Dict[str, Any]:
        """
        Safe debugging/demonstration mechanism to show outbound payload vs raw text metrics
        without logging sensitive values.
        """
        masked_text, mapping, records = pii_masker.mask_text(raw_text, document_id=document_id)

        # Check if raw values leaked into masked text
        has_leak = any(
            orig in masked_text
            for orig in mapping.values()
            if orig != "" and len(orig) > 3
        )

        entity_types = list({r.entity_type for r in records})

        return {
            "document_id": document_id,
            "raw_text_length": len(raw_text),
            "masked_text_length": len(masked_text),
            "masked_text_preview": masked_text[:400] + ("..." if len(masked_text) > 400 else ""),
            "detected_entities_count": len(mapping),
            "entity_types_found": entity_types,
            "has_raw_pii_in_outbound_payload": has_leak,
            "outbound_payload_preview": {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": masked_text[:250] + ("..." if len(masked_text) > 250 else "")
                            }
                        ]
                    }
                ],
                "privacy_guarantee": "Placeholders only, server-side mapping strictly detached."
            }
        }


privacy_wall = PrivacyWall()
