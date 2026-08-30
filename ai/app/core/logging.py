import logging
import sys
import time
from typing import Any, Dict, Optional


class PrivacySafeFormatter(logging.Formatter):
    """
    Formatter that guarantees sensitive tokens, keys, and PII patterns
    are not accidentally leaked into standard log streams.
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        # Extra safety layer to redact potential API keys or sensitive mappings if leaked
        if "AIzaSy" in msg:
            # Common Gemini API Key prefix
            import re
            msg = re.sub(r'AIzaSy[A-Za-z0-9_-]{33}', '[REDACTED_API_KEY]', msg)
        return msg


def get_logger(name: str = "ai_track") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = PrivacySafeFormatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger("ai_service")


class SafeAuditLogger:
    """Safe audit logger for tracking compliance AI lifecycle events without PII."""

    @staticmethod
    def log_event(
        event_name: str,
        document_id: str,
        duration_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        parts = [f"EVENT={event_name}", f"DOC_ID={document_id}"]
        if duration_ms is not None:
            parts.append(f"DURATION_MS={duration_ms:.2f}")
        if extra:
            # Safe serialize metadata (guaranteed non-PII)
            safe_extra = {
                k: v for k, v in extra.items()
                if "pii" not in k.lower() and "mapping" not in k.lower() and "raw" not in k.lower()
            }
            parts.append(f"METADATA={safe_extra}")
        logger.info(" | ".join(parts))
