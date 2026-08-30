from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"


class DecisionStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass
class PIIMapping:
    """Server-side mapping of placeholders to original values."""
    document_id: str
    placeholder: str
    original_value: str
    entity_type: str  # e.g., 'CLIENT', 'EMAIL', 'PHONE', 'ADDRESS', 'ACCOUNT', 'SSN', 'AMOUNT'
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Rule:
    """Compliance rule entity stored in the vector corpus."""
    id: str
    text: str
    category: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Disclosure:
    """Standard required disclosure entity stored in the vector corpus."""
    id: str
    text: str
    type: str  # e.g., 'SEC_RIA', 'RISK_OF_LOSS', 'PERFORMANCE_DISCLAIMER', 'CONFLICT_OF_INTEREST'
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Precedent:
    """Historically reviewed synthetic document for precedent search."""
    document_id: str
    masked_text: str
    decision: str  # 'approved' | 'rejected' | 'needs_revision'
    comment: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Flag:
    """Traceable compliance flag identified in the document."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    analysis_id: Optional[str] = None
    passage_excerpt: str = ""
    matched_rule_id: str = ""
    matched_rule: str = ""
    explanation: str = ""
    severity: SeverityLevel = SeverityLevel.MEDIUM


@dataclass
class DisclosureCheckResult:
    """Result of checking whether a required disclosure is present or missing."""
    disclosure_id: str
    disclosure_type: str
    disclosure_text: str
    is_present: bool
    similarity_score: float
    best_matching_passage: Optional[str] = None


@dataclass
class PrecedentMatch:
    """Precedent match returned from vector search."""
    document_id: str
    similarity: float
    previous_decision: str
    officer_comment: str
    excerpt: str = ""


@dataclass
class AIAnalysis:
    """AI analysis result entity (cached per document)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    summary: str = ""
    status: AnalysisStatus = AnalysisStatus.COMPLETED
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    flags: List[Flag] = field(default_factory=list)
    precedents: List[PrecedentMatch] = field(default_factory=list)
    disclosures_checked: List[DisclosureCheckResult] = field(default_factory=list)
    masked_text: str = ""
    error_message: Optional[str] = None
    retryable: bool = False
