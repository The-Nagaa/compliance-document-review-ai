from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ai.app.models.entities import AnalysisStatus
from ai.app.schemas.flags import FlagCreate, FlagResponse
from ai.app.schemas.retrieval import DisclosureCheck, PrecedentMatchResponse, RuleResult


class ProcessDocumentRequest(BaseModel):
    """Clean interface for document processing input."""
    document_id: str = Field(..., description="Unique document ID (from Backend/Data Eng).", min_length=1)
    extracted_text: str = Field(..., description="Clean extracted text from PDF/DOCX/XLSX.", min_length=1)


class LLMComplianceOutput(BaseModel):
    """Structured output expected from the LLM."""
    summary: str = Field(
        ...,
        description="A concise orienting summary of what the document is, its purpose, and claims.",
    )
    flags: List[FlagCreate] = Field(
        default_factory=list,
        description="List of identified compliance flags with passage, matched rule ID, matched rule, explanation, severity.",
    )


class AnalysisResponse(BaseModel):
    """Main API response for compliance AI analysis."""
    document_id: str
    status: AnalysisStatus
    summary: Optional[str] = None
    flags: List[FlagResponse] = Field(default_factory=list)
    precedents: List[PrecedentMatchResponse] = Field(default_factory=list)
    disclosures_checked: List[DisclosureCheck] = Field(default_factory=list)
    generated_at: Optional[datetime] = None
    cached: bool = False
    message: Optional[str] = None
    retryable: bool = False


class StatusResponse(BaseModel):
    """Status endpoint response."""
    document_id: str
    status: AnalysisStatus
    has_analysis: bool
    generated_at: Optional[datetime] = None
    flags_count: int = 0
    message: Optional[str] = None


class PrivacyInspectionResponse(BaseModel):
    """Inspection response demonstrating sanitized outbound payload vs raw text."""
    document_id: str
    raw_text_length: int
    masked_text_length: int
    masked_text_preview: str
    detected_entities_count: int
    entity_types_found: List[str]
    has_raw_pii_in_outbound_payload: bool
    outbound_payload_preview: Dict[str, Any]
