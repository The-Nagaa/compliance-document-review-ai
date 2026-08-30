from typing import Optional
from pydantic import BaseModel, Field


class RuleResult(BaseModel):
    """Schema for retrieved compliance rule."""
    id: str
    text: str
    category: str
    similarity: Optional[float] = None


class DisclosureCheck(BaseModel):
    """Schema for disclosure presence/absence check."""
    disclosure_id: str
    disclosure_type: str
    disclosure_text: str
    is_present: bool
    similarity_score: float
    best_matching_passage: Optional[str] = None


class PrecedentMatchResponse(BaseModel):
    """Schema for a matched historical precedent document."""
    document_id: str
    similarity: float
    previous_decision: str = Field(..., description="Decision made on precedent: approved, rejected, or needs_revision")
    officer_comment: str = Field(..., description="Historical compliance officer comment")
    excerpt: Optional[str] = None
