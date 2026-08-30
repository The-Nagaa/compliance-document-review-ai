from typing import Optional
from pydantic import BaseModel, Field, field_validator
from ai.app.models.entities import SeverityLevel


class FlagCreate(BaseModel):
    """Schema for a compliance flag generated during analysis."""
    passage: str = Field(
        ...,
        description="The exact triggering passage excerpt from the document.",
        min_length=3,
    )
    matched_rule_id: str = Field(
        ...,
        description="The identifier of the matched compliance rule (e.g., RULE-014).",
        min_length=2,
    )
    matched_rule: str = Field(
        ...,
        description="The title or text summary of the matched compliance rule.",
        min_length=3,
    )
    explanation: str = Field(
        ...,
        description="A concise one-line explanation of why the passage conflicts with the rule.",
        min_length=5,
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM,
        description="Severity level of the potential issue: low, medium, high.",
    )

    @field_validator("severity", mode="before")
    @classmethod
    def validate_severity(cls, v: str) -> SeverityLevel:
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower in ("low", "medium", "high"):
                return SeverityLevel(v_lower)
        return SeverityLevel.MEDIUM


class FlagResponse(BaseModel):
    """API response schema for a compliance flag."""
    id: str
    passage_excerpt: str
    matched_rule_id: str
    matched_rule: str
    explanation: str
    severity: SeverityLevel
