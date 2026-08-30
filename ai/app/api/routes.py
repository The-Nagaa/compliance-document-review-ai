"""
FastAPI Routes for AI Compliance Analysis Service.

Provides clean, standardized contracts for Backend and Frontend integration:
- POST /ai/analyze/{document_id}
- GET  /ai/analysis/{document_id}
- POST /ai/analyze/{document_id}/retry
- GET  /ai/status/{document_id}
- POST /ai/privacy-inspect
- GET  /ai/health
"""

from typing import Any, Dict
from fastapi import APIRouter, Body, HTTPException, Query, status
from ai.app.models.entities import AnalysisStatus
from ai.app.schemas.analysis import (
    AnalysisResponse,
    PrivacyInspectionResponse,
    ProcessDocumentRequest,
    StatusResponse,
)
from ai.app.services.analysis_service import analysis_service
from ai.app.services.privacy_wall import privacy_wall

router = APIRouter(prefix="/ai", tags=["AI Compliance Assist"])


@router.post(
    "/analyze/{document_id}",
    response_model=AnalysisResponse,
    summary="Process Document and Generate Compliance Analysis",
    description="Accepts extracted clean text, runs PII masking, vector retrieval, and grounded AI analysis.",
)
async def analyze_document(
    document_id: str,
    request: ProcessDocumentRequest = Body(...),
) -> AnalysisResponse:
    if request.document_id != document_id:
        # Guarantee URL path matches payload ID
        request.document_id = document_id

    return analysis_service.process_document(
        document_id=document_id,
        extracted_text=request.extracted_text,
        force_refresh=False,
    )


@router.get(
    "/analysis/{document_id}",
    response_model=AnalysisResponse,
    summary="Get Cached AI Analysis",
    description="Returns previously computed AI analysis for a document. Does NOT re-invoke Gemini.",
)
async def get_analysis(document_id: str) -> AnalysisResponse:
    analysis = analysis_service.get_analysis(document_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No AI analysis found for document '{document_id}'. Submit it via POST /ai/analyze/{document_id} first.",
        )
    return analysis


@router.post(
    "/analyze/{document_id}/retry",
    response_model=AnalysisResponse,
    summary="Retry AI Analysis",
    description="Bypasses cache and forces re-evaluation of document compliance analysis.",
)
async def retry_analysis(
    document_id: str,
    request: ProcessDocumentRequest = Body(...),
) -> AnalysisResponse:
    return analysis_service.retry_analysis(
        document_id=document_id,
        extracted_text=request.extracted_text,
    )


@router.get(
    "/status/{document_id}",
    response_model=StatusResponse,
    summary="Check AI Analysis Status",
    description="Quick check whether analysis exists, is completed, or is unavailable.",
)
async def get_status(document_id: str) -> StatusResponse:
    analysis = analysis_service.get_analysis(document_id)
    if not analysis:
        return StatusResponse(
            document_id=document_id,
            status=AnalysisStatus.PENDING,
            has_analysis=False,
            flags_count=0,
            message="Document has not been processed by AI yet.",
        )

    return StatusResponse(
        document_id=document_id,
        status=analysis.status,
        has_analysis=True,
        generated_at=analysis.generated_at,
        flags_count=len(analysis.flags),
        message=analysis.message,
    )


@router.post(
    "/privacy-inspect",
    response_model=PrivacyInspectionResponse,
    summary="Privacy Wall Inspection Preview",
    description="Demonstrates server-side PII masking and verifies that raw PII never leaks into outbound payloads.",
)
async def inspect_privacy(
    request: ProcessDocumentRequest = Body(...),
) -> PrivacyInspectionResponse:
    data = privacy_wall.inspect_sanitization(
        raw_text=request.extracted_text,
        document_id=request.document_id,
    )
    return PrivacyInspectionResponse(**data)


@router.get("/health", summary="Health Check")
async def health_check() -> Dict[str, Any]:
    from ai.app.core.config import settings
    from ai.app.repositories.vector_store import vector_store

    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "gemini_model": settings.GEMINI_MODEL,
        "gemini_api_key_configured": bool(
            settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_gemini_api_key_here"
        ),
        "corpus_loaded": {
            "rules": len(vector_store.rules),
            "disclosures": len(vector_store.disclosures),
            "precedents": len(vector_store.precedents),
        },
    }
