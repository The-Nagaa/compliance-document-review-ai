"""
FastAPI Application Entry Point for Compliance Document Review AI Track.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai.app.api.routes import router as ai_router
from ai.app.core.config import settings
from ai.app.core.logging import logger
from ai.app.repositories.analysis_cache import analysis_cache
from ai.app.repositories.vector_store import vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load seed data and analysis cache
    logger.info("Initializing AI Track Service...")
    vector_loaded = vector_store.load_from_disk()
    if not vector_loaded or len(vector_store.rules) == 0:
        logger.info("Vector store empty on startup. Running automatic seed generator...")
        from scripts.seed_corpus import seed_all
        seed_all()
    
    cache_loaded = analysis_cache.load_from_disk()
    if cache_loaded:
        logger.info(f"Loaded {analysis_cache.count()} cached analyses from disk.")

    yield

    # Shutdown: Persist cache
    logger.info("Shutting down AI Track Service. Persisting caches...")
    analysis_cache.save_to_disk()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Compliance Document Review AI Track: PII Masking, Vector-Grounded Analysis, Precedent Matching, and Degradation Handling.",
    lifespan=lifespan,
)

# CORS configuration for seamless frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error in API: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal AI service error. The document remains accessible for manual review.",
            "retryable": True,
        },
    )


# Mount AI API Router
app.include_router(ai_router)


@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/ai/health",
        "api_prefix": settings.API_PREFIX,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai.app.main:app", host="0.0.0.0", port=8000, reload=True)
