import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings:
    """AI Service Configuration Settings."""

    PROJECT_NAME: str = "Compliance Document Review App - AI Service"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/ai"

    # Gemini API settings
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TIMEOUT_SECONDS: float = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30.0"))
    GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "2"))

    # Vector Retrieval settings
    DISCLOSURE_SIMILARITY_THRESHOLD: float = float(os.getenv("DISCLOSURE_SIMILARITY_THRESHOLD", "0.75"))
    PRECEDENT_TOP_K: int = int(os.getenv("PRECEDENT_TOP_K", "3"))
    RULE_TOP_K: int = int(os.getenv("RULE_TOP_K", "5"))

    # Storage paths
    VECTOR_STORE_PATH: Path = Path(os.getenv("VECTOR_STORE_PATH", str(DATA_DIR / "vector_store.json")))
    ANALYSIS_CACHE_PATH: Path = Path(os.getenv("ANALYSIS_CACHE_PATH", str(DATA_DIR / "analysis_cache.json")))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
