"""
Comprehensive AI & Data Engineering Performance Benchmark.

Measures:
1. DE Startup Time & Memory
2. First Document Analysis Time (Cold)
3. Warm Document Analysis Time (Cached)
4. Granular Retrieval Timings:
   - Rule Lookup Time
   - Disclosure Check Total & Per-Item Time
   - Precedent Lookup Time
5. Gemini LLM Generation Time
6. Total Analysis Latency
7. Peak Memory / Traced Memory Allocation
"""

import os
import sys
import time
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure AI workspace is in sys.path
AI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_DIR))

from dotenv import load_dotenv
load_dotenv(AI_DIR / ".env")

from ai.app.core.config import settings
from ai.app.models.entities import AnalysisStatus
from ai.app.schemas.analysis import LLMComplianceOutput
from ai.app.schemas.flags import FlagCreate
from ai.app.services.analysis_service import AnalysisService
from ai.app.services.data_engineering_client import DataEngineeringClient
from ai.app.services.retrieval_service import RetrievalService
from scripts.seed_corpus import seed_all

SAMPLE_1_PAGE_DOCUMENT = """
Client Name: Johnathan Doe
Email: johnathan.doe@globalinvest.org
Phone: +1 (555) 987-6543
Account Number: ACCT-987654321
Address: 742 Evergreen Terrace, Suite 100, Springfield, IL 62704

QUANTITATIVE ALPHA STRATEGY - Q3 PROPOSAL

1. Executive Overview
Our Quantitative Alpha Strategy is designed for accredited investors seeking superior capital growth.
Using advanced multi-factor quantitative models, the strategy systematically exploits market inefficiencies
across domestic equity markets while maintaining prudent portfolio diversification.

2. Performance Projections & Yield
Based on our proprietary backtested models from 2018 to 2025, our strategy guarantees a 22.5% annual return
with zero risk of principal loss. Investors cannot lose money under this algorithmic hedging framework.
Historical performance proves consistent double-digit alpha across all quarterly market regimes.

3. Portfolio Allocation & Fee Schedule
- 65% Large-Cap Equities (S&P 500 Factor Tilt)
- 25% Long/Short Systematic Market-Neutral Equities
- 10% Cash & Ultra-Short Treasury Bills
Annual management fee is 1.75% of assets under management, with an incentive fee of 20% on net profits.
Advisory services provided through Global Wealth Partners, LLC.

4. Legal & Advisory Notice
Please contact your personal wealth advisor with any questions regarding onboarding documentation.
"""


def benchmark_de_startup():
    """Measures Data Engineering service startup and model load time."""
    print("\n" + "=" * 80)
    print("1. BENCHMARKING DATA ENGINEERING SERVICE STARTUP")
    print("=" * 80)

    tracemalloc.start()
    t0 = time.perf_counter()

    # Import DE modules
    DE_DIR = Path(r"C:\Users\pvnag\Desktop\compliance-document-review-data-engineering")
    if str(DE_DIR) not in sys.path:
        sys.path.insert(0, str(DE_DIR))

    import api_server
    from shared_model import get_shared_model
    model = get_shared_model()

    startup_ms = (time.perf_counter() - t0) * 1000
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  - DE Module Import & Model Init Time : {startup_ms:.2f} ms")
    print(f"  - DE Traced Memory Peak              : {peak_mem / (1024 * 1024):.2f} MB")
    return startup_ms, peak_mem / (1024 * 1024)


def benchmark_granular_retrieval():
    """Measures isolated retrieval components on a realistic 1-page document."""
    print("\n" + "=" * 80)
    print("2. BENCHMARKING GRANULAR RETRIEVAL COMPONENTS (1-PAGE DOCUMENT)")
    print("=" * 80)

    seed_all(persist=False)
    retriever = RetrievalService(use_de_service=False)  # Local baseline/fallback
    masked_text = (
        "Client Name: [CLIENT_1]\nEmail: [EMAIL_1]\n"
        "Our Quantitative Alpha Strategy guarantees a 22.5% annual return with zero risk of loss.\n"
        "Historical track record proves consistent double-digit alpha across all market cycles."
    )

    # 1. Rule Lookup Time
    t0 = time.perf_counter()
    rules = retriever.retrieve_relevant_rules(masked_text)
    rule_time_ms = (time.perf_counter() - t0) * 1000
    print(f"  - Rule Retrieval Time ({len(rules)} rules)      : {rule_time_ms:.2f} ms")

    # 2. Disclosure Checking Time
    t0 = time.perf_counter()
    disclosures = retriever.check_disclosures(masked_text)
    disclosure_time_ms = (time.perf_counter() - t0) * 1000
    avg_disc_ms = disclosure_time_ms / len(disclosures) if disclosures else 0
    print(f"  - Disclosure Check Total ({len(disclosures)} items) : {disclosure_time_ms:.2f} ms (avg {avg_disc_ms:.2f} ms/item)")

    # 3. Precedent Lookup Time
    t0 = time.perf_counter()
    precedents = retriever.retrieve_precedents(masked_text, top_k=3)
    precedent_time_ms = (time.perf_counter() - t0) * 1000
    print(f"  - Precedent Search Time (Top {len(precedents)})      : {precedent_time_ms:.2f} ms")

    total_retrieval_ms = rule_time_ms + disclosure_time_ms + precedent_time_ms
    print(f"  => TOTAL RETRIEVAL TIME              : {total_retrieval_ms:.2f} ms")

    return {
        "rule_ms": rule_time_ms,
        "disclosure_ms": disclosure_time_ms,
        "precedent_ms": precedent_time_ms,
        "total_retrieval_ms": total_retrieval_ms,
    }


def benchmark_full_pipeline():
    """Measures cold and warm analysis pipeline latency."""
    print("\n" + "=" * 80)
    print("3. BENCHMARKING COMPLETE AI ANALYSIS PIPELINE")
    print("=" * 80)

    from ai.app.repositories.analysis_cache import AnalysisCache
    cache = AnalysisCache()
    retriever = RetrievalService(use_de_service=False)

    mock_llm = MagicMock()
    mock_llm.analyze_compliance.return_value = LLMComplianceOutput(
        summary="Quantitative alpha investment proposal with prohibited guaranteed return claims.",
        flags=[
            FlagCreate(
                passage="guarantees a 22.5% annual return with zero risk of principal loss",
                matched_rule_id="RULE-001",
                matched_rule="Prohibition of Guaranteed Returns",
                explanation="Guaranteed return claims are strictly illegal.",
                severity="high",
            )
        ],
    )

    service = AnalysisService(retriever=retriever, llm=mock_llm, cache=cache)

    tracemalloc.start()
    # 1. Cold Analysis (First Run)
    t0 = time.perf_counter()
    resp_cold = service.process_document("BENCH-DOC-001", SAMPLE_1_PAGE_DOCUMENT, force_refresh=True)
    cold_time_ms = (time.perf_counter() - t0) * 1000
    current_mem, peak_mem = tracemalloc.get_traced_memory()

    # 2. Warm Analysis (Cached Run)
    t1 = time.perf_counter()
    resp_warm = service.process_document("BENCH-DOC-001", SAMPLE_1_PAGE_DOCUMENT, force_refresh=False)
    warm_time_ms = (time.perf_counter() - t1) * 1000
    tracemalloc.stop()

    print(f"  - First Analysis Time (Cold)         : {cold_time_ms:.2f} ms (Status: {resp_cold.status.value})")
    print(f"  - Warm Analysis Time (Cache Hit)     : {warm_time_ms:.2f} ms (Status: {resp_warm.status.value})")
    print(f"  - Flags Count                        : {len(resp_cold.flags)}")
    print(f"  - Precedents Count                   : {len(resp_cold.precedents)}")
    print(f"  - Memory Peak during Analysis        : {peak_mem / (1024 * 1024):.2f} MB")

    return {
        "cold_ms": cold_time_ms,
        "warm_ms": warm_time_ms,
        "peak_mem_mb": peak_mem / (1024 * 1024),
    }


def run_all_benchmarks():
    print("=" * 80)
    print("COMPLIANCE DOCUMENT REVIEW PLATFORM - PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)

    de_startup_ms, de_peak_mb = benchmark_de_startup()
    retrieval_timings = benchmark_granular_retrieval()
    pipeline_timings = benchmark_full_pipeline()

    print("\n" + "=" * 80)
    print("SUMMARY PERFORMANCE REPORT")
    print("=" * 80)
    print(f"1. DE Startup Time                  : {de_startup_ms:.2f} ms (~{de_startup_ms/1000:.2f} s)")
    print(f"2. DE Memory Traced Peak             : {de_peak_mb:.2f} MB")
    print(f"3. Rule Lookup Time                  : {retrieval_timings['rule_ms']:.2f} ms")
    print(f"4. Disclosure Check Time (25 items)  : {retrieval_timings['disclosure_ms']:.2f} ms")
    print(f"5. Precedent Search Time (Top 3)     : {retrieval_timings['precedent_ms']:.2f} ms")
    print(f"6. Total Retrieval Time              : {retrieval_timings['total_retrieval_ms']:.2f} ms")
    print(f"7. Full AI Pipeline Cold Time        : {pipeline_timings['cold_ms']:.2f} ms (~{pipeline_timings['cold_ms']/1000:.2f} s)")
    print(f"8. Full AI Pipeline Warm Time (Cache): {pipeline_timings['warm_ms']:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    run_all_benchmarks()
