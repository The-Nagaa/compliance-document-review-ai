# Compliance Document Review App - AI Track

> **Production-grade AI micro-module for financial compliance document review, featuring server-side PII masking, retrieval-grounded compliance analysis via Google Gemini free tier, missing disclosure detection, top-3 precedent matching, caching, and non-blocking graceful degradation.**

---

## 1. Architectural Overview & Retrieval Flow

```
                      +------------------------------------------+
                      |   Data Engineering / Backend Ingestion   |
                      |        (PDF / DOCX / XLSX Clean Text)    |
                      +--------------------+---------------------+
                                           |
                                           v
                             POST /ai/analyze/{document_id}
                                           |
                                           v
                      +------------------------------------------+
                      |       PRIVACY WALL & PII MASKER          |
                      |  - Regex + Heuristic Entity Detection    |
                      |  - Stable Placeholders ([CLIENT_1] etc.) |
                      |  - Server-Side Mapping Storage           |
                      +--------------------+---------------------+
                                           | (Masked Text ONLY)
                        +------------------+------------------+
                        |                                     |
                        v                                     v
         +------------------------------+     +------------------------------+
         |    VECTOR RETRIEVAL ENGINE   |     |    PRIVACY-SAFE GEMINI LLM   |
         |  - Persistent HTTP Pool      |     |  - System Instruction        |
         |  1. Rule Lookup (/rule-lookup|     |  - Pydantic Schema Validation|
         |  2. Parallel Disclosure Check|====>|  - Traceable Compliance Flags|
         |  3. Precedent Search (Top 3) |     |  - Summary Generation        |
         |  - Local Vector Fallback     |     |  - Error Degradation Handling|
         +--------------+---------------+     +--------------+---------------+
                        |                                     |
                        +------------------+------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |      ANALYSIS CACHE & ORCHESTRATION      |
                      |  - AI NEVER Decides (Human Officer Owns) |
                      |  - Graceful Degradation on API Failure   |
                      |  - Idempotent Cached Response            |
                      +--------------------+---------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |          BACKEND / FRONTEND API          |
                      |  - Flags with Passage/Rule/Reason        |
                      |  - Top 3 Similar Past Precedents        |
                      |  - Missing Disclosures & Summary         |
                      +------------------------------------------+
```

### Retrieval & Analysis Execution Flow
1. **Ingress & Privacy Wall**: When a document is submitted, the Privacy Wall detects sensitive entities (clients, emails, phones, SSNs, accounts, addresses, monetary amounts) and replaces them with deterministic placeholders (`[CLIENT_1]`, `[ACCOUNT_1]`). The raw PII mapping remains strictly on the server and is never transmitted outward.
2. **Retrieval Grounding**:
   - The pre-masked text is queried against Data Engineering's production retrieval endpoints using a persistent HTTP connection pool.
   - **Rule Lookup**: Fetches relevant compliance rules to ground the LLM analysis.
   - **Parallel Disclosure Check**: Checks document passages concurrently across all mandatory standard disclosures via a bounded thread pool, preserving deterministic index ordering.
   - **Precedent Search**: Fetches the Top 3 historical review precedents.
   - **Fallback Mechanism**: If the Data Engineering service is unreachable, retrieval seamlessly falls back to the local `vector_store.json` using matching 384-dimensional embeddings.
3. **LLM Analysis & Verification**: Grounded context is sent to Gemini (Free Tier) to generate an executive summary and traceable compliance flags.
4. **Caching & Egress**: The complete result is cached locally by document ID for instant retrieval upon subsequent requests.

---

## 2. Core Architectural Guarantees & Assistive Design

1. **Privacy-First Boundary**:
   - Document text is masked server-side *before* any text is sent to Gemini, Data Engineering, or embedding services.
   - Placeholders are deterministic (`[CLIENT_1]`, `[EMAIL_1]`, `[ACCOUNT_1]`, `[SSN_1]`, `[ADDRESS_1]`, `[AMOUNT_1]`).
   - The mapping (`[CLIENT_1] -> John Doe`) stays server-side and is **never** serialized to vendor APIs or the frontend.
2. **AI Remains Strictly Assistive (Human Decision-Maker Guarantee)**:
   - The AI system **never** makes, applies, or overrides final review verdicts (`approved`, `rejected`, `needs_revision`).
   - The AI serves solely as an assistive tool to highlight potential issues, providing exact passage excerpts, matched rule descriptions, severity levels, and reasoning.
   - The human compliance officer retains full authority and makes the final compliance decision.
3. **Non-Blocking Graceful Degradation**:
   - If the Gemini API is unavailable, rate-limited (HTTP 429), times out, or has an unconfigured API key, the review page continues to function.
   - Structured error statuses (`unavailable`, `rate_limited`, `failed`) are returned alongside `retryable` flags.
   - Retrieval assists (precedent matches and missing disclosure checks) remain functional even when LLM analysis is degraded.

---

## 3. Retrieval Architecture & Performance Optimizations

The retrieval pipeline incorporates several performance optimizations designed for high throughput and stability:

### A. Persistent HTTP Connection Pooling
- Handled in `DataEngineeringClient` via `httpx.Client` with configured connection limits (`max_keepalive_connections=20`, `max_connections=50`, `keepalive_expiry=30.0s`).
- Reuses open TCP/TLS connections across consecutive and parallel HTTP requests to the Data Engineering service, eliminating repeated handshake latencies.
- Implements thread-safe client lifecycle management with explicit `close()` and Python context manager (`__enter__` / `__exit__`) support.

### B. Parallel Disclosure Retrieval (Bounded Thread Pool)
- Evaluates document passages against all mandatory disclosures in parallel using a `ThreadPoolExecutor` (bounded to `max_workers=5`).
- Dispatches individual `/disclosure-check` requests concurrently across the persistent connection pool.

### C. Deterministic Result Ordering
- As parallel disclosure workers finish (`as_completed`), results are indexed by their original position.
- Results are reassembled in strict original sequence before returning, ensuring 100% deterministic output ordering regardless of thread scheduling.
- Logs unified execution metrics per document:
  ```text
  DISCLOSURE_PROFILING | total_ms=... | avg_request_ms=... | requests=25 | slowest=... (...)
  ```
  where `avg_request_ms` is the exact arithmetic mean duration of individual disclosure requests.

### D. Data Engineering Shared Embedding Model Reuse
- Data Engineering serves as the production single source of truth for rules, disclosures, and precedents.
- Data Engineering maintains a singleton instance of the `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) embedding model.
- Model weights are loaded once into memory during service initialization and reused across all requests, eliminating redundant model reloads.

### E. Retrieval Corpora Details
- **Rule Retrieval (Grounding)**: 34+ compliance rules spanning Prohibited Claims, Performance Standards, Required Disclosures, Testimonials, and Supervision (`POST /rule-lookup`).
- **Missing Disclosure Detection (Absence)**: 25 standard disclosures (SEC RIA, SIPC/FINRA, Risk of Loss, Form ADV, Tax Advice, etc.) queried against document chunks (`POST /disclosure-check`).
- **Precedent Search (Top 3)**: 100 historically reviewed synthetic submissions returning the top 3 most semantically similar precedents with prior officer comments (`POST /precedent-search`).
- **Local Fallback**: AI maintains a local `vector_store.json` using the matching 384D `all-MiniLM-L6-v2` embeddings for offline development and fallback.

---

## 4. Performance Benchmarking

A standalone benchmarking script is provided to profile component latencies, memory footprint, and pipeline throughput:

```powershell
python scripts/benchmark_performance.py
```

### Benchmark Scope
1. **Data Engineering Startup & Initialization**: Measures import and initial model weight loading time and traced peak memory usage.
2. **Granular Component Latencies**: Measures isolated retrieval latencies for Rule Lookup, 25-item Parallel Disclosure Check, and Top-3 Precedent Search.
3. **End-to-End Pipeline**: Measures Cold Document Analysis (first-time processing) vs Warm Cached Analysis (sub-millisecond cache hit).

> [!NOTE]
> **Environment-Dependent Measurements**: Benchmark latencies and memory figures depend heavily on hardware specifications (CPU architecture, RAM, SSD I/O, network). All benchmark timings should be interpreted as local relative measurements rather than absolute SLAs.

---

## 5. API Endpoints Contract

### Base URL: `http://localhost:8000/ai`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ai/analyze/{document_id}` | Submit extracted document text for masking, retrieval, and AI analysis. |
| `GET` | `/ai/analysis/{document_id}` | Retrieve cached AI analysis. Does not re-invoke external LLM calls. |
| `POST` | `/ai/analyze/{document_id}/retry` | Force re-analysis of document, bypassing cache. |
| `GET` | `/ai/status/{document_id}` | Lightweight status check (`pending`, `completed`, `unavailable`, `rate_limited`). |
| `POST` | `/ai/privacy-inspect` | Audit endpoint demonstrating sanitized payload vs raw text. |
| `GET` | `/ai/health` | Service health and corpus statistics. |

---

## 6. API Schemas

### `POST /ai/analyze/{document_id}` Request Body:
```json
{
  "document_id": "doc_10293",
  "extracted_text": "Marketing memorandum for Client: John Doe. Account #8839201. Guaranteed 20% returns."
}
```

### Successful Analysis Response (`status: "completed"`):
```json
{
  "document_id": "doc_10293",
  "status": "completed",
  "summary": "Marketing memorandum for [CLIENT_1] promoting a high-yield strategy with promised returns.",
  "flags": [
    {
      "id": "7bf3-...",
      "passage_excerpt": "Guaranteed 20% returns.",
      "matched_rule_id": "RULE-001",
      "matched_rule": "Prohibition of Guaranteed Returns",
      "explanation": "The passage makes an explicit guarantee of return which violates regulatory standards.",
      "severity": "high"
    }
  ],
  "precedents": [
    {
      "document_id": "PREC-DOC-0105",
      "similarity": 0.8421,
      "previous_decision": "rejected",
      "officer_comment": "REJECTED. Prohibited promissory return claims.",
      "excerpt": "Marketing memo for [CLIENT_1] claiming guaranteed returns..."
    }
  ],
  "disclosures_checked": [
    {
      "disclosure_id": "DISC-001",
      "disclosure_type": "SEC_RIA",
      "disclosure_text": "Advisory services offered through...",
      "is_present": false,
      "similarity_score": 0.231,
      "best_matching_passage": null
    }
  ],
  "generated_at": "2026-08-28T05:49:02.867963Z",
  "cached": false,
  "message": null,
  "retryable": false
}
```

### Degraded Analysis Response (`status: "unavailable"`):
```json
{
  "document_id": "doc_10293",
  "status": "unavailable",
  "summary": "AI summary unavailable.",
  "flags": [],
  "precedents": [ ... ],
  "disclosures_checked": [ ... ],
  "generated_at": "2026-08-28T05:52:10.123456Z",
  "cached": false,
  "message": "AI assist unavailable: Gemini API key is not configured. The document is still accessible for manual review.",
  "retryable": true
}
```

---

## 7. Known PII Masking Limitations

1. **Context-Free Single Names**: First names appearing without structural cues (e.g. "Bob said we should invest") are not aggressively masked to prevent false positives on standard financial terminology (e.g. "Dow Jones", "Treasury Bond").
2. **International Addresses**: Non-standard postal address structures without street/avenue/road suffixes or city/state/zip indicators may not be captured.
3. **Deliberately Obfuscated Entities**: Obfuscations like `john [at] gmail [dot] com` or `five-five-five-1234` require heavy NLP/NER models rather than regex/heuristics.

---

## 8. Setup & Execution Commands

### Prerequisites
- Python 3.11+
- Recommended: Virtual environment (`python -m venv venv`)

### Installation
```powershell
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env` and set configuration parameters:
```powershell
cp .env.example .env
```

### 1. Seed Local Fallback Corpus
Populate the local vector store with compliance rules, mandatory disclosures, and historical precedents (used for offline development and fallback):
```powershell
python scripts/seed_corpus.py
```

### 2. Run Test Suite
Run the full automated test suite (71+ unit and integration tests):
```powershell
python -m pytest -q
```

### 3. Run Performance Benchmark
Profile retrieval components, parallel execution throughput, and memory peak:
```powershell
python scripts/benchmark_performance.py
```

### 4. Run Acceptance Demo
Execute the full 13-step demonstration sequence:
```powershell
python scripts/acceptance_demo.py
```

### 5. Start the FastAPI Service
```powershell
uvicorn ai.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at: `http://localhost:8000/docs`

---

## 9. Cross-Track Integration Notes

- **Backend Track**:
  - Call `POST /ai/analyze/{document_id}` with `{ "document_id": "...", "extracted_text": "..." }` when an advisor submits a document.
  - Call `GET /ai/analysis/{document_id}` when loading the review dashboard.
  - If AI returns `status: "unavailable"`, render the review queue normally—never prevent the compliance officer from approving or rejecting the document.
- **Frontend Track**:
  - Display the `summary` in the assist card.
  - Render each item in `flags` with its `passage_excerpt`, `matched_rule`, `explanation`, and `severity` badge (`low`=yellow, `medium`=amber, `high`=red).
  - Render `precedents` (Top 3) with similarity score bar, previous decision chip, and historical officer comment.
  - If `status == "unavailable"` or `"rate_limited"`, show an alert banner with a `Retry AI Analysis` button calling `POST /ai/analyze/{document_id}/retry`.
- **Data Engineering Track**:
  - Data Engineering service runs on `http://localhost:5000` (configurable via `DATA_ENGINEERING_BASE_URL`).
  - Implements `/rule-lookup`, `/disclosure-check`, and `/precedent-search`.
  - Reuses a shared `all-MiniLM-L6-v2` 384-dimensional embedding model instance across endpoints.
- **DevOps Track**:
  - Dockerized deployment can mount `./data` as a volume for cached analyses.
