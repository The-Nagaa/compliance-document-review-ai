# Compliance Document Review App - AI Track

> **Production-grade AI micro-module for financial compliance document review, featuring server-side PII masking, retrieval-grounded compliance analysis via Google Gemini free tier, missing disclosure detection, top-3 precedent matching, caching, and non-blocking graceful degradation.**

---

## 1. Architectural Overview

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
         |  1. Applicable Rules Retrieval |     |  - System Instruction        |
         |  2. Missing Disclosures (Absence)  | Grounded |  - Pydantic Schema Validation|
         |  3. Top-3 Precedents Search  |====>|  - Traceable Compliance Flags|
         +--------------+---------------+     |  - Summary Generation        |
                        |                     +--------------+---------------+
                        +------------------+-----------------+
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
                      |  - Document Summary & Degraded States    |
                      +------------------------------------------+
```

---

## 2. Core Architectural Guarantees

1. **Privacy-First Boundary**:
   - Document text is masked server-side *before* any text is sent to Gemini or embedding APIs.
   - Placeholders are deterministic (`[CLIENT_1]`, `[EMAIL_1]`, `[ACCOUNT_1]`, `[SSN_1]`, `[ADDRESS_1]`, `[AMOUNT_1]`).
   - The mapping (`[CLIENT_1] -> John Doe`) stays server-side and is **never** serialized to vendor APIs or the frontend.
2. **AI Never Decides**:
   - The AI system **never** sets, recommends-and-applies, or pre-fills review verdicts (`Approved`, `Rejected`, `Needs Revision`).
   - All compliance flags provide exact evidence (triggering passage, matched rule ID, matched rule description, and one-line explanation).
   - The Compliance Officer makes the final decision.
3. **Non-Blocking Graceful Degradation**:
   - If the Gemini API is down, rate-limited (HTTP 429), times out, or has a missing API key, the review page continues to load normally.
   - Structured error statuses (`unavailable`, `rate_limited`, `failed`) are returned with retryability flags.
   - Local vector store assists (precedent matches and missing disclosure detection) continue to operate.

---

## 3. Vector Retrieval Architecture

The vector retrieval layer operates over three corpora:

### A. Rule Retrieval (Grounding)
- **Corpus**: 35 compliance rules spanning Prohibited Claims, Performance Standards, Required Disclosures, Testimonials, and Supervision.
- **Mechanism**: Paragraph-aware document chunks are embedded and queried against rule embeddings to provide the LLM with grounded context.

### B. Missing Disclosure Detection by Absence
- **Corpus**: 25 standard disclosures (SEC RIA, SIPC/FINRA, Risk of Loss, Form ADV, Tax Advice, etc.).
- **Mechanism**: Computes maximum cosine similarity across document passages against mandatory boilerplate disclosures. If the maximum similarity is below a configurable threshold (default `0.75`), the disclosure is automatically flagged as **MISSING**.

### C. Precedent Search (Top 3)
- **Corpus**: 100 historically reviewed synthetic submissions with historical officer comments and decisions (`approved`, `rejected`, `needs_revision`).
- **Mechanism**: Computes document-level semantic similarity and returns exactly the **Top 3 Most Similar Precedents**.

---

## 4. API Endpoints Contract

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

## 5. API Schemas

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
      "disclosure_type": "SEC_RIA_DISCLOSURE",
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
  "retryable": false
}
```

---

## 6. Known PII Masking Limitations

1. **Context-Free Single Names**: First names appearing without structural cues (e.g. "Bob said we should invest") are not aggressively masked to prevent false positives on standard financial terminology (e.g. "Dow Jones", "Treasury Bond").
2. **International Addresses**: Non-standard postal address structures without street/avenue/road suffixes or city/state/zip indicators may not be captured.
3. **Deliberately Obfuscated Entities**: Obfuscations like `john [at] gmail [dot] com` or `five-five-five-1234` require heavy NLP/NER models rather than regex/heuristics.

---

## 7. Setup & Execution Commands

### Prerequisites
- Python 3.11+
- Recommended: Virtual environment (`python -m venv venv`)

### Installation
```powershell
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env` and optionally set your Gemini API key:
```powershell
cp .env.example .env
```

### 1. Seed Corpus
Populate the vector store with 35 compliance rules, 25 disclosures, and 100 reviewed synthetic precedents:
```powershell
python scripts/seed_corpus.py
```

### 2. Run Test Suite
Run the complete unit and integration test suite:
```powershell
pytest tests/ -v --cov=ai.app
```

### 3. Run Acceptance Demo
Execute the full 13-step demonstration sequence:
```powershell
python scripts/acceptance_demo.py
```

### 4. Start the FastAPI Service
```powershell
uvicorn ai.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at: `http://localhost:8000/docs`

---

## 8. Cross-Track Integration Notes

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
  - Pass clean UTF-8 text extracted from PDF, DOCX, or XLSX directly into `extracted_text`.
- **DevOps Track**:
  - Dockerized deployment can mount `./data` as a volume for cached analyses and vector indices.
  - To migrate vector storage to PostgreSQL + pgvector, use the schema generated by `VectorStore.export_pgvector_sql()`.
