# EyeDropShop Brain — Prototype Handoff Guide

## 1. Project Overview

The EyeDropShop Brain is a centralized, governed knowledge system for future EyeDropShop AI agents.

It retrieves approved EyeDropShop information and uses a local language model to generate grounded answers with evidence and citations.

The current prototype supports:
- Product information
- Shipping and return policies
- Educational YouTube transcripts
- English/French product and policy knowledge
- Audience and approval controls
- Controlled response statuses
- Knowledge-gap logging
- API access
- Streamlit interface

If approved information is insufficient, the Brain returns `insufficient_knowledge` rather than guessing.

## 2. Current Prototype Scope

Current pilot:
- 30 selected products
- 60 bilingual product records
- 4 Canadian policy documents
- 23 educational transcripts
- 29 governed knowledge sources
- 549 searchable passages
- 20/20 automated system tests passing

This is a pilot subset, not the complete EyeDropShop catalog.

## 3. System Architecture

```text
User / Agent
     ↓
Brain API / Streamlit
     ↓
Retrieval
     ↓
PostgreSQL + pgvector
     ↓
Approved Evidence
     ↓
Qwen via Ollama
     ↓
Answer + Citations + Status
```

The Brain retrieves approved evidence first, then gives that evidence to the local language model.

## 4. Technology Stack

- **PostgreSQL** — primary knowledge database
- **pgvector** — semantic/vector search
- **BGE-M3** — embedding model
- **Qwen** — local language model
- **Ollama** — local model serving
- **FastAPI** — Brain API
- **Streamlit** — prototype interface
- **Python** — ingestion, retrieval, governance, API, and testing

## 5. Starting the Brain

### Simple startup

On the configured Mac, double-click:

`start_brain.command`

This activates the environment, selects PostgreSQL retrieval, and starts Streamlit.

The interface normally opens at:

`http://localhost:8501`

### Manual startup

```bash
conda activate eyedropshop-brain
export BRAIN_RETRIEVAL_BACKEND=postgres
streamlit run app.py
```

PostgreSQL and Ollama must also be available.

### API startup

```bash
export BRAIN_RETRIEVAL_BACKEND=postgres
uvicorn brain_api:app --host 127.0.0.1 --port 8000
```

Do not store API keys in Git, source code, or this document.

## 6. Knowledge Sources

- Products: `prototype_data/products/`
- Policies: `knowledge/converted/policies/`
- Original transcripts: `knowledge/Youtube_Transcripts/`
- Converted transcripts: `knowledge/converted/transcripts/`
- Governance registry: `governance/knowledge_registry.csv`

## 7. Knowledge Governance

Every source must have a governance record before it can be indexed.

Supported audiences:
- `internal`
- `doctor`
- `customer`
- `shared`

Supported approval states:
- `draft`
- `approved`
- `expired`
- `rejected`

Only approved, non-expired knowledge is eligible for retrieval.

Audience access:
- internal → internal + shared
- doctor → doctor + shared
- customer → customer + shared

## 8. Knowledge Update Procedure

For any new or modified knowledge:

1. **Receive the source**  
   Identify source, content type, language, country, audience, approval status, and version.

2. **Store the source**  
   Place it in the appropriate project folder and convert it if needed.

3. **Register governance**  
   Add or update the source in `governance/knowledge_registry.csv`.

4. **Validate**  
   Run:
   ```bash
   python ingestion_validator.py
   ```
   Do not continue if there is a blocking `FAIL`.

5. **Ingest**  
   For policies:
   ```bash
   python add_policies_to_database.py
   ```
   For transcripts:
   ```bash
   python add_transcripts_to_database.py
   ```

6. **Update PostgreSQL**
   ```bash
   python migrate_chroma_to_postgres.py --replace
   ```

7. **Run tests**
   ```bash
   python evaluate_brain_system.py
   ```
   Current expected result: `20/20 (100.0%)`.

8. **Manual verification**  
   Test representative questions in Streamlit and verify the answer, evidence, citation, audience, and status.

9. **Commit the update**  
   Commit the approved source, governance, ingestion, and test changes to Git.

### Updating existing knowledge

When replacing an existing policy or source:
- update its version/governance information;
- replace or expire the old source;
- re-run ingestion and PostgreSQL migration;
- confirm outdated passages are no longer retrieved;
- rerun automated and manual tests.

Old and new active policy versions should not coexist unless this is intentional.

## 9. Controlled Response Statuses

- `answerable` — approved evidence supports the answer
- `insufficient_knowledge` — not enough approved evidence
- `restricted` — outside the allowed response boundary, such as personalized medical advice
- `conflicting_sources` — approved sources disagree
- `human_review_required` — a person must review the case

For general customer-service gaps, the Brain can direct users to Customer Happiness.

For personalized medical questions, it directs users to an eye-care professional.

## 10. Testing

Run:

```bash
python evaluate_brain_system.py
```

Current result:

`20/20 tests passing`

Tests cover normal answers, missing knowledge, medical restrictions, conflicting sources, permissions, expired/draft content, authentication, authorization, transcript retrieval, and citations.

## 11. Known Limitations

- Only 30 selected products are included.
- The full EyeDropShop catalog is not yet indexed.
- Policy coverage is limited to the supplied Canadian shipping and return policies.
- 23 educational transcripts are included.
- Transcript text may contain transcription errors.
- Transcript citations do not yet include YouTube URLs.
- Wholesale knowledge still needs to be confirmed/integrated.
- The Brain does not provide personalized diagnosis or treatment.
- Missing approved information intentionally returns `insufficient_knowledge`.
- The current prototype is for controlled development and demonstration, not full production deployment.

## 12. Troubleshooting

**Streamlit does not start**
```bash
conda activate eyedropshop-brain
export BRAIN_RETRIEVAL_BACKEND=postgres
streamlit run app.py
```

**PostgreSQL retrieval fails**  
Confirm PostgreSQL is running and `BRAIN_RETRIEVAL_BACKEND=postgres` is set.

**The local model does not answer**  
Confirm Ollama is running and the configured Qwen model is available.

**API authentication error**  
Set the required API-key environment variable before starting Uvicorn.

**Knowledge update fails validation**  
Run `python ingestion_validator.py` and resolve all blocking `FAIL` results before indexing.
