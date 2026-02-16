# Youth Opportunity Matching Platform — Backend

Agno-based multi-agent system with FastAPI and AgentOS, using Supabase pgvector for knowledge storage.

## Setup

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and set:

- **SUPABASE_DB_URL**: Your Supabase PostgreSQL connection string (with pgvector)
  - Format: `postgresql+psycopg://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres`
  - Enable the pgvector extension in your Supabase project
- **OPENAI_API_KEY**: Your OpenAI API key (for embeddings and LLM)

### 3. Ingest data

Place your CSV files in `backend/data/`:

- `youth_profiles.csv` — Youth profile data
- `employer_data.csv` — Employer/opportunity data

Then run:

```bash
python scripts/ingest_data.py
```

### 4. Start the API

```bash
python main.py
```

The API runs at **http://localhost:7777**. Docs: http://localhost:7777/docs

## API Endpoints

- `GET /health` — Health check
- `GET /stats` — Platform statistics
- `POST /teams/youth-matching-platform/runs` — Run the matching team (non-streaming)
- `POST /teams/youth-matching-platform/runs` with `stream=true` — Run with streaming

## Candidate Response Format (JSON)

For candidate search/ranking requests, the backend is configured to return JSON (not markdown) so the frontend can render profile cards:

```json
{
  "candidates": [
    {
      "name": "Asha Kumar",
      "title": "Junior Data Analyst",
      "location": "Bengaluru",
      "summary": "Entry-level analyst with internship project experience.",
      "match_score": "88%",
      "skills": ["Python", "SQL", "Excel"],
      "languages": ["English", "Hindi"]
    }
  ],
  "notes": "Candidates ranked by skill, availability, and location fit."
}
```

If no matches are found:

```json
{
  "candidates": [],
  "notes": "No matching candidates found"
}
```

## Architecture

- **Youth Search Agent**: Searches youth profiles and ranks candidates
- **Employer Agent**: Helps employers find talent and post jobs
- **Team**: Routes requests to the appropriate agent
- **Knowledge bases**: Youth profiles and employer opportunities in pgvector (hybrid search)
