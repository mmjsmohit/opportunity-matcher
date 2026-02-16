# Youth Opportunity Matching Platform

AI-powered platform for matching youth with employment opportunities, built with Agno, FastAPI, Next.js, and AI Elements.

## Project structure

```
opportunity-matcher/
├── backend/          # Agno + FastAPI (port 7777)
│   ├── main.py       # FastAPI app with AgentOS
│   ├── tools/        # Custom YouthMatchingTools
│   ├── data/         # CSV data files
│   └── scripts/      # Data ingestion
└── frontend/         # Next.js + AI Elements (port 3000)
    └── app/          # Pages and API routes
```

## Quick start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with SUPABASE_DB_URL and OPENAI_API_KEY
python scripts/ingest_data.py
python main.py
```

### 2. Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

### 3. Environment

Create `frontend/.env.local`:

```
AGNO_API_URL=http://localhost:7777
```

## Features

- **Multi-agent matching**: Route-mode team with Youth Search and Employer agents
- **RAG with pgvector**: Hybrid search over youth profiles and employer opportunities
- **Custom tools**: Post jobs, send match notifications, get statistics
- **Chat UI**: AI Elements components with streaming support
