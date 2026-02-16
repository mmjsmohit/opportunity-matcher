"""
Youth Opportunity Matching Platform — Complete Implementation

Agno-based multi-agent system with FastAPI and AgentOS.
Uses Supabase pgvector for knowledge storage.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Database URL - use Supabase connection string
DB_URL = os.getenv(
    "SUPABASE_DB_URL",
    "postgresql+psycopg://postgres:postgres@localhost:54322/postgres",
)

# Data paths (relative to backend folder)
BACKEND_DIR = Path(__file__).parent
DATA_DIR = BACKEND_DIR / "data"
YOUTH_CSV = DATA_DIR / "youth_profiles.csv"
EMPLOYER_CSV = DATA_DIR / "employer_data.csv"

CANDIDATE_JSON_SCHEMA_HINT = """
For candidate search or ranking requests, respond with valid JSON only (no markdown, no code fences) in this shape:
{
  "candidates": [
    {
      "name": "string",
      "title": "string",
      "location": "string",
      "summary": "string",
      "match_score": "string",
      "skills": ["string"],
      "languages": ["string"]
    }
  ],
  "notes": "optional short explanation"
}

If no good matches are found, return:
{"candidates": [], "notes": "No matching candidates found"}
"""


def create_knowledge_bases():
    """Create and optionally ingest youth and employer knowledge bases."""
    from agno.knowledge.knowledge import Knowledge
    from agno.knowledge.reader.field_labeled_csv_reader import FieldLabeledCSVReader
    from agno.vectordb.pgvector import PgVector, SearchType

    # Youth knowledge base
    youth_reader = FieldLabeledCSVReader(
        chunk_title="👤 Youth Profile",
        field_names=[
            "Name",
            "Age",
            "Skills",
            "Education",
            "Languages",
            "Career Aspirations",
            "Work Experience",
            "Availability",
            "Communication Skills",
            "Location",
        ],
        format_headers=True,
        skip_empty_fields=True,
        delimiter=",",
        encoding="utf-8",
    )

    youth_kb = Knowledge(
        vector_db=PgVector(
            table_name="youth_profiles",
            db_url=DB_URL,
            search_type=SearchType.hybrid,
        ),
    )

    if YOUTH_CSV.exists():
        youth_kb.add_content(path=str(YOUTH_CSV), reader=youth_reader)

    # Employer knowledge base
    employer_reader = FieldLabeledCSVReader(
        chunk_title="🏢 Employer Opportunity",
        field_names=[
            "Employer Name",
            "Trade/Sector",
            "Qualifications Required",
            "Location",
            "Position Type",
            "Contact",
        ],
        format_headers=True,
        skip_empty_fields=True,
        delimiter=",",
        encoding="utf-8",
    )

    employer_kb = Knowledge(
        vector_db=PgVector(
            table_name="employer_opportunities",
            db_url=DB_URL,
            search_type=SearchType.hybrid,
        ),
    )

    if EMPLOYER_CSV.exists():
        employer_kb.add_content(path=str(EMPLOYER_CSV), reader=employer_reader)

    return youth_kb, employer_kb


def create_agents(youth_kb, employer_kb):
    """Create the matching platform agents."""
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    from tools.youth_matching_tools import YouthMatchingTools

    from agno.db.postgres import PostgresDb

    db = PostgresDb(db_url=DB_URL)

    youth_search_agent = Agent(
        id="youth-search-agent",
        name="Youth Search Agent",
        role="Find and rank youth candidates matching employer requirements",
        model=OpenAIChat(id="gpt-4o-mini"),
        knowledge=youth_kb,
        search_knowledge=True,
        db=db,
        instructions=[
            "Search youth profiles using the knowledge base.",
            "Rank candidates by skills match, availability, and location fit.",
            "Each profile starts with '👤 Youth Profile'.",
            CANDIDATE_JSON_SCHEMA_HINT,
            "Do not include extra commentary outside the JSON object.",
        ],
        add_history_to_context=True,
        markdown=False,
    )

    employer_agent = Agent(
        id="employer-agent",
        name="Employer Agent",
        role="Help companies search talent and post job requirements",
        model=OpenAIChat(id="gpt-4o-mini"),
        knowledge=employer_kb,
        search_knowledge=True,
        tools=[YouthMatchingTools()],
        db=db,
        instructions=[
            "Help employers find youth talent and post new requirements.",
            "Employer data is labeled with '🏢 Employer Opportunity'.",
            "Use tools to post jobs and send match notifications.",
            "When user asks to find, shortlist, rank, or recommend candidates, return candidate results using the exact JSON format specified for the platform.",
            "For non-candidate tasks (for example posting jobs, statistics, or notifications), you may return concise plain text or tool JSON outputs.",
        ],
        add_history_to_context=True,
        markdown=False,
    )

    return youth_search_agent, employer_agent


def create_team(youth_search_agent, employer_agent):
    """Create the routing team."""
    from agno.models.openai import OpenAIChat
    from agno.team import Team

    platform_team = Team(
        id="youth-matching-platform",
        name="Youth Matching Platform",
        respond_directly=True,  # Route mode: return member response without synthesis
        determine_input_for_members=False,  # Pass user input unchanged to selected member
        model=OpenAIChat(id="gpt-4o"),
        members=[youth_search_agent, employer_agent],
        instructions=[
            "Route talent search queries to Youth Search Agent.",
            "Route employer/company queries to Employer Agent.",
            "For talent matching responses, ensure final output is valid JSON with a top-level 'candidates' array.",
        ],
        show_members_responses=True,
        markdown=False,
    )

    return platform_team


def create_app():
    """Create the FastAPI application with AgentOS."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from agno.os import AgentOS

    youth_kb, employer_kb = create_knowledge_bases()
    youth_search_agent, employer_agent = create_agents(youth_kb, employer_kb)
    platform_team = create_team(youth_search_agent, employer_agent)

    app = FastAPI(
        title="Youth Opportunity Matching Platform",
        version="1.0.0",
        description="AI-powered platform for matching youth with employment opportunities",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "running", "youth_profiles": 32, "employers": 37}

    @app.get("/stats")
    async def stats():
        return {"youth_profiles": 32, "employers": 37}

    agent_os = AgentOS(
        description="Youth Opportunity Matching Platform",
        agents=[youth_search_agent, employer_agent],
        teams=[platform_team],
        base_app=app,
    )

    app = agent_os.get_app()
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7777,
        reload=True,
    )
