#!/usr/bin/env python3
"""
Ingest youth profiles and employer data into the vector knowledge bases.
Run this script after setting up Supabase and before starting the API.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DB_URL = os.getenv(
    "SUPABASE_DB_URL",
    "postgresql+psycopg://postgres:postgres@localhost:54322/postgres",
)

BACKEND_DIR = Path(__file__).parent.parent
YOUTH_CSV = BACKEND_DIR / "data" / "youth_profiles.csv"
EMPLOYER_CSV = BACKEND_DIR / "data" / "employer_data.csv"


def main():
    from agno.knowledge.knowledge import Knowledge
    from agno.knowledge.reader.field_labeled_csv_reader import FieldLabeledCSVReader
    from agno.vectordb.pgvector import PgVector, SearchType

    print("Ingesting data into knowledge bases...")

    if not YOUTH_CSV.exists():
        print(f"Warning: {YOUTH_CSV} not found. Skipping youth profiles.")
    else:
        youth_reader = FieldLabeledCSVReader(
            chunk_title="👤 Youth Profile",
            field_names=[
                "Name", "Age", "Skills", "Education", "Languages",
                "Career Aspirations", "Work Experience", "Availability",
                "Communication Skills", "Location",
            ],
            format_headers=True,
            skip_empty_fields=True,
        )
        youth_kb = Knowledge(
            vector_db=PgVector(
                table_name="youth_profiles",
                db_url=DB_URL,
                search_type=SearchType.hybrid,
            ),
        )
        youth_kb.add_content(path=str(YOUTH_CSV), reader=youth_reader)
        print(f"✓ Ingested youth profiles from {YOUTH_CSV}")

    if not EMPLOYER_CSV.exists():
        print(f"Warning: {EMPLOYER_CSV} not found. Skipping employer data.")
    else:
        employer_reader = FieldLabeledCSVReader(
            chunk_title="🏢 Employer Opportunity",
            field_names=[
                "Employer Name", "Trade/Sector", "Qualifications Required",
                "Location", "Position Type", "Contact",
            ],
            format_headers=True,
            skip_empty_fields=True,
        )
        employer_kb = Knowledge(
            vector_db=PgVector(
                table_name="employer_opportunities",
                db_url=DB_URL,
                search_type=SearchType.hybrid,
            ),
        )
        employer_kb.add_content(path=str(EMPLOYER_CSV), reader=employer_reader)
        print(f"✓ Ingested employer data from {EMPLOYER_CSV}")

    print("Done.")


if __name__ == "__main__":
    main()
