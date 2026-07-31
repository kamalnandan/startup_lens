"""
Neo4j Data Loader
Global Startup Intelligence Graph

Reads enriched YC company files, extracts entities and relationships
using GPT-4.1, and loads them into Neo4j.

Features:
- Resumable — tracks progress in a JSON file
- Handles errors gracefully
- Uses API key authentication for Azure OpenAI
- Sends full file content in one call
- Normalized entity names to reduce duplicates
"""

import os
import json
import time
import logging
import requests
from pathlib import Path
from neo4j import GraphDatabase
from app_config import get_required_setting

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_DIR = "input_enriched"  # change to "input_enriched" for full run
PROGRESS_FILE = "neo4j_load_progress.json"
LOG_FILE = "neo4j_load_log.txt"

NEO4J_URI = get_required_setting("NEO4J_URI")
NEO4J_USERNAME = get_required_setting("NEO4J_USERNAME")
NEO4J_PASSWORD = get_required_setting("NEO4J_PASSWORD")

AZURE_OPENAI_ENDPOINT = get_required_setting("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = get_required_setting("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = get_required_setting("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = get_required_setting("AZURE_OPENAI_API_KEY")

RATE_LIMIT_SECONDS = 0.5  # between files

# ── Progress Tracking ──────────────────────────────────────────────────────────

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"loaded": [], "failed": [], "skipped": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def log(message):
    logger.info(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# ── Entity Extraction ──────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are extracting structured startup data from a company profile.

Extract entities and relationships from the text below and return ONLY valid JSON.
No preamble, no markdown, no explanation — just the JSON object.

Return this exact structure:
{
  "company": {
    "name": "company name",
    "description": "what the company does in one sentence",
    "status": "Active|Acquired|Public|Dead|Unknown",
    "stage": "Early|Growth|Public|Unknown",
    "team_size": 0,
    "yc_batch": "Winter 2009 or null"
  },
  "founders": ["founder name 1", "founder name 2"],
  "investors": ["investor name 1", "investor name 2"],
  "industries": ["industry 1", "industry 2"],
  "locations": [{"city": "city name", "country": "country name"}],
  "technologies": ["technology 1", "technology 2"],
  "acquisitions": [{"acquirer": "company name", "year": 2020, "amount": "1B or unknown"}],
  "funding_rounds": [{"round": "Series A", "amount": "10M or unknown", "year": 2015}],
  "competitors": ["competitor 1", "competitor 2"]
}

Extraction rules:
- Use null for unknown values EXCEPT for amount and round fields — use "unknown" string instead
- Keep lists empty [] if no data found
- Extract only what is explicitly mentioned — do not infer
- For team_size use integer or 0 if unknown
- For city use "Unknown" string if not mentioned
- For year use 0 if unknown
- For status: Active=still operating, Acquired=bought by another company, Public=listed on stock exchange, Dead=shut down

Entity normalization rules — CRITICAL, always follow these:
- FOUNDERS: Always use full official name. "P. Collison" → "Patrick Collison", "Chesky" → "Brian Chesky"
- INVESTORS: Always use full official name. "Sequoia" → "Sequoia Capital", "a16z" → "Andreessen Horowitz", "Tiger" → "Tiger Global"
- Y COMBINATOR: Always use exactly "Y Combinator" — never "YC", "YC Accelerator", "Y-Combinator", "YC (Y Combinator)"
- COMPANIES: Use the most widely known official name. "FB" → "Facebook", "GOOG" → "Google"
- Remove duplicate entities — if the same person/company appears twice with different names, include only once with the full official name

Text:
{text}"""


def extract_entities(text: str, company_name: str) -> dict:
    """Use GPT-4.1 to extract entities from full file content."""
    try:
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

        # Send full file content — GPT-4.1 has 128K context window
        prompt = EXTRACTION_PROMPT.replace("{text}", text)

        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "api-key": AZURE_OPENAI_API_KEY
            },
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0,
                "response_format": {"type": "json_object"}
            },
            timeout=60
        )

        response_json = response.json()

        if "error" in response_json:
            log(f"  API error for {company_name}: {response_json['error']}")
            return None

        content = response_json["choices"][0]["message"]["content"]
        return json.loads(content)

    except Exception as e:
        log(f"  Extraction failed for {company_name}: {e}")
        return None

# ── Neo4j Loading ──────────────────────────────────────────────────────────────

def load_company(session, data: dict, filename: str):
    """Load a company and all its relationships into Neo4j."""

    company = data.get("company", {})
    company_name = company.get("name", filename)

    if not company_name:
        return

    # Create Company node
    session.run("""
        MERGE (c:Company {name: $name})
        SET c.description = $description,
            c.status = $status,
            c.stage = $stage,
            c.team_size = $team_size,
            c.yc_batch = $yc_batch,
            c.filename = $filename
    """,
        name=company_name,
        description=company.get("description", ""),
        status=company.get("status", "Unknown"),
        stage=company.get("stage", "Unknown"),
        team_size=company.get("team_size", 0),
        yc_batch=company.get("yc_batch"),
        filename=filename
    )

    # Create Founder nodes and relationships
    founders = data.get("founders", [])
    for founder_name in founders:
        if not founder_name:
            continue
        session.run("""
            MERGE (f:Founder {name: $name})
            WITH f
            MATCH (c:Company {name: $company})
            MERGE (f)-[:FOUNDED]->(c)
        """, name=founder_name, company=company_name)

    # Create co-founded relationships between founders
    if len(founders) > 1:
        for i in range(len(founders)):
            for j in range(i + 1, len(founders)):
                session.run("""
                    MERGE (f1:Founder {name: $name1})
                    MERGE (f2:Founder {name: $name2})
                    MERGE (f1)-[:CO_FOUNDED_WITH]->(f2)
                """, name1=founders[i], name2=founders[j])

    # Create Investor nodes and relationships
    for investor_name in data.get("investors", []):
        if not investor_name:
            continue
        session.run("""
            MERGE (i:Investor {name: $name})
            WITH i
            MATCH (c:Company {name: $company})
            MERGE (i)-[:INVESTED_IN]->(c)
        """, name=investor_name, company=company_name)

    # Create Industry nodes and relationships
    for industry_name in data.get("industries", []):
        if not industry_name:
            continue
        session.run("""
            MERGE (ind:Industry {name: $name})
            WITH ind
            MATCH (c:Company {name: $company})
            MERGE (c)-[:OPERATES_IN]->(ind)
        """, name=industry_name, company=company_name)

    # Create Location nodes and relationships
    for loc in data.get("locations", []):
        if not loc or not loc.get("country"):
            continue
        city = loc.get("city") or "Unknown"
        country = loc.get("country", "Unknown")
        session.run("""
            MERGE (l:Location {city: $city, country: $country})
            WITH l
            MATCH (c:Company {name: $company})
            MERGE (c)-[:HEADQUARTERED_IN]->(l)
        """,
            city=city,
            country=country,
            company=company_name
        )

    # Create Technology nodes and relationships
    for tech_name in data.get("technologies", []):
        if not tech_name:
            continue
        session.run("""
            MERGE (t:Technology {name: $name})
            WITH t
            MATCH (c:Company {name: $company})
            MERGE (c)-[:USES]->(t)
        """, name=tech_name, company=company_name)

    # Create YC Batch node and relationship
    yc_batch = company.get("yc_batch")
    if yc_batch:
        session.run("""
            MERGE (b:Batch {name: $name})
            WITH b
            MATCH (c:Company {name: $company})
            MERGE (c)-[:PART_OF]->(b)
        """, name=yc_batch, company=company_name)

    # Create Acquisition relationships
    for acq in data.get("acquisitions", []):
        if not acq or not acq.get("acquirer"):
            continue
        session.run("""
            MERGE (acquirer:Company {name: $acquirer})
            WITH acquirer
            MATCH (c:Company {name: $company})
            MERGE (c)-[r:ACQUIRED_BY]->(acquirer)
            SET r.year = $year, r.amount = $amount
        """,
            acquirer=acq.get("acquirer", "Unknown"),
            company=company_name,
            year=acq.get("year") or 0,
            amount=acq.get("amount") or "unknown"
        )

    # Create Funding Round nodes
    for round_data in data.get("funding_rounds", []):
        if not round_data:
            continue
        session.run("""
            MATCH (c:Company {name: $company})
            MERGE (fe:FundingEvent {round: $round, year: $year, company: $company})
            MERGE (c)-[r:RAISED]->(fe)
            SET r.amount = $amount
        """,
            company=company_name,
            round=round_data.get("round") or "Unknown",
            amount=round_data.get("amount") or "unknown",
            year=round_data.get("year") or 0
        )

    # Create Competitor relationships
    for competitor_name in data.get("competitors", []):
        if not competitor_name:
            continue
        session.run("""
            MERGE (comp:Company {name: $name})
            WITH comp
            MATCH (c:Company {name: $company})
            MERGE (c)-[:COMPETES_WITH]->(comp)
        """, name=competitor_name, company=company_name)

# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_loader():
    progress = load_progress()
    processed = set(progress["loaded"] + progress["failed"] + progress["skipped"])

    files = sorted(Path(INPUT_DIR).glob("*.txt"))
    total = len(files)

    log(f"\n{'='*60}")
    log(f"Neo4j Data Loader")
    log(f"Input directory: {INPUT_DIR}")
    log(f"Total files: {total}")
    log(f"Already processed: {len(processed)}")
    log(f"Remaining: {total - len(processed)}")
    log(f"{'='*60}\n")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Create indexes for fast lookups
    with driver.session() as session:
        session.run("CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)")
        session.run("CREATE INDEX founder_name IF NOT EXISTS FOR (f:Founder) ON (f.name)")
        session.run("CREATE INDEX investor_name IF NOT EXISTS FOR (i:Investor) ON (i.name)")
        session.run("CREATE INDEX batch_name IF NOT EXISTS FOR (b:Batch) ON (b.name)")
        session.run("CREATE INDEX industry_name IF NOT EXISTS FOR (ind:Industry) ON (ind.name)")
        log("Indexes created/verified")

    for i, filepath in enumerate(files):
        filename = filepath.name

        if filename in processed:
            print(f"[{i+1}/{total}] Skipping: {filename}")
            continue

        try:
            text = filepath.read_text(encoding="utf-8")
            company_name = filename.replace(".txt", "").replace("-", " ").title()

            print(f"[{i+1}/{total}] Processing: {company_name} ({len(text):,} chars)")

            data = extract_entities(text, company_name)
            if not data:
                progress["failed"].append(filename)
                save_progress(progress)
                continue

            with driver.session() as session:
                load_company(session, data, filename)

            progress["loaded"].append(filename)
            save_progress(progress)
            log(f"  ✓ Loaded: {company_name}")

        except Exception as e:
            log(f"  ! Failed: {filename} — {e}")
            progress["failed"].append(filename)
            save_progress(progress)

        time.sleep(RATE_LIMIT_SECONDS)

    driver.close()

    log(f"\n{'='*60}")
    log(f"Loading Complete!")
    log(f"✓ Loaded:  {len(progress['loaded'])}")
    log(f"! Failed:  {len(progress['failed'])}")
    log(f"- Skipped: {len(progress['skipped'])}")
    log(f"{'='*60}\n")

if __name__ == "__main__":
    run_loader()