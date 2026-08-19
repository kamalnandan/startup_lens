"""
Neo4j Query Layer
Global Startup Intelligence Graph

Converts natural language questions to Cypher queries,
executes them against Neo4j, and synthesizes natural language answers.

Components:
1. Query classifier — local vs global
2. Cypher generator — GPT-4.1 converts question to Cypher
3. Query executor — runs Cypher against Neo4j
4. Answer synthesizer — GPT-4.1 converts results to natural language
"""

import json
import logging
import requests
from neo4j import GraphDatabase
from app_config import get_required_setting

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

NEO4J_URI = get_required_setting("NEO4J_URI")
NEO4J_USERNAME = get_required_setting("NEO4J_USERNAME")
NEO4J_PASSWORD = get_required_setting("NEO4J_PASSWORD")

AZURE_OPENAI_ENDPOINT = get_required_setting("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = get_required_setting("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = get_required_setting("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = get_required_setting("AZURE_OPENAI_API_KEY")

MAX_CYPHER_RETRIES = 3
MAX_RESULTS = 50

# ── LLM Call ──────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
    """Call GPT-4.1 with system + user prompt using API key auth."""
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        },
        json={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0
        },
        timeout=60
    )
    return response.json()["choices"][0]["message"]["content"].strip()

# ── Schema (cached in system prompt) ──────────────────────────────────────────

GRAPH_SCHEMA = """
You are an expert Neo4j Cypher query generator for a startup intelligence knowledge graph.

## Graph Schema

### Node Types and Properties:
- (Company {name, description, status, stage, team_size, yc_batch, filename})
  status values: Active, Acquired, Public, Dead, Unknown
  stage values: Early, Growth, Public, Unknown

- (Founder {name})
- (Investor {name})
- (Industry {name})
- (Location {city, country})
- (Technology {name})
- (Batch {name}) — e.g. "Winter 2009", "Summer 2021"
- (FundingEvent {round, year, amount, company})
  round values: Seed, Series A, Series B, Series C, Series D, IPO, etc.

### Relationship Types:
- (Founder)-[:FOUNDED]->(Company)
- (Founder)-[:CO_FOUNDED_WITH]->(Founder)
- (Investor)-[:INVESTED_IN]->(Company)
- (Company)-[:OPERATES_IN]->(Industry)
- (Company)-[:HEADQUARTERED_IN]->(Location)
- (Company)-[:USES]->(Technology)
- (Company)-[:PART_OF]->(Batch)
- (Company)-[:ACQUIRED_BY]->(Company)
- (Company)-[:RAISED]->(FundingEvent)
- (Company)-[:COMPETES_WITH]->(Company)

## Cypher Generation Rules:
1. Always return meaningful properties, not just IDs
2. Use LIMIT 100 on all queries to avoid large result sets
3. Use case-insensitive matching: toLower(n.name) CONTAINS toLower("value")
4. For pattern queries, use aggregation (count, collect) to summarize results
5. Return results as flat list of properties, not nested objects
6. For multi-hop queries, chain MATCH clauses
7. If counting or ranking, use ORDER BY and LIMIT

## Examples:

Question: Who founded Stripe?
Cypher:
MATCH (f:Founder)-[:FOUNDED]->(c:Company {name: "Stripe"})
RETURN f.name AS founder, c.name AS company, c.yc_batch AS batch

Question: Which investors backed both Stripe and Airbnb?
Cypher:
MATCH (i:Investor)-[:INVESTED_IN]->(c1:Company {name: "Stripe"})
MATCH (i)-[:INVESTED_IN]->(c2:Company {name: "Airbnb"})
RETURN i.name AS investor

Question: Which YC founders from India built fintech companies?
Cypher:
MATCH (f:Founder)-[:FOUNDED]->(c:Company)-[:HEADQUARTERED_IN]->(l:Location {country: "India"})
MATCH (c)-[:OPERATES_IN]->(ind:Industry)
WHERE toLower(ind.name) CONTAINS "fintech" OR toLower(ind.name) CONTAINS "finance" OR toLower(ind.name) CONTAINS "payments"
RETURN f.name AS founder, c.name AS company, c.yc_batch AS batch, ind.name AS industry
ORDER BY c.yc_batch

Question: Which industries are most represented across YC companies?
Cypher:
MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
RETURN ind.name AS industry, count(c) AS company_count
ORDER BY company_count DESC
LIMIT 20

Question: Which investors have the most portfolio companies?
Cypher:
MATCH (i:Investor)-[:INVESTED_IN]->(c:Company)
RETURN i.name AS investor, count(c) AS portfolio_size, collect(c.name)[0..5] AS sample_companies
ORDER BY portfolio_size DESC
LIMIT 20

Question: What are common patterns among failed YC startups?
Cypher:
MATCH (c:Company)
WHERE c.status = "Dead"
OPTIONAL MATCH (c)-[:OPERATES_IN]->(ind:Industry)
OPTIONAL MATCH (c)-[:HEADQUARTERED_IN]->(l:Location)
RETURN c.name AS company, c.yc_batch AS batch,
       collect(DISTINCT ind.name) AS industries,
       l.country AS country
LIMIT 100

Question: Who are the most connected founders across YC?
Cypher:
MATCH (f:Founder)-[:FOUNDED]->(c:Company)
WITH f, count(c) AS companies_founded, collect(c.name) AS companies
WHERE companies_founded > 1
RETURN f.name AS founder, companies_founded, companies
ORDER BY companies_founded DESC
LIMIT 20

Question: How many funding rounds did GoCardless have?
Cypher:
MATCH (c:Company)-[:RAISED]->(fe:FundingEvent)
WHERE toLower(c.name) CONTAINS toLower("gocardless")
RETURN c.name AS company, count(fe) AS funding_rounds,
       collect(fe.round) AS rounds, collect(fe.year) AS years

Question: What is Stripe's funding history?
Cypher:
MATCH (c:Company)-[:RAISED]->(fe:FundingEvent)
WHERE toLower(c.name) CONTAINS toLower("stripe")
RETURN c.name AS company, fe.round AS round, 
       fe.year AS year, fe.amount AS amount
ORDER BY fe.year

Question: Which companies raised a Series B?
Cypher:
MATCH (c:Company)-[:RAISED]->(fe:FundingEvent)
WHERE toLower(fe.round) CONTAINS "series b"
RETURN c.name AS company, fe.year AS year, fe.amount AS amount
ORDER BY fe.year DESC
LIMIT 50

Question: Which companies raised the most funding rounds?
Cypher:
MATCH (c:Company)-[:RAISED]->(fe:FundingEvent)
WITH c, count(fe) AS rounds
WHERE rounds > 2
RETURN c.name AS company, rounds
ORDER BY rounds DESC
LIMIT 20


Note: When asked about external investors or VC portfolios, always exclude 
"Y Combinator" from results since it is the accelerator, not an external investor.
"""

# ── Step 1: Query Classification ───────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You classify startup research questions as LOCAL or GLOBAL.

LOCAL: About a specific named company, founder, or investor.
Examples: "Who founded Stripe?", "What does Airbnb do?", "Which batch was Razorpay in?"

GLOBAL: About patterns, trends, aggregations, or comparisons across many companies.
Examples: "Which industries dominate YC?", "Common failure patterns?", "Most active investors?"

Reply with only LOCAL or GLOBAL."""

def classify_query(question: str) -> str:
    result = call_llm(CLASSIFIER_SYSTEM, question, max_tokens=10)
    method = "local" if result.upper().startswith("LOCAL") else "global"
    logger.info(f"Query classified as: {method}")
    return method

# ── Step 2: Cypher Generation ──────────────────────────────────────────────────

CYPHER_SYSTEM = GRAPH_SCHEMA + """
Generate a single valid Cypher query for the question below.
Return ONLY the Cypher query — no explanation, no markdown, no backticks.
"""

def generate_cypher(question: str, error_context: str = None, previous_cypher: str = None) -> str:
    if error_context:
        user_prompt = f"""Question: {question}

Previous Cypher attempt failed with error: {error_context}
Previous Cypher: {previous_cypher}

Generate a corrected Cypher query."""
    else:
        user_prompt = f"Question: {question}"

    cypher = call_llm(CYPHER_SYSTEM, user_prompt, max_tokens=500)
    cypher = cypher.replace("```cypher", "").replace("```", "").strip()
    logger.info(f"Generated Cypher: {cypher}")
    return cypher

# ── Step 3: Query Execution ────────────────────────────────────────────────────

def execute_cypher(cypher: str, driver) -> list:
    """Execute Cypher query and return results as list of dicts."""
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(record) for record in result]

# ── Step 4: Answer Synthesis ───────────────────────────────────────────────────

# SYNTHESIS_SYSTEM = """You are an expert analyst of the startup ecosystem.
# You are given a user question and raw data from a knowledge graph of YC startups.
# Synthesize a comprehensive, insightful answer from the data.

# Rules:
# - Be specific — name actual companies, founders, investors from the data
# - Identify patterns and trends where relevant
# - Be concise but thorough
# - If data is empty, say so honestly
# - Do not make up information not present in the data
# """

SYNTHESIS_SYSTEM = """You are an analyst of the startup ecosystem.
You are given a user question and raw data from a knowledge graph.
Synthesize an answer STRICTLY from the provided data only.

CRITICAL RULES:
- ONLY use information present in the data provided
- If the data is insufficient, say so explicitly
- Do NOT supplement with your own knowledge about companies or founders
- Do NOT make assumptions about data not present
- If a company's funding amount is unknown, say "amount not available"
- Never invent or estimate figures not in the data
"""

def synthesize_answer(question: str, results: list) -> str:
    if not results:
        return "No results found for this query. The data may not contain enough information to answer this question."

    results_text = json.dumps(results[:MAX_RESULTS], indent=2)
#     user_prompt = f"""Question: {question}

# Data from knowledge graph:
# {results_text}

# Provide a comprehensive answer based on this data."""
    user_prompt = f"""Question: {question}

Data from knowledge graph:
{results_text}

IMPORTANT: Base your answer STRICTLY on the data above. 
If the data doesn't contain enough information, say so explicitly.
Do not use any external knowledge."""

    return call_llm(SYNTHESIS_SYSTEM, user_prompt, max_tokens=2000)

# ── Main Query Pipeline ────────────────────────────────────────────────────────

def query(question: str) -> dict:
    """
    Full query pipeline:
    1. Classify question
    2. Generate Cypher
    3. Execute against Neo4j
    4. Synthesize answer
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        # Step 1 — Classify
        method = classify_query(question)

        # Step 2 & 3 — Generate Cypher with retry on failure
        cypher = None
        results = []
        error = None

        for attempt in range(MAX_CYPHER_RETRIES):
            try:
                cypher = generate_cypher(
                    question,
                    error_context=error,
                    previous_cypher=cypher
                )
                results = execute_cypher(cypher, driver)
                error = None
                break
            except Exception as e:
                error = str(e)
                logger.warning(f"Cypher attempt {attempt + 1} failed: {error}")

        # Step 4 — Synthesize
        answer = synthesize_answer(question, results)

        return {
            "question": question,
            "method": method,
            "cypher": cypher,
            "cypher_result": results,
            "result_count": len(results),
            "answer": answer
        }

    finally:
        driver.close()


# ── CLI for testing ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "Who founded Stripe and what does it do?",
        "Which investors backed both Stripe and Airbnb?",
        "Which industries are most represented across YC companies?",
        "Which YC founders from India built fintech companies?",
        "Who are the most connected founders across YC?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("="*60)

        result = query(question)

        print(f"Method: {result['method']}")
        print(f"Cypher: {result['cypher']}")
        print(f"Results: {result['result_count']} records")
        print(f"\nAnswer:\n{result['answer']}")