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
import itertools
import logging
import re
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
2. Use LIMIT 50 on all queries to avoid large result sets
3. Use case-insensitive exact matching for short category names such as AI:
   toLower(n.name) = "ai". Use CONTAINS only for longer free-text search terms.
4. For pattern queries, use aggregation (count, collect) to summarize results
5. Return results as flat list of properties, not nested objects
6. For multi-hop queries, chain MATCH clauses
7. If counting or ranking, use ORDER BY and LIMIT
8. When asked about failed/dead startups, filter on c.status = "Dead"
9. When asked about successful startups, filter on c.status IN ["Active", "Public", "Acquired"]
10. Available status values: Active, Acquired, Public, Dead, Unknown
11. Preserve the user's boolean intent. "B2B SaaS" means companies matching both
    B2B AND SaaS, not either category.
12. team_size = 0 means unknown. Exclude it from employee-count comparisons.

## Examples:

Question: Who founded Stripe?
Cypher:
MATCH (f:Founder)-[:FOUNDED]->(c:Company {name: "Stripe"})
RETURN f.name AS founder, c.name AS company, c.yc_batch AS batch

Question: Which YC founders from India built fintech companies?
Cypher:
MATCH (f:Founder)-[:FOUNDED]->(c:Company)-[:HEADQUARTERED_IN]->(l:Location {country: "India"})
MATCH (c)-[:OPERATES_IN]->(ind:Industry)
WHERE toLower(ind.name) CONTAINS "fintech" OR toLower(ind.name) CONTAINS "finance" OR toLower(ind.name) CONTAINS "payments"
RETURN f.name AS founder, c.name AS company, c.yc_batch AS batch, ind.name AS industry
ORDER BY c.yc_batch

Question: Compare AI companies in San Francisco and New York.
Cypher:
MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
WHERE toLower(ind.name) IN ["ai", "artificial intelligence"]
MATCH (c)-[:HEADQUARTERED_IN]->(l:Location)
WHERE toLower(l.city) IN ["san francisco", "new york"]
RETURN l.city AS city, count(DISTINCT c) AS ai_company_count,
       collect(DISTINCT c.name)[0..10] AS sample_companies
ORDER BY ai_company_count DESC

Question: Which active B2B SaaS companies have fewer than 20 employees?
Cypher:
MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
WITH c, collect(toLower(ind.name)) AS industries
WHERE c.status = "Active"
  AND any(name IN industries WHERE name CONTAINS "b2b")
  AND any(name IN industries WHERE name CONTAINS "saas")
  AND c.team_size > 0 AND c.team_size < 20
RETURN DISTINCT c.name AS company, c.team_size AS team_size
ORDER BY c.team_size
LIMIT 50

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

Question: Which industries have the most failed (Dead) YC startups?
Cypher:
MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
WHERE c.status = "Dead"
RETURN ind.name AS industry, count(c) AS dead_count
ORDER BY dead_count DESC
LIMIT 20

Question: How many YC companies are Active vs Dead vs Acquired?
Cypher:
MATCH (c:Company)
RETURN c.status AS status, count(c) AS company_count
ORDER BY company_count DESC

Question: What are common patterns among failed YC startups?
Cypher:
MATCH (c:Company)
WHERE c.status = "Dead"
OPTIONAL MATCH (c)-[:OPERATES_IN]->(ind:Industry)
OPTIONAL MATCH (c)-[:HEADQUARTERED_IN]->(l:Location)
OPTIONAL MATCH (c)-[:PART_OF]->(b:Batch)
RETURN ind.name AS industry, l.country AS country, b.name AS batch,
       count(c) AS dead_count
ORDER BY dead_count DESC
LIMIT 50

Question: What is the breakdown of YC companies by stage?
Cypher:
MATCH (c:Company)
RETURN c.stage AS stage, count(c) AS company_count
ORDER BY company_count DESC

Question: Which YC companies were acquired and by whom?
Cypher:
MATCH (c:Company)-[:ACQUIRED_BY]->(acquirer:Company)
RETURN c.name AS company, acquirer.name AS acquired_by, c.yc_batch AS batch
ORDER BY c.yc_batch DESC
LIMIT 100

Question: Who are the most connected founders across YC?
Cypher:
MATCH (f:Founder)-[:FOUNDED]->(c:Company)
WITH f, count(c) AS companies_founded, collect(c.name) AS companies
WHERE companies_founded > 1
RETURN f.name AS founder, companies_founded, companies
ORDER BY companies_founded DESC
LIMIT 20

Question: Which companies raised Series A funding?
Cypher:
MATCH (c:Company)-[:RAISED]->(fe:FundingEvent)
WHERE toLower(fe.round) CONTAINS "series a"
RETURN c.name AS company, fe.year AS year, fe.amount AS amount, c.yc_batch AS batch
ORDER BY fe.year DESC
LIMIT 50

Question: What is Stripe's funding history?
Cypher:
MATCH (c:Company)-[:RAISED]->(fe:FundingEvent)
WHERE toLower(c.name) CONTAINS toLower("stripe")
RETURN c.name AS company, fe.round AS round, 
       fe.year AS year, fe.amount AS amount
ORDER BY fe.year

Question: Which companies raised the most funding rounds?
Cypher:
MATCH (c:Company)-[:RAISED]->(fe:FundingEvent)
WITH c, count(fe) AS rounds
WHERE rounds > 2
RETURN c.name AS company, rounds
ORDER BY rounds DESC
LIMIT 20

Question: Which technologies are most used by YC companies?
Cypher:
MATCH (c:Company)-[:USES]->(t:Technology)
RETURN t.name AS technology, count(c) AS company_count
ORDER BY company_count DESC
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


def enforce_result_limit(cypher: str) -> str:
    """Ensure the final result-producing clause is capped in Neo4j."""
    clean_cypher = cypher.rstrip().rstrip(";")
    trailing_limit = re.search(
        r"\bLIMIT\s+(\d+)\s*$",
        clean_cypher,
        flags=re.IGNORECASE,
    )
    if trailing_limit:
        limit = min(int(trailing_limit.group(1)), MAX_RESULTS)
        return (
            f"{clean_cypher[:trailing_limit.start()]}"
            f"LIMIT {limit}"
        )
    return f"{clean_cypher}\nLIMIT {MAX_RESULTS}"


def executable_cypher_text(cypher: str) -> str:
    """Mask literals and comments before checking Cypher operators."""
    output = []
    index = 0
    state = None
    escaped = False

    while index < len(cypher):
        character = cypher[index]
        following = cypher[index + 1] if index + 1 < len(cypher) else ""

        if state in {"'", '"', "`"}:
            output.append(" ")
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == state:
                state = None
            index += 1
            continue
        if state == "line_comment":
            output.append("\n" if character == "\n" else " ")
            if character == "\n":
                state = None
            index += 1
            continue
        if state == "block_comment":
            output.append(" ")
            if character == "*" and following == "/":
                output.append(" ")
                state = None
                index += 2
            else:
                index += 1
            continue

        if character in {"'", '"', "`"}:
            output.append(" ")
            state = character
            index += 1
        elif character == "/" and following == "/":
            output.extend((" ", " "))
            state = "line_comment"
            index += 2
        elif character == "/" and following == "*":
            output.extend((" ", " "))
            state = "block_comment"
            index += 2
        else:
            output.append(character)
            index += 1

    return "".join(output)


def validate_cypher_semantics(question: str, cypher: str) -> None:
    """Reject common queries that execute successfully but change user intent."""
    if re.search(
        r"\bUNION\b",
        executable_cypher_text(cypher),
        flags=re.IGNORECASE,
    ):
        raise ValueError("Use one result-producing query instead of UNION.")

    if re.search(r"\bAI\b", question, flags=re.IGNORECASE) and re.search(
        r"\bCONTAINS\s+(?:toLower\()?['\"]ai['\"]",
        cypher,
        flags=re.IGNORECASE,
    ):
        raise ValueError(
            'Use an exact category match for AI; CONTAINS "ai" matches unrelated words.'
        )

    if re.search(r"\bB2B\s+SaaS\b", question, flags=re.IGNORECASE):
        lower_cypher = cypher.lower()
        if "b2b" not in lower_cypher or "saas" not in lower_cypher:
            raise ValueError("B2B SaaS requires explicit filters for both categories.")

        category_property = r"(?:\b\w+\.\w+|toLower\([^)]*\))"
        b2b_predicate = (
            rf"{category_property}\s*(?:(?:=|CONTAINS)\s*['\"]b2b['\"]|"
            rf"IN\s*\[[^\]]*['\"]b2b['\"][^\]]*\])"
        )
        saas_predicate = (
            rf"{category_property}\s*(?:(?:=|CONTAINS)\s*['\"]saas['\"]|"
            rf"IN\s*\[[^\]]*['\"]saas['\"][^\]]*\])"
        )
        category_or = (
            rf"{b2b_predicate}.{{0,160}}\bOR\b.{{0,160}}{saas_predicate}|"
            rf"{saas_predicate}.{{0,160}}\bOR\b.{{0,160}}{b2b_predicate}"
        )
        if re.search(category_or, cypher, flags=re.IGNORECASE | re.DOTALL):
            raise ValueError("B2B SaaS requires both categories, not an OR condition.")

        category_in = (
            rf"{category_property}\s+IN\s*\[[^\]]*['\"]b2b['\"]"
            rf"[^\]]*['\"]saas['\"][^\]]*\]"
        )
        reverse_category_in = (
            rf"{category_property}\s+IN\s*\[[^\]]*['\"]saas['\"]"
            rf"[^\]]*['\"]b2b['\"][^\]]*\]"
        )
        if re.search(
            f"{category_in}|{reverse_category_in}",
            cypher,
            flags=re.IGNORECASE,
        ):
            raise ValueError("B2B SaaS requires both categories, not IN membership.")

        literal_membership_or = (
            r"['\"]b2b['\"]\s+IN\s+\w+.{0,120}\bOR\b.{0,120}"
            r"['\"]saas['\"]\s+IN\s+\w+|"
            r"['\"]saas['\"]\s+IN\s+\w+.{0,120}\bOR\b.{0,120}"
            r"['\"]b2b['\"]\s+IN\s+\w+"
        )
        if re.search(
            literal_membership_or,
            cypher,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            raise ValueError("B2B SaaS requires both categories, not OR membership.")

    if re.search(r"(?:fewer|less)\s+than\s+\d+\s+employees", question, flags=re.IGNORECASE):
        has_upper_bound = re.search(r"team_size\s*<", cypher, flags=re.IGNORECASE)
        excludes_unknown = re.search(
            r"team_size\s*>\s*0|0\s*<\s*\w*\.?team_size",
            cypher,
            flags=re.IGNORECASE,
        )
        if has_upper_bound and not excludes_unknown:
            raise ValueError("Exclude unknown team sizes with c.team_size > 0.")

# ── Step 3: Query Execution ────────────────────────────────────────────────────

def execute_cypher(cypher: str, driver) -> list:
    """Execute Cypher and stream no more rows than synthesis can consume."""
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(record) for record in itertools.islice(result, MAX_RESULTS)]

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
- The provided data contains matching records. Never claim that there are no
  results or that the requested property is unavailable when a record contains it.
- If only part of the question can be answered, answer that part first, then
  identify the specific missing fields.
- Do NOT supplement with your own knowledge about companies or founders
- Do NOT make assumptions about data not present
- If a company's funding amount is unknown, say "amount not available"
- Never invent or estimate figures not in the data
"""


FALSE_NO_DATA_PATTERNS = (
    r"^\s*the data (?:provided )?does not contain enough information",
    r"^\s*based on the (?:provided )?data,? there (?:is|are) insufficient",
    r"there (?:is|are) insufficient information to determine",
    r"^\s*no results (?:were )?found",
    r"cannot determine .+ from the (?:provided )?data",
)


def has_substantive_results(results: list) -> bool:
    return any(
        value not in (None, "", [], {})
        for row in results
        for value in row.values()
    )


def falsely_claims_no_data(answer: str) -> bool:
    return no_data_claim_start(answer) is not None


def no_data_claim_start(answer: str):
    starts = []
    for pattern in FALSE_NO_DATA_PATTERNS:
        match = re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL)
        if match:
            starts.append(match.start())
    return min(starts) if starts else None


def answer_references_results(answer: str, results: list, question: str = "") -> bool:
    """Check whether an answer grounds a partial response in a returned value."""
    normalized_answer = answer.casefold()
    normalized_question = question.casefold()
    for row in results:
        for key, value in row.items():
            if str(key).casefold() in {"status", "stage"}:
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, str) and len(item.strip()) >= 3:
                    normalized_item = item.strip().casefold()
                    if (
                        normalized_item not in normalized_question
                        and normalized_item in normalized_answer
                    ):
                        return True
                elif isinstance(item, bool):
                    normalized_item = str(item).casefold()
                    if (
                        normalized_item not in normalized_question
                        and normalized_item in normalized_answer
                    ):
                        return True
                elif isinstance(item, (int, float)):
                    numeric = re.escape(str(item))
                    pattern = rf"(?<![\d.]){numeric}(?![\d.])"
                    if (
                        not re.search(pattern, question)
                        and re.search(pattern, answer)
                    ):
                        return True
    return False


def is_zero_count_aggregate(results: list, cypher: str = None) -> bool:
    """Recognize a single aggregate row that represents zero matching entities."""
    if (
        len(results) != 1
        or not results[0]
        or not cypher
        or not re.search(r"\bcount\s*\(", cypher, flags=re.IGNORECASE)
    ):
        return False

    populated_values = [
        value for value in results[0].values() if value not in (None, "", [], {})
    ]
    return bool(populated_values) and all(value == 0 for value in populated_values)


def format_grounded_results(results: list) -> str:
    """Render graph records directly when synthesis contradicts the data."""
    visible_results = results[:MAX_RESULTS]
    lines = [f"Found {len(results)} matching record{'s' if len(results) != 1 else ''}:"]

    for index, row in enumerate(visible_results, start=1):
        fields = []
        for key, value in row.items():
            if value in (None, "", [], {}):
                continue
            label = key.replace("_", " ").capitalize()
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            fields.append(f"{label}: {value}")
        lines.append(f"{index}. {'; '.join(fields)}")

    if len(results) > len(visible_results):
        lines.append(f"Showing the first {len(visible_results)} of {len(results)} records.")

    return "\n".join(lines)


def synthesize_answer(question: str, results: list, cypher: str = None) -> str:
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
These are {len(results)} matching records. Do not say there are no results or
that a populated field is unavailable.
Do not use any external knowledge."""

    answer = call_llm(SYNTHESIS_SYSTEM, user_prompt, max_tokens=2000)
    contradicts_results = (
        has_substantive_results(results)
        and not is_zero_count_aggregate(results, cypher)
        and falsely_claims_no_data(answer)
        and not answer_references_results(
            answer[:no_data_claim_start(answer)],
            results,
            question,
        )
    )
    if contradicts_results:
        logger.warning("Synthesis contradicted non-empty graph results; using grounded fallback")
        return format_grounded_results(results)
    return answer

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
                cypher = enforce_result_limit(cypher)
                validate_cypher_semantics(question, cypher)
                results = execute_cypher(cypher, driver)
                error = None
                break
            except Exception as e:
                error = str(e)
                logger.warning(f"Cypher attempt {attempt + 1} failed: {error}")

        if error:
            raise RuntimeError(
                f"Unable to generate and execute a valid Cypher query after "
                f"{MAX_CYPHER_RETRIES} attempts: {error}"
            )

        # Step 4 — Synthesize
        answer = synthesize_answer(question, results, cypher)

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