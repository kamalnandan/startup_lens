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
AI_CATEGORY_ALIASES = frozenset({"ai", "artificial intelligence"})
CANONICAL_AI_CATEGORY = "Artificial Intelligence"
CATEGORY_RESULT_FIELDS = frozenset(
    {"category", "categories", "industry", "industries", "technology", "technologies"}
)
CYPHER_STRING_LITERAL_PATTERN = (
    r"""(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')"""
)


def cypher_string_literals(text: str) -> list[str]:
    """Extract quoted Cypher string values while respecting their quote type."""
    return [
        match.group(0)[1:-1]
        for match in re.finditer(CYPHER_STRING_LITERAL_PATTERN, text)
    ]

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
3. Treat AI and Artificial Intelligence as the same category. For any AI filter,
   use case-insensitive exact alias matching:
   toLower(n.name) IN ["ai", "artificial intelligence"].
   Never use CONTAINS or match only one alias.
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
13. Return the properties used to satisfy filters so the answer can explain why
    each record matched (for example, t.name AS technology and ind.name AS industry).
14. Batch names use full forms such as "Winter 2023" and "Summer 2023", never
    abbreviations such as "W23" or "S23".
15. Whenever counting Company nodes, use count(DISTINCT c) to avoid duplicate
    companies introduced by relationship traversal.
16. When filtering company records by status, return c.status AS status.
17. When grouping or ranking Industry or Technology categories, combine the AI
    aliases under one canonical label before aggregation:
    CASE WHEN toLower(n.name) IN ["ai", "artificial intelligence"]
         THEN "Artificial Intelligence" ELSE n.name END AS category.
18. Match user-supplied entity names and locations case-insensitively. This
    includes Company, Founder, Investor, Industry, and Technology names, plus
    Location city and country. Lowercase both the property and literals:
    toLower(c.name) IN ["stripe", "razorpay"].

## Examples:

Question: Who founded Stripe?
Cypher:
MATCH (f:Founder)-[:FOUNDED]->(c:Company)
WHERE toLower(c.name) = "stripe"
RETURN f.name AS founder, c.name AS company, c.yc_batch AS batch

Question: Who founded Stripe and Razorpay?
Cypher:
MATCH (f:Founder)-[:FOUNDED]->(c:Company)
WHERE toLower(c.name) IN ["stripe", "razorpay"]
RETURN f.name AS founder, c.name AS company, c.yc_batch AS batch
ORDER BY c.name, founder

Question: Which YC founders from India built fintech companies?
Cypher:
MATCH (f:Founder)-[:FOUNDED]->(c:Company)-[:HEADQUARTERED_IN]->(l:Location)
MATCH (c)-[:OPERATES_IN]->(ind:Industry)
WHERE toLower(l.country) = "india"
  AND toLower(ind.name) IN ["fintech", "finance", "payments"]
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
RETURN DISTINCT c.name AS company, c.team_size AS team_size,
       c.status AS status
ORDER BY c.team_size
LIMIT 50

Question: Which YC companies use Python and operate in fintech?
Cypher:
MATCH (c:Company)-[:USES]->(t:Technology)
WHERE toLower(t.name) = "python"
MATCH (c)-[:OPERATES_IN]->(ind:Industry)
WHERE toLower(ind.name) IN ["fintech", "finance", "payments"]
RETURN DISTINCT c.name AS company, c.yc_batch AS batch,
       t.name AS technology, ind.name AS industry
ORDER BY c.yc_batch
LIMIT 50

Question: Which industries are most represented across YC companies?
Cypher:
MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
RETURN CASE WHEN toLower(ind.name) IN ["ai", "artificial intelligence"]
            THEN "Artificial Intelligence" ELSE ind.name END AS industry,
       count(DISTINCT c) AS company_count
ORDER BY company_count DESC
LIMIT 20

Question: Which investors have the most portfolio companies?
Cypher:
MATCH (i:Investor)-[:INVESTED_IN]->(c:Company)
RETURN i.name AS investor, count(DISTINCT c) AS portfolio_size, collect(DISTINCT c.name)[0..5] AS sample_companies
ORDER BY portfolio_size DESC
LIMIT 20

Question: Which industries have the most failed (Dead) YC startups?
Cypher:
MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
WHERE c.status = "Dead"
RETURN CASE WHEN toLower(ind.name) IN ["ai", "artificial intelligence"]
            THEN "Artificial Intelligence" ELSE ind.name END AS industry,
       count(DISTINCT c) AS dead_count
ORDER BY dead_count DESC
LIMIT 20

Question: How many YC companies are Active vs Dead vs Acquired?
Cypher:
MATCH (c:Company)
RETURN c.status AS status, count(DISTINCT c) AS company_count
ORDER BY company_count DESC

Question: What are common patterns among failed YC startups?
Cypher:
MATCH (c:Company)
WHERE c.status = "Dead"
OPTIONAL MATCH (c)-[:OPERATES_IN]->(ind:Industry)
OPTIONAL MATCH (c)-[:HEADQUARTERED_IN]->(l:Location)
OPTIONAL MATCH (c)-[:PART_OF]->(b:Batch)
RETURN CASE WHEN toLower(ind.name) IN ["ai", "artificial intelligence"]
            THEN "Artificial Intelligence" ELSE ind.name END AS industry,
       l.country AS country, b.name AS batch,
       count(DISTINCT c) AS dead_count
ORDER BY dead_count DESC
LIMIT 50

Question: What is the breakdown of YC companies by stage?
Cypher:
MATCH (c:Company)
RETURN c.stage AS stage, count(DISTINCT c) AS company_count
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
WITH f, count(DISTINCT c) AS companies_founded, collect(DISTINCT c.name) AS companies
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
RETURN CASE WHEN toLower(t.name) IN ["ai", "artificial intelligence"]
            THEN "Artificial Intelligence" ELSE t.name END AS technology,
       count(DISTINCT c) AS company_count
ORDER BY company_count DESC
LIMIT 20

Note: When asked about external investors or VC portfolios, always exclude
"Y Combinator" case-insensitively with toLower(i.name) <> "y combinator",
since it is the accelerator, not an external investor.
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


def expand_batch_abbreviations(question: str) -> str:
    """Expand YC batch aliases to the names stored in the graph."""
    seasons = {"w": "Winter", "s": "Summer"}

    def replace_batch(match):
        season = seasons[match.group(1).casefold()]
        year = 2000 + int(match.group(2))
        return f"{season} {year}"

    return re.sub(
        r"\b([WS])(\d{2})\b",
        replace_batch,
        question,
        flags=re.IGNORECASE,
    )


def generate_cypher(question: str, error_context: str = None, previous_cypher: str = None) -> str:
    generation_question = expand_batch_abbreviations(question)
    if error_context:
        user_prompt = f"""Question: {generation_question}

Previous Cypher attempt failed with error: {error_context}
Previous Cypher: {previous_cypher}

Generate a corrected Cypher query."""
    else:
        user_prompt = f"Question: {generation_question}"

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

    query_parts = re.split(
        r"\bRETURN\b",
        cypher,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    filtering_cypher = query_parts[0]
    return_clause = query_parts[1] if len(query_parts) > 1 else ""
    company_variables = set(
        re.findall(
            r"\(\s*(\w+)\s*:\s*Company\b",
            cypher,
            flags=re.IGNORECASE,
        )
    )
    case_insensitive_fields = {
        "Company": ("name",),
        "Founder": ("name",),
        "Investor": ("name",),
        "Industry": ("name",),
        "Technology": ("name",),
        "Location": ("city", "country"),
    }
    entity_variables = re.findall(
        r"\(\s*(\w+)\s*:\s*(Company|Founder|Investor|Industry|Technology|Location)\b",
        cypher,
        flags=re.IGNORECASE,
    )
    normalized_fields = {
        label.casefold(): fields
        for label, fields in case_insensitive_fields.items()
    }
    for variable, label in entity_variables:
        for field in normalized_fields[label.casefold()]:
            raw_property_filter = re.search(
                (
                    rf"\b{re.escape(variable)}\.{re.escape(field)}\b\s*"
                    r"(?:=|<>|!=|\bIN\b|\bCONTAINS\b|"
                    r"\bSTARTS\s+WITH\b|\bENDS\s+WITH\b)"
                ),
                filtering_cypher,
                flags=re.IGNORECASE,
            )
            reversed_raw_filter = re.search(
                (
                    r"['\"][^'\"]+['\"]\s*(?:=|<>|!=|\bCONTAINS\b|"
                    r"\bSTARTS\s+WITH\b|\bENDS\s+WITH\b)\s*"
                    rf"\b{re.escape(variable)}\.{re.escape(field)}\b"
                ),
                filtering_cypher,
                flags=re.IGNORECASE,
            )
            property_map_filter = re.search(
                (
                    rf"\(\s*{re.escape(variable)}\s*:\s*{re.escape(label)}\s*"
                    rf"\{{[^}}]*\b{re.escape(field)}\s*:"
                ),
                filtering_cypher,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if raw_property_filter or reversed_raw_filter or property_map_filter:
                raise ValueError(
                    f"Match {label}.{field} case-insensitively with "
                    f"toLower({variable}.{field}) and lowercase literals."
                )
            for values in re.findall(
                (
                    rf"toLower\(\s*{re.escape(variable)}\.{re.escape(field)}\s*\)"
                    r"\s+IN\s*(\[[^\]]+\])"
                ),
                filtering_cypher,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                literals = cypher_string_literals(values)
                if not literals or any(value != value.lower() for value in literals):
                    raise ValueError(
                        f"Use lowercase literals with "
                        f"toLower({variable}.{field}) IN."
                    )
            for literal_token in re.findall(
                (
                    rf"toLower\(\s*{re.escape(variable)}\.{re.escape(field)}\s*\)"
                    r"\s*(?:=|<>|!=|\bCONTAINS\b|\bSTARTS\s+WITH\b|"
                    rf"\bENDS\s+WITH\b)\s*({CYPHER_STRING_LITERAL_PATTERN})"
                ),
                filtering_cypher,
                flags=re.IGNORECASE,
            ):
                literal = literal_token[1:-1]
                if literal != literal.lower():
                    raise ValueError(
                        f"Use a lowercase literal with "
                        f"toLower({variable}.{field})."
                    )
            for literal_token in re.findall(
                (
                    rf"({CYPHER_STRING_LITERAL_PATTERN})\s*"
                    r"(?:=|<>|!=|\bCONTAINS\b|\bSTARTS\s+WITH\b|"
                    r"\bENDS\s+WITH\b)\s*"
                    rf"toLower\(\s*{re.escape(variable)}\.{re.escape(field)}\s*\)"
                ),
                filtering_cypher,
                flags=re.IGNORECASE,
            ):
                literal = literal_token[1:-1]
                if literal != literal.lower():
                    raise ValueError(
                        f"Use a lowercase literal with "
                        f"toLower({variable}.{field})."
                    )
    for variable in company_variables:
        if re.search(
            (
                rf"\bcount\s*\(\s*{re.escape(variable)}"
                rf"(?:\.\w+)?\s*\)"
            ),
            cypher,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                f"Count companies with count(DISTINCT {variable}) to avoid duplicates."
            )
        if re.search(
            (
                rf"\bcount\s*\(\s*CASE\b.{{0,500}}?"
                rf"(?:\bTHEN|\bELSE)\s+{re.escape(variable)}"
                rf"(?:\s+ELSE\s+NULL)?\s+END\s*\)"
            ),
            cypher,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            raise ValueError(
                "Count conditional Company nodes with "
                f"count(DISTINCT CASE ... THEN {variable} END)."
            )
    company_count_intent = re.search(
        (
            r"(?:\b(?:how many|number of|count|most|compare)\b.{0,80}"
            r"\b(?:companies|startups)\b)"
            r"|(?:\b(?:company|startup)\s+count\b)"
        ),
        question,
        flags=re.IGNORECASE,
    ) or re.search(
        r"\bAS\s+(?:company|startup)_count\b",
        return_clause,
        flags=re.IGNORECASE,
    )
    if (
        company_variables
        and company_count_intent
        and re.search(r"\bcount\s*\(", cypher, flags=re.IGNORECASE)
        and not any(
            (
                re.search(
                    rf"\bcount\s*\(\s*DISTINCT\s+{re.escape(variable)}\s*\)",
                    cypher,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    (
                        rf"\bcount\s*\(\s*DISTINCT\s+CASE\b.{{0,500}}?"
                        rf"\bTHEN\s+{re.escape(variable)}"
                        rf"(?:\s+ELSE\s+NULL)?\s+END\s*\)"
                    ),
                    cypher,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            )
            for variable in company_variables
        )
    ):
        variable = sorted(company_variables)[0]
        raise ValueError(
            f"Count companies with count(DISTINCT {variable}) to avoid duplicates."
        )
    if (
        company_variables
        and company_count_intent
        and re.search(r"\bcount\s*\(\s*\*\s*\)", cypher, flags=re.IGNORECASE)
    ):
        variable = sorted(company_variables)[0]
        raise ValueError(
            f"Count companies with count(DISTINCT {variable}) to avoid duplicates."
        )

    returned_company_variables = {
        variable
        for variable in company_variables
        if re.search(
            rf"\b{re.escape(variable)}\.name\b",
            return_clause,
            flags=re.IGNORECASE,
        )
    }
    for variable in company_variables:
        projected_name_aliases = re.findall(
            rf"\b{re.escape(variable)}\.name\s+AS\s+(\w+)\b",
            filtering_cypher,
            flags=re.IGNORECASE,
        )
        if any(
            re.search(
                rf"\b{re.escape(alias)}\b",
                return_clause,
                flags=re.IGNORECASE,
            )
            for alias in projected_name_aliases
        ):
            returned_company_variables.add(variable)
    status_filtered_variables = set(
        re.findall(
            r"\b(\w+)\.status\b",
            filtering_cypher,
            flags=re.IGNORECASE,
        )
    )
    status_filtered_variables.update(
        variable
        for variable in company_variables
        if re.search(
            (
                rf"\(\s*{re.escape(variable)}\s*:\s*Company\s*"
                rf"\{{[^}}]*\bstatus\s*:"
            ),
            filtering_cypher,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    for variable in returned_company_variables & status_filtered_variables:
        direct_status = re.search(
            rf"\b{re.escape(variable)}\.status\s+AS\s+status\b",
            return_clause,
            flags=re.IGNORECASE,
        )
        projected_status = re.search(
            rf"\b{re.escape(variable)}\.status\s+AS\s+status\b",
            filtering_cypher,
            flags=re.IGNORECASE,
        ) and re.search(r"\bstatus\b", return_clause, flags=re.IGNORECASE)
        if not direct_status and not projected_status:
            raise ValueError("Return the filtered company status AS status.")

    batch_aliases = re.findall(r"\b([WS])(\d{2})\b", question, flags=re.IGNORECASE)
    for season_code, short_year in batch_aliases:
        season = "Winter" if season_code.casefold() == "w" else "Summer"
        full_batch = f"{season} {2000 + int(short_year)}"
        abbreviation = f"{season_code}{short_year}"
        batch_filter = (
            rf"(?:\b\w+\.)?(?:name|yc_batch)\s*(?:=|IN)\s*"
            rf"(?:\[[^\]]{{0,200}})?['\"]{re.escape(full_batch)}['\"]|"
            rf"(?:name|yc_batch)\s*:\s*['\"]{re.escape(full_batch)}['\"]"
        )
        if re.search(
            rf"['\"]{re.escape(abbreviation)}['\"]",
            cypher,
            flags=re.IGNORECASE,
        ) or not re.search(
            batch_filter,
            filtering_cypher,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            raise ValueError(
                f'Use the stored batch name "{full_batch}" instead of "{abbreviation}".'
            )

    category_variables = set(
        re.findall(
            r"\((\w+)\s*:\s*(?:Industry|Technology)\b",
            cypher,
            flags=re.IGNORECASE,
        )
    )
    category_variables.update(
        re.findall(
            (
                r"-\s*\[[^\]]*:\s*(?:OPERATES_IN|USES)\b[^\]]*\]\s*"
                r"->\s*\(\s*(\w+)\b"
            ),
            cypher,
            flags=re.IGNORECASE,
        )
    )
    while True:
        projected_category_variables = {
            alias
            for variable in category_variables
            for alias in re.findall(
                rf"\b{re.escape(variable)}\s+AS\s+(\w+)\b",
                cypher,
                flags=re.IGNORECASE,
            )
        }
        if projected_category_variables.issubset(category_variables):
            break
        category_variables.update(projected_category_variables)
    ai_intent = re.search(
        r"\bAI\b|\bArtificial\s+Intelligence\b",
        question,
        flags=re.IGNORECASE,
    )
    ai_company_name_intent = re.search(
        (
            r"\b(?:compan(?:y|ies)|startups?)\s+"
            r"(?:(?:is|are)\s+)?(?:named|called)\s+"
            r"['\"]?(?:AI|Artificial\s+Intelligence)\b"
        ),
        question,
        flags=re.IGNORECASE,
    )
    company_name_values = []
    for variable in company_variables:
        company_name_values.extend(
            re.findall(
                (
                    rf"(?:toLower\(\s*)?\b{re.escape(variable)}\.name\b"
                    r"\s*\)?\s*(?:=|CONTAINS)\s*['\"]([^'\"]+)['\"]"
                ),
                filtering_cypher,
                flags=re.IGNORECASE,
            )
        )
        company_name_values.extend(
            re.findall(
                (
                    rf"\(\s*{re.escape(variable)}\s*:\s*Company\s*"
                    r"\{[^}]*\bname\s*:\s*['\"]([^'\"]+)['\"]"
                ),
                filtering_cypher,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
    ai_company_name_reference = any(
        re.search(
            r"\bAI\b|\bArtificial\s+Intelligence\b",
            value,
            flags=re.IGNORECASE,
        )
        and value.casefold() not in AI_CATEGORY_ALIASES
        for value in company_name_values
    )
    ai_mentions = re.findall(
        r"\bAI\b|\bArtificial\s+Intelligence\b",
        question,
        flags=re.IGNORECASE,
    )
    explicit_ai_company_name_only = bool(
        ai_company_name_intent and len(ai_mentions) == 1
    )
    question_without_company_names = question
    for value in company_name_values:
        if value.casefold() in AI_CATEGORY_ALIASES:
            continue
        question_without_company_names = re.sub(
            re.escape(value),
            "",
            question_without_company_names,
            flags=re.IGNORECASE,
        )
    ai_company_reference_only = bool(
        ai_company_name_reference
        and not re.search(
            r"\bAI\b|\bArtificial\s+Intelligence\b",
            question_without_company_names,
            flags=re.IGNORECASE,
        )
    )
    ai_alias_filter = any(
        variable in category_variables
        and AI_CATEGORY_ALIASES.issubset(
            {
                value.casefold()
                for value in re.findall(r"['\"]([^'\"]+)['\"]", values)
            }
        )
        for variable, values in re.findall(
            (
                r"toLower\(\s*(\w+)\.name\s*\)\s+IN\s*"
                r"(\[[^\]]+\])"
            ),
            filtering_cypher,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if (
        ai_intent
        and not explicit_ai_company_name_only
        and not ai_company_reference_only
        and not ai_alias_filter
    ):
        raise ValueError(
            "Match both AI category aliases exactly with "
            'IN ["ai", "artificial intelligence"].'
        )

    if re.search(r"\bcount\s*\(", cypher, flags=re.IGNORECASE):
        projection_clauses = re.findall(
            (
                r"\b(?:WITH|RETURN)\b(.*?)(?="
                r"\b(?:MATCH|WHERE|WITH|RETURN|ORDER\s+BY|LIMIT|UNWIND|CALL)\b"
                r"|\Z)"
            ),
            cypher,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for variable in category_variables:
            category_cases = [
                expression
                for expression in re.findall(
                    r"(\bCASE\b.{0,800}?\bEND)\s+AS\s+\w+\b",
                    cypher,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if re.search(
                    rf"\b{re.escape(variable)}\.name\b",
                    expression,
                    flags=re.IGNORECASE,
                )
            ]
            noncanonical_projection = False
            for clause in projection_clauses:
                clause_without_cases = re.sub(
                    r"\bCASE\b.{0,800}?\bEND",
                    "",
                    clause,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if re.search(
                    rf"\b{re.escape(variable)}\.name\b",
                    clause_without_cases,
                    flags=re.IGNORECASE,
                ):
                    noncanonical_projection = True
                    break
            invalid_category_case = any(
                not AI_CATEGORY_ALIASES.issubset(
                    {
                        value.casefold()
                        for value in re.findall(
                            r"['\"]([^'\"]+)['\"]",
                            expression,
                        )
                    }
                )
                or not re.search(
                    r"\bTHEN\s+['\"]Artificial Intelligence['\"]",
                    expression,
                    flags=re.IGNORECASE,
                )
                for expression in category_cases
            )
            if not noncanonical_projection and not invalid_category_case:
                continue
            raise ValueError(
                "Normalize AI and Artificial Intelligence to the canonical "
                "Artificial Intelligence label before category aggregation."
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

    mentions_companies = re.search(
        r"\b(?:companies|startups)\b",
        question,
        flags=re.IGNORECASE,
    )
    aggregate_intent = re.search(
        (
            r"\b(?:how many|count|breakdown|average|avg|median|sum|total|"
            r"percentage|percent|ratio)\b|"
            r"\bnumber of(?:\s+\w+){0,3}\s+(?:companies|startups)\b"
        ),
        question,
        flags=re.IGNORECASE,
    )
    fintech_company_list = mentions_companies and not aggregate_intent and re.search(
        r"\bfintech\b",
        question,
        flags=re.IGNORECASE,
    )
    if fintech_company_list:
        filtering_cypher = re.split(
            r"\bRETURN\b",
            cypher,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        fintech_synonyms = ("fintech", "finance", "payments")
        missing_synonyms = [
            synonym
            for synonym in fintech_synonyms
            if synonym not in filtering_cypher.casefold()
        ]
        if missing_synonyms:
            raise ValueError(
                "Treat fintech as Fintech, Finance, or Payments industries."
            )

        return_clause = re.split(r"\bRETURN\b", cypher, flags=re.IGNORECASE)[-1]
        company_variables = re.findall(
            r"\b(\w+)\.name\s+AS\s+company\b",
            return_clause,
            flags=re.IGNORECASE,
        )
        quoted_term = lambda term: rf"['\"]{term}['\"]"
        industry_variables = re.findall(
            r"\((\w+)\s*:\s*Industry\b",
            filtering_cypher,
            flags=re.IGNORECASE,
        )
        filtered_industry_variables = []
        for variable in industry_variables:
            connected_to_returned_company = any(
                re.search(
                    (
                        rf"\(\s*{re.escape(company)}\b[^)]*\)\s*-\s*"
                        rf"\[[^\]]*OPERATES_IN[^\]]*\]\s*->\s*"
                        rf"\(\s*{re.escape(variable)}\s*:\s*Industry\b|"
                        rf"\(\s*{re.escape(variable)}\s*:\s*Industry\b[^)]*\)\s*"
                        rf"<-\s*\[[^\]]*OPERATES_IN[^\]]*\]\s*-\s*"
                        rf"\(\s*{re.escape(company)}\b[^)]*\)"
                    ),
                    filtering_cypher,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                for company in company_variables
            )
            if not connected_to_returned_company:
                continue
            industry_property = (
                rf"toLower\(\s*{re.escape(variable)}\.name\s*\)"
            )
            in_lists = re.findall(
                rf"{industry_property}\s+IN\s*(\[[^\]]*\])",
                filtering_cypher,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if any(
                all(
                    re.search(quoted_term(term), list_text)
                    for term in fintech_synonyms
                )
                for list_text in in_lists
            ):
                filtered_industry_variables.append(variable)

        if not filtered_industry_variables:
            raise ValueError(
                "Filter the returned company's industry with "
                "IN [Fintech, Finance, Payments]."
            )

        if not any(
            re.search(
                rf"\b{re.escape(variable)}\.name\s+AS\s+industry\b",
                return_clause,
                flags=re.IGNORECASE,
            )
            for variable in filtered_industry_variables
        ):
            raise ValueError("Return the matched industry name AS industry.")

        if re.search(r"\bpython\b", question, flags=re.IGNORECASE):
            technology_variables = re.findall(
                r"\((\w+)\s*:\s*Technology\b",
                filtering_cypher,
                flags=re.IGNORECASE,
            )
            matched_technology_variables = []
            for variable in technology_variables:
                connected_to_returned_company = any(
                    re.search(
                        (
                            rf"\(\s*{re.escape(company)}\b[^)]*\)\s*-\s*"
                            rf"\[[^\]]*USES[^\]]*\]\s*->\s*"
                            rf"\(\s*{re.escape(variable)}\s*:\s*Technology\b"
                        ),
                        filtering_cypher,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    for company in company_variables
                )
                python_filter = re.search(
                    (
                        rf"toLower\(\s*{re.escape(variable)}\.name\s*\)"
                        rf"\s*=\s*['\"]python['\"]"
                    ),
                    filtering_cypher,
                    flags=re.IGNORECASE,
                )
                if connected_to_returned_company and python_filter:
                    matched_technology_variables.append(variable)

            if not matched_technology_variables:
                raise ValueError(
                    "Filter the returned company's matched Technology as Python."
                )
            if not any(
                re.search(
                    rf"\b{re.escape(variable)}\.name\s+AS\s+technology\b",
                    return_clause,
                    flags=re.IGNORECASE,
                )
                for variable in matched_technology_variables
            ):
                raise ValueError("Return the matched technology name AS technology.")

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

# ── Result Normalization ──────────────────────────────────────────────────────

def normalize_category_aliases(value, category_field=False):
    """Canonicalize known category aliases without modifying stored graph data."""
    if isinstance(value, str):
        if category_field and value.casefold() in AI_CATEGORY_ALIASES:
            return CANONICAL_AI_CATEGORY
        return value
    if isinstance(value, list):
        normalized = [
            normalize_category_aliases(item, category_field=category_field)
            for item in value
        ]
        if not category_field:
            return normalized
        deduplicated = []
        seen_strings = set()
        for item in normalized:
            if isinstance(item, str):
                key = item.casefold()
                if key in seen_strings:
                    continue
                seen_strings.add(key)
            deduplicated.append(item)
        return deduplicated
    if isinstance(value, tuple):
        return tuple(
            normalize_category_aliases(item, category_field=category_field)
            for item in value
        )
    if isinstance(value, dict):
        return {
            key: normalize_category_aliases(
                item,
                category_field=str(key).casefold() in CATEGORY_RESULT_FIELDS,
            )
            for key, item in value.items()
        }
    return value


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
                results = normalize_category_aliases(execute_cypher(cypher, driver))
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