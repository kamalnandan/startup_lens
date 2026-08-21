# 🔭 StartupLens

> The YC ecosystem explorer — research 20 years of Y Combinator through an AI-powered knowledge graph.

---

## What is StartupLens?

StartupLens helps founders, YC applicants, and investors research the Y Combinator ecosystem like never before. Ask complex, multi-hop questions about 5,700+ YC companies — questions that no search engine or spreadsheet can answer.

Built on a Neo4j knowledge graph with a natural language query layer powered by GPT-4.1, StartupLens converts your questions into graph queries, executes them, and synthesizes insightful answers.

---

## Who is this for?

- **YC applicants** — Research batches, industries, and trends before applying
- **Aspiring founders** — Learn from patterns across 20 years of YC companies
- **Angel investors & scouts** — Discover YC companies by industry, location, or founder background
- **Startup ecosystem researchers** — Study patterns, failure rates, and ecosystem trends

---

## Example Questions

- *"What industries are trending in recent YC batches?"*
- *"Which YC founders from India built fintech companies?"*
- *"What is the survival rate of YC companies by industry?"*
- *"Which investors have the largest YC portfolios?"*
- *"What are common patterns among failed YC startups?"*
- *"Which countries outside the US produce the most YC companies?"*

---

## Architecture

```
Input Data (6000+ YC company profiles + Wikipedia enrichment)
        ↓
Entity Extraction (GPT-4.1)
        ↓
Neo4j Knowledge Graph
(Companies, Founders, Investors, Industries, Batches, Locations, Technologies)
        ↓
Natural Language Query
        ↓
GPT-4.1 → Cypher Query → Neo4j → GPT-4.1 Synthesis
        ↓
Answer
        ↓
FastAPI + Streamlit UI
```

---

## Graph Schema

### Node Types
| Node | Properties |
|---|---|
| `Company` | name, description, status, stage, team_size, yc_batch |
| `Founder` | name |
| `Investor` | name |
| `Industry` | name |
| `Location` | city, country |
| `Technology` | name |
| `Batch` | name (e.g. "Winter 2009") |
| `FundingEvent` | round, year, amount |

### Relationship Types
| Relationship | Meaning |
|---|---|
| `(Founder)-[:FOUNDED]->(Company)` | Founder started this company |
| `(Founder)-[:CO_FOUNDED_WITH]->(Founder)` | Founders co-founded together |
| `(Investor)-[:INVESTED_IN]->(Company)` | Investor backed this company |
| `(Company)-[:OPERATES_IN]->(Industry)` | Company's industry |
| `(Company)-[:HEADQUARTERED_IN]->(Location)` | Company's location |
| `(Company)-[:USES]->(Technology)` | Technology used |
| `(Company)-[:PART_OF]->(Batch)` | YC batch |
| `(Company)-[:ACQUIRED_BY]->(Company)` | Acquisition |
| `(Company)-[:RAISED]->(FundingEvent)` | Funding round |
| `(Company)-[:COMPETES_WITH]->(Company)` | Competitors |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Knowledge Graph | Neo4j (self-hosted on Azure VM) |
| LLM | GPT-4.1 via Azure OpenAI |
| Entity Extraction | GPT-4.1 with structured JSON output |
| Query Layer | Natural language → Cypher → Neo4j |
| API | FastAPI (Python) |
| UI | Streamlit |
| Data Enrichment | Wikipedia API |
| Infrastructure | Azure (VM, OpenAI, Web Apps) |

---

## Project Structure

```
startuplens/
├── neo4j_loader.py          # Extracts entities from text files and loads into Neo4j
├── neo4j_query.py           # Natural language → Cypher → answer pipeline
├── neo4j_dedup.py           # Entity deduplication and normalization
├── neo4j_fastapi.py         # FastAPI query API
├── streamlit_app_v2.py      # Streamlit UI
├── enrich_companies.py      # Wikipedia enrichment pipeline
├── settings.yaml            # GraphRAG configuration (legacy)
├── prompts/                 # Custom GraphRAG prompts (legacy)
├── input/                   # Raw YC company profiles
├── input_enriched/          # Wikipedia-enriched company profiles
└── .env                     # API keys (not committed)
```

---

## Setup

### Prerequisites
- Python 3.10+
- Neo4j 5.x (local or Azure VM)
- Azure OpenAI access (GPT-4.1 + text-embedding-3-large)
- APOC plugin for Neo4j

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/startuplens.git
cd startuplens

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:
```env
AZURE_OPENAI_API_KEY=your-azure-openai-key
NEO4J_PASSWORD=your-neo4j-password
```

Update `neo4j_loader.py` and `neo4j_query.py` with your Neo4j URI and Azure OpenAI endpoint.

### Load Data

```bash
# Load YC company profiles into Neo4j
python neo4j_loader.py

# Run deduplication
python neo4j_dedup.py
```

### Run

```bash
# Start FastAPI server
uvicorn neo4j_fastapi:app --reload

# Start Streamlit UI (in a separate terminal)
streamlit run streamlit_app_v2.py
```

Open `http://localhost:8501` in your browser.

### Query regression tests

Run the deterministic query-generation and answer-grounding tests:

```bash
python -m unittest tests.test_neo4j_query -v
```

After deploying, run the representative live-query suite with credentials supplied
through the environment:

```bash
RUN_LIVE_API_TESTS=1 \
STARTUPLENS_API_URL=https://your-api-host \
STARTUPLENS_API_KEY=your-api-key \
python -m unittest tests.test_live_query_regression -v
```

---

## Data Pipeline

### 1. Raw Data
6,000+ YC company profiles scraped from Y Combinator's website (2005–2025), covering:
- Company name, description, industry, location
- YC batch, team size, status
- Founder names

### 2. Wikipedia Enrichment
Each company profile is enriched with Wikipedia data:
- Full funding history
- Acquisition details
- Founder backgrounds
- Key milestones

### 3. Entity Extraction
GPT-4.1 extracts structured entities from each enriched profile:
- Companies, founders, investors
- Industries, technologies, locations
- Funding rounds, acquisitions, competitors

### 4. Knowledge Graph
Extracted entities and relationships are loaded into Neo4j with:
- Normalized entity names (entity resolution)
- Indexed nodes for fast lookups
- Deduplication using fuzzy matching

---

## Query Pipeline

```
User question
      ↓
1. Classify (local vs global) — GPT-4.1
      ↓
2. Generate Cypher — GPT-4.1 + schema + few-shot examples
      ↓
3. Execute — Neo4j
      ↓
4. Synthesize answer — GPT-4.1
      ↓
Natural language answer
```

---

## Roadmap

- [ ] Enrich all 5,700 companies with Crunchbase data (funding, investors, valuations)
- [ ] Batch trend analysis and visualizations
- [ ] "Companies like yours" feature for YC applicants
- [ ] Neo4j vector index for semantic search
- [ ] Export answers to PDF/Excel
- [ ] **P1 post-alpha:** Session-aware follow-up questions using session IDs,
      bounded TTL-based history, entity/pronoun resolution, context clearing,
      and multi-turn regression tests

---

## License

MIT

---

## Author

Built by Kamal — Customer Success Architect at Microsoft, exploring the intersection of knowledge graphs, LLMs, and startup intelligence.
