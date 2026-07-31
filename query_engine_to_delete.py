import os
from openai import AzureOpenAI
from neo4j import GraphDatabase
from app_config import get_required_setting

# ==========================================
# CONFIGURATIONS (Update with your Azure details)
# ==========================================
AZURE_ENDPOINT = get_required_setting("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = get_required_setting("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = get_required_setting("AZURE_OPENAI_API_VERSION")

ADVANCED_MODEL_DEPLOYMENT = get_required_setting("AZURE_OPENAI_DEPLOYMENT")
MINI_MODEL_DEPLOYMENT = get_required_setting("AZURE_OPENAI_DEPLOYMENT")

NEO4J_URI = get_required_setting("NEO4J_URI")
NEO4J_AUTH = (
    get_required_setting("NEO4J_USERNAME"),
    get_required_setting("NEO4J_PASSWORD"),
)

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

SCHEMA_PROMPT = """
You are an expert Cypher query generator for a Neo4j database tracking the YC startup ecosystem.
NODES:
- (:Company {id: "airbnb", location: "San Francisco, CA, USA"})
- (:Founder {id: "brian_chesky"})
- (:Investor {id: "sequoia_capital"})
- (:Batch {id: "winter_2009"})
- (:Industry {id: "consumer_travel_leisure_and_tourism"})

RELATIONSHIPS:
- (:Founder)-[:FOUNDED]->(:Company)
- (:Investor)-[:INVESTED_IN]->(:Company)
- (:Company)-[:PART_OF_BATCH]->(:Batch)
- (:Company)-[:IN_INDUSTRY]->(:Industry)

RULES:
1. Always normalize entity names to lowercase snake_case or clean lowercase strings for ids.
2. Only return the executable Cypher query inside a single clean string block. No markdown backticks.
"""

def generate_cypher(user_question: str) -> str:
    """Translates high-level natural language questions into raw executable Cypher logic."""
    response = client.chat.completions.create(
        model=ADVANCED_MODEL_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user", "content": f"Generate a Cypher query for this question: '{user_question}'"}
        ],
        temperature=0.0
    )
    query = response.choices.message.content.strip()
    return query.replace("```cypher", "").replace("```", "").strip()

def run_cypher_query(cypher_query: str):
    """Executes code traversals over network sockets directly into Azure."""
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]
    except Exception as e:
        return f"Database Execution Error: {str(e)}"

def synthesize_answer(user_question: str, cypher_query: str, graph_results: list) -> str:
    """Combines database arrays and the prompt query into a precise conversational response."""
    prompt = f"""
    You are a professional venture capital intelligence bot. Answer the user's question accurately using ONLY the verified database records provided below.
    User Question: {user_question}
    Database Raw Records: {graph_results}
    """
    response = client.chat.completions.create(
        model=MINI_MODEL_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices.message.content.strip()

def ask_graph_rag(user_question: str) -> str:
    print(f"\n[Question]: {user_question}")
    cypher_query = generate_cypher(user_question)
    print(f"[Generated Cypher]: {cypher_query}")
    raw_results = run_cypher_query(cypher_query)
    final_answer = synthesize_answer(user_question, cypher_query, raw_results)
    return final_answer

if __name__ == "__main__":
    # Test primary complex multi-hop relational question
    test_q = "Can you tell me the startups that were invested into by the investor of Airbnb?"
    print(f"\n[Answer]:\n{ask_graph_rag(test_q)}\n")
    driver.close()
