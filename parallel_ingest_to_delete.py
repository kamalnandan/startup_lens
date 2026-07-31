import os
import concurrent.futures
from openai import AzureOpenAI
from neo4j import GraphDatabase
from tqdm import tqdm
from schema import KnowledgeGraph
from app_config import get_required_setting

# ==========================================
# CONFIGURATIONS (Update with your Azure details)
# ==========================================
AZURE_ENDPOINT = get_required_setting("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = get_required_setting("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = get_required_setting("AZURE_OPENAI_API_VERSION")

#MINI_MODEL_DEPLOYMENT = "gpt-4o-mini" 
ADVANCED_MODEL_DEPLOYMENT = get_required_setting("AZURE_OPENAI_DEPLOYMENT")
MINI_MODEL_DEPLOYMENT = get_required_setting("AZURE_OPENAI_DEPLOYMENT")

CORPUS_DIR = "./input_test"     
MAX_WORKERS = 3                    

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

import json  # Ensure you import json at the top of parallel_ingest.py if not present

def extract_full_startup_graph(file_name: str, file_content: str) -> KnowledgeGraph:
    """Uses native Azure OpenAI to fetch structured text, manually validating schema objects."""
    base_name = os.path.splitext(file_name)[0]
    inferred_company = base_name.lower().replace(" ", "_")
    
    prompt = f"""
    Analyze the full text and metadata profile for the startup file '{file_name}'.
    
    Extract:
    1. The core company itself (force entity ID to be exactly '{inferred_company}').
    2. Founders, Investors, Batches, Industries, and Locations.
    3. Metrics: year founded, domain of work, summary description, team size, corporate stage.
    4. Relationships: Track funding interaction parameters like investment_amount and valuation on edges.
    
    Full Profile Content:
    {file_content}
    """
    
    # We drop down to standard .create, forcing JSON output via system instructions
    completion = client.chat.completions.create(
        model=MINI_MODEL_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a master knowledge graph data architect mapping corporate intelligence. You MUST reply ONLY with a raw JSON object matching the requested schema. Do not enclose in markdown blocks."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},  # Forces the model to return valid JSON
        temperature=0.0
    )
    
    # Handle both object-style and raw dictionary indexing safely
    if isinstance(completion.choices, list):
        choice = completion.choices[0]
    else:
        choice = completion.choices[0]

    # Extract text content safely depending on response object structure
    if hasattr(choice, "message"):
        raw_json_str = choice.message.content
    else:
        raw_json_str = choice["message"]["content"]
        
    # Standardize and validate using our strict Pydantic model configurations
    json_data = json.loads(raw_json_str)
    return KnowledgeGraph.model_validate(json_data)

def write_to_neo4j(tx, graph_data: KnowledgeGraph):
    if not graph_data:
        return
        
    # 1. Upsert Nodes
    for node in graph_data.nodes:
        props = {
            "year_founded": node.year_founded,
            "domain_of_work": node.domain_of_work,
            "description": node.description,
            "team_size": node.team_size,
            "status_or_stage": node.status_or_stage,
        }
        
        # FIXED: Loop through the strict list attributes model and map into a flat dict for Neo4j
        if node.extra_attributes:
            for attr in node.extra_attributes:
                if attr.key and attr.value:
                    props[attr.key] = attr.value
            
        cleaned_props = {k: v for k, v in props.items() if v is not None}
        query = f"MERGE (n:{node.label} {{id: $id}}) SET n += $properties"
        tx.run(query, id=node.id, properties=cleaned_props)
        
    # 2. Upsert Edges
    for edge in graph_data.edges:
        edge_props = {
            "investment_amount": edge.investment_amount,
            "share_price": edge.share_price,
            "valuation": edge.valuation,
            "date_occurred": edge.date_occurred
        }
        cleaned_edge_props = {k: v for k, v in edge_props.items() if v is not None}
        query = f"""
        MATCH (source {{id: $source}}) MATCH (target {{id: $target}})
        MERGE (source)-[r:{edge.relationship}]->(target) SET r += $properties
        """
        tx.run(query, source=edge.source, target=edge.target, properties=cleaned_edge_props)

def process_single_file(file_name: str):
    file_path = os.path.join(CORPUS_DIR, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip(): return True
        
        extracted_graph = extract_full_startup_graph(file_name, content)
        with driver.session() as session:
            session.execute_write(write_to_neo4j, extracted_graph)
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed on {file_name}: {str(e)}")
        return False

def main():
    if not os.path.exists(CORPUS_DIR):
        print(f"Error: Directory '{CORPUS_DIR}' not found.")
        return

    all_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
    print(f"Found {len(all_files)} files. Starting native parallel Azure ingestion...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_file, all_files), total=len(all_files), desc="Uploading to Azure"))
    print(f"\nPipeline Finished! Successfully loaded {sum(1 for r in results if r)} portfolios.")

if __name__ == "__main__":
    main()
    driver.close()
