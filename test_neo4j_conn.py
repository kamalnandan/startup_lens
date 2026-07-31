from neo4j import GraphDatabase
from app_config import get_required_setting

driver = GraphDatabase.driver(
    get_required_setting("NEO4J_URI"),
    auth=(
        get_required_setting("NEO4J_USERNAME"),
        get_required_setting("NEO4J_PASSWORD"),
    ),
)

with driver.session() as session:
    result = session.run("RETURN 'Connected!' AS message")
    print(result.single()["message"])

driver.close()