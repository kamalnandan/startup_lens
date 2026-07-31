"""
Neo4j Entity Deduplication Script
Global Startup Intelligence Graph

Finds similar entity names in Neo4j and merges duplicate nodes.
Run this after loading all companies to clean up entity resolution issues.

Uses fuzzy string matching to find candidates, then merges them in Neo4j.
"""

import logging
from neo4j import GraphDatabase
from app_config import get_required_setting

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

NEO4J_URI = get_required_setting("NEO4J_URI")
NEO4J_USERNAME = get_required_setting("NEO4J_USERNAME")
NEO4J_PASSWORD = get_required_setting("NEO4J_PASSWORD")

# Similarity threshold — 0.85 means 85% similar
SIMILARITY_THRESHOLD = 0.85

# Known aliases to merge — add more as you discover them
KNOWN_ALIASES = {
    # Y Combinator variants
    "YC": "Y Combinator",
    "YC Accelerator": "Y Combinator",
    "YC (Y Combinator)": "Y Combinator",
    "YC Startup Accelerator": "Y Combinator",
    "Y-Combinator": "Y Combinator",
    "YCombinator": "Y Combinator",

    # Investor aliases
    "Sequoia": "Sequoia Capital",
    "Sequoia Cap": "Sequoia Capital",
    "a16z": "Andreessen Horowitz",
    "Andreessen-Horowitz": "Andreessen Horowitz",
    "Tiger": "Tiger Global",
    "Tiger Global Management": "Tiger Global",
    "Accel Partners": "Accel",
    "Accel Ventures": "Accel",
    "GV": "Google Ventures",
    "Google Ventures": "GV",
    "SV Angel": "SV Angel",

    # Company aliases
    "FB": "Facebook",
    "Meta Platforms": "Meta",
    "Alphabet": "Google",
    "GOOG": "Google",
}

# ── Deduplication ──────────────────────────────────────────────────────────────

def merge_nodes(session, canonical_name: str, duplicate_name: str, label: str):
    """Merge duplicate node into canonical node, transferring all relationships."""
    logger.info(f"  Merging '{duplicate_name}' → '{canonical_name}'")

    # Transfer all relationships from duplicate to canonical
    session.run(f"""
        MATCH (canonical:{label} {{name: $canonical}})
        MATCH (duplicate:{label} {{name: $duplicate}})
        WHERE canonical <> duplicate
        
        // Transfer outgoing relationships
        WITH canonical, duplicate
        MATCH (duplicate)-[r]->(other)
        WHERE other <> canonical
        CALL apoc.merge.relationship(canonical, type(r), {{}}, properties(r), other) YIELD rel
        
        RETURN count(rel)
    """, canonical=canonical_name, duplicate=duplicate_name)

    session.run(f"""
        MATCH (canonical:{label} {{name: $canonical}})
        MATCH (duplicate:{label} {{name: $duplicate}})
        WHERE canonical <> duplicate
        
        // Transfer incoming relationships
        WITH canonical, duplicate
        MATCH (other)-[r]->(duplicate)
        WHERE other <> canonical
        CALL apoc.merge.relationship(other, type(r), {{}}, properties(r), canonical) YIELD rel
        
        RETURN count(rel)
    """, canonical=canonical_name, duplicate=duplicate_name)

    # Delete duplicate node
    session.run(f"""
        MATCH (duplicate:{label} {{name: $duplicate}})
        DETACH DELETE duplicate
    """, duplicate=duplicate_name)


def apply_known_aliases(session):
    """Apply known alias mappings to merge duplicates."""
    logger.info("Applying known aliases...")

    for label in ["Investor", "Founder", "Company"]:
        # Get all nodes of this type
        result = session.run(f"MATCH (n:{label}) RETURN n.name AS name")
        existing_names = {record["name"] for record in result}

        for alias, canonical in KNOWN_ALIASES.items():
            if alias in existing_names and canonical in existing_names:
                logger.info(f"  [{label}] Found alias: '{alias}' → '{canonical}'")
                merge_nodes(session, canonical, alias, label)
            elif alias in existing_names and canonical not in existing_names:
                # Rename alias to canonical
                logger.info(f"  [{label}] Renaming: '{alias}' → '{canonical}'")
                session.run(f"""
                    MATCH (n:{label} {{name: $alias}})
                    SET n.name = $canonical
                """, alias=alias, canonical=canonical)


def find_fuzzy_duplicates(session, label: str, threshold: float = 0.85):
    """Find potential duplicate nodes using Neo4j's string similarity."""
    logger.info(f"Finding fuzzy duplicates for {label}...")

    # Use Neo4j's apoc.text.jaroWinklerDistance for similarity
    result = session.run(f"""
        MATCH (a:{label}), (b:{label})
        WHERE a.name < b.name
        AND apoc.text.jaroWinklerDistance(toLower(a.name), toLower(b.name)) > $threshold
        RETURN a.name AS name1, b.name AS name2,
               apoc.text.jaroWinklerDistance(toLower(a.name), toLower(b.name)) AS similarity
        ORDER BY similarity DESC
        LIMIT 50
    """, threshold=threshold)

    candidates = [(r["name1"], r["name2"], r["similarity"]) for r in result]
    return candidates


def print_duplicate_report(driver):
    """Print a report of potential duplicates for manual review."""
    logger.info("\n" + "="*60)
    logger.info("DUPLICATE DETECTION REPORT")
    logger.info("="*60)

    with driver.session() as session:
        for label in ["Investor", "Founder", "Company"]:
            candidates = find_fuzzy_duplicates(session, label)
            if candidates:
                logger.info(f"\n{label} duplicates ({len(candidates)} found):")
                for name1, name2, similarity in candidates:
                    logger.info(f"  {similarity:.2f} | '{name1}' ↔ '{name2}'")
            else:
                logger.info(f"\n{label}: No duplicates found above threshold")


def run_deduplication(auto_merge: bool = False):
    """
    Run the deduplication pipeline.
    
    Args:
        auto_merge: If True, automatically merge fuzzy matches above threshold.
                   If False (default), only apply known aliases and print report.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        # Step 1 — Apply known aliases
        apply_known_aliases(session)

    # Step 2 — Print duplicate report for review
    print_duplicate_report(driver)

    if auto_merge:
        logger.info("\nAuto-merging fuzzy duplicates...")
        with driver.session() as session:
            for label in ["Investor", "Founder"]:  # be conservative — skip Company
                candidates = find_fuzzy_duplicates(session, label)
                for name1, name2, similarity in candidates:
                    if similarity > 0.92:  # only merge very high confidence
                        # Keep the longer/more complete name as canonical
                        canonical = name1 if len(name1) >= len(name2) else name2
                        duplicate = name2 if canonical == name1 else name1
                        merge_nodes(session, canonical, duplicate, label)

    driver.close()
    logger.info("\nDeduplication complete!")


if __name__ == "__main__":
    # Run with auto_merge=False first to review the report
    # Then run with auto_merge=True to apply merges
    run_deduplication(auto_merge=False)