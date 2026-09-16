"""Load assertion-based extractions into a namespaced prototype graph.

Every label this writes carries a "V2" prefix, so the prototype graph lives
alongside the production graph in the same database without touching it. The
wipe operation is likewise scoped to those labels.

The model keeps provenance on the fact rather than on the company:

    (:V2Company)-[:ASSERTS]->(:V2Fact)-[:ABOUT]->(:V2Person|V2Place|...)

A fact node carries the predicate, the typed value, its granularity or role,
and the verbatim span that states it. Entity nodes exist so that questions
spanning companies ("which investors backed both") remain graph traversals
rather than string comparisons, but the assertion stays the unit of truth:
an answer can always be traced back to the sentence that supports it.

Usage:
    python graph_model_v2.py --extraction prototype_extraction_v2.json
    python graph_model_v2.py --wipe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neo4j import GraphDatabase

from app_config import get_setting
import extraction_schema


# The prototype is meant to run against a throwaway instance, so the connection
# is resolved at call time from flags or V2-specific variables, and only falls
# back to the production ones when nothing else is given. Reading them at import
# time would make the module unusable without production credentials present.
def connection_settings(args=None) -> dict:
    """Resolve the target instance from flags, V2 variables, then production."""
    def pick(flag, *names):
        if args is not None and getattr(args, flag, None):
            return getattr(args, flag)
        for name in names:
            value = get_setting(name, "")
            if value:
                return value
        return ""

    return {
        "uri": pick("uri", "NEO4J_V2_URI", "NEO4J_URI"),
        "username": pick("username", "NEO4J_V2_USERNAME", "NEO4J_USERNAME"),
        "password": pick("password", "NEO4J_V2_PASSWORD", "NEO4J_PASSWORD"),
        "database": pick("database", "NEO4J_V2_DATABASE") or "neo4j",
    }

PREFIX = "V2"

# Values that name an entity get a node so they can be traversed. Values that
# are prose, money, dates or counts stay on the fact, because they are
# measurements rather than things other companies can also point at.
ENTITY_LABELS = {
    extraction_schema.PERSON: f"{PREFIX}Person",
    extraction_schema.ORGANISATION: f"{PREFIX}Organisation",
    extraction_schema.PLACE: f"{PREFIX}Place",
    extraction_schema.CATEGORY: f"{PREFIX}Category",
}

V2_LABELS = (
    f"{PREFIX}Company",
    f"{PREFIX}Fact",
    *ENTITY_LABELS.values(),
)

FACT_PROPERTIES = (
    "predicate",
    "value_type",
    "granularity",
    "role",
    "qualifier",
    "source_span",
    "source_doc",
    "conflicts_with",
    "asserted",
    "agreement",
    "run_count",
)


def _fact_key(company: str, assertion: dict) -> str:
    """Stable identity for a fact, so re-loading updates rather than duplicates."""
    return "|".join(
        [
            company.casefold(),
            assertion["predicate"],
            str(assertion["value"]).casefold(),
            str(assertion.get("role") or ""),
            str(assertion.get("qualifier") or ""),
        ]
    )


def create_constraints(session) -> None:
    session.run(
        f"CREATE CONSTRAINT {PREFIX.lower()}_company_name IF NOT EXISTS "
        f"FOR (c:{PREFIX}Company) REQUIRE c.name IS UNIQUE"
    )
    session.run(
        f"CREATE CONSTRAINT {PREFIX.lower()}_fact_key IF NOT EXISTS "
        f"FOR (f:{PREFIX}Fact) REQUIRE f.key IS UNIQUE"
    )
    for label in ENTITY_LABELS.values():
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
        )


def load_company(session, company: str, source_doc: str, assertions: list) -> int:
    session.run(
        f"MERGE (c:{PREFIX}Company {{name: $name}}) SET c.source_doc = $doc",
        name=company,
        doc=source_doc,
    )

    written = 0
    for assertion in assertions:
        properties = {
            key: assertion.get(key)
            for key in FACT_PROPERTIES
            if assertion.get(key) is not None
        }
        properties["source_doc"] = source_doc
        review = assertion.get("review")
        if review:
            properties["review"] = list(review)

        session.run(
            f"""
            MERGE (f:{PREFIX}Fact {{key: $key}})
            SET f += $properties, f.value = $value
            WITH f
            MATCH (c:{PREFIX}Company {{name: $company}})
            MERGE (c)-[:ASSERTS]->(f)
            """,
            key=_fact_key(company, assertion),
            properties=properties,
            value=assertion["value"],
            company=company,
        )

        label = ENTITY_LABELS.get(assertion["value_type"])
        if label and isinstance(assertion["value"], str):
            session.run(
                f"""
                MATCH (f:{PREFIX}Fact {{key: $key}})
                MERGE (e:{label} {{name: $value}})
                MERGE (f)-[:ABOUT]->(e)
                """,
                key=_fact_key(company, assertion),
                value=assertion["value"],
            )
        written += 1

    return written


def wipe(session) -> int:
    total = 0
    for label in V2_LABELS:
        record = session.run(
            f"MATCH (n:{label}) DETACH DELETE n RETURN count(n) AS deleted"
        ).single()
        total += record["deleted"] if record else 0
    return total


def summarise(session) -> dict:
    counts = {}
    for label in V2_LABELS:
        counts[label] = session.run(
            f"MATCH (n:{label}) RETURN count(n) AS c"
        ).single()["c"]
    counts["production_Company"] = session.run(
        "MATCH (n:Company) RETURN count(n) AS c"
    ).single()["c"]
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", type=Path)
    parser.add_argument("--wipe", action="store_true")
    parser.add_argument("--uri", help="Bolt URI of the target instance.")
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--database")
    args = parser.parse_args()

    settings = connection_settings(args)
    if not settings["uri"]:
        parser.error("No target instance: pass --uri or set NEO4J_V2_URI.")

    is_production = settings["uri"] == get_setting("NEO4J_URI", "")
    print(f"target: {settings['uri']} db={settings['database']}"
          f"{'  [PRODUCTION INSTANCE]' if is_production else ''}")

    driver = GraphDatabase.driver(
        settings["uri"], auth=(settings["username"], settings["password"])
    )
    try:
        with driver.session(database=settings["database"]) as session:
            if args.wipe:
                print(f"wiped {wipe(session)} prototype nodes")

            if args.extraction:
                report = json.loads(args.extraction.read_text(encoding="utf-8"))
                create_constraints(session)
                for entry in report["companies"]:
                    if entry.get("error"):
                        print(f"  ! {entry['company']}: {entry['error']}")
                        continue
                    written = load_company(
                        session,
                        entry["company"],
                        f"{entry['company']}.txt",
                        entry["accepted"],
                    )
                    print(f"  {entry['company']}: {written} facts")

            print(json.dumps(summarise(session), indent=2))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
