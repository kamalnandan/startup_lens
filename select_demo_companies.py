"""Choosing which companies to extract first.

Extracting all 5,378 documents is roughly 45 hours of serial API calls, which
is not a thing to start two days before a demo. Prominence is the cheap way
out: the companies an audience will actually ask about are a small, knowable
subset, and on the long tail "no data" is an honest answer that demonstrates
the property the system is being shown off for.

Ranked from the production graph, which already knows who raised what, who
went public, and who employs whom.
"""

import argparse
import json
import pathlib
import re
import sys

from neo4j import GraphDatabase

import app_config

# Everything the benchmark asks about, so the numbers stay comparable, plus
# the five the prototype was built on.
REQUIRED = {"airbnb", "stripe", "coinbase", "doordash", "dropbox"}

MULTIPLIERS = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def parse_amount(raw) -> float:
    """Read '$1.4 billion', '$500M', '2,000,000' as a number."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).lower().replace(",", "").replace("$", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(k|m|b|t|thousand|million|billion|trillion)?",
                      text)
    if not match:
        return 0.0
    value = float(match.group(1))
    word = (match.group(2) or "")[:1]
    return value * MULTIPLIERS.get(word, 1.0)


QUERY = """
MATCH (c:Company)
OPTIONAL MATCH (c)-[r:RAISED]->()
WITH c, collect(r.amount) AS amounts
OPTIONAL MATCH (c)<-[:INVESTED_IN]-(i)
WITH c, amounts, count(DISTINCT i) AS investors
RETURN c.name AS name, c.filename AS filename, c.status AS status,
       c.team_size AS team_size, c.yc_batch AS yc_batch,
       amounts, investors
"""


def score(record: dict) -> float:
    """Rank by the evidence of consequence the graph already holds.

    Headcount carries this, because it is the only prominence signal the
    graph actually has for everyone: just 314 of 5,975 companies have any
    funding edge at all, so ranking on money raised ranks the completeness of
    the funding data instead of the companies. It put Coub and Nomiku in the
    demo set and left out Reddit, Flexport and Rippling, all of which employ
    thousands.

    Going public is still the strongest single event. Money raised stays as a
    bonus rather than the backbone, and investor count stands in for
    attention.
    """
    import math

    points = 0.0

    try:
        team = float(record.get("team_size") or 0)
    except (TypeError, ValueError):
        team = 0.0
    points += math.log10(team + 1) * 30

    status = (record.get("status") or "").lower()
    if status == "public":
        points += 55
    elif status == "acquired":
        points += 22
    elif status == "dead":
        points -= 25

    raised = sum(parse_amount(a) for a in (record.get("amounts") or []))
    points += math.log10(raised + 1) * 4
    points += min(record.get("investors") or 0, 20) * 1.0

    return points


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--input-dir", type=pathlib.Path,
                        help="Only pick companies whose document exists here.")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)

    driver = GraphDatabase.driver(
        app_config.get_required_setting("NEO4J_URI"),
        auth=(app_config.get_required_setting("NEO4J_USERNAME"),
              app_config.get_required_setting("NEO4J_PASSWORD")))
    try:
        with driver.session() as session:
            records = [dict(r) for r in session.run(QUERY)]
    finally:
        driver.close()

    available = None
    if args.input_dir:
        available = {p.name.lower() for p in args.input_dir.glob("*.txt")}
        available |= {p.stem.lower() for p in args.input_dir.glob("*.txt")}

    def document_for(record: dict):
        """The document this company was built from, if it is on disk."""
        for candidate in (record.get("filename"), record.get("name")):
            if not candidate:
                continue
            stem = pathlib.Path(str(candidate)).stem.lower()
            if available is None or stem in available:
                return f"{stem}.txt"
        return None

    def wears_its_acquirer_name(name: str, document: str) -> bool:
        """True when a company node carries the identity of whoever bought it.

        Production has companies named after their acquirer: the node called
        'Meta Platforms, Inc.' with 83,553 employees is built from
        thread-2.txt, a small YC startup Meta bought, and 'Palantir
        Technologies Inc.' is kimono-labs.txt. Ranking by headcount hands
        those two the acquirer's size and puts them near the top, so they are
        dropped here rather than demonstrated on stage.
        """
        flat = re.sub(r"[^a-z0-9]", "", name.lower())
        stem = re.sub(r"[^a-z0-9]", "", pathlib.Path(document).stem.lower())
        return not (stem.startswith(flat[:10]) or flat.startswith(stem[:10]))

    ranked = []
    for record in records:
        name = (record.get("name") or "").strip()
        if not name:
            continue
        document = document_for(record)
        if available is not None and document is None:
            continue
        if document and wears_its_acquirer_name(name, document):
            continue
        ranked.append({
            "name": name,
            "document": document,
            "status": record.get("status"),
            "team_size": record.get("team_size"),
            "yc_batch": record.get("yc_batch"),
            "raised": sum(parse_amount(a) for a in (record.get("amounts") or [])),
            "investors": record.get("investors"),
            "score": round(score(record), 2),
        })

    ranked.sort(key=lambda row: row["score"], reverse=True)
    chosen = ranked[:args.count]

    have = {row["name"].lower() for row in chosen}
    for row in ranked:
        if row["name"].lower() in REQUIRED and row["name"].lower() not in have:
            chosen.append(row)

    print(f"{len(ranked)} candidates, choosing {len(chosen)}")
    for position, row in enumerate(chosen, 1):
        print(f'{position:4}. {row["name"][:34]:34} {str(row["status"] or "")[:8]:8} '
              f'${row["raised"]/1e6:>10,.1f}m  inv={row["investors"]:>3}  '
              f'{row["score"]:>6.1f}')

    if args.output:
        args.output.write_text(json.dumps(chosen, indent=1), encoding="utf-8")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
