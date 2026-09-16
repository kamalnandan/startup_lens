"""Answering questions from the assertion graph.

Having the right facts in the graph and giving the right answer are different
achievements, and only the second one is the product. The benchmark scorer
measured the first: it confirmed that 15 of the 21 previously-wrong questions
have their supporting facts present as asserted claims. This asks the harder
question - whether a query written against the assertion shape can find them.

The shape is harder than production's. There, a company's founding date is a
property one hop away. Here every claim is its own node, so the same question
is three hops through a node whose meaning lives in a `predicate` string. That
is a more expressive graph and a more difficult target, and nothing so far has
tested whether it can be queried at all.
"""

import argparse
import json
import pathlib
import re
import sys

from neo4j import GraphDatabase

import assertion_query
import graph_model_v2
import neo4j_query

MAX_ATTEMPTS = 3
MAX_RESULTS = 50
MAX_PARTS = 6

DECOMPOSE_SYSTEM = """You split a question into the independent facts it asks for.

A question like "State when and where DoorDash's journey began and its YC
batch" asks for three separate things. Answered as one query they stand or
fall together, and a graph missing any one of them returns nothing for all
three. Split so each can be found, or missed, on its own.

Rules:
- Each part must be answerable without reading the others. Name the company in
  every part; never write "it", "its" or "the company".
- Never narrow the question. A question about which companies do something is
  asking for all of them, and must stay plural - "name a company that went
  public" is a different and much worse question than "which companies went
  public". Keep superlatives, counts and "all" exactly as written.
- A question that asks one thing about many companies is ONE part, not one
  part per company. Only split when genuinely different facts are requested.
- Split only what is genuinely separable. A question asking for one fact
  returns one part.
- Preserve the qualifiers that make a part precise. "the city where Airbnb
  says it was born" is not "where is Airbnb located".
- At most %d parts.

Return ONLY a JSON array of strings.""" % MAX_PARTS


def decompose(question: str) -> list:
    """Split a question into independently answerable parts.

    Returns the question unchanged if it does not split, so the simple path
    costs one extra call and nothing else.
    """
    try:
        raw = neo4j_query.call_llm(DECOMPOSE_SYSTEM, question, max_tokens=400)
        parts = json.loads(raw.replace("```json", "").replace("```", "").strip())
    except Exception:                     # noqa: BLE001 - fall back to whole
        return [question]
    parts = [str(p).strip() for p in parts if str(p).strip()]
    return parts[:MAX_PARTS] or [question]


CYPHER_SYSTEM = assertion_query.GRAPH_SCHEMA + """
Generate a single valid Cypher query answering the question below.

Write for this graph, not a conventional one. Values live on the fact, not on
the company: a founding date is (c:V2Company)-[:ASSERTS]->(f:V2Fact) with
f.predicate = 'founded_on', and the answer is f.value.

Return the source_span alongside any value you return, so the answer can cite
the sentence it came from.

Return ONLY the Cypher query - no explanation, no markdown, no backticks.
"""


def generate_cypher(question: str, error: str = None,
                    previous: str = None) -> str:
    if error:
        prompt = (f"Question: {question}\n\n"
                  f"Previous Cypher failed: {error}\n"
                  f"Previous Cypher: {previous}\n\n"
                  "Generate a corrected Cypher query.")
    else:
        prompt = f"Question: {question}"
    cypher = neo4j_query.call_llm(CYPHER_SYSTEM, prompt, max_tokens=600)
    return cypher.replace("```cypher", "").replace("```", "").strip()


def _limited(cypher: str) -> str:
    clean = cypher.rstrip().rstrip(";")
    if re.search(r"\bLIMIT\s+\d+\s*$", clean, flags=re.IGNORECASE):
        return clean
    return f"{clean}\nLIMIT {MAX_RESULTS}"


def _readonly(cypher: str) -> None:
    banned = ("CREATE", "DELETE", "MERGE", "SET ", "REMOVE", "DROP")
    upper = cypher.upper()
    for word in banned:
        if re.search(rf"\b{word.strip()}\b", upper):
            raise ValueError(f"Refusing to run a query containing {word.strip()}.")


def ask_one(question: str, driver, database: str) -> dict:
    """Answer one indivisible question, or abstain, recording every attempt."""
    attempts = []
    error = None
    cypher = None

    for _ in range(MAX_ATTEMPTS):
        try:
            cypher = generate_cypher(question, error, cypher)
            _readonly(cypher)
            assertion_query.validate_asserted_filter(cypher)
            with driver.session(database=database) as session:
                rows = [dict(r) for r in session.run(_limited(cypher))]
        except Exception as failure:      # noqa: BLE001 - reported, not raised
            error = str(failure)
            attempts.append({"cypher": cypher, "error": error})
            continue

        attempts.append({"cypher": cypher, "rows": len(rows)})
        if not rows:
            # An empty result is an answer: the graph does not support a claim
            # here. Retrying would only invite a looser query that finds
            # something adjacent and calls it the answer.
            return {"question": question, "status": "no_data",
                    "cypher": cypher, "rows": [], "attempts": attempts}
        return {"question": question, "status": "found", "cypher": cypher,
                "rows": rows[:MAX_RESULTS], "attempts": attempts}

    return {"question": question, "status": "abstained", "error": error,
            "cypher": cypher, "attempts": attempts}


def ask(question: str, driver, database: str, split: bool = True) -> dict:
    """Answer a question by answering each fact it asks for separately.

    A compound question asked as a single query is all-or-nothing: one leg of
    the join finding nothing empties the whole result, so a question that asks
    for three facts and can only support two answers none of them. Asking each
    part on its own means a gap costs exactly the part it belongs to.
    """
    parts = decompose(question) if split else [question]
    answered = [ask_one(part, driver, database) for part in parts]

    supported = [part for part in answered if part["status"] == "found"]
    if not supported:
        return {"question": question, "status": "no_data", "parts": answered}

    answer = synthesize_parts(question, supported, answered)
    return {
        "question": question,
        "status": "answered" if len(supported) == len(answered) else "partial",
        "parts": answered,
        "answer": answer,
    }


def synthesize_parts(question: str, supported: list, answered: list) -> str:
    """Write the answer from what was found, naming what was not.

    The parts that found nothing are passed in deliberately. Leaving them out
    would let the answer read as complete when it is not, and a fluent partial
    answer that never admits its gap is the failure this system exists to
    avoid.
    """
    blocks = []
    for part in answered:
        if part["status"] == "found":
            blocks.append(f"Question part: {part['question']}\n"
                          f"{neo4j_query.format_grounded_results(part['rows'])}")
        else:
            blocks.append(f"Question part: {part['question']}\n"
                          "NO DATA - the graph holds no supported fact for "
                          "this part.")
    instruction = (
        "Answer the whole question in continuous prose. The parts below are "
        "working notes, not a template: do not restate them as headings, do "
        "not number them, and do not repeat the phrase NO DATA. Where a part "
        "has no data, say that this answer does not cover it - do not say the "
        "sources do not state it, because a query finding nothing is not "
        "proof that no source says it. Never fill a gap from prior knowledge."
    )
    return neo4j_query.call_llm(
        neo4j_query.SYNTHESIS_SYSTEM,
        f"Question: {question}\n\n{instruction}\n\n" + "\n\n".join(blocks),
        max_tokens=800)



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=pathlib.Path,
                        help="JSON file with a 'questions' list.")
    parser.add_argument("--question", action="append", default=[],
                        help="Ask a single question directly.")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-split", action="store_true",
                        help="Ask each question whole, as the first version did.")
    args = parser.parse_args(argv)

    questions = list(args.question)
    if args.questions:
        loaded = json.loads(args.questions.read_text(encoding="utf-8"))
        for record in loaded.get("questions", loaded):
            questions.append(record["question"] if isinstance(record, dict)
                             else record)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        parser.error("Nothing to ask.")

    settings = graph_model_v2.connection_settings()
    driver = GraphDatabase.driver(
        settings["uri"], auth=(settings["username"], settings["password"]))

    results = []
    try:
        for number, question in enumerate(questions, 1):
            print(f"[{number}/{len(questions)}] {question[:90]}", flush=True)
            result = ask(question, driver, settings["database"],
                         split=not args.no_split)
            parts = len(result.get("parts", []))
            print(f"    -> {result['status']} ({parts} parts)", flush=True)
            results.append(result)
    finally:
        driver.close()

    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(json.dumps(counts, indent=1))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": counts, "results": results}, indent=1),
            encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
