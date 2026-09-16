"""Score the prototype graph against the benchmark questions it got wrong.

This does not ask whether the system answers well - it asks the narrower
question the extraction work was meant to move: are the facts each answer
needs actually present, and are they asserted rather than withheld?

A question is only as answerable as its hardest claim, so a claim counts as
covered only when every distinctive term in it - the numbers, the dates, the
proper nouns - is found in some fact the graph asserts about that company.
"""

import argparse
import json
import re
from pathlib import Path

COMPANIES = ("coinbase", "airbnb", "dropbox", "doordash", "stripe")

# Words that appear in every claim and distinguish nothing.
STOPWORDS = frozenset("""
a an the and or but of in on at to for with from by as is are was were be been
its it his her their this that these those says said say lists listed give
given gives state states stated according report reports reported which what
when where who whom how many much both two three at least about
""".split())

TERM = re.compile(r"[A-Za-z][A-Za-z.'&-]*|\d[\d,.$%]*")

# The same alias problem places have, in organisation names. The scorer knows
# about it so it does not report a fact as absent when only the spelling
# differs; the graph itself still needs the gazetteer layer to fix it properly.
ORG_ALIASES = {
    "y combinator": "yc",
    "andreessen horowitz": "a16z",
    "new york stock exchange": "nyse",
    "securities and exchange commission": "sec",
}


def claim_terms(claim: str, subject: str = "") -> list:
    """The terms in a claim that an answer would have to get right.

    The subject is dropped: every claim about DoorDash says "DoorDash", so
    finding it proves nothing about whether the graph knows the claim.
    """
    terms = []
    for token in TERM.findall(claim):
        bare = token.strip(".,'&-")
        if bare.lower().endswith("'s"):
            bare = bare[:-2]
        if not bare or bare.lower() in STOPWORDS:
            continue
        if subject and bare.casefold() == subject.casefold():
            continue
        # A lowercase common word carries no burden of proof; a number, a
        # proper noun or an acronym does.
        if bare[0].isdigit() or bare[0].isupper():
            terms.append(bare)
    return terms


def _haystack(facts: list) -> str:
    parts = []
    for fact in facts:
        parts.append(str(fact.get("value", "")))
        parts.append(str(fact.get("role", "")))
        parts.append(str(fact.get("qualifier", "")))
        parts.append(str(fact.get("source_span", "")))
        parts.extend(str(s) for s in fact.get("supporting_spans", []))
        parts.extend(str(s) for s in fact.get("also_written_as", []))
    text = " ".join(parts).casefold()
    for long_form, short_form in ORG_ALIASES.items():
        if long_form in text:
            text += " " + short_form
    return text


def _found(term: str, haystack: str) -> bool:
    bare = term.casefold().strip(".,'&-")
    if bare in haystack:
        return True
    # "APIs" and "API" are the same term; a claim should not count as missing
    # because it pluralised what the document wrote once.
    if len(bare) > 2 and bare.endswith("s") and bare[:-1] in haystack:
        return True
    # "51,323,531" and "51323531" are the same number; so are "$250" and "250".
    stripped = bare.replace(",", "").replace("$", "")
    return bool(stripped) and stripped in haystack.replace(",", "").replace("$", "")


def company_of(question: str) -> str:
    lowered = question.casefold()
    for company in COMPANIES:
        if company in lowered:
            return company
    return ""


def score(extraction: dict, adjudications: list) -> dict:
    by_company = {}
    for entry in extraction["companies"]:
        facts = entry.get("accepted", [])
        by_company[entry["company"]] = {
            "asserted": [f for f in facts if f.get("asserted")],
            "withheld": [f for f in facts if not f.get("asserted")],
        }

    results = []
    for item in adjudications:
        company = company_of(item["question"])
        if company not in by_company:
            continue
        asserted = _haystack(by_company[company]["asserted"])
        withheld = _haystack(by_company[company]["withheld"])

        claims = []
        for claim in item.get("expected_claims", []):
            terms = claim_terms(claim, subject=company)
            missing = [t for t in terms if not _found(t, asserted)]
            recoverable = [t for t in missing if _found(t, withheld)]
            claims.append({
                "claim": claim,
                "terms": len(terms),
                "missing": missing,
                "recoverable_from_withheld": recoverable,
            })

        supported = [c for c in claims if not c["missing"]]
        blocked_only_by_withholding = [
            c for c in claims
            if c["missing"] and len(c["recoverable_from_withheld"]) == len(c["missing"])
        ]
        if len(supported) == len(claims):
            status = "answerable"
        elif len(supported) + len(blocked_only_by_withholding) == len(claims):
            status = "withheld"
        elif supported:
            status = "partial"
        else:
            status = "missing"

        results.append({
            "id": item["id"],
            "company": company,
            "previous_verdict": item["verdict"],
            "question": item["question"],
            "status": status,
            "claims_supported": len(supported),
            "claims_total": len(claims),
            "detail": claims,
        })

    summary = {}
    for row in results:
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    return {"summary": summary, "questions": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--verdicts", nargs="*", default=["incorrect"],
                        help="Which previous verdicts to re-score.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    extraction = json.loads(args.extraction.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    adjudications = [a for a in benchmark["adjudications"]
                     if a["verdict"] in set(args.verdicts)]

    report = score(extraction, adjudications)
    for row in report["questions"]:
        print(f"{row['status']:<11} {row['id']:<14} "
              f"{row['claims_supported']}/{row['claims_total']}  "
              f"{row['question'][:64]}")
    print()
    print(json.dumps(report["summary"], indent=2))

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
