"""Reading an assertion graph safely.

The prototype graph stores what the documents say, including the parts they
do not say: Airbnb's ticker is in there as a fact marked unasserted, because
the model supplied ABNB from memory and the document never wrote it. That
design only protects anyone if reading it respects the mark, so the filter
cannot be left to a prompt instruction that the model follows most of the
time. It is enforced here.
"""

import re

import extraction_schema

PREFIX = "V2"
FACT_LABEL = f"{PREFIX}Fact"

# The schema the Cypher generator is shown. It describes a graph of claims
# rather than a graph of companies: a fact carries its own predicate, its own
# evidence and its own status, so the question "how do you know" always has an
# answer stored next to the answer itself.
GRAPH_SCHEMA = f"""
Graph shape
-----------
({PREFIX}Company)-[:ASSERTS]->({PREFIX}Fact)-[:ABOUT]->(entity)

A {PREFIX}Fact is one claim taken from one sentence. Its properties:
  predicate     what is claimed, one of the list below
  value         the claimed value
  value_type    person | organisation | place | category | text | money |
                date | count | identifier
  role          for places: {" | ".join(extraction_schema.PLACE_ROLES)}
                for people: {" | ".join(extraction_schema.FOUNDER_ROLES)}
  granularity   for dates: day | month | year
  source_span   the sentence the claim was taken from
  source_doc    the document that sentence came from
  asserted      whether the claim is supported by its sentence
  agreement     how many independent extraction runs found it
  run_count     how many runs there were

Facts whose value names a person, organisation, place or category also point
at a node for it via :ABOUT, so those can be traversed and counted. Facts
whose value is a date, a sum of money, a count or free text have NO :ABOUT
relationship at all - their value is the answer, and requiring the hop
silently returns nothing.

Examples
--------
When did Airbnb say it was founded? The value is the answer, no :ABOUT:
  MATCH (c:{PREFIX}Company)-[:ASSERTS]->(f:{PREFIX}Fact)
  WHERE c.name = 'airbnb' AND f.predicate = 'founded_on' AND f.asserted = true
  RETURN f.value AS founded_on, f.source_span AS evidence

Where is Stripe based? A place fact, so the entity can be traversed - but the
role is what makes it an answer to "based":
  MATCH (c:{PREFIX}Company)-[:ASSERTS]->(f:{PREFIX}Fact)-[:ABOUT]->(p:{PREFIX}Place)
  WHERE c.name = 'stripe' AND f.predicate = 'located_in'
    AND f.role = 'headquarters' AND f.asserted = true
  RETURN p.name AS place, f.source_span AS evidence

Company names are stored lowercase. Compare with toLower() on the question's
spelling, never on c.name alone.

Predicates
----------
{chr(10).join(f"  {name}: {text}"
              for name, text in extraction_schema.PREDICATE_DEFINITIONS.items())}

Rules
-----
Every MATCH on a {PREFIX}Fact MUST filter on asserted = true. Facts with
asserted = false are claims the document did not actually support - they are
kept only so a human can review them, and returning one states something no
source says.

A place is meaningless without its role. Filter on it: a question about where
a company is based means role = 'headquarters', not any place it touches.

Place names are stored as the document wrote them, so a city usually carries
its region: 'San Francisco, CA, USA', 'Dublin, Ireland'. Matching a city with
equality therefore finds almost nothing. Use
toLower(p.name) CONTAINS 'san francisco'.

Never filter a value with a case-sensitive CONTAINS. Coinbase's developer
tools are stored as 'Coinbase Developer Platform', so CONTAINS 'developer'
finds nothing. Use toLower(f.value) CONTAINS 'developer' when you must.

Better still, do not guess the document's wording. For predicates that hold a
list - offers, uses, operates_in, competes_with, invested_in_by - return every
value for the company and let the answer select from them. A substring filter
that guesses wrong is indistinguishable from the fact being absent, and the
difference matters: one is a gap in the graph, the other is a bug.
"""

# A MATCH that binds a fact and never constrains asserted is the failure this
# module exists to prevent, so the check is deliberately literal: find the
# variables bound to a fact, then require each one to be constrained.
_FACT_BINDING = re.compile(
    r"\(\s*(\w+)\s*:\s*" + FACT_LABEL + r"\b", re.IGNORECASE
)


def fact_variables(cypher: str) -> list:
    """Variables bound to a fact node."""
    seen = []
    for name in _FACT_BINDING.findall(cypher):
        if name not in seen:
            seen.append(name)
    return seen


def _constrains_asserted(cypher: str, variable: str) -> bool:
    # Accepts `f.asserted = true`, `f.asserted` as a bare predicate, and an
    # inline `{asserted: true}` on the pattern itself.
    patterns = (
        rf"\b{re.escape(variable)}\s*\.\s*asserted\b",
        rf"\(\s*{re.escape(variable)}\s*:\s*{FACT_LABEL}\s*\{{[^}}]*\basserted\b",
    )
    return any(re.search(p, cypher, re.IGNORECASE) for p in patterns)


def validate_asserted_filter(cypher: str) -> None:
    """Reject a query that reads facts without respecting their status.

    Withheld facts are not merely lower quality - they are claims the source
    never made. Returning one is exactly the confident-wrong answer the whole
    assertion model exists to prevent, so this fails closed.
    """
    unguarded = [name for name in fact_variables(cypher)
                 if not _constrains_asserted(cypher, name)]
    if unguarded:
        listed = ", ".join(unguarded)
        raise ValueError(
            f"Fact variable(s) {listed} are read without filtering on "
            f"asserted. Add {unguarded[0]}.asserted = true - facts marked "
            f"false are claims their source does not support."
        )


class Conflict(dict):
    """Two asserted facts that cannot both be the answer."""


def resolve_single_valued(facts: list) -> tuple:
    """Pick the answer for a predicate that can only have one, or refuse.

    Extraction can assert two different values for a predicate the schema says
    is single-valued - two founding dates, two tickers - when a document is
    genuinely inconsistent or two sentences were read differently. Choosing
    between them by agreement count would be guessing with a number attached,
    so an unresolved disagreement is returned as a conflict for the caller to
    abstain on rather than a value.

    Dates are the exception worth making: 'August 2008' and '2008' do not
    disagree, one is simply coarser, so the finer one wins.
    """
    if not facts:
        return None, None
    if len(facts) == 1:
        return facts[0], None

    distinct = []
    for fact in facts:
        for kept in distinct:
            if _compatible(kept, fact):
                if _finer(fact, kept):
                    distinct[distinct.index(kept)] = fact
                break
        else:
            distinct.append(fact)

    if len(distinct) == 1:
        return distinct[0], None
    return None, Conflict(
        predicate=facts[0].get("predicate"),
        values=[f.get("value") for f in distinct],
        spans=[f.get("source_span") for f in distinct],
    )


def _compatible(left: dict, right: dict) -> bool:
    if left.get("value_type") == extraction_schema.DATE:
        return _dates_agree(left.get("value"), right.get("value"))
    if left.get("value_type") == extraction_schema.PLACE:
        return extraction_schema.places_agree(left.get("value"),
                                              right.get("value"))
    return extraction_schema._same_value(left.get("value"), right.get("value"))


_MONTH_NAMES = {
    name: number
    for number, names in enumerate(
        (("january", "jan"), ("february", "feb"), ("march", "mar"),
         ("april", "apr"), ("may",), ("june", "jun"), ("july", "jul"),
         ("august", "aug"), ("september", "sept", "sep"), ("october", "oct"),
         ("november", "nov"), ("december", "dec")), start=1)
    for name in names
}


def date_parts(value) -> tuple:
    """Pull (year, month, day) out of a date however it was written.

    The graph holds `2008`, `August 2008`, `March 23, 2018` and `2021-04-14`
    side by side, because each one is quoted the way its sentence wrote it.
    Comparing those as strings makes every coarse date look like it disagrees
    with the finer date it actually contains, so they are reduced to the parts
    they state and compared there. A part nobody stated is None, which is
    absence, not a mismatch.
    """
    text = str(value or "").strip().lower()
    if not text:
        return (None, None, None)

    iso = re.match(r"^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?$", text)
    if iso:
        return (int(iso.group(1)), int(iso.group(2)),
                int(iso.group(3)) if iso.group(3) else None)

    year = None
    years = re.findall(r"\b(1[6-9]\d{2}|20\d{2})\b", text)
    if years:
        year = int(years[-1])

    month = None
    for name, number in _MONTH_NAMES.items():
        if re.search(rf"\b{name}\b", text):
            month = number
            break

    day = None
    for candidate in re.findall(r"\b(\d{1,2})\b", text):
        if 1 <= int(candidate) <= 31:
            day = int(candidate)
            break

    return (year, month, day)


def _dates_agree(left, right) -> bool:
    """True when two dates never contradict on a part they both state."""
    for mine, theirs in zip(date_parts(left), date_parts(right)):
        if mine is not None and theirs is not None and mine != theirs:
            return False
    return True


def _date_detail(value) -> int:
    return sum(1 for part in date_parts(value) if part is not None)


def _finer(candidate: dict, current: dict) -> bool:
    if candidate.get("value_type") == extraction_schema.DATE:
        # Ranked by what the value states rather than the granularity
        # property, because the property can be missing and the value cannot.
        return _date_detail(candidate.get("value")) > _date_detail(
            current.get("value"))
    if candidate.get("value_type") == extraction_schema.PLACE:
        return (len(extraction_schema.place_components(candidate.get("value")))
                > len(extraction_schema.place_components(current.get("value"))))
    return False
