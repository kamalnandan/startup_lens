"""
Semantic type contract for the startup intelligence graph.

The question space is unbounded, but the property space is closed: an answer can
only ever assert a value that came from one of the properties declared in the
graph schema. This module assigns a semantic type to every one of those
properties once, then derives generic rules from that contract:

* A ``RETURN`` alias is the claim label the synthesizer reads, so the alias's
  requested fact type must be carriable by the source property's type.
* A fact type that no property carries is unmodeled, so any question asking for
  it must be answered with an explicit "not available" rather than a value
  borrowed from an unrelated property.

Nothing here is specific to a company, a question, or an observed failure.
Adding a property to the graph means adding one entry to ``PROPERTY_FACT_TYPES``.
"""

import re

# ── Fact types ────────────────────────────────────────────────────────────────

PROVENANCE = "provenance"
PROPER_NAME = "proper_name"
FREE_TEXT = "free_text"
CATEGORICAL = "categorical"
COHORT_LABEL = "cohort_label"
ROUND_LABEL = "round_label"
COUNT = "count"
MONEY = "money"
YEAR = "year"
DATE = "date"
PLACE = "place"
IDENTIFIER = "identifier"
STATEMENT = "statement"

# Human-readable names used in validation and abstention messages.
FACT_TYPE_LABELS = {
    PROVENANCE: "internal source provenance",
    PROPER_NAME: "a proper name",
    FREE_TEXT: "descriptive prose",
    CATEGORICAL: "a categorical label",
    COHORT_LABEL: "a YC batch label",
    ROUND_LABEL: "a funding round label",
    COUNT: "a count",
    MONEY: "a monetary amount",
    YEAR: "a year",
    DATE: "a calendar date",
    PLACE: "a place",
    IDENTIFIER: "an external identifier such as a ticker symbol",
    STATEMENT: "an official statement such as a mission or policy",
}

# ── The closed property contract ──────────────────────────────────────────────
# Every property declared in GRAPH_SCHEMA, typed exactly once.
# COUNT properties declare the subject they count so a count of one kind of
# thing can never be presented as a count of another.

PROPERTY_FACT_TYPES = {
    ("company", "name"): PROPER_NAME,
    ("company", "description"): FREE_TEXT,
    ("company", "status"): CATEGORICAL,
    ("company", "stage"): CATEGORICAL,
    ("company", "team_size"): COUNT,
    ("company", "yc_batch"): COHORT_LABEL,
    ("company", "filename"): PROVENANCE,
    ("founder", "name"): PROPER_NAME,
    ("investor", "name"): PROPER_NAME,
    ("industry", "name"): CATEGORICAL,
    ("technology", "name"): CATEGORICAL,
    ("location", "city"): PLACE,
    ("location", "country"): PLACE,
    ("batch", "name"): COHORT_LABEL,
    ("fundingevent", "round"): ROUND_LABEL,
    ("fundingevent", "year"): YEAR,
    ("fundingevent", "amount"): MONEY,
    # FundingEvent.company stores the *name* of the company that raised the
    # round. It is a proper name, never an amount, valuation, or identifier.
    ("fundingevent", "company"): PROPER_NAME,
}

# Counts declare what they count.
PROPERTY_COUNT_SUBJECTS = {
    ("company", "team_size"): "people",
}

# Relationship properties are shared across relationship types.
RELATIONSHIP_PROPERTY_FACT_TYPES = {
    "amount": MONEY,
    "year": YEAR,
}

# Node labels that exist in the graph, used to recognise modeled entities.
NODE_LABELS = frozenset(
    {"company", "founder", "investor", "industry", "location", "technology",
     "batch", "fundingevent"}
)

# Relationships that connect peers of the same label. A projection reached only
# through one of these has left the subject the question anchored on.
PEER_RELATIONSHIPS = frozenset({"CO_FOUNDED_WITH", "COMPETES_WITH"})


# Relationship types and the properties they carry. Relationships with no
# properties cannot answer questions about the order, timing, or role of the
# connection they represent.
RELATIONSHIP_PROPERTIES = {
    "FOUNDED": frozenset(),
    "CO_FOUNDED_WITH": frozenset(),
    "INVESTED_IN": frozenset(),
    "OPERATES_IN": frozenset(),
    "HEADQUARTERED_IN": frozenset(),
    "USES": frozenset(),
    "PART_OF": frozenset(),
    "ACQUIRED_BY": frozenset({"year", "amount"}),
    "RAISED": frozenset({"amount"}),
    "COMPETES_WITH": frozenset(),
}

# Wording that asks about the order, timing, or role of a connection rather
# than the connection itself.
RELATIONSHIP_ASPECT_MARKERS = frozenset(
    {"later", "originally", "original", "first", "earlier", "initially",
     "subsequently", "applied", "joined", "added", "order", "sequence",
     "before", "after", "which one", "who applied"}
)

# Question wording that names a relationship with no stored attributes.
RELATIONSHIP_SUBJECT_MARKERS = {
    "FOUNDED": ("founded", "founder", "founders", "founding", "co-founder",
                "cofounder", "co-founders", "cofounders"),
    "INVESTED_IN": ("invested", "investor", "investors", "backed"),
    "USES": ("uses", "technology", "technologies"),
}

# Wording that directs a question at a person rather than an organisation.
PERSON_QUESTION_MARKERS = frozenset({"who", "whom", "whose"})


def unmodeled_relationship_aspects(question: str) -> list:
    """Return notes for relationship attributes the graph does not store."""
    if not question:
        return []
    normalized = " ".join(_tokenize(question))
    tokens = set(normalized.split())
    if not any(
        marker in normalized if " " in marker else marker in tokens
        for marker in RELATIONSHIP_ASPECT_MARKERS
    ):
        return []

    notes = []
    for relationship, markers in RELATIONSHIP_SUBJECT_MARKERS.items():
        if RELATIONSHIP_PROPERTIES.get(relationship):
            continue
        if any(marker in tokens for marker in markers):
            notes.append(
                f"The {relationship} relationship stores no order, date, or "
                f"role. The data cannot show who came first, who joined later, "
                f"or in what capacity. Do not infer it from the record order."
            )
    if not notes and tokens & PERSON_QUESTION_MARKERS:
        # FOUNDED is the only relationship the contract gives between a person
        # and a company, so any person-directed aspect question lands on it.
        notes.append(
            "The FOUNDED relationship stores no order, date, or role. The data "
            "cannot show who came first, who joined later, or in what "
            "capacity. Do not infer it from the record order."
        )
    return notes


def property_fact_type(label: str, property_name: str):
    """Return the semantic type of a node property, or None when unknown."""
    return PROPERTY_FACT_TYPES.get((label.casefold(), property_name.casefold()))


def count_subject_for_property(label: str, property_name: str):
    """Return what a COUNT-typed property counts."""
    return PROPERTY_COUNT_SUBJECTS.get((label.casefold(), property_name.casefold()))


# ── Requested fact types inferred from wording ────────────────────────────────
# Ordered most specific first; the first marker found in an alias or question
# phrase determines the requested type.

FACT_TYPE_MARKERS = (
    (IDENTIFIER, (
        "ticker", "tickers", "symbol", "symbols", "cusip", "isin", "cik",
    )),
    (STATEMENT, (
        "mission", "vision", "purpose", "policy", "policies", "terms",
        "conditions", "guarantee", "pledge", "commitment", "principle",
        "principles", "values",
    )),
    (MONEY, (
        "price", "prices", "pricing", "valuation", "valued", "proceeds",
        "revenue", "volume", "worth", "cost", "fee", "fees", "amount",
        "raised", "funding", "cap", "capitalization",
    )),
    (DATE, (
        "date", "dates", "day", "month", "birthday", "anniversary",
    )),
    (COUNT, (
        "count", "number", "total", "headcount", "quantity",
        "how many",
    )),
    (YEAR, (
        "year", "years",
    )),
    (PLACE, (
        "city", "cities", "country", "countries", "region", "regions",
        "location", "locations", "headquarters", "hq", "based",
    )),
    (COHORT_LABEL, (
        "batch", "cohort",
    )),
    (ROUND_LABEL, (
        "round", "rounds",
    )),
    (CATEGORICAL, (
        "industry", "industries", "sector", "sectors", "category",
        "categories", "technology", "technologies", "status", "stage",
    )),
    (PROPER_NAME, (
        "name", "names", "founder", "founders", "investor", "investors",
        "company", "companies", "acquirer",
    )),
)

# Words that describe how something is counted rather than what is counted.
COUNT_MARKER_WORDS = frozenset(
    {"count", "number", "total", "quantity", "how", "many", "of", "the", "a"}
)

# Subjects that all mean "people at the company".
PEOPLE_SUBJECTS = frozenset(
    {"people", "person", "employee", "employees", "team", "teams", "staff",
     "headcount", "member", "members", "worker", "workers", "size",
     "employment"}
)


def _tokenize(text: str) -> list:
    return [token for token in re.split(r"[^a-z0-9]+", text.casefold()) if token]


def infer_requested_fact_type(phrase: str):
    """Infer the fact type a phrase asks for, or None when it is not clear.

    The head noun of an alias determines its meaning, so ``funding_round_count``
    is a count rather than an amount. Unknown wording deliberately returns None
    so validation fails open rather than rejecting queries it cannot classify.
    """
    if not phrase:
        return None
    tokens = _tokenize(phrase)
    if not tokens:
        return None

    for candidate in (tokens[-1:], tokens):
        candidate_tokens = set(candidate)
        normalized = " ".join(candidate)
        for fact_type, markers in FACT_TYPE_MARKERS:
            for marker in markers:
                if " " in marker:
                    if marker in normalized:
                        return fact_type
                elif marker in candidate_tokens:
                    return fact_type
    return None


def infer_count_subject(phrase: str):
    """Return what a count-style phrase counts, or None when unclear."""
    tokens = [token for token in _tokenize(phrase) if token not in COUNT_MARKER_WORDS]
    if not tokens:
        return None
    subject = tokens[-1]
    if subject in PEOPLE_SUBJECTS:
        return "people"
    return subject.rstrip("s") or subject


def count_subjects_match(source_subject, requested_subject) -> bool:
    """Two counts are interchangeable only when they count the same thing."""
    if not source_subject or not requested_subject:
        return True
    if source_subject in PEOPLE_SUBJECTS and requested_subject in PEOPLE_SUBJECTS:
        return True
    return source_subject.rstrip("s") == requested_subject.rstrip("s")


# ── Compatibility ─────────────────────────────────────────────────────────────
# What each stored type is allowed to be presented as. Types absent from the
# values of this map are carried by no property and are therefore unmodeled.

COMPATIBLE_REQUESTED_TYPES = {
    PROVENANCE: frozenset(),
    PROPER_NAME: frozenset({PROPER_NAME}),
    FREE_TEXT: frozenset({FREE_TEXT}),
    CATEGORICAL: frozenset({CATEGORICAL, PROPER_NAME}),
    COHORT_LABEL: frozenset({COHORT_LABEL}),
    ROUND_LABEL: frozenset({ROUND_LABEL}),
    COUNT: frozenset({COUNT}),
    MONEY: frozenset({MONEY}),
    YEAR: frozenset({YEAR}),
    PLACE: frozenset({PLACE, PROPER_NAME}),
}

UNMODELED_FACT_TYPES = frozenset(
    {IDENTIFIER, STATEMENT, DATE}
)


def is_compatible(source_type, requested_type) -> bool:
    """Whether a stored value may be presented as the requested kind of fact."""
    if source_type is None or requested_type is None:
        return True
    if source_type == requested_type:
        return source_type not in (PROVENANCE,)
    return requested_type in COMPATIBLE_REQUESTED_TYPES.get(source_type, frozenset())


def unmodeled_reason(requested_type) -> str:
    """Explain why a requested fact type cannot be answered from the graph."""
    return (
        f"The graph stores no property holding {FACT_TYPE_LABELS[requested_type]}"
    )


# ── Cypher projection analysis ────────────────────────────────────────────────

def variable_labels(cypher: str) -> dict:
    """Map each pattern variable to the node label it is bound to."""
    labels = {}
    for variable, label in re.findall(
        r"\(\s*(\w+)\s*:\s*(\w+)", cypher
    ):
        labels.setdefault(variable, label)
    return labels


def _split_top_level(clause: str) -> list:
    parts = []
    depth = 0
    current = []
    in_quote = None
    for character in clause:
        if in_quote:
            current.append(character)
            if character == in_quote:
                in_quote = None
            continue
        if character in "\"'":
            in_quote = character
            current.append(character)
            continue
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        if character == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(character)
    if current:
        parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def return_projections(cypher: str) -> list:
    """Return (expression, alias) pairs from the final RETURN clause."""
    match = re.search(r"\bRETURN\b(.*)", cypher, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    clause = re.split(
        r"\b(?:ORDER\s+BY|SKIP|LIMIT)\b",
        match.group(1),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    projections = []
    for part in _split_top_level(clause):
        alias_match = re.search(
            r"^(.*?)\s+AS\s+(\w+)\s*$", part, flags=re.IGNORECASE | re.DOTALL
        )
        if alias_match:
            projections.append(
                (alias_match.group(1).strip(), alias_match.group(2).strip())
            )
        else:
            projections.append((part, None))
    return projections


def carried_projections(cypher: str) -> list:
    """Return (expression, alias) pairs from every WITH and RETURN clause.

    Values renamed in a WITH clause reach the answer under the new alias, so
    projection rules must see them too.
    """
    projections = list(return_projections(cypher))
    for clause in re.findall(
        r"\bWITH\b(.*?)(?=\b(?:OPTIONAL\s+)?MATCH\b|\bWHERE\b|\bWITH\b|"
        r"\bRETURN\b|\bORDER\s+BY\b|$)",
        cypher,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        for part in _split_top_level(clause):
            alias_match = re.search(
                r"^(.*?)\s+AS\s+(\w+)\s*$", part, flags=re.IGNORECASE | re.DOTALL
            )
            if alias_match:
                projections.append(
                    (alias_match.group(1).strip(), alias_match.group(2).strip())
                )
            else:
                projections.append((part, None))
    return projections


def projection_source(expression: str, labels: dict):
    """Describe what a projection expression reads from the graph.

    Returns ``(fact_type, count_subject, description)``. ``fact_type`` is None
    when the expression cannot be attributed to a typed property, which makes
    dependent checks fail open.
    """
    aggregate = re.match(r"^\s*count\s*\(", expression, flags=re.IGNORECASE)
    references = re.findall(r"\b(\w+)\.(\w+)\b", expression)

    if aggregate:
        # A computed count is produced by the query itself, so its alias names
        # the slice being counted rather than a stored quantity. Only stored
        # counts declare a fixed subject that may not be relabelled.
        return COUNT, None, "a computed count"

    for variable, property_name in references:
        label = labels.get(variable)
        if not label:
            continue
        fact_type = property_fact_type(label, property_name)
        if fact_type is None:
            continue
        subject = count_subject_for_property(label, property_name)
        return fact_type, subject, f"{label.casefold()}.{property_name.casefold()}"

    return None, None, expression.strip()


# ── Question-side coverage ────────────────────────────────────────────────────

MARKERS_BY_FACT_TYPE = {fact_type: markers for fact_type, markers in FACT_TYPE_MARKERS}

# Wording that asks for calendar precision the graph does not store. Kept
# deliberately explicit so ordinary "when" questions still use modeled years.
DATE_PRECISION_PHRASES = (
    "what date", "which date", "exact date", "specific date", "founding date",
    "date of founding", "month and year", "what month", "which month",
    "what day", "which day", "full date",
)


def unmodeled_facts_in_question(question: str) -> list:
    """Return the fact types a question asks for that no property can carry."""
    if not question:
        return []
    normalized = " ".join(_tokenize(question))
    tokens = set(normalized.split())
    found = []
    for fact_type in (IDENTIFIER, STATEMENT, DATE):
        markers = MARKERS_BY_FACT_TYPE.get(fact_type, ())
        matched = any(
            marker in normalized if " " in marker else marker in tokens
            for marker in markers
        )
        if fact_type == DATE:
            matched = any(phrase in normalized for phrase in DATE_PRECISION_PHRASES)
        if matched:
            found.append(fact_type)
    return found


def synthesis_constraints(unmodeled_types=None, question: str = "") -> list:
    """Build synthesis rules from the contract, not from observed failures.

    Fact types that no property carries are unmodeled for every question, so
    their constraints are unconditional rather than triggered by wording.
    """
    constraints = [
        "A YC batch label such as \"Summer 2013\" identifies a cohort. Never "
        "present it as a founding, listing, or event date.",
        "A company description is prose about what a company does. Never "
        "present it as an official mission, policy, or terms statement, and "
        "never infer customer segments, products, offerings, or capabilities "
        "it does not state.",
        "Never mention internal source provenance such as file names, and "
        "never present one as a fact about a company.",
        "A count of records in the graph is not a real-world total. Only state "
        "a count as the answer when the graph models the thing being counted.",
        "A linked Location is the company's headquarters. Never present it as "
        "where a company started, was founded, or was born.",
    ]
    for fact_type in sorted(set(unmodeled_types or ()) | UNMODELED_FACT_TYPES):
        constraints.append(
            f"{unmodeled_reason(fact_type)}. If the question asks for it, say "
            f"plainly that it is not available in the data instead of "
            f"substituting another value."
        )
    constraints.extend(unmodeled_relationship_aspects(question))
    return constraints
