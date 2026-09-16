"""Assertion-based extraction schema for the startup knowledge graph.

The original loader extracted into ten fixed slots. Facts the source stated
but the slots could not hold were discarded, and facts the slots required but
the source did not state were invented to fill them. Both failures are
structural, so this schema changes the unit of extraction.

The unit here is an *assertion*: one predicate about one subject, carrying the
verbatim span of source text that states it. A span that does not literally
occur in the document is rejected, which turns "extract only what is stated"
from a prompt instruction into a check the loader enforces.

Three further rules follow from the failures this replaces:

- Values are typed, and a date carries its own granularity, so a year is never
  mistaken for a calendar date and never discarded for failing to be one.
- Places carry a role, so a market is never stored as a headquarters.
- Absent is null and never zero, so "unknown" is distinguishable from a real
  measurement.
"""

from __future__ import annotations

import json
import re
import unicodedata


# ── Value types ────────────────────────────────────────────────────────────────

TEXT = "text"
PERSON = "person"
ORGANISATION = "organisation"
PLACE = "place"
MONEY = "money"
DATE = "date"
COUNT = "count"
IDENTIFIER = "identifier"
CATEGORY = "category"

VALUE_TYPES = frozenset(
    {TEXT, PERSON, ORGANISATION, PLACE, MONEY, DATE, COUNT, IDENTIFIER, CATEGORY}
)

# Date granularity is part of the value. A year-granular date answers "which
# year" truthfully and "which day" not at all, and the distinction is only
# representable if it travels with the value.
DAY = "day"
MONTH = "month"
YEAR = "year"
DATE_GRANULARITIES = (DAY, MONTH, YEAR)


# ── Predicates ─────────────────────────────────────────────────────────────────

# Each predicate declares the type of value it carries and whether a subject
# may hold more than one. Extending the graph means adding a row here, not
# reshaping the loader.
PREDICATES = {
    "name":            {"type": ORGANISATION, "multi": False},
    "description":     {"type": TEXT,         "multi": False},
    "mission":         {"type": TEXT,         "multi": False},
    "status":          {"type": CATEGORY,     "multi": False},
    "stage":           {"type": CATEGORY,     "multi": False},
    "team_size":       {"type": COUNT,        "multi": False, "counts": "people"},
    "yc_batch":        {"type": CATEGORY,     "multi": False},
    "founded_on":      {"type": DATE,         "multi": False},
    # A company's story usually starts before the company does: Airbnb's
    # air mattress in October 2007, incorporated in August 2008. Recording
    # only the later date makes "how did it begin" unanswerable, and putting
    # both under founded_on would make the founding date ambiguous instead.
    "originated_on":   {"type": DATE,         "multi": False},
    "founded_by":      {"type": PERSON,       "multi": True},
    "operates_in":     {"type": CATEGORY,     "multi": True},
    "uses":            {"type": CATEGORY,     "multi": True},
    # What a company sells is not the sector it sells into. Stripe Terminal
    # and Coinbase's developer API are named products, and with only
    # operates_in and description to hold them they were dropped entirely.
    "offers":          {"type": TEXT,         "multi": True},
    "invested_in_by":  {"type": ORGANISATION, "multi": True},
    "competes_with":   {"type": ORGANISATION, "multi": True},
    "acquired_by":     {"type": ORGANISATION, "multi": False},
    "located_in":      {"type": PLACE,        "multi": True},
    "ticker":          {"type": IDENTIFIER,   "multi": False},
    "listed_on":       {"type": ORGANISATION, "multi": False},
    # listed_on names the exchange, which left the date a listing actually
    # happened with nowhere to go - so "when did DBX start trading" had no
    # answer even though the document said it plainly.
    "first_traded_on": {"type": DATE,         "multi": False},
    "ipo_price":       {"type": MONEY,        "multi": False},
    "ipo_proceeds":    {"type": MONEY,        "multi": False},
    "ipo_share_count": {"type": COUNT,        "multi": False, "counts": "shares"},
    "listing_count":   {"type": COUNT,        "multi": False, "counts": "listings"},
    "market_count":    {"type": COUNT,        "multi": False, "counts": "markets"},
    "raised":          {"type": MONEY,        "multi": True},
}

# A place means nothing without the role it plays. Collapsing every place into
# a headquarters is what turned Airbnb's markets into twelve head offices.
PLACE_ROLES = ("headquarters", "office", "market", "founding_location",
               "incorporation")

# A person's connection to a company likewise has a role and an order. The
# previous schema stored neither, so "who applied first" had to be guessed.
FOUNDER_ROLES = ("original_applicant", "founder", "co_founder", "later_co_founder")

# Words that name an arrangement rather than a place. "Remote" is a working
# model, not somewhere a company is located.
NON_PLACE_VALUES = frozenset({"remote", "remote-first", "distributed", "none",
                              "n/a", "unknown", "worldwide", "global"})


def _same_value(left, right) -> bool:
    """Compare two assertion values for practical equality."""
    if isinstance(left, str) and isinstance(right, str):
        return _normalise(left) == _normalise(right)
    return left == right


# A verbatim span proves the sentence exists, not that it supports the claim.
# A sentence can name someone precisely in order to say they were NOT involved,
# which reads as confirmation to a extractor matching on names. These phrases
# mark a span as stating something hypothetical, denied, or undone.
COUNTERFACTUAL_MARKERS = (
    "supposed to", "was to be", "would have", "planned to", "intended to",
    "parted ways", "never ", "declined", "turned down", "backed out",
    "withdrew", "rejected", "instead of", "rather than", "but left",
    "before leaving", "failed to", "did not", "didn't", "no longer",
    "former", "formerly", "ex-", "stepped down", "resigned", "departed",
    "considered", "rumou", "reportedly", "allegedly", "expected to",
)


def counterfactual_markers_in(span: str) -> list:
    """Return markers suggesting the span denies or hypothesises the fact.

    Matching is word-bounded because these markers are cheap to trip over:
    "former" inside "roommates and former schoolmates" says nothing about
    whether the sentence supports the claim, and an unbounded substring search
    cannot tell that from "the former chief executive".
    """
    normalised = _normalise(span)
    found = []
    for marker in COUNTERFACTUAL_MARKERS:
        stem = marker.strip()
        pattern = r"\b" + re.escape(stem) + (r"" if stem.endswith("-") else r"\b")
        if re.search(pattern, normalised):
            found.append(stem)
    return found


class ExtractionError(ValueError):
    """Raised when a model response violates the schema contract."""


# ── Span verification ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Fold quotes, dashes and whitespace so spans survive transcription."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def span_occurs_in(span: str, document: str) -> bool:
    """Return whether a quoted span genuinely appears in the document."""
    if not span or not span.strip():
        return False
    return _normalise(span) in _normalise(document)


# ── Assertion validation ───────────────────────────────────────────────────────

REQUIRED_FIELDS = ("predicate", "value", "source_span")

# Models return these as strings when they mean the field is absent, and a
# literal "None" is not a granularity or a role.
NULL_TOKENS = frozenset({"none", "null", "n/a", "na", "unknown", ""})


def _blank(value) -> bool:
    return value is None or (isinstance(value, str)
                             and value.strip().lower() in NULL_TOKENS)


def validate_assertion(assertion: dict, document: str) -> dict:
    """Validate one assertion against the schema and its source document.

    Returns the normalised assertion. Raises ExtractionError describing the
    first violation, so a caller can drop the assertion and keep the rest.
    """
    if not isinstance(assertion, dict):
        raise ExtractionError("Assertion must be an object.")

    for field in REQUIRED_FIELDS:
        if field not in assertion or assertion[field] in (None, ""):
            raise ExtractionError(f"Assertion is missing {field}.")

    predicate = assertion["predicate"]
    declared = PREDICATES.get(predicate)
    if declared is None:
        raise ExtractionError(f"Unknown predicate {predicate!r}.")

    # The predicate already declares its type, so asking the model to restate it
    # only creates a way to be wrong about something we know: a reply saying
    # "string" instead of "place" once cost every location in the document.
    # A stated type that disagrees is a real signal though - it means the model
    # had a different kind of thing in mind - so it is recorded, not ignored.
    stated_type = assertion.get("value_type")
    value_type = declared["type"]
    assertion["value_type"] = value_type
    if (not _blank(stated_type) and stated_type in VALUE_TYPES
            and stated_type != value_type):
        assertion.setdefault("review", []).append(
            f"extracted as {stated_type}, stored as {value_type}"
        )

    for field in ("granularity", "role", "qualifier"):
        if _blank(assertion.get(field)):
            assertion.pop(field, None)

    span = assertion["source_span"]
    if not span_occurs_in(span, document):
        raise ExtractionError(
            f"Source span for {predicate!r} does not occur in the document."
        )

    markers = counterfactual_markers_in(span)
    if markers:
        # A marker is a reason to look closely, not a verdict. Withholding on
        # the marker alone made it one that nothing could overturn, and
        # "roommates and former schoolmates" then cost Airbnb its origin date.
        # So the fact is withheld pending the verification pass, which asks the
        # precise question - does this sentence assert this claim? - and can
        # release it. If verification never runs, it stays withheld.
        assertion.setdefault("review", []).append(
            f"span may not assert this fact: {', '.join(sorted(markers))}"
        )
        assertion["needs_check"] = True
        assertion["asserted"] = False
    else:
        assertion.setdefault("asserted", True)

    if value_type == DATE:
        granularity = assertion.get("granularity")
        if granularity not in DATE_GRANULARITIES:
            raise ExtractionError(
                f"A date needs a granularity in {DATE_GRANULARITIES}."
            )

    if value_type == COUNT:
        value = assertion["value"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ExtractionError(f"Count for {predicate!r} must be an integer.")
        if value < 0:
            raise ExtractionError(f"Count for {predicate!r} cannot be negative.")

    if value_type == PLACE:
        if _normalise(str(assertion["value"])) in NON_PLACE_VALUES:
            raise ExtractionError(
                f"{assertion['value']!r} names a working arrangement, not a place."
            )
        role = assertion.get("role")
        if role not in PLACE_ROLES:
            # A place whose role is unrecognised is still a stated fact. Drop
            # the role rather than the assertion, and mark it for review, so a
            # vocabulary gap never silently deletes evidence.
            assertion["role"] = None
            assertion.setdefault("review", []).append(
                f"unrecognised place role {role!r}"
            )

    if predicate == "founded_by":
        role = assertion.get("role")
        if role is not None and role not in FOUNDER_ROLES:
            assertion["role"] = None
            assertion.setdefault("review", []).append(
                f"unrecognised founder role {role!r}"
            )

    return assertion


def validate_extraction(payload: dict, document: str) -> tuple:
    """Split a model response into accepted assertions and rejections.

    Rejecting rather than failing keeps one bad assertion from discarding a
    document, and the rejection list is the signal for improving extraction.
    """
    if not isinstance(payload, dict):
        raise ExtractionError("Extraction payload must be an object.")

    assertions = payload.get("assertions")
    if not isinstance(assertions, list):
        raise ExtractionError("Extraction payload needs an 'assertions' list.")

    accepted, rejected = [], []
    seen = {}
    for assertion in assertions:
        try:
            validated = validate_assertion(assertion, document)
        except ExtractionError as error:
            rejected.append(
                {"assertion": assertion, "reason": str(error)}
            )
            continue

        predicate = validated["predicate"]
        if PREDICATES[predicate]["multi"]:
            # The same fact is often stated in several places. Keep the most
            # specific statement rather than one row per mention.
            key = (predicate, _normalise(str(validated["value"])))
            previous = seen.get(key)
            if previous is not None:
                if previous.get("role") is None and validated.get("role"):
                    previous["role"] = validated["role"]
                    previous["source_span"] = validated["source_span"]
                    previous.pop("review", None)
                continue
            seen[key] = validated
        else:
            first = seen.get(predicate)
            if first is not None:
                # A source can state a single-valued fact more than once, and
                # the statements can disagree. Keeping both with their spans
                # preserves the disagreement for the reader to resolve;
                # discarding one would manufacture false certainty.
                if _same_value(first["value"], validated["value"]):
                    continue
                validated["conflicts_with"] = first["source_span"]
                first.setdefault("conflicts_with", validated["source_span"])
                accepted.append(validated)
                continue
            seen[predicate] = validated

        accepted.append(validated)

    return accepted, rejected


# ── Extraction prompt ──────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """Extract factual assertions about {company} from the document below.

Return ONLY valid JSON of this shape:
{{"assertions": [
  {{"predicate": "...", "value": ..., "value_type": "...", "source_span": "...",
    "granularity": "...", "role": "...", "qualifier": "..."}}
]}}

Rules, all of which are checked after you answer:

1. Every assertion MUST include "source_span": a span copied VERBATIM from the
   document that states the fact. An assertion whose span is not found in the
   document is discarded. Never paraphrase, correct, or shorten mid-word.
2. Extract a fact ONLY if the document states it. Do not infer, calculate,
   combine, or complete facts from your own knowledge.
3. Omit a predicate entirely when the document does not state it. Never emit a
   placeholder, a zero, or "unknown".
4. Use these predicates and value types only:
{predicate_table}
5. Dates carry "granularity": "day", "month", or "year" — whichever the
   document actually supports. "in 2012" is year. "June 2012" is month.
6. Places carry "role": {place_roles}. A market or service area is NOT a
   headquarters. If the document says the company is remote or has no physical
   headquarters, do not emit a headquarters.
7. "founded_by" may carry "role": {founder_roles}. Use "original_applicant"
   only when the document says that person applied or started first, and
   "later_co_founder" only when it says they joined afterwards. Omit the role
   when the document states no order.
8. Counts are integers and must count the thing the predicate names. Use
   "qualifier" to record what the count is over when the document scopes it.
9. When the document contradicts itself, emit both assertions with their own
   spans. Do not silently choose one.
10. "ipo_price" is the price of ONE share. Total money raised is
   "ipo_proceeds". Do not use one for the other.
11. Do not emit a place for a working arrangement such as "Remote" or
   "Distributed". Those are not locations.

Document:
{document}"""


# Extraction picks a predicate while juggling 23 of them over a whole document,
# and reliably confuses money it received with money it spent or was fined.
# Verification asks one narrow question about one fact, so it needs a sharper
# statement of what each predicate means than the extraction table gives.
PREDICATE_DEFINITIONS = {
    "name": "the company's own name",
    "description": "what the company does",
    "mission": "the company's stated vision or mission",
    "status": "the company's operating status, such as public or acquired",
    "stage": "the company's funding or maturity stage",
    "team_size": "how many people the company employs",
    "yc_batch": "the Y Combinator batch the company attended",
    "founded_on": "the date the company was founded or incorporated",
    "originated_on": "the date the company's story began BEFORE it was formally "
                     "founded - when the idea started or the first version "
                     "launched; only use this when the document also gives a "
                     "later, separate founding date",
    "founded_by": "a person who founded the company",
    "operates_in": "an industry or sector the company operates in",
    "uses": "a technology or tag associated with the company",
    "offers": "a named product or service the company sells or provides, such "
              "as a platform, an app or a developer API; NOT the sector it "
              "works in, NOT a company it owns",
    "invested_in_by": "an organisation that invested IN the company",
    "competes_with": "a company named as a competitor",
    "acquired_by": "the organisation that ACQUIRED this company; NOT a company "
                   "this company bought, NOT a subsidiary or holding company it created",
    "located_in": "a place the company has a stated connection to",
    "ticker": "the company's stock ticker symbol",
    "listed_on": "the stock exchange the company is listed on",
    "first_traded_on": "the date the company's shares first traded, or were "
                       "expected to first trade, on an exchange",
    "ipo_price": "the price of ONE share at IPO",
    "ipo_proceeds": "the total money the company raised in its IPO",
    "ipo_share_count": "how many shares were offered at IPO",
    "listing_count": "how many listings are on the company's platform",
    "market_count": "how many markets, cities or countries the company operates in; "
                    "NOT users, NOT customers, NOT employees",
    "raised": "money the company RECEIVED as investment or funding; NOT money it "
              "paid, spent, was fined, lost, or agreed to pay for an acquisition",
}

VERIFICATION_PROMPT = """You are checking extracted facts about {company}.

For each numbered item you get a claim and the sentence it was taken from.
Decide ONLY this: does that sentence actually assert that claim?

Answer "no" when the sentence is about something else that merely resembles the
claim - money paid rather than received, a company acquired rather than an
acquirer, users counted rather than markets - or when the sentence denies or
hypothesises the claim rather than stating it.

Do not use outside knowledge. Judge the sentence alone. The claim may be true in
the world and still fail, if this sentence does not state it.

Return ONLY valid JSON:
{{"verdicts": [{{"id": 1, "supports": true}}, {{"id": 2, "supports": false,
  "reason": "sentence describes a fine, not funding"}}]}}

Items:
{items}"""


def build_verification_prompt(company: str, assertions: list) -> str:
    """Ask, for each assertion, whether its span really asserts it."""
    lines = []
    for index, assertion in enumerate(assertions, start=1):
        meaning = PREDICATE_DEFINITIONS.get(assertion["predicate"], assertion["predicate"])
        lines.append(
            f"{index}. claim: {assertion['predicate']} = {assertion['value']!r}"
            f"\n   meaning: {meaning}"
            f"\n   sentence: \"{assertion['source_span']}\""
        )
    return VERIFICATION_PROMPT.format(company=company, items="\n".join(lines))


def apply_verdicts(assertions: list, verdicts: list) -> list:
    """Mark assertions their span does not support as unasserted."""
    by_id = {}
    for verdict in verdicts or []:
        try:
            by_id[int(verdict.get("id"))] = verdict
        except (TypeError, ValueError):
            continue

    for index, assertion in enumerate(assertions, start=1):
        verdict = by_id.get(index)
        if verdict is None:
            # No verdict is not evidence of support, but discarding the fact
            # would punish it for a truncated response, so flag it instead.
            assertion.setdefault("review", []).append("not verified")
            continue
        if not verdict.get("supports", True):
            reason = verdict.get("reason") or "span does not assert this claim"
            assertion.setdefault("review", []).append(f"unsupported: {reason}")
            assertion["asserted"] = False
        elif assertion.pop("needs_check", False):
            # Held back only because the sentence contained a suspicious word.
            # Asked directly, the model says the sentence does assert the
            # claim, and that answer is better evidence than the word was.
            assertion.setdefault("review", []).append(
                "marker checked: span does assert this fact"
            )
            assertion["asserted"] = True
    return assertions


# ── Place normalisation (layer 1: structural) ─────────────────────────────────
#
# One city arrives written three ways in one document - "San Francisco",
# "San Francisco, California", "San Francisco, CA, USA" - because three
# different sentences mention it. Stored as written, that is three
# headquarters, and a question about where the company is based has three
# answers. Collapsing them needs no knowledge of San Francisco specifically:
# place strings are almost always "City[, Region][, Country]", so splitting on
# commas and expanding a *bounded* set of abbreviations gets there structurally.
#
# What this layer deliberately cannot do is resolve aliases that the string
# does not contain - NYC, Bengaluru/Bangalore, Bombay/Mumbai. Those need a
# gazetteer, and are left to a later layer rather than guessed at here.

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut",
    "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii",
    "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine",
    "md": "maryland", "ma": "massachusetts", "mi": "michigan",
    "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota",
    "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

COUNTRY_ALIASES = {
    "usa": "united states", "u.s.": "united states", "u.s.a.": "united states",
    "us": "united states", "united states of america": "united states",
    "america": "united states",
    "uk": "united kingdom", "u.k.": "united kingdom",
    "great britain": "united kingdom", "britain": "united kingdom",
    "uae": "united arab emirates",
    "roi": "ireland", "republic of ireland": "ireland",
    "prc": "china", "people's republic of china": "china",
    "south korea": "korea", "republic of korea": "korea",
    "the netherlands": "netherlands", "holland": "netherlands",
}


def place_components(value) -> tuple:
    """Split a place string into normalised, expanded components."""
    components = []
    for part in str(value).split(","):
        part = _normalise(part)
        if not part:
            continue
        part = US_STATES.get(part.rstrip("."), part)
        part = COUNTRY_ALIASES.get(part, part)
        if part not in components:
            components.append(part)
    return tuple(components)


def _is_subsequence(shorter: tuple, longer: tuple) -> bool:
    it = iter(longer)
    return all(component in it for component in shorter)


def places_agree(left, right) -> bool:
    """Whether two place strings are the same place written differently.

    The test is subsequence containment rather than position-by-position
    equality, because "San Francisco, United States" omits the state that
    "San Francisco, California, United States" states - the components line up
    in order, not in slots. Two places that both name a region and name
    different ones stay separate: Springfield, Illinois is not Springfield,
    Missouri, and South San Francisco is not San Francisco.
    """
    left_parts, right_parts = place_components(left), place_components(right)
    if not left_parts or not right_parts:
        return False
    if left_parts[0] != right_parts[0]:
        return False
    if len(left_parts) > len(right_parts):
        left_parts, right_parts = right_parts, left_parts
    return _is_subsequence(left_parts, right_parts)


def merge_place_assertions(assertions: list) -> list:
    """Collapse differently-written spellings of one place into one fact.

    The surviving fact is the most specific spelling, and it carries every
    span that mentioned the place, so the evidence for it grows rather than
    being thrown away with the duplicate.
    """
    places, others = [], []
    for assertion in assertions:
        if (assertion.get("value_type") == PLACE
                and isinstance(assertion.get("value"), str)):
            places.append(assertion)
        else:
            others.append(assertion)

    clusters = []
    for assertion in places:
        # Role matters here in a way it does not for agreement: a city that is
        # an office is not the same fact as a city that is a headquarters.
        key = (assertion.get("predicate"), assertion.get("role"))
        for cluster in clusters:
            if cluster["key"] == key and places_agree(
                cluster["members"][0]["value"], assertion["value"]
            ):
                cluster["members"].append(assertion)
                break
        else:
            clusters.append({"key": key, "members": [assertion]})

    merged = []
    for cluster in clusters:
        members = sorted(
            cluster["members"],
            key=lambda a: (len(place_components(a["value"])), len(a["value"])),
            reverse=True,
        )
        winner = dict(members[0])
        spans, variants = [], []
        for member in members:
            for span in [member.get("source_span")] + list(
                member.get("supporting_spans", [])
            ):
                if span and span not in spans:
                    spans.append(span)
            if member["value"] != winner["value"] and member["value"] not in variants:
                variants.append(member["value"])
        if len(spans) > 1:
            winner["supporting_spans"] = spans[1:]
        if variants:
            winner["also_written_as"] = variants
            winner["agreement"] = max(m.get("agreement", 1) for m in members)
        merged.append(winner)

    return others + merged


def assertion_identity(assertion: dict) -> tuple:
    """Identity used to decide whether two runs found the same fact.

    Role and span are excluded deliberately. Runs often agree that Ehrsam is a
    founder while disagreeing on whether the document called him a later one,
    and quote neighbouring sentences for the same fact. Keying on those would
    report agreement as disagreement and discard facts every run found.
    """
    value = assertion.get("value")
    if isinstance(value, str):
        value = _normalise(value)
    return (assertion.get("predicate"), value)


def merge_runs(runs: list, min_agreement: int = 2) -> tuple:
    """Keep facts that several independent runs found.

    Extraction is not reproducible even at temperature zero, so a single pass
    silently varies: one run records a founder's role, the next drops it. Facts
    that recur are the ones the document supports plainly. Facts seen once are
    kept but marked unasserted, because a single sighting is as likely to be an
    omission by the other runs as an invention by this one.
    """
    total_runs = len(runs)
    seen = {}

    def find_identity(assertion):
        identity = assertion_identity(assertion)
        if identity in seen or assertion.get("value_type") != PLACE:
            return identity
        # A run that wrote "San Francisco" and a run that wrote
        # "San Francisco, CA, USA" agreed; only the spelling differed. Matching
        # them here means the fact is seen twice and survives the vote, instead
        # of being two facts each seen once and both discarded.
        for key, entry in seen.items():
            candidate = entry["assertion"]
            if (key[0] == identity[0]
                    and candidate.get("role") == assertion.get("role")
                    and places_agree(candidate.get("value"), assertion["value"])):
                return key
        return identity

    for run_assertions in runs:
        for assertion in run_assertions:
            identity = find_identity(assertion)
            existing = seen.get(identity)
            if existing is None:
                seen[identity] = {"assertion": dict(assertion), "count": 1}
                continue
            existing["count"] += 1
            # Prefer the richer sighting: a run that captured the role or the
            # granularity saw more of what the document said.
            current = existing["assertion"]
            for field in ("role", "granularity", "qualifier"):
                if not current.get(field) and assertion.get(field):
                    current[field] = assertion[field]
            if assertion.get("value_type") == PLACE:
                # Keep the spelling that says the most. The run that wrote
                # "San Francisco, California" located the company; the run that
                # wrote "San Francisco" only narrowed it down.
                incoming = assertion["value"]
                if incoming != current["value"]:
                    if (len(place_components(incoming))
                            > len(place_components(current["value"]))):
                        current["value"], incoming = incoming, current["value"]
                    variants = current.setdefault("also_written_as", [])
                    if incoming not in variants:
                        variants.append(incoming)
                span = assertion.get("source_span")
                if span and span != current.get("source_span"):
                    spans = current.setdefault("supporting_spans", [])
                    if span not in spans:
                        spans.append(span)

    agreed, contested = [], []
    for entry in seen.values():
        assertion = entry["assertion"]
        assertion["agreement"] = entry["count"]
        assertion["run_count"] = total_runs
        if entry["count"] >= min_agreement:
            agreed.append(assertion)
        else:
            assertion.setdefault("review", []).append(
                f"seen in {entry['count']} of {total_runs} runs"
            )
            assertion["asserted"] = False
            contested.append(assertion)
    # Within a single run the same city can appear under three spellings from
    # three different sentences, and each spelling recurs across runs - so all
    # three survive the vote as separate facts. Collapsing them here is what
    # turns Airbnb's three headquarters back into one.
    return merge_place_assertions(agreed), contested


# Sweeping a whole document for 23 predicates at once is an unstable search:
# each run's attention lands differently, so runs disagree, facts nobody asked
# for go missing, and nothing ever compares two places to notice it called both
# a headquarters. Asking a narrow question per group makes the task closer to
# verification, which was reliable for exactly that reason. Grouping also means
# every place is judged against the others in one reply.
PREDICATE_GROUPS = {
    "identity": {
        "predicates": ("name", "description", "mission", "status", "stage",
                       "yc_batch", "founded_on", "originated_on", "team_size"),
        "question": "What does this document say about what the company is, "
                    "when it was founded, its YC batch, and how big it is?",
        "guidance": "Give founded_on the finest granularity the document "
                    "supports, and emit both if it states a year in one place "
                    "and a month in another. If the document describes the idea "
                    "starting or a first version launching BEFORE a separate, "
                    "later founding date, record the earlier one as "
                    "originated_on and the later one as founded_on.",
    },
    "people": {
        "predicates": ("founded_by",),
        "question": "Who does this document name as founding the company?",
        "guidance": "Consider the founders together and decide their order. "
                    "Mark someone 'original_applicant' only if the document "
                    "says they applied or started first, and 'later_co_founder' "
                    "only if it says they joined afterwards. Do NOT list someone "
                    "the document says was never actually a founder, or who "
                    "left before the company started.",
    },
    "places": {
        "predicates": ("located_in",),
        "question": "Which places does this document connect to the company?",
        "guidance": "List every place together, then assign each a role. A "
                    "company has ONE headquarters at a time; if several places "
                    "appear, at most one is the headquarters and the rest are "
                    "offices, markets, founding locations or incorporations. "
                    "If the document gives a former and a current headquarters, "
                    "only the current one is 'headquarters'.",
    },
    "money_in": {
        "predicates": ("raised", "ipo_price", "ipo_proceeds", "ipo_share_count"),
        "question": "What money did the company RECEIVE, and what are its IPO "
                    "figures?",
        "guidance": "Only money the company received as investment counts as "
                    "'raised'. Money it paid, was fined, lost, or spent on an "
                    "acquisition is NOT raised - omit it entirely. 'ipo_price' "
                    "is the price of one share; total money raised at IPO is "
                    "'ipo_proceeds'. State explicitly if the document gives a "
                    "share count.",
    },
    "market": {
        "predicates": ("ticker", "listed_on", "first_traded_on",
                       "market_count", "listing_count"),
        "question": "Is the company publicly listed, and what scale figures "
                    "does the document give?",
        "guidance": "A ticker must appear literally in the span. "
                    "'first_traded_on' is the date trading began or was "
                    "expected to begin, which is a different fact from the "
                    "exchange. 'market_count' counts markets, cities or "
                    "countries - never users, customers or employees. "
                    "'listing_count' counts listings on the platform.",
    },
    "relations": {
        "predicates": ("invested_in_by", "competes_with", "acquired_by",
                       "operates_in", "uses", "offers"),
        "question": "Which other organisations, industries and technologies "
                    "does the document connect to the company, and what does "
                    "the company sell?",
        "guidance": "'acquired_by' means an organisation acquired THIS company. "
                    "A company this one bought, or a holding company it created, "
                    "is not an acquirer - omit it. 'offers' is a named product "
                    "or service the company provides; the sector it works in is "
                    "'operates_in', not an offering.",
    },
}

TARGETED_PROMPT = """{question}

Company: {company}

Extract ONLY these predicates, ignoring everything else in the document:
{predicate_table}

{guidance}

Rules, all checked after you answer:

1. Every assertion MUST include "source_span": text copied VERBATIM from the
   document that states the fact. Spans not found in the document are discarded.
2. Extract a fact ONLY if the document states it. Never infer, calculate, or
   use outside knowledge. Omit anything the document does not state.
3. Prefer a sentence that states the fact plainly. Do not use a sentence that
   denies it, supposes it, or describes it as planned or former.
4. Dates carry "granularity": "day", "month" or "year".
5. Places carry "role": {place_roles}. founded_by may carry "role":
   {founder_roles}.
6. Counts are integers, and must count the thing the predicate names.
7. If the document contradicts itself, emit both with their own spans.

Return ONLY valid JSON:
{{"assertions": [{{"predicate": "...", "value": ..., "value_type": "...",
  "source_span": "...", "granularity": "...", "role": "...", "qualifier": "..."}}]}}

Document:
{document}"""


def build_targeted_prompt(company: str, document: str, group: str) -> str:
    """Prompt for one group of related predicates rather than all of them."""
    spec = PREDICATE_GROUPS[group]
    rows = []
    for predicate in spec["predicates"]:
        declared = PREDICATES[predicate]
        row = f"   - {predicate}: {declared['type']}"
        if declared["multi"]:
            row += " (may repeat)"
        if "counts" in declared:
            row += f" (counts {declared['counts']})"
        rows.append(row + f" - {PREDICATE_DEFINITIONS[predicate]}")
    return TARGETED_PROMPT.format(
        question=spec["question"],
        company=company,
        predicate_table="\n".join(rows),
        guidance=spec["guidance"],
        document=document,
        place_roles=", ".join(PLACE_ROLES),
        founder_roles=", ".join(FOUNDER_ROLES),
    )


def _predicate_table() -> str:
    rows = []
    for predicate, declared in PREDICATES.items():
        row = f"   - {predicate}: {declared['type']}"
        if declared["multi"]:
            row += " (may repeat)"
        if "counts" in declared:
            row += f" (counts {declared['counts']})"
        rows.append(row)
    return "\n".join(rows)


def build_extraction_prompt(company: str, document: str) -> str:
    return EXTRACTION_PROMPT.format(
        company=company,
        document=document,
        predicate_table=_predicate_table(),
        place_roles=", ".join(PLACE_ROLES),
        founder_roles=", ".join(FOUNDER_ROLES),
    )


__all__ = [
    "PREDICATES",
    "PREDICATE_DEFINITIONS",
    "PLACE_ROLES",
    "FOUNDER_ROLES",
    "VALUE_TYPES",
    "DATE_GRANULARITIES",
    "ExtractionError",
    "span_occurs_in",
    "validate_assertion",
    "validate_extraction",
    "build_extraction_prompt",
    "build_verification_prompt",
    "apply_verdicts",
    "assertion_identity",
    "merge_runs",
]
