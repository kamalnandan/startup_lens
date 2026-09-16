"""Validation and deterministic checks for the source-backed gold benchmark."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


EXPECTED_CATEGORIES = {
    "identity": "identity",
    "funding": "funding_status",
    "company": "company_facts",
    "compound": "compound",
}
VOLATILITY_LEVELS = {"low", "medium", "high"}
REQUIRED_SOURCE_FIELDS = {
    "title",
    "url",
    "publisher",
    "accessed_date",
    "evidence",
    "supports_claims",
}


def load_benchmark(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Gold benchmark must be a JSON array.")
    return records


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_https_url(value: Any) -> bool:
    if not _is_nonempty_string(value):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _valid_iso_date(value: Any) -> bool:
    if not _is_nonempty_string(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def validate_benchmark(
    records: list[dict[str, Any]],
    *,
    expected_count: int = 100,
) -> list[str]:
    errors: list[str] = []
    if len(records) != expected_count:
        errors.append(f"Expected {expected_count} records, found {len(records)}.")

    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    category_counts = {category: 0 for category in EXPECTED_CATEGORIES.values()}

    for index, record in enumerate(records):
        location = f"record[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{location}: expected an object.")
            continue

        record_id = record.get("id")
        if not _is_nonempty_string(record_id):
            errors.append(f"{location}: missing id.")
            record_id = location
        elif record_id in seen_ids:
            errors.append(f"{record_id}: duplicate id.")
        else:
            seen_ids.add(record_id)

        id_match = (
            re.fullmatch(r"(identity|funding|company|compound)-\d{3}", record_id)
            if isinstance(record_id, str)
            else None
        )
        if not id_match:
            errors.append(f"{record_id}: id does not match the benchmark convention.")

        category = record.get("category")
        if category not in category_counts:
            errors.append(f"{record_id}: unsupported category {category!r}.")
        else:
            category_counts[category] += 1
        if id_match and category != EXPECTED_CATEGORIES[id_match.group(1)]:
            errors.append(f"{record_id}: category does not match its id prefix.")

        question = record.get("question")
        if not _is_nonempty_string(question):
            errors.append(f"{record_id}: missing question.")
        elif question.casefold() in seen_questions:
            errors.append(f"{record_id}: duplicate question.")
        else:
            seen_questions.add(question.casefold())

        for field in ("expected_answer_summary", "notes"):
            if not _is_nonempty_string(record.get(field)):
                errors.append(f"{record_id}: missing {field}.")

        if not _valid_iso_date(record.get("as_of_date")):
            errors.append(f"{record_id}: as_of_date must be an ISO date.")
        if record.get("volatility") not in VOLATILITY_LEVELS:
            errors.append(f"{record_id}: invalid volatility.")

        claims = record.get("expected_claims")
        if not isinstance(claims, list) or not claims:
            errors.append(f"{record_id}: expected_claims must be non-empty.")
            claims = []
        elif not all(_is_nonempty_string(claim) for claim in claims):
            errors.append(f"{record_id}: every expected claim must be text.")

        terms = record.get("required_answer_terms")
        if not isinstance(terms, list) or not terms:
            errors.append(f"{record_id}: required_answer_terms must be non-empty.")
        elif not all(
            _is_nonempty_string(term) and term == term.casefold() for term in terms
        ):
            errors.append(
                f"{record_id}: required answer terms must be non-empty lowercase text."
            )

        variants = record.get("acceptable_variants")
        if not isinstance(variants, dict):
            errors.append(f"{record_id}: acceptable_variants must be an object.")
        else:
            for canonical, alternatives in variants.items():
                if not _is_nonempty_string(canonical) or not isinstance(
                    alternatives, list
                ):
                    errors.append(f"{record_id}: invalid acceptable variant entry.")
                    continue
                if canonical not in (terms if isinstance(terms, list) else []):
                    errors.append(
                        f"{record_id}: variant key {canonical!r} is not a required term."
                    )
                if not all(_is_nonempty_string(value) for value in alternatives):
                    errors.append(f"{record_id}: acceptable variants must be text.")

        answer_patterns = record.get("required_answer_patterns")
        if answer_patterns is not None:
            if not isinstance(answer_patterns, list) or not answer_patterns:
                errors.append(
                    f"{record_id}: required_answer_patterns must be a non-empty array."
                )
            else:
                for pattern_index, pattern in enumerate(answer_patterns):
                    if not _is_nonempty_string(pattern):
                        errors.append(
                            f"{record_id}: required answer pattern "
                            f"{pattern_index} must be non-empty text."
                        )
                        continue
                    try:
                        re.compile(pattern, re.IGNORECASE)
                    except re.error as error:
                        errors.append(
                            f"{record_id}: invalid required answer pattern "
                            f"{pattern_index}: {error}."
                        )

        sources = record.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{record_id}: sources must be non-empty.")
            sources = []

        covered_claims: set[int] = set()
        for source_index, source in enumerate(sources):
            source_location = f"{record_id}.sources[{source_index}]"
            if not isinstance(source, dict):
                errors.append(f"{source_location}: expected an object.")
                continue
            missing = REQUIRED_SOURCE_FIELDS - source.keys()
            if missing:
                errors.append(
                    f"{source_location}: missing {', '.join(sorted(missing))}."
                )
            if not _valid_https_url(source.get("url")):
                errors.append(f"{source_location}: source URL must use HTTPS.")
            for field in ("title", "publisher", "evidence"):
                if not _is_nonempty_string(source.get(field)):
                    errors.append(f"{source_location}: missing {field}.")
            if not _valid_iso_date(source.get("accessed_date")):
                errors.append(f"{source_location}: accessed_date must be an ISO date.")
            supports_claims = source.get("supports_claims")
            if not isinstance(supports_claims, list) or not supports_claims:
                errors.append(f"{source_location}: supports_claims must be non-empty.")
                continue
            for claim_index in supports_claims:
                if (
                    not isinstance(claim_index, int)
                    or claim_index < 0
                    or claim_index >= len(claims)
                ):
                    errors.append(
                        f"{source_location}: invalid claim index {claim_index!r}."
                    )
                else:
                    covered_claims.add(claim_index)

        uncovered = sorted(set(range(len(claims))) - covered_claims)
        if uncovered:
            errors.append(f"{record_id}: claims without sources: {uncovered}.")

    for category, count in category_counts.items():
        if count != 25:
            errors.append(f"Category {category!r} must contain 25 records, found {count}.")

    return errors


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def contains_term(text: str, term: str) -> bool:
    normalized_term = normalize_text(term)
    pattern = re.escape(normalized_term)
    if normalized_term[:1].isalnum():
        pattern = rf"(?<!\w){pattern}"
    if normalized_term[-1:].isalnum():
        pattern = rf"{pattern}(?!\w)"
    return bool(re.search(pattern, text))


def evaluate_answer(
    record: dict[str, Any],
    answer: str,
) -> dict[str, Any]:
    normalized_answer = normalize_text(answer)
    variants = record.get("acceptable_variants", {})
    matched_terms: list[str] = []
    missing_terms: list[str] = []
    matched_patterns: list[str] = []
    missing_patterns: list[str] = []

    for term in record["required_answer_terms"]:
        candidates = [term, *variants.get(term, [])]
        if any(contains_term(normalized_answer, candidate) for candidate in candidates):
            matched_terms.append(term)
        else:
            missing_terms.append(term)

    for pattern in record.get("required_answer_patterns", []):
        if re.search(pattern, answer, re.IGNORECASE):
            matched_patterns.append(pattern)
        else:
            missing_patterns.append(pattern)

    total = len(record["required_answer_terms"])
    return {
        "passed_deterministic_checks": (
            bool(answer.strip()) and not missing_terms and not missing_patterns
        ),
        "term_coverage": round(len(matched_terms) / total, 4) if total else 0.0,
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "matched_patterns": matched_patterns,
        "missing_patterns": missing_patterns,
        "manual_fact_review_required": True,
    }
