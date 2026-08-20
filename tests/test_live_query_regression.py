import os
import re
import unittest

import requests


RUN_LIVE_TESTS = os.environ.get("RUN_LIVE_API_TESTS") == "1"
API_URL = os.environ.get("STARTUPLENS_API_URL", "").rstrip("/")
API_KEY = os.environ.get("STARTUPLENS_API_KEY", "")

FALSE_NO_DATA = re.compile(
    r"^\s*(?:(?:based on|according to) the (?:provided )?data[:,]?\s*)?"
    r"(?:no results|the data (?:provided )?does not contain enough|"
    r"there (?:is|are) insufficient information|"
    r"(?:it is not possible to|cannot|unable to) determine)",
    flags=re.IGNORECASE | re.DOTALL,
)


def has_normalized_fintech_filter(cypher):
    query_parts = re.split(
        r"\bRETURN\b",
        cypher,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    filtering_cypher = query_parts[0]
    return_clause = query_parts[1] if len(query_parts) > 1 else ""
    company_variables = re.findall(
        r"\b(\w+)\.name\s+AS\s+company\b",
        return_clause,
        flags=re.IGNORECASE,
    )
    industry_variables = re.findall(
        r"\((\w+)\s*:\s*Industry\b",
        filtering_cypher,
        flags=re.IGNORECASE,
    )
    for variable in industry_variables:
        connected_to_returned_company = any(
            re.search(
                (
                    rf"\(\s*{re.escape(company)}\b[^)]*\)\s*-\s*"
                    rf"\[[^\]]*OPERATES_IN[^\]]*\]\s*->\s*"
                    rf"\(\s*{re.escape(variable)}\s*:\s*Industry\b|"
                    rf"\(\s*{re.escape(variable)}\s*:\s*Industry\b[^)]*\)\s*"
                    rf"<-\s*\[[^\]]*OPERATES_IN[^\]]*\]\s*-\s*"
                    rf"\(\s*{re.escape(company)}\b[^)]*\)"
                ),
                filtering_cypher,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for company in company_variables
        )
        if not connected_to_returned_company:
            continue
        property_pattern = (
            rf"toLower\(\s*{re.escape(variable)}\.name\s*\)"
        )
        for values in re.findall(
            rf"{property_pattern}\s+IN\s*(\[[^\]]*\])",
            filtering_cypher,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            if all(
                re.search(
                    rf"['\"]{term}['\"]",
                    values,
                )
                for term in ("fintech", "finance", "payments")
            ) and re.search(
                rf"\b{re.escape(variable)}\.name\s+AS\s+industry\b",
                return_clause,
                flags=re.IGNORECASE,
            ):
                return True
    return False


def has_distinct_company_count(cypher):
    company_variables = re.findall(
        r"\((\w+)\s*:\s*Company\b",
        cypher,
        flags=re.IGNORECASE,
    )
    return any(
        re.search(
            rf"\bcount\s*\(\s*DISTINCT\s+{re.escape(variable)}\s*\)",
            cypher,
            flags=re.IGNORECASE,
        )
        for variable in company_variables
    )


CASES = (
    {
        "query": "Which YC companies build developer tools?",
        "min_results": 1,
        "required_terms": ("developer",),
    },
    {
        "query": "What are the top 10 industries by number of YC companies?",
        "min_results": 10,
        "required_terms": ("b2b", "fintech"),
    },
    {
        "query": "Compare the AI startup ecosystem in San Francisco and New York.",
        "min_results": 2,
        "required_terms": ("san francisco", "new york"),
        "forbidden_cypher": (r'CONTAINS\s+["\']ai["\']',),
    },
    {
        "query": (
            "Which active B2B SaaS companies from the W23 or S23 batches "
            "have fewer than 20 employees?"
        ),
        "min_results": 1,
        "forbidden_terms": ("team size: 0",),
        "required_filter_cypher": (
            r'["\']Winter 2023["\']',
            r'["\']Summer 2023["\']',
        ),
        "required_cypher": (r"\b\w+\.status\s+AS\s+status\b",),
        "forbidden_cypher": (
            r'["\']W23["\']',
            r'["\']S23["\']',
            r'["\']b2b["\'].{0,120}\bOR\b.{0,120}["\']saas["\']',
            (
                r'(?:\b\w+\.\w+|toLower\([^)]*\))\s+IN\s*\['
                r'[^\]]*["\']b2b["\'][^\]]*["\']saas["\']'
            ),
            (
                r'["\']b2b["\']\s+IN\s+\w+.{0,120}\bOR\b.{0,120}'
                r'["\']saas["\']\s+IN\s+\w+'
            ),
        ),
    },
    {
        "query": (
            "Who founded Stripe, and what other information is available "
            "about its founders?"
        ),
        "min_results": 2,
        "required_terms": ("patrick collison", "john collison"),
        "allow_grounded_partial": True,
    },
    {
        "query": "Which YC companies use Python and operate in fintech?",
        "min_results": 1,
        "required_terms": ("python", "fintech"),
        "required_filter_terms": ("fintech", "finance", "payments"),
        "required_cypher": (
            r"\bAS\s+technology\b",
            r"\bAS\s+industry\b",
        ),
        "require_fintech_industry_in": True,
        "forbidden_terms": (
            "does not contain enough information",
            "does not specify",
            "missing fields",
        ),
    },
    {
        "query": "Which batches produced the most healthcare companies?",
        "min_results": 1,
        "required_terms": ("healthcare",),
        "require_distinct_company_count": True,
    },
    {
        "query": "Show companies that raised funding but are now inactive.",
        "min_results": 1,
        "required_cypher": (r"\b\w+\.status\s+AS\s+status\b",),
        "forbidden_terms": (
            "insufficient information to determine",
            "does not specify",
            "field is missing",
            "status is unavailable",
        ),
    },
    {
        "query": "What technologies are most commonly used by developer-tools companies?",
        "min_results": 1,
        "required_patterns": (r"\btechnolog(?:y|ies)\b",),
    },
    {
        "query": "Which YC companies are building nuclear-powered consumer smartphones?",
        "expected_results": 0,
        "required_terms": ("no results",),
    },
    {
        "query": "Which investors have the most YC portfolio companies?",
        "min_results": 5,
        "required_patterns": (r"\b(?:investor|portfolio)\w*\b",),
    },
    {
        "query": "Which YC companies were acquired and by whom?",
        "min_results": 1,
        "required_patterns": (r"\bacquir(?:ed|er|ers|ing)\b",),
    },
    {
        "query": "How many YC companies are Active vs Dead vs Acquired?",
        "min_results": 3,
        "required_terms": ("active", "dead", "acquired"),
    },
    {
        "query": "Which YC companies headquartered in India operate in fintech?",
        "min_results": 1,
        "required_terms": ("india",),
        "required_patterns": (r"\b(?:fintech|finance|payments)\b",),
        "required_filter_terms": ("fintech", "finance", "payments"),
        "required_cypher": (r"\bAS\s+industry\b",),
        "require_fintech_industry_in": True,
    },
    {
        "query": "What is Stripe's funding history?",
        "min_results": 1,
        "required_terms": ("stripe",),
        "required_patterns": (r"\b(?:funding|seed|series|round)\w*\b",),
    },
    {
        "query": "Which founders started more than one YC company?",
        "min_results": 1,
        "required_patterns": (r"\bfounder\w*\b",),
    },
    {
        "query": "Which active YC companies in San Francisco use AI?",
        "min_results": 1,
        "required_terms": ("san francisco", "ai"),
        "required_cypher": (r"\b\w+\.status\s+AS\s+status\b",),
        "forbidden_cypher": (r'CONTAINS\s+["\']ai["\']',),
    },
    {
        "query": "Which investors backed healthcare companies?",
        "min_results": 1,
        "required_patterns": (r"\b(?:investor|backed|invested)\w*\b",),
        "required_terms": ("healthcare",),
    },
    {
        "query": "Which W21 companies are now public or acquired?",
        "min_results": 1,
        "required_patterns": (r"\b(?:public|acquired)\b",),
        "required_cypher": (r"\b\w+\.status\s+AS\s+status\b",),
        "required_filter_cypher": (r'["\']Winter 2021["\']',),
        "forbidden_cypher": (r'["\']W21["\']',),
    },
    {
        "query": "Compare healthcare companies in W21 and S21.",
        "min_results": 2,
        "required_terms": ("winter 2021", "summer 2021"),
        "required_filter_cypher": (
            r'["\']Winter 2021["\']',
            r'["\']Summer 2021["\']',
        ),
        "require_distinct_company_count": True,
    },
    {
        "query": "Which YC companies have more than 500 employees?",
        "min_results": 1,
        "required_patterns": (r"\b(?:employees|team size)\b",),
    },
    {
        "query": "Which dead consumer companies came from S20?",
        "min_results": 1,
        "required_patterns": (r"\b(?:dead|inactive)\b",),
        "required_terms": ("consumer",),
        "required_cypher": (r"\b\w+\.status\s+AS\s+status\b",),
        "required_filter_cypher": (r'["\']Summer 2020["\']',),
        "forbidden_cypher": (r'["\']S20["\']',),
        "forbidden_terms": (
            "does not specify",
            "field is missing",
            "status is unavailable",
            "status is missing",
        ),
    },
    {
        "query": "Who invested in Airbnb?",
        "min_results": 1,
        "required_terms": ("airbnb",),
        "required_patterns": (r"\b(?:investor|invested|investment)\w*\b",),
    },
    {
        "query": "What is the breakdown of YC companies by stage?",
        "min_results": 3,
        "required_terms": ("stage",),
    },
    {
        "query": "Which technologies are most used by YC companies?",
        "min_results": 5,
        "required_patterns": (r"\btechnolog(?:y|ies)\b",),
    },
)


def contradicts_results(answer, case):
    if not FALSE_NO_DATA.search(answer):
        return False

    answer_lower = answer.lower()
    return not (
        case.get("allow_grounded_partial")
        and all(term in answer_lower for term in case.get("required_terms", ()))
    )


class LiveAssertionTests(unittest.TestCase):
    def test_grounded_partial_answer_is_allowed_for_configured_case(self):
        case = {
            "allow_grounded_partial": True,
            "required_terms": ("patrick collison", "john collison"),
        }
        answer = (
            "Cannot determine other biographical details, but Patrick Collison "
            "and John Collison founded Stripe."
        )

        self.assertFalse(contradicts_results(answer, case))

    def test_prefixed_blanket_no_data_claim_is_rejected(self):
        answer = "Based on the provided data, no results were found."

        self.assertTrue(contradicts_results(answer, {}))


@unittest.skipUnless(
    RUN_LIVE_TESTS and API_URL and API_KEY,
    "Set RUN_LIVE_API_TESTS=1, STARTUPLENS_API_URL, and STARTUPLENS_API_KEY",
)
class LiveQueryRegressionTests(unittest.TestCase):
    def test_representative_queries(self):
        failures = []

        for case in CASES:
            query = case["query"]
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    headers={"X-API-Key": API_KEY},
                    json={"query": query},
                    timeout=90,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as error:
                failures.append(f"{query}: request failed: {error}")
                continue

            answer = payload.get("answer", "")
            answer_lower = answer.lower()
            result_count = payload.get("result_count")

            if not answer:
                failures.append(f"{query}: empty answer")
            if result_count is None:
                failures.append(f"{query}: missing result_count")
                continue
            if result_count > 0 and contradicts_results(answer, case):
                failures.append(f"{query}: answer contradicts {result_count} results")
            if result_count > 50:
                failures.append(f"{query}: returned {result_count} results; maximum is 50")

            expected = case.get("expected_results")
            minimum = case.get("min_results")
            if expected is not None and result_count != expected:
                failures.append(f"{query}: expected {expected} results, got {result_count}")
            if minimum is not None and result_count < minimum:
                failures.append(f"{query}: expected at least {minimum} results, got {result_count}")

            for term in case.get("required_terms", ()):
                if term not in answer_lower:
                    failures.append(f"{query}: answer is missing {term!r}")
            for pattern in case.get("required_patterns", ()):
                if not re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL):
                    failures.append(
                        f"{query}: answer does not match required pattern {pattern!r}"
                    )
            for term in case.get("forbidden_terms", ()):
                if term in answer_lower:
                    failures.append(f"{query}: answer unexpectedly contains {term!r}")
            cypher = payload.get("cypher", "")
            if not cypher:
                failures.append(f"{query}: missing generated Cypher")
            if (
                case.get("require_fintech_industry_in")
                and not has_normalized_fintech_filter(cypher)
            ):
                failures.append(
                    f"{query}: Cypher is missing normalized fintech industry IN filter"
                )
            if (
                case.get("require_distinct_company_count")
                and not has_distinct_company_count(cypher)
            ):
                failures.append(
                    f"{query}: Cypher is missing count(DISTINCT Company)"
                )
            filtering_cypher = re.split(
                r"\bRETURN\b",
                cypher,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            for pattern in case.get("required_filter_cypher", ()):
                if not re.search(
                    pattern,
                    filtering_cypher,
                    flags=re.IGNORECASE | re.DOTALL,
                ):
                    failures.append(
                        f"{query}: filter Cypher does not match required pattern {pattern!r}"
                    )
            for term in case.get("required_filter_terms", ()):
                if term.casefold() not in filtering_cypher.casefold():
                    failures.append(
                        f"{query}: filter Cypher is missing required term {term!r}"
                    )
            for pattern in case.get("required_cypher", ()):
                if not re.search(pattern, cypher, flags=re.IGNORECASE | re.DOTALL):
                    failures.append(
                        f"{query}: Cypher does not match required pattern {pattern!r}"
                    )
            for pattern in case.get("forbidden_cypher", ()):
                if re.search(pattern, cypher, flags=re.IGNORECASE | re.DOTALL):
                    failures.append(
                        f"{query}: generated Cypher matches forbidden pattern {pattern!r}"
                    )

        self.assertEqual([], failures, "\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
