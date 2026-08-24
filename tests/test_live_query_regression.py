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
        r"\(\s*(\w+)\s*:\s*Company\b",
        cypher,
        flags=re.IGNORECASE,
    )
    return any(
        (
            re.search(
                rf"\bcount\s*\(\s*DISTINCT\s+{re.escape(variable)}\s*\)",
                cypher,
                flags=re.IGNORECASE,
            )
            or re.search(
                (
                    rf"\bcount\s*\(\s*DISTINCT\s+CASE\b.{{0,500}}?"
                    rf"\bTHEN\s+{re.escape(variable)}"
                    rf"(?:\s+ELSE\s+NULL)?\s+END\s*\)"
                ),
                cypher,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        for variable in company_variables
    )


def has_ai_alias_filter(cypher):
    filtering_cypher = re.split(
        r"\bRETURN\b",
        cypher,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    for values in re.findall(
        r"toLower\(\s*\w+\.name\s*\)\s+IN\s*(\[[^\]]+\])",
        filtering_cypher,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        aliases = {
            value.casefold()
            for value in re.findall(r"['\"]([^'\"]+)['\"]", values)
        }
        if {"ai", "artificial intelligence"}.issubset(aliases):
            return True
    return False


def has_canonical_ai_category(cypher, output_alias):
    expressions = re.findall(
        (
            r"(\bCASE\b.{0,800}?\bEND)\s+AS\s+"
            rf"{re.escape(output_alias)}\b"
        ),
        cypher,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return any(
        {"ai", "artificial intelligence"}.issubset(
            {
                value.casefold()
                for value in re.findall(r"['\"]([^'\"]+)['\"]", expression)
            }
        )
        and re.search(
            r"\bTHEN\s+['\"]Artificial Intelligence['\"]",
            expression,
            flags=re.IGNORECASE,
        )
        for expression in expressions
    )


def has_separate_healthcare_fintech_counts(cypher):
    company_variables = re.findall(
        r"\(\s*(\w+)\s*:\s*Company\b",
        cypher,
        flags=re.IGNORECASE,
    )
    for variable in company_variables:
        conditional_counts = re.findall(
            (
                r"\bcount\s*\(\s*DISTINCT\s+CASE\b(.{0,800}?)"
                rf"\bTHEN\s+{re.escape(variable)}"
                r"(?:\s+ELSE\s+NULL)?\s+END\s*\)"
            ),
            cypher,
            flags=re.IGNORECASE | re.DOTALL,
        )
        healthcare_indices = {
            index
            for index, expression in enumerate(conditional_counts)
            if "health" in expression.casefold()
            and not any(
                term in expression.casefold()
                for term in ("fintech", "finance", "payments")
            )
        }
        fintech_indices = {
            index
            for index, expression in enumerate(conditional_counts)
            if all(
                term in expression.casefold()
                for term in ("fintech", "finance", "payments")
            )
            and "health" not in expression.casefold()
        }
        if healthcare_indices and fintech_indices:
            return True
    return False


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
        "canonical_ai_category": "industry",
    },
    {
        "query": "Compare the AI startup ecosystem in San Francisco and New York.",
        "min_results": 2,
        "required_terms": ("san francisco", "new york"),
        "require_ai_alias_filter": True,
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
        "required_terms": ("python",),
        "required_patterns": (r"\b(?:fintech|finance|payments)\b",),
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
        "canonical_ai_category": "technology",
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
        "require_ai_alias_filter": True,
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
        "canonical_ai_category": "technology",
    },
    {
        "query": "How many YC companies use AI?",
        "min_results": 1,
        "required_patterns": (r"\b(?:AI|Artificial Intelligence)\b",),
        "require_ai_alias_filter": True,
        "require_distinct_company_count": True,
        "equivalent_answer_number_group": "ai-company-count",
        "answer_number_pattern": r"\b([\d,]+)\s+YC companies\b",
    },
    {
        "query": "How many YC companies use Artificial Intelligence?",
        "min_results": 1,
        "required_patterns": (r"\b(?:AI|Artificial Intelligence)\b",),
        "require_ai_alias_filter": True,
        "require_distinct_company_count": True,
        "equivalent_answer_number_group": "ai-company-count",
        "answer_number_pattern": r"\b([\d,]+)\s+YC companies\b",
    },
    {
        "query": "Which AI companies are headquartered in New York?",
        "min_results": 1,
        "required_terms": ("new york",),
        "require_ai_alias_filter": True,
        "required_filter_terms": ("new york",),
    },
    {
        "query": "Who founded Scale AI?",
        "min_results": 1,
        "required_terms": ("scale ai",),
        "required_filter_terms": ("scale ai",),
        "forbid_ai_alias_filter": True,
    },
    {
        "query": "Which AI companies compete with Scale AI?",
        "expected_results": 0,
        "required_terms": ("no results",),
        "require_ai_alias_filter": True,
        "required_filter_terms": ("scale ai",),
    },
    {
        "query": "Which YC companies use Kubernetes?",
        "min_results": 1,
        "required_terms": ("kubernetes",),
        "required_filter_terms": ("kubernetes",),
        "forbid_ai_alias_filter": True,
    },
    {
        "query": "Which industries have the most active YC companies?",
        "min_results": 5,
        "required_patterns": (r"\bindustr(?:y|ies)\b",),
        "canonical_ai_category": "industry",
        "require_distinct_company_count": True,
        "required_filter_cypher": (
            r"\b\w+\.status\s*=\s*[\"']Active[\"']",
        ),
        "forbidden_cypher": (
            r"\b\w+\.status\s+IN\s*\[[^\]]*[\"']Active[\"']",
        ),
    },
    {
        "query": "Which technologies are used by Stripe?",
        "min_results": 1,
        "required_terms": ("stripe",),
        "required_patterns": (r"\btechnolog(?:y|ies)\b",),
        "required_filter_terms": ("stripe",),
        "forbid_ai_alias_filter": True,
    },
    {
        "query": "Compare Artificial Intelligence companies in W21 and S21.",
        "min_results": 2,
        "required_terms": ("winter 2021", "summer 2021"),
        "require_ai_alias_filter": True,
        "require_distinct_company_count": True,
        "required_filter_cypher": (
            r'["\']Winter 2021["\']',
            r'["\']Summer 2021["\']',
        ),
        "forbidden_cypher": (
            r'["\']W21["\']',
            r'["\']S21["\']',
        ),
    },
    {
        "query": (
            "List active Artificial Intelligence companies with fewer "
            "than 50 employees."
        ),
        "min_results": 1,
        "required_patterns": (r"\b(?:AI|Artificial Intelligence)\b",),
        "require_ai_alias_filter": True,
        "required_cypher": (r"\b\w+\.status\s+AS\s+status\b",),
        "required_filter_cypher": (
            r"\b\w+\.status\s*=\s*[\"']Active[\"']",
            r"\b\w+\.team_size\s*>\s*0\b",
            r"\b\w+\.team_size\s*<\s*50\b",
        ),
        "forbidden_terms": ("team size: 0",),
    },
    {
        "query": (
            "Can you tell me who founded Stripe? Also tell me who were "
            "the founders of RazorPay!"
        ),
        "min_results": 4,
        "required_terms": (
            "john collison",
            "patrick collison",
            "harshil mathur",
            "shashank kumar",
        ),
        "required_filter_cypher": (
            r"toLower\(\s*\w+\.name\s*\)\s+IN\s*\[",
            r"[\"']stripe[\"']",
            r"[\"']razorpay[\"']",
        ),
        "forbidden_cypher": (
            r"\b\w+\.name\s+IN\s*\[",
        ),
    },
    {
        "query": "Who founded Airbnb and Coinbase?",
        "min_results": 5,
        "required_terms": (
            "airbnb",
            "coinbase",
            "brian chesky",
            "joe gebbia",
            "nathan blecharczyk",
            "brian armstrong",
            "fred ehrsam",
        ),
        "required_filter_cypher": (
            r"toLower\(\s*\w+\.name\s*\)\s+IN\s*\[",
            r"[\"']airbnb[\"']",
            r"[\"']coinbase[\"']",
        ),
    },
    {
        "query": (
            "Compare the status, stage, and team size of Stripe, Airbnb, "
            "and Coinbase."
        ),
        "min_results": 3,
        "required_terms": ("stripe", "airbnb", "coinbase", "active", "growth"),
        "required_patterns": (
            r"\b(?:team size|employees)\b",
        ),
        "required_cypher": (
            r"\b\w+\.status\s+AS\s+status\b",
            r"\b\w+\.stage\s+AS\s+stage\b",
            r"\b\w+\.team_size\s+AS\s+team_size\b",
        ),
        "required_filter_cypher": (
            r"toLower\(\s*\w+\.name\s*\)\s+IN\s*\[",
            r"[\"']stripe[\"']",
            r"[\"']airbnb[\"']",
            r"[\"']coinbase[\"']",
        ),
    },
    {
        "query": (
            "Who founded Stripe, which YC batch was it in, and what is "
            "its current status?"
        ),
        "min_results": 2,
        "required_terms": ("john collison", "patrick collison", "summer 2009"),
        "required_patterns": (r"\bactive\b",),
        "required_cypher": (
            r"\b\w+\.yc_batch\s+AS\s+batch\b",
            r"\b\w+\.status\s+AS\s+status\b",
        ),
        "required_filter_cypher": (
            r"toLower\(\s*\w+\.name\s*\)\s*=\s*[\"']stripe[\"']",
        ),
    },
    {
        "query": "Which investors backed Stripe and which backed Airbnb?",
        "min_results": 2,
        "required_terms": (
            "stripe",
            "airbnb",
            "sequoia capital",
            "andreessen horowitz",
        ),
        "required_patterns": (r"\binvest(?:or|ed|ment)\w*\b",),
        "required_cypher": (
            r"\b\w+\.name\s+AS\s+investor\b",
            r"\b\w+\.name\s+AS\s+company\b",
        ),
        "required_filter_cypher": (
            r"toLower\(\s*\w+\.name\s*\)\s+IN\s*\[",
            r"[\"']stripe[\"']",
            r"[\"']airbnb[\"']",
        ),
    },
    {
        "query": (
            "Compare AI company counts in San Francisco, New York, and London."
        ),
        "min_results": 3,
        "required_terms": ("san francisco", "new york", "london"),
        "required_patterns": (
            (
                r"(?:san francisco.{0,80}\b\d[\d,]*\b|"
                r"\b\d[\d,]*\b.{0,80}san francisco)"
            ),
            (
                r"(?:new york.{0,80}\b\d[\d,]*\b|"
                r"\b\d[\d,]*\b.{0,80}new york)"
            ),
            (
                r"(?:london.{0,80}\b\d[\d,]*\b|"
                r"\b\d[\d,]*\b.{0,80}london)"
            ),
        ),
        "require_ai_alias_filter": True,
        "require_distinct_company_count": True,
        "required_cypher": (
            r"\b\w+\.city\s+AS\s+city\b",
        ),
        "required_filter_cypher": (
            r"toLower\(\s*\w+\.city\s*\)\s+IN\s*\[",
            r"[\"']san francisco[\"']",
            r"[\"']new york[\"']",
            r"[\"']london[\"']",
        ),
    },
    {
        "query": (
            "For W21 and S21, compare healthcare and fintech company counts."
        ),
        "min_results": 2,
        "required_terms": (
            "winter 2021",
            "summer 2021",
            "healthcare",
            "fintech",
        ),
        "required_patterns": (
            r"(?:.*\b\d{1,3}(?:,\d{3})*\b){4}",
        ),
        "required_filter_cypher": (
            r"[\"']Winter 2021[\"']",
            r"[\"']Summer 2021[\"']",
            r"health",
            r"[\"']fintech[\"']",
            r"[\"']finance[\"']",
            r"[\"']payments[\"']",
        ),
        "require_distinct_company_count": True,
        "require_separate_healthcare_fintech_counts": True,
        "forbidden_cypher": (
            r"[\"']W21[\"']",
            r"[\"']S21[\"']",
        ),
    },
    {
        "query": (
            "List active AI companies in New York with fewer than 20 employees."
        ),
        "min_results": 1,
        "required_terms": ("new york",),
        "required_patterns": (r"\b(?:AI|Artificial Intelligence)\b",),
        "require_ai_alias_filter": True,
        "required_cypher": (r"\b\w+\.status\s+AS\s+status\b",),
        "required_filter_cypher": (
            r"toLower\(\s*\w+\.city\s*\)\s*=\s*[\"']new york[\"']",
            r"\b\w+\.status\s*=\s*[\"']Active[\"']",
            r"\b\w+\.team_size\s*>\s*0\b",
            r"\b\w+\.team_size\s*<\s*20\b",
        ),
    },
    {
        "query": "Who founded RazorPay, and who invested in Airbnb?",
        "min_results": 1,
        "required_terms": (
            "harshil mathur",
            "shashank kumar",
            "airbnb",
            "sequoia capital",
        ),
        "required_patterns": (r"\binvest(?:or|ed|ment)\w*\b",),
        "required_cypher": (
            r"collect\(\s*DISTINCT\s+\w+\.name\s*\)\s+AS\s+founders\b",
            r"collect\(\s*DISTINCT\s+\w+\.name\s*\)\s+AS\s+investors\b",
            (
                r"toLower\(\s*\w+\.name\s*\)\s*=\s*"
                r"[\"']razorpay[\"']"
            ),
            (
                r"toLower\(\s*\w+\.name\s*\)\s*=\s*"
                r"[\"']airbnb[\"']"
            ),
        ),
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
    def test_combined_category_count_is_not_treated_as_separate_counts(self):
        cypher = """
        MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
        RETURN count(DISTINCT CASE
                     WHEN toLower(ind.name) IN [
                       "healthcare", "fintech", "finance", "payments"
                     ]
                     THEN c END) AS combined_count
        """

        self.assertFalse(has_separate_healthcare_fintech_counts(cypher))

    def test_separate_category_counts_are_recognized(self):
        cypher = """
        MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
        RETURN count(DISTINCT CASE
                     WHEN toLower(ind.name) CONTAINS "health"
                     THEN c END) AS healthcare_count,
               count(DISTINCT CASE
                     WHEN toLower(ind.name) IN [
                       "fintech", "finance", "payments"
                     ]
                     THEN c END) AS fintech_count
        """

        self.assertTrue(has_separate_healthcare_fintech_counts(cypher))

    def test_combined_and_healthcare_counts_are_not_separate_fintech_counts(self):
        cypher = """
        MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
        RETURN count(DISTINCT CASE
                     WHEN toLower(ind.name) IN [
                       "healthcare", "fintech", "finance", "payments"
                     ]
                     THEN c END) AS combined_count,
               count(DISTINCT CASE
                     WHEN toLower(ind.name) CONTAINS "health"
                     THEN c END) AS healthcare_count
        """

        self.assertFalse(has_separate_healthcare_fintech_counts(cypher))

    def test_partial_combined_and_fintech_counts_are_not_separate_healthcare_counts(
        self,
    ):
        cypher = """
        MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
        RETURN count(DISTINCT CASE
                     WHEN toLower(ind.name) IN ["healthcare", "fintech"]
                     THEN c END) AS combined_count,
               count(DISTINCT CASE
                     WHEN toLower(ind.name) IN [
                       "fintech", "finance", "payments"
                     ]
                     THEN c END) AS fintech_count
        """

        self.assertFalse(has_separate_healthcare_fintech_counts(cypher))

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
        equivalent_answer_numbers = {}

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
            if (
                case.get("require_separate_healthcare_fintech_counts")
                and not has_separate_healthcare_fintech_counts(cypher)
            ):
                failures.append(
                    f"{query}: Cypher is missing separate distinct healthcare "
                    "and normalized fintech counts"
                )
            if case.get("require_ai_alias_filter") and not has_ai_alias_filter(cypher):
                failures.append(
                    f"{query}: Cypher is missing exact AI alias normalization"
                )
            if case.get("forbid_ai_alias_filter") and has_ai_alias_filter(cypher):
                failures.append(
                    f"{query}: Cypher unexpectedly applies AI category normalization"
                )
            canonical_alias = case.get("canonical_ai_category")
            if canonical_alias and not has_canonical_ai_category(
                cypher, canonical_alias
            ):
                failures.append(
                    f"{query}: Cypher does not canonicalize AI as {canonical_alias}"
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
            equivalence_group = case.get("equivalent_answer_number_group")
            if equivalence_group:
                number_match = re.search(
                    case["answer_number_pattern"],
                    answer,
                    flags=re.IGNORECASE,
                )
                if not number_match:
                    failures.append(
                        f"{query}: answer does not expose a comparable aggregate"
                    )
                else:
                    answer_number = int(number_match.group(1).replace(",", ""))
                    previous = equivalent_answer_numbers.setdefault(
                        equivalence_group, answer_number
                    )
                    if previous != answer_number:
                        failures.append(
                            f"{query}: aggregate {answer_number} does not match "
                            f"{equivalence_group} result {previous}"
                        )

        self.assertEqual([], failures, "\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
