import importlib
import os
import unittest
from unittest.mock import MagicMock, patch


REQUIRED_SETTINGS = (
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
)

for setting in REQUIRED_SETTINGS:
    os.environ.setdefault(setting, f"test-{setting.lower()}")

neo4j_query = importlib.import_module("neo4j_query")


class CypherSemanticsTests(unittest.TestCase):
    def test_rejects_broad_success_filter_for_active_question(self):
        with self.assertRaisesRegex(ValueError, 'status = "Active"'):
            neo4j_query.validate_cypher_semantics(
                "Which industries have the most active YC companies?",
                """
                MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                WHERE c.status IN ["Active", "Public", "Acquired"]
                RETURN ind.name AS industry,
                       count(DISTINCT c) AS company_count
                """,
            )

    def test_rejects_reordered_broad_filter_for_active_question(self):
        with self.assertRaisesRegex(ValueError, 'status = "Active"'):
            neo4j_query.validate_cypher_semantics(
                "Which active YC companies use Kubernetes?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                WHERE c.status IN ["Public", "Active"]
                  AND toLower(t.name) = "kubernetes"
                RETURN c.name AS company, c.status AS status
                """,
            )

    def test_does_not_treat_active_investors_as_active_companies(self):
        neo4j_query.validate_cypher_semantics(
            "Which investors are most active?",
            """
            MATCH (i:Investor)<-[:HAS_INVESTOR]-(c:Company)
            RETURN i.name AS investor, count(DISTINCT c) AS company_count
            """,
        )

    def test_rejects_wrong_statuses_for_active_question(self):
        with self.assertRaisesRegex(ValueError, 'status = "Active"'):
            neo4j_query.validate_cypher_semantics(
                "Which active YC companies use Kubernetes?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                WHERE c.status IN ["Public", "Acquired"]
                  AND toLower(t.name) = "kubernetes"
                RETURN c.name AS company, c.status AS status
                """,
            )

    def test_accepts_exact_active_filter(self):
        neo4j_query.validate_cypher_semantics(
            "Which active YC companies use Kubernetes?",
            """
            MATCH (c:Company)-[:USES]->(t:Technology)
            WHERE c.status = "Active"
              AND toLower(t.name) = "kubernetes"
            RETURN c.name AS company, c.status AS status
            """,
        )

    def test_accepts_lowercased_expanded_batch_filter(self):
        neo4j_query.validate_cypher_semantics(
            "Compare W21 and S21 companies.",
            """
            MATCH (c:Company)-[:PART_OF]->(b:Batch)
            WHERE toLower(b.name) IN ["winter 2021", "summer 2021"]
            RETURN b.name AS batch, count(DISTINCT c) AS company_count
            """,
        )

    def test_rejects_wrong_case_direct_expanded_batch_filter(self):
        with self.assertRaisesRegex(ValueError, "Winter 2021"):
            neo4j_query.validate_cypher_semantics(
                "Which W21 companies are active?",
                """
                MATCH (c:Company)
                WHERE c.yc_batch = "winter 2021"
                  AND c.status = "Active"
                RETURN c.name AS company, c.status AS status
                """,
            )

    def test_rejects_case_sensitive_multi_company_name_filter(self):
        with self.assertRaisesRegex(ValueError, "case-insensitively"):
            neo4j_query.validate_cypher_semantics(
                "Who founded Stripe and RazorPay?",
                """
                MATCH (f:Founder)-[:FOUNDED]->(c:Company)
                WHERE c.name IN ["Stripe", "RazorPay"]
                RETURN f.name AS founder, c.name AS company
                """,
            )

    def test_rejects_case_sensitive_company_name_equality(self):
        with self.assertRaisesRegex(ValueError, "Company.name case-insensitively"):
            neo4j_query.validate_cypher_semantics(
                "Who founded Stripe and RazorPay?",
                """
                MATCH (f:Founder)-[:FOUNDED]->(c:Company)
                WHERE c.name = "Stripe" OR c.name = "RazorPay"
                RETURN f.name AS founder, c.name AS company
                """,
            )

    def test_rejects_reversed_case_sensitive_company_name_equality(self):
        with self.assertRaisesRegex(ValueError, "Company.name case-insensitively"):
            neo4j_query.validate_cypher_semantics(
                "Who founded Stripe?",
                """
                MATCH (c:Company)
                WHERE "Stripe" = c.name
                RETURN c.name AS company
                """,
            )

    def test_rejects_mixed_case_reversed_lowercase_filter(self):
        with self.assertRaisesRegex(ValueError, "lowercase literal"):
            neo4j_query.validate_cypher_semantics(
                "Who founded Stripe?",
                """
                MATCH (c:Company)
                WHERE "Stripe" = toLower(c.name)
                RETURN c.name AS company
                """,
            )

    def test_accepts_reversed_case_insensitive_company_filter(self):
        neo4j_query.validate_cypher_semantics(
            "Who founded Stripe?",
            """
            MATCH (c:Company)
            WHERE "stripe" = toLower(c.name)
            RETURN c.name AS company
            """,
        )

    def test_rejects_mixed_case_apostrophe_company_filter(self):
        with self.assertRaisesRegex(ValueError, "lowercase literal"):
            neo4j_query.validate_cypher_semantics(
                "Tell me about McDonald's.",
                """
                MATCH (c:Company)
                WHERE "McDonald's" = toLower(c.name)
                RETURN c.name AS company
                """,
            )

    def test_accepts_lowercase_apostrophe_company_filter(self):
        neo4j_query.validate_cypher_semantics(
            "Tell me about McDonald's.",
            """
            MATCH (c:Company)
            WHERE "mcdonald's" = toLower(c.name)
            RETURN c.name AS company
            """,
        )

    def test_rejects_reversed_mixed_case_string_operator(self):
        with self.assertRaisesRegex(ValueError, "lowercase literal"):
            neo4j_query.validate_cypher_semantics(
                "Find companies whose name appears in Stripe.",
                """
                MATCH (c:Company)
                WHERE "Stripe" CONTAINS toLower(c.name)
                RETURN c.name AS company
                """,
            )

    def test_rejects_case_sensitive_company_property_map(self):
        with self.assertRaisesRegex(ValueError, "Company.name case-insensitively"):
            neo4j_query.validate_cypher_semantics(
                "Who founded RazorPay?",
                """
                MATCH (f:Founder)-[:FOUNDED]->(
                    c:Company {name: "RazorPay"}
                )
                RETURN f.name AS founder
                """,
            )

    def test_rejects_case_sensitive_investor_name_filter(self):
        with self.assertRaisesRegex(ValueError, "Investor.name case-insensitively"):
            neo4j_query.validate_cypher_semantics(
                "Which companies did Sequoia Capital invest in?",
                """
                MATCH (i:Investor)-[:INVESTED_IN]->(c:Company)
                WHERE i.name = "Sequoia Capital"
                RETURN c.name AS company
                """,
            )

    def test_rejects_case_sensitive_location_filter(self):
        with self.assertRaisesRegex(ValueError, "Location.country case-insensitively"):
            neo4j_query.validate_cypher_semantics(
                "Which companies are headquartered in India?",
                """
                MATCH (c:Company)-[:HEADQUARTERED_IN]->(l:Location)
                WHERE l.country = "India"
                RETURN c.name AS company
                """,
            )

    def test_accepts_case_insensitive_named_entity_filters(self):
        neo4j_query.validate_cypher_semantics(
            "Which companies did Sequoia Capital invest in India?",
            """
            MATCH (i:Investor)-[:INVESTED_IN]->(c:Company)
            MATCH (c)-[:HEADQUARTERED_IN]->(l:Location)
            WHERE toLower(i.name) = "sequoia capital"
              AND toLower(l.country) = "india"
            RETURN c.name AS company
            """,
        )

    def test_rejects_mixed_case_literals_in_lowercase_company_filter(self):
        with self.assertRaisesRegex(ValueError, "lowercase literals"):
            neo4j_query.validate_cypher_semantics(
                "Who founded Stripe and RazorPay?",
                """
                MATCH (f:Founder)-[:FOUNDED]->(c:Company)
                WHERE toLower(c.name) IN ["stripe", "RazorPay"]
                RETURN f.name AS founder, c.name AS company
                """,
            )

    def test_accepts_case_insensitive_multi_company_name_filter(self):
        neo4j_query.validate_cypher_semantics(
            "Who founded Stripe and RazorPay?",
            """
            MATCH (f:Founder)-[:FOUNDED]->(c:Company)
            WHERE toLower(c.name) IN ["stripe", "razorpay"]
            RETURN f.name AS founder, c.name AS company
            """,
        )

    def test_accepts_lowercase_unicode_company_name_filter(self):
        neo4j_query.validate_cypher_semantics(
            "Who founded Straße and Stripe?",
            """
            MATCH (f:Founder)-[:FOUNDED]->(c:Company)
            WHERE toLower(c.name) IN ["straße", "stripe"]
            RETURN f.name AS founder, c.name AS company
            """,
        )

    def test_allows_case_sensitive_in_filter_for_non_company_name(self):
        neo4j_query.validate_cypher_semantics(
            "Which companies are Active or Public?",
            """
            MATCH (c:Company)
            WHERE c.status IN ["Active", "Public"]
            RETURN c.name AS company, c.status AS status
            """,
        )

    def test_rejects_non_distinct_company_count(self):
        with self.assertRaisesRegex(ValueError, r"count\(DISTINCT c\)"):
            neo4j_query.validate_cypher_semantics(
                "Which batches produced the most healthcare companies?",
                """
                MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                MATCH (c)-[:PART_OF]->(b:Batch)
                RETURN b.name AS batch, count(c) AS company_count
                """,
            )

    def test_accepts_distinct_company_count(self):
        neo4j_query.validate_cypher_semantics(
            "Which batches produced the most healthcare companies?",
            """
            MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
            MATCH (c)-[:PART_OF]->(b:Batch)
            RETURN b.name AS batch, count(DISTINCT c) AS company_count
            """,
        )

    def test_rejects_company_property_count(self):
        with self.assertRaisesRegex(ValueError, r"count\(DISTINCT c\)"):
            neo4j_query.validate_cypher_semantics(
                "How many healthcare companies are there?",
                """
                MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                RETURN count(c.name) AS company_count
                """,
            )

    def test_rejects_distinct_company_property_count(self):
        with self.assertRaisesRegex(ValueError, r"count\(DISTINCT c\)"):
            neo4j_query.validate_cypher_semantics(
                "Which batches produced the most healthcare companies?",
                """
                MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                MATCH (c)-[:PART_OF]->(b:Batch)
                RETURN b.name AS batch, count(DISTINCT c.name) AS company_count
                """,
            )

    def test_rejects_distinct_non_company_count(self):
        with self.assertRaisesRegex(ValueError, r"count\(DISTINCT c\)"):
            neo4j_query.validate_cypher_semantics(
                "Which batches produced the most healthcare companies?",
                """
                MATCH (c:Company)-[:PART_OF]->(b:Batch)
                RETURN b.name AS batch, count(DISTINCT b) AS company_count
                """,
            )

    def test_rejects_count_star_for_company_question(self):
        with self.assertRaisesRegex(ValueError, r"count\(DISTINCT c\)"):
            neo4j_query.validate_cypher_semantics(
                "How many healthcare companies are there?",
                """
                MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                RETURN count(*) AS company_count
                """,
            )

    def test_rejects_count_star_for_singular_company_wording(self):
        with self.assertRaisesRegex(ValueError, r"count\(DISTINCT c\)"):
            neo4j_query.validate_cypher_semantics(
                "What is the company count by batch?",
                """
                MATCH (c:Company)-[:PART_OF]->(b:Batch)
                RETURN b.name AS batch, count(*) AS company_count
                """,
            )

    def test_accepts_related_entity_count_for_company_ranking(self):
        neo4j_query.validate_cypher_semantics(
            "Which companies raised the most funding rounds?",
            """
            MATCH (c:Company)-[:RAISED]->(fe:FundingEvent)
            RETURN c.name AS company, count(fe) AS funding_round_count
            """,
        )

    def test_requires_status_evidence_for_filtered_company_list(self):
        with self.assertRaisesRegex(ValueError, "company status AS status"):
            neo4j_query.validate_cypher_semantics(
                "Which dead consumer companies came from S20?",
                """
                MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                WHERE c.status = "Dead" AND c.yc_batch = "Summer 2020"
                RETURN c.name AS company, ind.name AS industry
                """,
            )

    def test_accepts_status_evidence_for_filtered_company_list(self):
        neo4j_query.validate_cypher_semantics(
            "Which dead consumer companies came from S20?",
            """
            MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
            WHERE c.status = "Dead" AND c.yc_batch = "Summer 2020"
            RETURN c.name AS company, ind.name AS industry,
                   c.status AS status
            """,
        )

    def test_requires_status_evidence_after_with_name_projection(self):
        with self.assertRaisesRegex(ValueError, "company status AS status"):
            neo4j_query.validate_cypher_semantics(
                "Which companies are dead?",
                """
                MATCH (c:Company)
                WHERE c.status = "Dead"
                WITH c.name AS company
                RETURN company
                """,
            )

    def test_accepts_status_evidence_after_with_projection(self):
        neo4j_query.validate_cypher_semantics(
            "Which companies are dead?",
            """
            MATCH (c:Company)
            WHERE c.status = "Dead"
            WITH c.name AS company, c.status AS status
            RETURN company, status
            """,
        )

    def test_requires_status_evidence_for_negative_predicate(self):
        with self.assertRaisesRegex(ValueError, "company status AS status"):
            neo4j_query.validate_cypher_semantics(
                "Which companies are not active?",
                """
                MATCH (c:Company)
                WHERE c.status <> "Active"
                RETURN c.name AS company
                """,
            )

    def test_requires_status_evidence_for_property_map(self):
        with self.assertRaisesRegex(ValueError, "company status AS status"):
            neo4j_query.validate_cypher_semantics(
                "Which companies are dead?",
                """
                MATCH (c:Company {status: "Dead"})
                RETURN c.name AS company
                """,
            )

    def test_requires_status_evidence_with_startup_alias(self):
        with self.assertRaisesRegex(ValueError, "company status AS status"):
            neo4j_query.validate_cypher_semantics(
                "Which startups are dead?",
                """
                MATCH (c:Company {status: "Dead"})
                RETURN c.name AS startup
                """,
            )

    def test_status_aggregate_does_not_require_company_record_evidence(self):
        neo4j_query.validate_cypher_semantics(
            "How many active companies are there?",
            """
            MATCH (c:Company)
            WHERE c.status = "Active"
            RETURN count(DISTINCT c) AS company_count
            """,
        )

    def test_expands_batch_abbreviations(self):
        self.assertEqual(
            neo4j_query.expand_batch_abbreviations(
                "Compare W23 with S23 and W09."
            ),
            "Compare Winter 2023 with Summer 2023 and Winter 2009.",
        )

    def test_rejects_case_sensitive_fintech_filter(self):
        with self.assertRaisesRegex(ValueError, "Industry.name case-insensitively"):
            neo4j_query.validate_cypher_semantics(
                "Which fintech companies are headquartered in India?",
                """
                MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                WHERE ind.name IN ["fintech", "finance", "payments"]
                RETURN c.name AS company, ind.name AS industry
                """,
            )

    def test_rejects_mixed_case_literals_with_lowercased_property(self):
        with self.assertRaisesRegex(ValueError, "lowercase literals"):
            neo4j_query.validate_cypher_semantics(
                "Which fintech companies are headquartered in India?",
                """
                MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                WHERE toLower(ind.name) IN ["Fintech", "Finance", "Payments"]
                RETURN c.name AS company, ind.name AS industry
                """,
            )

    def test_average_fintech_question_is_treated_as_aggregate(self):
        neo4j_query.validate_cypher_semantics(
            "What is the average team size of fintech companies?",
            """
            MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
            WHERE toLower(ind.name) IN ["fintech", "finance", "payments"]
            RETURN avg(c.team_size) AS average_team_size
            """,
        )

    def test_number_of_employees_does_not_bypass_fintech_list_validation(self):
        with self.assertRaisesRegex(ValueError, "Fintech, Finance, or Payments"):
            neo4j_query.validate_cypher_semantics(
                "Which fintech companies have a number of employees above 100?",
                "MATCH (c:Company) WHERE c.team_size > 100 RETURN c.name AS company",
            )

    def test_generate_cypher_uses_expanded_batch_names(self):
        with patch.object(
            neo4j_query,
            "call_llm",
            return_value="MATCH (b:Batch {name: 'Winter 2023'}) RETURN b.name",
        ) as call_llm:
            neo4j_query.generate_cypher("Which companies joined W23?")

        self.assertIn("Winter 2023", call_llm.call_args.args[1])
        self.assertNotIn("W23", call_llm.call_args.args[1])

    def test_rejects_abbreviated_batch_values(self):
        with self.assertRaisesRegex(ValueError, "stored batch name"):
            neo4j_query.validate_cypher_semantics(
                "Which companies joined W23 or S23?",
                """
                MATCH (c)-[:PART_OF]->(b:Batch)
                WHERE b.name IN ["W23", "S23"]
                RETURN c.name
                """,
            )

    def test_accepts_expanded_batch_values(self):
        neo4j_query.validate_cypher_semantics(
            "Which companies joined W23 or S23?",
            """
            MATCH (c)-[:PART_OF]->(b:Batch)
            WHERE b.name IN ["Winter 2023", "Summer 2023"]
            RETURN c.name
            """,
        )

    def test_rejects_batch_name_only_returned_as_constant(self):
        with self.assertRaisesRegex(ValueError, "stored batch name"):
            neo4j_query.validate_cypher_semantics(
                "Which companies joined W23 or S23?",
                """
                MATCH (c)-[:PART_OF]->(b:Batch)
                WHERE b.name = "Winter 2023"
                RETURN c.name, "Summer 2023" AS note
                """,
            )

    def test_rejects_batch_name_bound_without_filtering(self):
        with self.assertRaisesRegex(ValueError, "stored batch name"):
            neo4j_query.validate_cypher_semantics(
                "Which companies joined W23?",
                """
                WITH "Winter 2023" AS requested_batch
                MATCH (c:Company)
                RETURN c.name
                """,
            )

    def test_adds_server_side_result_limit(self):
        cypher = "MATCH (c:Company) RETURN c.name"

        self.assertEqual(
            neo4j_query.enforce_result_limit(cypher),
            "MATCH (c:Company) RETURN c.name\nLIMIT 50",
        )

    def test_caps_existing_trailing_limit(self):
        cypher = "MATCH (c:Company) RETURN c.name LIMIT 100;"

        self.assertEqual(
            neo4j_query.enforce_result_limit(cypher),
            "MATCH (c:Company) RETURN c.name LIMIT 50",
        )

    def test_adds_final_limit_after_intermediate_limit(self):
        cypher = (
            "MATCH (c) WITH c LIMIT 50 "
            "MATCH (c)-[:USES]->(t) RETURN c.name, t.name"
        )

        self.assertTrue(
            neo4j_query.enforce_result_limit(cypher).endswith(
                "RETURN c.name, t.name\nLIMIT 50"
            )
        )

    def test_rejects_union_queries(self):
        with self.assertRaisesRegex(ValueError, "instead of UNION"):
            neo4j_query.validate_cypher_semantics(
                "Which companies are active?",
                (
                    "MATCH (c:Company {status: 'Active'}) RETURN c.name "
                    "UNION MATCH (c:Company {status: 'Public'}) RETURN c.name"
                ),
            )

    def test_allows_union_inside_string_literal(self):
        neo4j_query.validate_cypher_semantics(
            "Which companies are in Union City?",
            """
            MATCH (c:Company)-[:HEADQUARTERED_IN]->(l:Location)
            WHERE toLower(l.city) = "union city"
            RETURN c.name
            """,
        )

    def test_rejects_contains_for_short_ai_category(self):
        with self.assertRaisesRegex(ValueError, "both AI category aliases"):
            neo4j_query.validate_cypher_semantics(
                "Compare AI startups in San Francisco and New York.",
                'MATCH (c)-[:OPERATES_IN]->(i) WHERE toLower(i.name) CONTAINS "ai" RETURN c',
            )

    def test_rejects_single_exact_ai_alias(self):
        with self.assertRaisesRegex(ValueError, "both AI category aliases"):
            neo4j_query.validate_cypher_semantics(
                "Which companies use AI?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                WHERE toLower(t.name) = "ai"
                RETURN c.name AS company, t.name AS technology
                """,
            )

    def test_accepts_both_exact_ai_aliases(self):
        neo4j_query.validate_cypher_semantics(
            "Which companies use Artificial Intelligence?",
            """
            MATCH (c:Company)-[:USES]->(t:Technology)
            WHERE toLower(t.name) IN ["ai", "artificial intelligence"]
            RETURN c.name AS company, t.name AS technology
            """,
        )

    def test_accepts_explicit_company_named_ai(self):
        neo4j_query.validate_cypher_semantics(
            "Which company is named AI?",
            """
            MATCH (c:Company)
            WHERE toLower(c.name) = "ai"
            RETURN c.name AS company
            """,
        )

    def test_accepts_explicit_company_named_artificial_intelligence(self):
        neo4j_query.validate_cypher_semantics(
            "Which company is named Artificial Intelligence?",
            """
            MATCH (c:Company)
            WHERE toLower(c.name) = "artificial intelligence"
            RETURN c.name AS company
            """,
        )

    def test_accepts_explicit_companies_named_ai(self):
        neo4j_query.validate_cypher_semantics(
            "Which companies are named AI?",
            """
            MATCH (c:Company)
            WHERE toLower(c.name) = "ai"
            RETURN c.name AS company
            """,
        )

    def test_accepts_company_name_ending_in_ai(self):
        neo4j_query.validate_cypher_semantics(
            "What is Scale AI's funding history?",
            """
            MATCH (c:Company)-[:RAISED]->(fe:FundingEvent)
            WHERE toLower(c.name) = "scale ai"
            RETURN c.name AS company, fe.round AS round
            """,
        )

    def test_company_name_ai_filter_does_not_satisfy_category_intent(self):
        with self.assertRaisesRegex(ValueError, "both AI category aliases"):
            neo4j_query.validate_cypher_semantics(
                "Which companies use AI?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                WHERE toLower(c.name) = "ai"
                RETURN c.name AS company, t.name AS technology
                """,
            )

    def test_company_proper_name_does_not_suppress_ai_category_intent(self):
        with self.assertRaisesRegex(ValueError, "both AI category aliases"):
            neo4j_query.validate_cypher_semantics(
                "Which AI companies compete with Scale AI?",
                """
                MATCH (c:Company)-[:COMPETES_WITH]->(rival:Company)
                WHERE toLower(rival.name) = "scale ai"
                RETURN c.name AS company
                """,
            )

    def test_rejects_ai_alias_filter_on_company_name(self):
        with self.assertRaisesRegex(ValueError, "both AI category aliases"):
            neo4j_query.validate_cypher_semantics(
                "Which companies use AI?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                WHERE toLower(c.name) IN ["ai", "artificial intelligence"]
                RETURN c.name AS company, t.name AS technology
                """,
            )

    def test_rejects_raw_category_aggregation(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            neo4j_query.validate_cypher_semantics(
                "Which technologies are most used by YC companies?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                RETURN t.name AS technology,
                       count(DISTINCT c) AS company_count
                """,
            )

    def test_accepts_canonical_ai_category_aggregation(self):
        neo4j_query.validate_cypher_semantics(
            "Which technologies are most used by YC companies?",
            """
            MATCH (c:Company)-[:USES]->(t:Technology)
            RETURN CASE
                     WHEN toLower(t.name) IN ["ai", "artificial intelligence"]
                     THEN "Artificial Intelligence"
                     ELSE t.name
                   END AS technology,
                   count(DISTINCT c) AS company_count
            """,
        )

    def test_prompt_and_validator_support_conditional_batch_category_counts(self):
        self.assertIn(
            "For W21 and S21, compare healthcare and fintech company counts.",
            neo4j_query.GRAPH_SCHEMA,
        )
        neo4j_query.validate_cypher_semantics(
            "For W21 and S21, compare healthcare and fintech company counts.",
            """
            MATCH (c:Company)-[:PART_OF]->(b:Batch)
            MATCH (c)-[:OPERATES_IN]->(ind:Industry)
            WHERE b.name IN ["Winter 2021", "Summer 2021"]
            RETURN b.name AS batch,
                   count(DISTINCT CASE
                     WHEN toLower(ind.name) CONTAINS "health" THEN c END
                   ) AS healthcare_count,
                   count(DISTINCT CASE
                     WHEN toLower(ind.name) IN [
                       "fintech", "finance", "payments"
                     ] THEN c END
                   ) AS fintech_count
            """,
        )

    def test_rejects_noncanonical_case_category_aggregation(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            neo4j_query.validate_cypher_semantics(
                "Which technologies are most used by YC companies?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                RETURN CASE WHEN toLower(t.name) = "ai"
                            THEN "AI" ELSE t.name END AS technology,
                       count(DISTINCT c) AS company_count
                """,
            )

    def test_rejects_raw_category_alias_aggregation(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            neo4j_query.validate_cypher_semantics(
                "Which technologies are most used by YC companies?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                RETURN t.name AS category,
                       count(DISTINCT c) AS company_count
                """,
            )

    def test_rejects_wrapped_raw_category_aggregation(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            neo4j_query.validate_cypher_semantics(
                "Which technologies are most used by YC companies?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                RETURN toLower(t.name) AS technology,
                       count(DISTINCT c) AS company_count
                """,
            )

    def test_rejects_raw_category_with_additional_grouping(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            neo4j_query.validate_cypher_semantics(
                "Which industries are most common by company stage?",
                """
                MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                RETURN ind.name AS industry, c.stage AS stage,
                       count(DISTINCT c) AS company_count
                """,
            )

    def test_rejects_raw_category_projected_through_with(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            neo4j_query.validate_cypher_semantics(
                "Which technologies are most used by YC companies?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                WITH t.name AS category, count(DISTINCT c) AS company_count
                RETURN category, company_count
                """,
            )

    def test_rejects_aliased_category_node_aggregation(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            neo4j_query.validate_cypher_semantics(
                "Which technologies are most used by YC companies?",
                """
                MATCH (c:Company)-[:USES]->(t:Technology)
                WITH t AS category, count(DISTINCT c) AS company_count
                RETURN category.name AS technology, company_count
                """,
            )

    def test_accepts_canonical_category_projected_through_with(self):
        neo4j_query.validate_cypher_semantics(
            "Which technologies are most used by YC companies?",
            """
            MATCH (c:Company)-[:USES]->(t:Technology)
            WITH CASE
                   WHEN toLower(t.name) IN ["artificial intelligence", "ai"]
                   THEN "Artificial Intelligence"
                   ELSE t.name
                 END AS category,
                 count(DISTINCT c) AS company_count
            RETURN category, company_count
            """,
        )

    def test_accepts_canonical_industry_aggregation_with_other_groups(self):
        neo4j_query.validate_cypher_semantics(
            "What are common patterns among failed YC startups?",
            """
            MATCH (c:Company)
            WHERE c.status = "Dead"
            OPTIONAL MATCH (c)-[:OPERATES_IN]->(ind:Industry)
            OPTIONAL MATCH (c)-[:HEADQUARTERED_IN]->(l:Location)
            RETURN CASE
                     WHEN toLower(ind.name) IN ["ai", "artificial intelligence"]
                     THEN "Artificial Intelligence"
                     ELSE ind.name
                   END AS industry,
                   l.country AS country,
                   count(DISTINCT c) AS dead_count
            """,
        )

    def test_rejects_or_for_b2b_saas(self):
        with self.assertRaisesRegex(ValueError, "both categories"):
            neo4j_query.validate_cypher_semantics(
                "Which B2B SaaS companies are active?",
                'MATCH (c)-[:OPERATES_IN]->(i) WHERE i.name = "B2B" OR i.name = "SaaS" RETURN c',
            )

    def test_rejects_in_membership_for_b2b_saas(self):
        with self.assertRaisesRegex(ValueError, "IN membership"):
            neo4j_query.validate_cypher_semantics(
                "Which B2B SaaS companies are active?",
                """
                MATCH (c)-[:OPERATES_IN]->(i)
                WHERE toLower(i.name) IN ["b2b", "saas"]
                RETURN c
                """,
            )

    def test_accepts_all_predicate_for_b2b_saas(self):
        neo4j_query.validate_cypher_semantics(
            "Which B2B SaaS companies are active?",
            """
            MATCH (c)-[:OPERATES_IN]->(i)
            WITH c, collect(toLower(i.name)) AS industries
            WHERE all(category IN ["b2b", "saas"] WHERE category IN industries)
            RETURN c
            """,
        )

    def test_rejects_literal_or_membership_for_b2b_saas(self):
        with self.assertRaisesRegex(ValueError, "OR membership"):
            neo4j_query.validate_cypher_semantics(
                "Which B2B SaaS companies are active?",
                """
                MATCH (c)-[:OPERATES_IN]->(i)
                WITH c, collect(toLower(i.name)) AS industries
                WHERE "b2b" IN industries OR "saas" IN industries
                RETURN c
                """,
            )

    def test_rejects_singleton_in_or_for_b2b_saas(self):
        with self.assertRaisesRegex(ValueError, "OR condition"):
            neo4j_query.validate_cypher_semantics(
                "Which B2B SaaS companies are active?",
                """
                MATCH (c)-[:OPERATES_IN]->(i)
                WHERE toLower(i.name) IN ["b2b"]
                   OR toLower(i.name) IN ["saas"]
                RETURN c
                """,
            )

    def test_rejects_unknown_team_sizes_from_employee_filter(self):
        with self.assertRaisesRegex(ValueError, "unknown team sizes"):
            neo4j_query.validate_cypher_semantics(
                "Which companies have fewer than 20 employees?",
                "MATCH (c:Company) WHERE c.team_size < 20 RETURN c",
            )

    def test_accepts_conjunctive_categories_and_known_team_sizes(self):
        neo4j_query.validate_cypher_semantics(
            "Which B2B SaaS companies have fewer than 20 employees?",
            """
            MATCH (c)-[:OPERATES_IN]->(b2b), (c)-[:OPERATES_IN]->(saas)
            WHERE b2b.name = "B2B" AND saas.name = "SaaS"
              AND c.team_size > 0 AND c.team_size < 20
            RETURN c
            """,
        )

    def test_requires_filter_evidence_for_python_fintech(self):
        with self.assertRaisesRegex(ValueError, "matched industry"):
            neo4j_query.validate_cypher_semantics(
                "Which companies use Python and operate in fintech?",
                """
                MATCH (c)-[:USES]->(t:Technology)
                WHERE toLower(t.name) = "python"
                MATCH (c)-[:OPERATES_IN]->(ind:Industry)
                WHERE toLower(ind.name) IN ["fintech", "finance", "payments"]
                RETURN c.name AS company
                """,
            )

    def test_requires_filter_evidence_for_name_phrasing(self):
        with self.assertRaisesRegex(ValueError, "matched industry"):
            neo4j_query.validate_cypher_semantics(
                "Name YC companies using Python in fintech.",
                """
                MATCH (c)-[:USES]->(t:Technology)
                MATCH (c)-[:OPERATES_IN]->(ind:Industry)
                WHERE toLower(t.name) = "python"
                  AND toLower(ind.name) IN ["fintech", "finance", "payments"]
                RETURN c.name AS company
                """,
            )

    def test_accepts_filter_evidence_for_python_fintech(self):
        neo4j_query.validate_cypher_semantics(
            "Which companies use Python and operate in fintech?",
            """
            MATCH (c)-[:USES]->(t:Technology)
            WHERE toLower(t.name) = "python"
            MATCH (c)-[:OPERATES_IN]->(ind:Industry)
            WHERE toLower(ind.name) IN ["fintech", "finance", "payments"]
            RETURN c.name AS company, t.name AS technology, ind.name AS industry
            """,
        )

    def test_rejects_incomplete_fintech_synonyms(self):
        with self.assertRaisesRegex(ValueError, "Fintech, Finance, or Payments"):
            neo4j_query.validate_cypher_semantics(
            "Which companies use Python and operate in fintech?",
            """
            MATCH (c)-[:USES]->(t:Technology)
            MATCH (c)-[:OPERATES_IN]->(ind:Industry)
            WHERE toLower(t.name) = "python"
              AND toLower(ind.name) CONTAINS "fintech"
            RETURN c.name AS company, t.name AS technology,
                   ind.name AS industry
            """,
            )

    def test_rejects_conjunctive_fintech_synonyms(self):
        with self.assertRaisesRegex(ValueError, "returned company's industry"):
            neo4j_query.validate_cypher_semantics(
                "Which companies use Python and operate in fintech?",
                """
                MATCH (c)-[:USES]->(t:Technology)
                MATCH (c)-[:OPERATES_IN]->(ind:Industry)
                WHERE toLower(t.name) = "python"
                  AND toLower(ind.name) CONTAINS "fintech"
                  AND toLower(ind.name) CONTAINS "finance"
                  AND toLower(ind.name) CONTAINS "payments"
                RETURN c.name AS company, t.name AS technology,
                       ind.name AS industry
                """,
            )

    def test_rejects_narrow_fintech_filter_without_python(self):
            with self.assertRaisesRegex(ValueError, "Fintech, Finance, or Payments"):
                neo4j_query.validate_cypher_semantics(
                    "Which YC companies headquartered in India operate in fintech?",
                    """
                    MATCH (c)-[:HEADQUARTERED_IN]->(l:Location)
                    MATCH (c)-[:OPERATES_IN]->(ind:Industry)
                    WHERE toLower(l.country) = "india"
                      AND toLower(ind.name) CONTAINS "fintech"
                    RETURN c.name AS company, ind.name AS industry
                    """,
                )

    def test_accepts_fintech_synonyms_without_python(self):
            neo4j_query.validate_cypher_semantics(
                "Which YC companies headquartered in India operate in fintech?",
                """
                MATCH (c)-[:HEADQUARTERED_IN]->(l:Location)
                MATCH (c)-[:OPERATES_IN]->(ind:Industry)
                WHERE toLower(l.country) = "india"
                  AND toLower(ind.name) IN ["fintech", "finance", "payments"]
                RETURN c.name AS company, ind.name AS industry
                """,
            )

    def test_what_fintech_companies_requires_normalized_filter(self):
            with self.assertRaisesRegex(ValueError, "Fintech, Finance, or Payments"):
                neo4j_query.validate_cypher_semantics(
                    "What fintech companies are headquartered in India?",
                    """
                    MATCH (c:Company)-[:HEADQUARTERED_IN]->(l:Location)
                    WHERE toLower(l.country) = "india"
                    RETURN c.name AS company
                    """,
                )

    def test_rejects_fintech_list_on_non_industry_property(self):
            with self.assertRaisesRegex(ValueError, "returned company's industry"):
                neo4j_query.validate_cypher_semantics(
                    "Which fintech companies are headquartered in India?",
                    """
                    MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                    WHERE toLower(c.name) IN ["fintech", "finance", "payments"]
                    RETURN c.name AS company, ind.name AS industry
                    """,
                )

    def test_rejects_disconnected_fintech_industry(self):
        with self.assertRaisesRegex(ValueError, "returned company's industry"):
                neo4j_query.validate_cypher_semantics(
                    "Which fintech companies are headquartered in India?",
                    """
                    MATCH (c:Company), (ind:Industry)
                    WHERE toLower(ind.name) IN ["fintech", "finance", "payments"]
                    RETURN c.name AS company, ind.name AS industry
                    """,
                )

    def test_rejects_fintech_industry_for_unrelated_company(self):
        with self.assertRaisesRegex(ValueError, "returned company's industry"):
                neo4j_query.validate_cypher_semantics(
                    "Which fintech companies are headquartered in India?",
                    """
                    MATCH (c:Company),
                          (other:Company)-[:OPERATES_IN]->(ind:Industry)
                    WHERE toLower(ind.name) IN ["fintech", "finance", "payments"]
                    RETURN c.name AS company, ind.name AS industry
                    """,
                )

    def test_rejects_fabricated_industry_evidence(self):
        with self.assertRaisesRegex(ValueError, "industry name AS industry"):
                neo4j_query.validate_cypher_semantics(
                    "Which fintech companies are headquartered in India?",
                    """
                    MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                    WHERE toLower(ind.name) IN ["fintech", "finance", "payments"]
                    RETURN c.name AS company, "not evidence" AS industry
                    """,
                )

    def test_rejects_fabricated_technology_evidence(self):
        with self.assertRaisesRegex(ValueError, "technology name AS technology"):
                neo4j_query.validate_cypher_semantics(
                    "Which companies use Python and operate in fintech?",
                    """
                    MATCH (c:Company)-[:USES]->(t:Technology)
                    MATCH (c)-[:OPERATES_IN]->(ind:Industry)
                    WHERE toLower(t.name) = "python"
                      AND toLower(ind.name) IN ["fintech", "finance", "payments"]
                    RETURN c.name AS company, ind.name AS industry,
                           "not evidence" AS technology
                    """,
                )

    def test_compare_activity_still_requires_fintech_filter(self):
        with self.assertRaisesRegex(ValueError, "Fintech, Finance, or Payments"):
                neo4j_query.validate_cypher_semantics(
                    "Which fintech companies compare prices for consumers?",
                    "MATCH (c:Company) RETURN c.name AS company",
                )

    def test_accepts_fintech_synonym_in_list(self):
        neo4j_query.validate_cypher_semantics(
            "Which companies use Python and operate in fintech?",
            """
            MATCH (c)-[:USES]->(t:Technology)
            MATCH (c)-[:OPERATES_IN]->(ind:Industry)
            WHERE toLower(t.name) = "python"
              AND toLower(ind.name) IN ["fintech", "finance", "payments"]
            RETURN c.name AS company, t.name AS technology,
                   ind.name AS industry
            """,
        )

    def test_does_not_require_industry_for_python_payment_product(self):
        neo4j_query.validate_cypher_semantics(
            "Which Python companies use Stripe for payments?",
            """
            MATCH (c)-[:USES]->(t:Technology)
            WHERE toLower(t.name) = "python"
            RETURN c.name AS company, t.name AS technology
            """,
        )

    def test_does_not_require_record_evidence_for_python_fintech_aggregate(self):
        neo4j_query.validate_cypher_semantics(
            "How many companies use Python versus operate in fintech?",
            """
            MATCH (c:Company)
            OPTIONAL MATCH (c)-[:USES]->(t:Technology)
            OPTIONAL MATCH (c)-[:OPERATES_IN]->(ind:Industry)
            RETURN count(DISTINCT CASE WHEN toLower(t.name) = "python"
                         THEN c END) AS python_count,
                   count(DISTINCT CASE WHEN toLower(ind.name) CONTAINS "fintech"
                         THEN c END) AS fintech_count
            """,
        )

    def test_accepts_conditional_distinct_company_count_with_else_null(self):
        neo4j_query.validate_cypher_semantics(
            "How many healthcare companies are there?",
            """
            MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
            RETURN count(
                DISTINCT CASE WHEN ind.name = "Healthcare" THEN c ELSE NULL END
            ) AS healthcare_count
            """,
        )

    def test_rejects_non_distinct_conditional_count_alongside_distinct_count(self):
        with self.assertRaisesRegex(
            ValueError, "Count conditional Company nodes"
        ):
            neo4j_query.validate_cypher_semantics(
                "How many healthcare companies are there?",
                """
                MATCH (c:Company)-[:OPERATES_IN]->(ind:Industry)
                RETURN count(DISTINCT c) AS total,
                       count(CASE WHEN ind.name = "Healthcare"
                             THEN c END) AS healthcare_count
                """,
            )


class CategoryNormalizationTests(unittest.TestCase):
    def test_normalizes_ai_aliases_recursively(self):
        results = [
            {
                "technology": "AI",
                "technologies": ["AI", "Artificial Intelligence", "Python"],
                "company": "AI",
            }
        ]

        normalized = neo4j_query.normalize_category_aliases(results)

        self.assertEqual(
            normalized,
            [
                {
                    "technology": "Artificial Intelligence",
                    "technologies": ["Artificial Intelligence", "Python"],
                    "company": "AI",
                }
            ],
        )
        self.assertEqual(results[0]["technology"], "AI")


class AnswerSynthesisTests(unittest.TestCase):
    def test_empty_results_do_not_call_llm(self):
        with patch.object(neo4j_query, "call_llm") as call_llm:
            answer = neo4j_query.synthesize_answer("Any matches?", [])

        call_llm.assert_not_called()
        self.assertIn("No results found", answer)

    def test_false_no_data_claim_uses_grounded_fallback(self):
        results = [
            {
                "company": "ExampleCo",
                "round": "Seed",
                "year": 2023,
                "status": "Dead",
            }
        ]
        with patch.object(
            neo4j_query,
            "call_llm",
            return_value="The data provided does not contain enough information to answer.",
        ):
            answer = neo4j_query.synthesize_answer(
                "Show companies that raised funding but are now inactive.",
                results,
            )

        self.assertIn("Found 1 matching record", answer)
        self.assertIn("Company: ExampleCo", answer)
        self.assertIn("Status: Dead", answer)

    def test_valid_grounded_answer_is_preserved(self):
        expected = "ExampleCo raised a Seed round in 2023."
        with patch.object(neo4j_query, "call_llm", return_value=expected):
            answer = neo4j_query.synthesize_answer(
                "Which company raised a seed round?",
                [{"company": "ExampleCo", "round": "Seed", "year": 2023}],
            )

        self.assertEqual(answer, expected)

    def test_grounded_partial_answer_is_preserved(self):
        expected = (
            "Patrick Collison founded Stripe. I cannot determine other "
            "biographical details from the provided data."
        )
        with patch.object(neo4j_query, "call_llm", return_value=expected):
            answer = neo4j_query.synthesize_answer(
                "Who founded Stripe, and what else is known about the founder?",
                [{"founder": "Patrick Collison", "company": "Stripe"}],
            )

        self.assertEqual(answer, expected)

    def test_zero_count_aggregate_does_not_claim_one_match(self):
        for alias in (
            "company_count",
            "companyCount",
            "count(c)",
            "total_companies",
            "matching_companies",
            "num_companies",
        ):
            with self.subTest(alias=alias):
                expected = "No results were found."
                with patch.object(neo4j_query, "call_llm", return_value=expected):
                    answer = neo4j_query.synthesize_answer(
                        "How many companies match?",
                        [{alias: 0}],
                        f"MATCH (c:Company) RETURN count(c) AS `{alias}`",
                    )

                self.assertEqual(answer, expected)

    def test_numeric_grounded_partial_answer_is_preserved(self):
        expected = (
            "There are 12 companies, but I cannot determine total funding from "
            "the provided data."
        )
        with patch.object(neo4j_query, "call_llm", return_value=expected):
            answer = neo4j_query.synthesize_answer(
                "How many companies are there and how much did they raise?",
                [{"company_count": 12, "total_funding": None}],
            )

        self.assertEqual(answer, expected)

    def test_generic_status_does_not_ground_no_data_claim(self):
        with patch.object(
            neo4j_query,
            "call_llm",
            return_value=(
                "Cannot determine the active companies from the provided data."
            ),
        ):
            answer = neo4j_query.synthesize_answer(
                "Which companies are active?",
                [{"company": "ExampleCo", "status": "Active"}],
            )

        self.assertIn("Company: ExampleCo", answer)

    def test_subject_mention_inside_no_data_claim_is_not_grounding(self):
        with patch.object(
            neo4j_query,
            "call_llm",
            return_value=(
                "Cannot determine who founded Stripe from the provided data."
            ),
        ):
            answer = neo4j_query.synthesize_answer(
                "Who founded Stripe?",
                [{"founder": "Patrick Collison", "company": "Stripe"}],
            )

        self.assertIn("Founder: Patrick Collison", answer)

    def test_subject_preamble_before_no_data_claim_is_not_grounding(self):
        with patch.object(
            neo4j_query,
            "call_llm",
            return_value=(
                "Stripe is the company in question. Cannot determine who "
                "founded Stripe from the provided data."
            ),
        ):
            answer = neo4j_query.synthesize_answer(
                "Who founded Stripe?",
                [{"founder": "Patrick Collison", "company": "Stripe"}],
            )

        self.assertIn("Founder: Patrick Collison", answer)

    def test_fallback_discloses_truncation(self):
        results = [{"company": f"Company {index}"} for index in range(55)]

        answer = neo4j_query.format_grounded_results(results)

        self.assertIn("Showing the first 50 of 55 records.", answer)
        self.assertNotIn("Company 54", answer)


class QueryPipelineTests(unittest.TestCase):
    def test_semantic_failure_regenerates_before_execution(self):
        driver = MagicMock()
        invalid = (
            'MATCH (c)-[:OPERATES_IN]->(i) '
            'WHERE toLower(i.name) CONTAINS "ai" RETURN c LIMIT 100'
        )
        valid = (
            'MATCH (c)-[:OPERATES_IN]->(i) '
            'WHERE toLower(i.name) IN ["ai", "artificial intelligence"] '
            'RETURN c LIMIT 100'
        )

        with (
            patch.object(neo4j_query.GraphDatabase, "driver", return_value=driver),
            patch.object(neo4j_query, "classify_query", return_value="global"),
            patch.object(
                neo4j_query,
                "generate_cypher",
                side_effect=[invalid, valid],
            ) as generate,
            patch.object(
                neo4j_query,
                "execute_cypher",
                return_value=[{"company": "ExampleCo"}],
            ) as execute,
            patch.object(
                neo4j_query,
                "synthesize_answer",
                return_value="ExampleCo",
            ),
        ):
            result = neo4j_query.query("Which AI companies are active?")

        self.assertEqual(generate.call_count, 2)
        execute.assert_called_once_with(
            neo4j_query.enforce_result_limit(valid),
            driver,
        )
        self.assertEqual(result["result_count"], 1)
        driver.close.assert_called_once()

    def test_exhausted_semantic_retries_raise_error(self):
        driver = MagicMock()
        invalid = (
            'MATCH (c)-[:OPERATES_IN]->(i) '
            'WHERE toLower(i.name) CONTAINS "ai" RETURN c'
        )

        with (
            patch.object(neo4j_query.GraphDatabase, "driver", return_value=driver),
            patch.object(neo4j_query, "classify_query", return_value="global"),
            patch.object(neo4j_query, "generate_cypher", return_value=invalid),
            patch.object(neo4j_query, "execute_cypher") as execute,
        ):
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                neo4j_query.query("Which AI companies are active?")

        execute.assert_not_called()
        driver.close.assert_called_once()

    def test_execution_streams_at_most_max_results(self):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        session.run.return_value = (
            {"company": f"Company {index}"}
            for index in range(neo4j_query.MAX_RESULTS + 10)
        )

        results = neo4j_query.execute_cypher(
            "MATCH (c:Company) RETURN c.name AS company",
            driver,
        )

        self.assertEqual(len(results), neo4j_query.MAX_RESULTS)
        self.assertEqual(results[-1]["company"], "Company 49")


if __name__ == "__main__":
    unittest.main()
