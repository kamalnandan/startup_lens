"""Regression tests for the schema type contract.

These tests are deliberately split into two groups. The first group replays
Cypher observed in the gold benchmark. The second group uses wording and
property pairings that were never observed, to prove the rules are driven by
the schema's semantic types rather than by a list of known-bad pairs.
"""

import importlib
import os
import unittest


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
schema_contract = importlib.import_module("schema_contract")


def validate(cypher, question="Test question"):
    neo4j_query.validate_cypher_semantics(question, cypher)


CONTRACT_ERROR = r"(cannot be returned as|internal source provenance|peer relationship)"


class ContractCoverageTests(unittest.TestCase):
    """Every declared property must have a semantic type."""

    def test_every_schema_property_is_typed(self):
        for (label, prop), fact_type in schema_contract.PROPERTY_FACT_TYPES.items():
            self.assertIsNotNone(
                fact_type, f"{label}.{prop} has no declared fact type"
            )

    def test_unmodeled_types_carry_no_property(self):
        carried = set(schema_contract.PROPERTY_FACT_TYPES.values())
        for fact_type in schema_contract.UNMODELED_FACT_TYPES:
            self.assertNotIn(
                fact_type,
                carried,
                f"{fact_type} is declared unmodeled but a property carries it",
            )


class ObservedFailureTests(unittest.TestCase):
    """Cypher taken from benchmark answers that were materially wrong."""

    def test_rejects_filename_as_founding_date(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate(
                "MATCH (c:Company) WHERE toLower(c.name) = 'stripe' WITH c "
                "RETURN c.filename AS founding_date"
            )

    def test_rejects_funding_company_as_ticker(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate(
                "MATCH (c:Company)-[:RAISED]->(fe:FundingEvent) "
                "RETURN fe.company AS ticker"
            )

    def test_rejects_description_as_mission(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate("MATCH (c:Company) RETURN c.description AS mission")

    def test_rejects_team_size_as_listing_count(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate("MATCH (c:Company) RETURN c.team_size AS active_listing_count")

    def test_rejects_amount_as_valuation(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate(
                "MATCH (c:Company)-[:RAISED]->(fe:FundingEvent) "
                "RETURN fe.company AS valuation"
            )

    def test_rejects_peer_target_projection(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate(
                "MATCH (f:Founder) WHERE toLower(f.name) = 'tony xu' WITH f MATCH (f)-[:CO_FOUNDED_WITH]->"
                "(other:Founder) "
                "WITH collect(other.name) AS co_founders "
                "RETURN co_founders"
            )


class UnseenVariantTests(unittest.TestCase):
    """Pairings never observed in the benchmark, derived only from types."""

    def test_rejects_description_as_policy(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate("MATCH (c:Company) RETURN c.description AS privacy_policy")

    def test_rejects_year_as_launch_date(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate(
                "MATCH (c:Company)-[:RAISED]->(fe:FundingEvent) "
                "RETURN fe.year AS launch_date"
            )

    def test_rejects_stage_as_revenue(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate("MATCH (c:Company) RETURN c.stage AS annual_revenue")

    def test_rejects_investor_name_as_stock_symbol(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate("MATCH (i:Investor) RETURN i.name AS stock_symbol")

    def test_rejects_location_city_as_headcount(self):
        with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
            validate("MATCH (l:Location) RETURN l.city AS employee_count")

    def test_rejects_any_provenance_projection(self):
        for alias in ("source", "origin", "c_file", "document"):
            with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
                validate(f"MATCH (c:Company) RETURN c.filename AS {alias}")

    def test_rejects_team_size_relabelled_to_other_countable(self):
        for alias in ("product_count", "office_count", "customer_count"):
            with self.assertRaisesRegex(ValueError, CONTRACT_ERROR):
                validate(f"MATCH (c:Company) RETURN c.team_size AS {alias}")


class NonRegressionTests(unittest.TestCase):
    """Legitimate projections must keep validating."""

    def test_allows_matching_types(self):
        validate("MATCH (c:Company) RETURN c.name AS company, c.stage AS stage")

    def test_allows_team_size_as_people_count(self):
        validate("MATCH (c:Company) RETURN c.team_size AS employee_count")

    def test_allows_amount_as_round_size(self):
        validate(
            "MATCH (c:Company)-[:RAISED]->(fe:FundingEvent) "
            "RETURN c.name AS company, fe.amount AS size"
        )

    def test_allows_computed_count_named_for_its_filter(self):
        validate(
            "MATCH (c:Company)-[:OPERATES_IN]->(i:Industry) WHERE toLower(i.name) = 'healthcare' WITH c, i "
            "RETURN count(DISTINCT c) AS healthcare_count"
        )

    def test_allows_year_as_year(self):
        validate(
            "MATCH (c:Company)-[:RAISED]->(fe:FundingEvent) "
            "RETURN c.name AS company, fe.year AS year"
        )

    def test_allows_competitor_when_also_reached_directly(self):
        validate(
            "MATCH (c:Company) WHERE toLower(c.name) = 'stripe' WITH c-[:COMPETES_WITH]->(rival:Company) "
            "RETURN c.name AS company, rival.name AS competitor"
        )

    def test_unknown_wording_fails_open(self):
        self.assertIsNone(schema_contract.infer_requested_fact_type("zork_flimflam"))
        validate("MATCH (c:Company) RETURN c.description AS zork_flimflam")


class SynthesisConstraintTests(unittest.TestCase):
    def test_unmodeled_types_are_always_constrained(self):
        constraints = " ".join(schema_contract.synthesis_constraints())
        for fact_type in schema_contract.UNMODELED_FACT_TYPES:
            label = schema_contract.FACT_TYPE_LABELS[fact_type]
            self.assertIn(label, constraints)

    def test_batch_is_never_presented_as_a_date(self):
        constraints = " ".join(schema_contract.synthesis_constraints())
        self.assertIn("batch", constraints.casefold())

    def test_person_aspect_question_gets_founded_note(self):
        notes = schema_contract.unmodeled_relationship_aspects(
            "Who applied to Y Combinator to build Coinbase?"
        )
        self.assertTrue(any("FOUNDED" in note for note in notes))

    def test_plain_question_gets_no_aspect_note(self):
        self.assertEqual(
            schema_contract.unmodeled_relationship_aspects(
                "What industry is Stripe in?"
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
