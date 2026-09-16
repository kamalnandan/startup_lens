import unittest

import assertion_query


def fact(**kwargs):
    base = {"predicate": "founded_on", "value": "2008-08",
            "value_type": "date", "granularity": "month",
            "source_span": "Founded in August of 2008"}
    base.update(kwargs)
    return base


class AssertedFilterTests(unittest.TestCase):
    """Withheld facts are claims the source never made.

    Airbnb's ticker sits in the graph right now as an unasserted fact, because
    the model supplied ABNB from memory and the document never wrote it.
    Returning it would be precisely the confident-wrong answer the assertion
    model exists to prevent, so a query that reads facts without checking their
    status must fail rather than run.
    """

    def test_accepts_an_explicit_filter(self):
        assertion_query.validate_asserted_filter(
            "MATCH (c:V2Company)-[:ASSERTS]->(f:V2Fact) "
            "WHERE f.asserted = true RETURN f.value"
        )

    def test_accepts_an_inline_filter(self):
        assertion_query.validate_asserted_filter(
            "MATCH (c:V2Company)-[:ASSERTS]->(f:V2Fact {asserted: true}) "
            "RETURN f.value"
        )

    def test_rejects_an_unguarded_read(self):
        with self.assertRaisesRegex(ValueError, "asserted"):
            assertion_query.validate_asserted_filter(
                "MATCH (c:V2Company)-[:ASSERTS]->(f:V2Fact) RETURN f.value"
            )

    def test_rejects_when_only_one_of_two_is_guarded(self):
        with self.assertRaisesRegex(ValueError, r"\bg\b"):
            assertion_query.validate_asserted_filter(
                "MATCH (c:V2Company)-[:ASSERTS]->(f:V2Fact), "
                "(c)-[:ASSERTS]->(g:V2Fact) "
                "WHERE f.asserted = true RETURN f.value, g.value"
            )

    def test_ignores_queries_that_read_no_facts(self):
        assertion_query.validate_asserted_filter(
            "MATCH (c:V2Company) RETURN c.name"
        )


class SingleValueTests(unittest.TestCase):
    def test_one_fact_is_the_answer(self):
        chosen, conflict = assertion_query.resolve_single_valued([fact()])
        self.assertEqual(chosen["value"], "2008-08")
        self.assertIsNone(conflict)

    def test_a_coarser_date_does_not_contradict_a_finer_one(self):
        chosen, conflict = assertion_query.resolve_single_valued([
            fact(value="2008", granularity="year"),
            fact(value="2008-08", granularity="month"),
        ])
        self.assertIsNone(conflict)
        self.assertEqual(chosen["value"], "2008-08")

    def test_dates_as_the_documents_actually_write_them(self):
        # These are the exact forms sitting in the prototype graph. The first
        # version of this compared date strings directly, which made every
        # coarse date look like it contradicted the finer one containing it -
        # a false conflict, and a false abstention, on the commonest case
        # there is. The tests missed it by using ISO dates the graph does not
        # contain.
        for value, expected in (
            ("2008", (2008, None, None)),
            ("August 2008", (2008, 8, None)),
            ("March 23, 2018", (2018, 3, 23)),
            ("2021-04-14", (2021, 4, 14)),
            ("December 2020", (2020, 12, None)),
        ):
            self.assertEqual(assertion_query.date_parts(value), expected, value)

    def test_a_month_and_its_year_are_one_answer(self):
        chosen, conflict = assertion_query.resolve_single_valued([
            fact(value="2008", granularity="year"),
            fact(value="August 2008", granularity="month"),
        ])
        self.assertIsNone(conflict)
        self.assertEqual(chosen["value"], "August 2008")

    def test_different_months_of_one_year_still_conflict(self):
        chosen, conflict = assertion_query.resolve_single_valued([
            fact(value="August 2008", granularity="month"),
            fact(value="September 2008", granularity="month"),
        ])
        self.assertIsNone(chosen)
        self.assertEqual(len(conflict["values"]), 2)

    def test_a_missing_granularity_property_does_not_decide_it(self):
        # The value states its own precision; the property is a convenience
        # that may not be there.
        chosen, conflict = assertion_query.resolve_single_valued([
            fact(value="2007", granularity=None),
            fact(value="October 2007", granularity=None),
        ])
        self.assertIsNone(conflict)
        self.assertEqual(chosen["value"], "October 2007")

    def test_two_real_answers_are_a_conflict_not_a_choice(self):
        chosen, conflict = assertion_query.resolve_single_valued([
            fact(value="2008-08", granularity="month"),
            fact(value="2009-03", granularity="month"),
        ])
        self.assertIsNone(chosen)
        self.assertEqual(sorted(conflict["values"]), ["2008-08", "2009-03"])

    def test_agreement_does_not_break_a_tie(self):
        # Picking the more-agreed value would be guessing with a number
        # attached; the caller should abstain instead.
        _, conflict = assertion_query.resolve_single_valued([
            fact(value="COIN", value_type="identifier", agreement=3),
            fact(value="CBSE", value_type="identifier", agreement=1),
        ])
        self.assertIsNotNone(conflict)

    def test_spellings_of_one_place_are_not_a_conflict(self):
        chosen, conflict = assertion_query.resolve_single_valued([
            fact(value="San Francisco", value_type="place", granularity=None),
            fact(value="San Francisco, CA, USA", value_type="place",
                 granularity=None),
        ])
        self.assertIsNone(conflict)
        self.assertEqual(chosen["value"], "San Francisco, CA, USA")

    def test_no_facts_is_not_a_conflict(self):
        chosen, conflict = assertion_query.resolve_single_valued([])
        self.assertIsNone(chosen)
        self.assertIsNone(conflict)


class SchemaTests(unittest.TestCase):
    def test_schema_tells_the_model_the_rule(self):
        self.assertIn("asserted = true", assertion_query.GRAPH_SCHEMA)

    def test_schema_lists_every_predicate(self):
        import extraction_schema
        for predicate in extraction_schema.PREDICATES:
            self.assertIn(predicate, assertion_query.GRAPH_SCHEMA)


if __name__ == "__main__":
    unittest.main()
