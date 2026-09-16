"""Tests for the assertion-based extraction schema."""

import importlib
import unittest


extraction_schema = importlib.import_module("extraction_schema")


DOCUMENT = (
    "Coinbase was founded in June 2012 by Brian Armstrong. "
    "Fred Ehrsam later joined as a co-founder. "
    "Brian Armstrong enrolled in the Y Combinator programme. "
    "Location: Los Angeles, CA, USA. "
    "The company opened an office in London. "
    "It trades under the symbol COIN."
)


def assertion(**overrides):
    base = {
        "predicate": "founded_by",
        "value": "Brian Armstrong",
        "value_type": "person",
        "source_span": "founded in June 2012 by Brian Armstrong",
    }
    base.update(overrides)
    return base


class SpanVerificationTests(unittest.TestCase):
    def test_accepts_verbatim_span(self):
        self.assertTrue(
            extraction_schema.span_occurs_in("Fred Ehrsam later joined", DOCUMENT)
        )

    def test_tolerates_whitespace_and_quote_differences(self):
        self.assertTrue(
            extraction_schema.span_occurs_in(
                "Fred  Ehrsam\nlater joined", DOCUMENT
            )
        )

    def test_rejects_invented_span(self):
        self.assertFalse(
            extraction_schema.span_occurs_in(
                "Coinbase was founded by Jane Doe", DOCUMENT
            )
        )

    def test_rejects_empty_span(self):
        self.assertFalse(extraction_schema.span_occurs_in("   ", DOCUMENT))

    def test_fabricated_fact_is_rejected_even_when_well_formed(self):
        payload = {
            "assertions": [
                assertion(
                    predicate="ticker",
                    value="FAKE",
                    value_type="identifier",
                    source_span="It trades under the symbol FAKE.",
                )
            ]
        }
        accepted, rejected = extraction_schema.validate_extraction(payload, DOCUMENT)
        self.assertEqual(accepted, [])
        self.assertIn("does not occur", rejected[0]["reason"])


class TypeContractTests(unittest.TestCase):
    def test_rejects_unknown_predicate(self):
        with self.assertRaisesRegex(extraction_schema.ExtractionError, "Unknown predicate"):
            extraction_schema.validate_assertion(
                assertion(predicate="revenue"), DOCUMENT
            )

    def test_value_type_comes_from_the_predicate_not_the_reply(self):
        # A stated type that disagrees is recorded, not fatal: discarding the
        # fact would lose real content over a field the schema already knows.
        validated = extraction_schema.validate_assertion(
            assertion(value_type="money"), DOCUMENT
        )
        self.assertEqual(validated["value_type"], "person")
        self.assertIn("extracted as money, stored as person", validated["review"])

    def test_unrecognised_value_type_is_ignored(self):
        validated = extraction_schema.validate_assertion(
            assertion(value_type="string"), DOCUMENT
        )
        self.assertEqual(validated["value_type"], "person")
        self.assertNotIn("review", validated)

    def test_missing_value_type_is_filled_in(self):
        payload = assertion()
        payload.pop("value_type", None)
        validated = extraction_schema.validate_assertion(payload, DOCUMENT)
        self.assertEqual(validated["value_type"], "person")

    def test_string_none_is_treated_as_absent(self):
        validated = extraction_schema.validate_assertion(
            assertion(role="None", qualifier="None"), DOCUMENT
        )
        self.assertNotIn("role", validated)
        self.assertNotIn("qualifier", validated)

    def test_date_requires_granularity(self):
        with self.assertRaisesRegex(extraction_schema.ExtractionError, "granularity"):
            extraction_schema.validate_assertion(
                assertion(
                    predicate="founded_on",
                    value="2012-06",
                    value_type="date",
                ),
                DOCUMENT,
            )

    def test_date_keeps_its_granularity(self):
        validated = extraction_schema.validate_assertion(
            assertion(
                predicate="founded_on",
                value="2012-06",
                value_type="date",
                granularity="month",
            ),
            DOCUMENT,
        )
        self.assertEqual(validated["granularity"], "month")

    def test_count_must_be_an_integer(self):
        with self.assertRaisesRegex(extraction_schema.ExtractionError, "must be an integer"):
            extraction_schema.validate_assertion(
                assertion(
                    predicate="team_size",
                    value="6112",
                    value_type="count",
                    source_span="Location: Los Angeles, CA, USA",
                ),
                DOCUMENT,
            )


class PlaceRoleTests(unittest.TestCase):
    def test_working_arrangement_is_not_a_place(self):
        with self.assertRaisesRegex(extraction_schema.ExtractionError, "not a place"):
            extraction_schema.validate_assertion(
                assertion(
                    predicate="located_in",
                    value="Remote",
                    value_type="place",
                    role="headquarters",
                    source_span="Location: Los Angeles, CA, USA",
                ),
                DOCUMENT,
            )

    def test_office_is_not_a_headquarters(self):
        validated = extraction_schema.validate_assertion(
            assertion(
                predicate="located_in",
                value="London",
                value_type="place",
                role="office",
                source_span="opened an office in London",
            ),
            DOCUMENT,
        )
        self.assertEqual(validated["role"], "office")

    def test_unknown_role_keeps_the_fact_and_flags_review(self):
        validated = extraction_schema.validate_assertion(
            assertion(
                predicate="located_in",
                value="London",
                value_type="place",
                role="satellite",
                source_span="opened an office in London",
            ),
            DOCUMENT,
        )
        self.assertIsNone(validated["role"])
        self.assertTrue(validated["review"])


class AggregationTests(unittest.TestCase):
    def test_repeated_value_merges_and_keeps_the_specific_role(self):
        payload = {
            "assertions": [
                assertion(),
                assertion(
                    role="original_applicant",
                    source_span="Brian Armstrong enrolled in the Y Combinator programme",
                ),
            ]
        }
        accepted, rejected = extraction_schema.validate_extraction(payload, DOCUMENT)
        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["role"], "original_applicant")

    def test_founder_order_is_preserved_from_the_source(self):
        payload = {
            "assertions": [
                assertion(role="original_applicant"),
                assertion(
                    value="Fred Ehrsam",
                    role="later_co_founder",
                    source_span="Fred Ehrsam later joined as a co-founder",
                ),
            ]
        }
        accepted, _ = extraction_schema.validate_extraction(payload, DOCUMENT)
        roles = {a["value"]: a["role"] for a in accepted}
        self.assertEqual(roles["Brian Armstrong"], "original_applicant")
        self.assertEqual(roles["Fred Ehrsam"], "later_co_founder")

    def test_conflicting_single_values_are_both_kept(self):
        payload = {
            "assertions": [
                assertion(
                    predicate="founded_on",
                    value="2012-06",
                    value_type="date",
                    granularity="month",
                ),
                assertion(
                    predicate="founded_on",
                    value="2012",
                    value_type="date",
                    granularity="year",
                    source_span="Brian Armstrong enrolled in the Y Combinator programme",
                ),
            ]
        }
        accepted, rejected = extraction_schema.validate_extraction(payload, DOCUMENT)
        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 2)
        self.assertTrue(all(a.get("conflicts_with") for a in accepted))

    def test_identical_repeated_single_value_is_collapsed(self):
        payload = {"assertions": [assertion(predicate="name", value="Coinbase",
                                            value_type="organisation"),
                                  assertion(predicate="name", value="Coinbase",
                                            value_type="organisation")]}
        accepted, rejected = extraction_schema.validate_extraction(payload, DOCUMENT)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])


class CounterfactualTests(unittest.TestCase):
    DOC = (
        "Ben Reeves was originally supposed to be part of the founding team, "
        "but parted ways with Armstrong. "
        "Coinbase was founded in June 2012 by Brian Armstrong."
    )

    def test_denied_involvement_is_not_asserted(self):
        validated = extraction_schema.validate_assertion(
            {
                "predicate": "founded_by",
                "value": "Ben Reeves",
                "value_type": "person",
                "source_span": (
                    "Ben Reeves was originally supposed to be part of the "
                    "founding team, but parted ways with Armstrong"
                ),
            },
            self.DOC,
        )
        self.assertFalse(validated["asserted"])
        self.assertTrue(validated["review"])

    def test_plain_statement_is_asserted(self):
        validated = extraction_schema.validate_assertion(
            {
                "predicate": "founded_by",
                "value": "Brian Armstrong",
                "value_type": "person",
                "source_span": "Coinbase was founded in June 2012 by Brian Armstrong",
            },
            self.DOC,
        )
        self.assertTrue(validated["asserted"])

    def test_counterfactual_fact_is_kept_not_discarded(self):
        payload = {
            "assertions": [
                {
                    "predicate": "founded_by",
                    "value": "Ben Reeves",
                    "value_type": "person",
                    "source_span": "Ben Reeves was originally supposed to be part of the founding team",
                }
            ]
        }
        accepted, rejected = extraction_schema.validate_extraction(payload, self.DOC)
        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 1)
        self.assertFalse(accepted[0]["asserted"])


class VerificationTests(unittest.TestCase):
    def _fact(self, predicate, value, span):
        return {"predicate": predicate, "value": value, "source_span": span}

    def test_prompt_states_what_the_predicate_means(self):
        prompt = extraction_schema.build_verification_prompt(
            "coinbase",
            [self._fact("raised", 50000000, "pay a US$50 million penalty")],
        )
        self.assertIn("NOT money it paid", prompt)
        self.assertIn("penalty", prompt)

    def test_unsupported_verdict_marks_fact_unasserted(self):
        facts = [self._fact("raised", 50000000, "pay a US$50 million penalty")]
        extraction_schema.apply_verdicts(
            facts, [{"id": 1, "supports": False, "reason": "a fine"}]
        )
        self.assertFalse(facts[0]["asserted"])
        self.assertIn("unsupported: a fine", facts[0]["review"])

    def test_supported_verdict_leaves_fact_alone(self):
        facts = [self._fact("raised", 5000000, "received a US$5 million Series A")]
        extraction_schema.apply_verdicts(facts, [{"id": 1, "supports": True}])
        self.assertNotIn("asserted", facts[0])

    def test_missing_verdict_is_flagged_not_trusted(self):
        facts = [self._fact("raised", 1, "x"), self._fact("raised", 2, "y")]
        extraction_schema.apply_verdicts(facts, [{"id": 1, "supports": True}])
        self.assertIn("not verified", facts[1]["review"])
        self.assertNotIn("asserted", facts[1])


class AgreementTests(unittest.TestCase):
    def _fact(self, predicate, value, **extra):
        fact = {"predicate": predicate, "value": value, "source_span": "s"}
        fact.update(extra)
        return fact

    def test_fact_in_two_of_three_runs_is_asserted(self):
        runs = [
            [self._fact("founded_by", "Brian Armstrong")],
            [self._fact("founded_by", "Brian Armstrong")],
            [],
        ]
        agreed, contested = extraction_schema.merge_runs(runs)
        self.assertEqual(len(agreed), 1)
        self.assertEqual(agreed[0]["agreement"], 2)
        self.assertEqual(contested, [])

    def test_fact_seen_once_is_contested_not_discarded(self):
        runs = [[self._fact("raised", 400000000)], [], []]
        agreed, contested = extraction_schema.merge_runs(runs)
        self.assertEqual(agreed, [])
        self.assertEqual(len(contested), 1)
        self.assertFalse(contested[0]["asserted"])

    def test_role_from_the_richer_run_is_kept(self):
        runs = [
            [self._fact("founded_by", "Fred Ehrsam")],
            [self._fact("founded_by", "Fred Ehrsam", role="later_co_founder")],
        ]
        agreed, _ = extraction_schema.merge_runs(runs)
        self.assertEqual(agreed[0]["role"], "later_co_founder")

    def test_identity_ignores_case_and_span(self):
        first = self._fact("founded_by", "Brian Armstrong", source_span="a")
        second = self._fact("founded_by", "brian  armstrong", source_span="b")
        self.assertEqual(
            extraction_schema.assertion_identity(first),
            extraction_schema.assertion_identity(second),
        )


class PromptTests(unittest.TestCase):
    def test_prompt_lists_every_predicate(self):
        prompt = extraction_schema.build_extraction_prompt("Coinbase", DOCUMENT)
        for predicate in extraction_schema.PREDICATES:
            self.assertIn(predicate, prompt)

    def test_prompt_includes_the_document(self):
        prompt = extraction_schema.build_extraction_prompt("Coinbase", DOCUMENT)
        self.assertIn("Fred Ehrsam later joined", prompt)


class PlaceNormalisationTests(unittest.TestCase):
    def place(self, value, span, role="headquarters"):
        return {"predicate": "located_in", "value": value, "value_type": "place",
                "role": role, "source_span": span}

    def test_expands_state_and_country_abbreviations(self):
        self.assertEqual(
            extraction_schema.place_components("San Francisco, CA, USA"),
            ("san francisco", "california", "united states"),
        )

    def test_shorter_spelling_is_the_same_place(self):
        self.assertTrue(
            extraction_schema.places_agree("San Francisco", "San Francisco, CA, USA")
        )
        self.assertTrue(
            extraction_schema.places_agree(
                "San Francisco, United States", "San Francisco, California, USA"
            )
        )

    def test_same_city_in_different_states_stays_separate(self):
        self.assertFalse(
            extraction_schema.places_agree(
                "Springfield, Illinois", "Springfield, Missouri"
            )
        )

    def test_a_different_city_is_not_absorbed(self):
        self.assertFalse(
            extraction_schema.places_agree(
                "San Francisco, California", "South San Francisco, California"
            )
        )

    def test_spellings_collapse_to_the_most_specific(self):
        merged = extraction_schema.merge_place_assertions([
            self.place("San Francisco", "Headquartered in San Francisco"),
            self.place("San Francisco, CA, USA", "Location: San Francisco, CA, USA"),
            self.place("San Francisco, California", "based in San Francisco, California"),
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["value"], "San Francisco, CA, USA")
        self.assertEqual(len(merged[0]["supporting_spans"]), 2)
        self.assertEqual(len(merged[0]["also_written_as"]), 2)

    def test_roles_are_not_collapsed_together(self):
        merged = extraction_schema.merge_place_assertions([
            self.place("Dublin", "Headquartered in Dublin", role="headquarters"),
            self.place("Dublin, Ireland", "an office in Dublin, Ireland", role="office"),
        ])
        self.assertEqual(len(merged), 2)

    def test_runs_that_spelled_it_differently_count_as_agreement(self):
        agreed, contested = extraction_schema.merge_runs([
            [self.place("San Francisco", "Headquartered in San Francisco")],
            [self.place("San Francisco, California", "based in San Francisco, California")],
        ], min_agreement=2)
        self.assertEqual(len(agreed), 1)
        self.assertEqual(agreed[0]["value"], "San Francisco, California")
        self.assertEqual(agreed[0]["agreement"], 2)
        self.assertEqual(contested, [])

    def test_non_places_are_untouched(self):
        facts = [{"predicate": "name", "value": "Airbnb", "value_type": "text",
                  "source_span": "Airbnb"}]
        self.assertEqual(extraction_schema.merge_place_assertions(facts), facts)


class SchemaCoverageTests(unittest.TestCase):
    """A predicate nobody asks for is a predicate nobody extracts.

    Every gap found so far had this shape: a fact stated plainly in the
    document with nowhere in the schema to put it. These guard the other half
    of that - a predicate that exists but never reaches a prompt.
    """

    def test_every_predicate_belongs_to_exactly_one_group(self):
        seen = {}
        for group, spec in extraction_schema.PREDICATE_GROUPS.items():
            for predicate in spec["predicates"]:
                self.assertIn(predicate, extraction_schema.PREDICATES,
                              f"{group} asks for unknown predicate {predicate}")
                self.assertNotIn(predicate, seen,
                                 f"{predicate} is in both {seen.get(predicate)} "
                                 f"and {group}")
                seen[predicate] = group
        missing = set(extraction_schema.PREDICATES) - set(seen)
        self.assertEqual(missing, set(),
                         f"no group ever asks for {sorted(missing)}")

    def test_every_predicate_is_defined(self):
        missing = set(extraction_schema.PREDICATES) - set(
            extraction_schema.PREDICATE_DEFINITIONS
        )
        self.assertEqual(missing, set())

    def test_new_predicates_are_extractable(self):
        for predicate in ("offers", "originated_on", "first_traded_on"):
            self.assertIn(predicate, extraction_schema.PREDICATES)
            group = next(g for g, s in extraction_schema.PREDICATE_GROUPS.items()
                         if predicate in s["predicates"])
            prompt = extraction_schema.build_targeted_prompt("Stripe", "doc", group)
            self.assertIn(predicate, prompt)


class MarkerVerificationTests(unittest.TestCase):
    """A suspicious word is a reason to check, not a verdict."""

    SPAN = ("In October 2007, in San Francisco, roommates and former "
            "schoolmates Brian Chesky and Joe Gebbia came up with an idea.")

    def flagged(self):
        return extraction_schema.validate_assertion({
            "predicate": "originated_on", "value": "2007-10",
            "granularity": "month", "source_span": self.SPAN,
        }, self.SPAN)

    def test_marker_matching_is_word_bounded(self):
        self.assertEqual(
            extraction_schema.counterfactual_markers_in("he considered it"),
            ["considered"],
        )
        # "formerly" contains "former", but "reconsidered" is not "considered".
        self.assertEqual(
            extraction_schema.counterfactual_markers_in("the deal reconsidered"),
            [],
        )

    def test_a_marker_withholds_pending_a_check(self):
        assertion = self.flagged()
        self.assertFalse(assertion["asserted"])
        self.assertTrue(assertion["needs_check"])

    def test_verification_can_release_a_flagged_fact(self):
        assertion = self.flagged()
        extraction_schema.apply_verdicts([assertion], [{"id": 1, "supports": True}])
        self.assertTrue(assertion["asserted"])
        self.assertNotIn("needs_check", assertion)

    def test_verification_can_confirm_the_marker_was_right(self):
        assertion = self.flagged()
        extraction_schema.apply_verdicts(
            [assertion], [{"id": 1, "supports": False, "reason": "denies it"}]
        )
        self.assertFalse(assertion["asserted"])

    def test_an_unverified_flagged_fact_stays_withheld(self):
        assertion = self.flagged()
        extraction_schema.apply_verdicts([assertion], [])
        self.assertFalse(assertion["asserted"])


if __name__ == "__main__":
    unittest.main()
