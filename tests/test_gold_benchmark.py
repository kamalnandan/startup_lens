import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from evaluate_gold_benchmark import run_case
from gold_benchmark import evaluate_answer, load_benchmark, validate_benchmark


def valid_record():
    return {
        "id": "identity-001",
        "category": "identity",
        "question": "Who founded ExampleCo?",
        "expected_answer_summary": "Jane Doe founded ExampleCo.",
        "expected_claims": ["Jane Doe founded ExampleCo."],
        "required_answer_terms": ["jane doe", "exampleco"],
        "acceptable_variants": {"exampleco": ["example co"]},
        "as_of_date": "2026-08-24",
        "volatility": "low",
        "sources": [
            {
                "title": "About ExampleCo",
                "url": "https://example.com/about",
                "publisher": "ExampleCo",
                "accessed_date": "2026-08-24",
                "evidence": "The company identifies Jane Doe as its founder.",
                "supports_claims": [0],
            }
        ],
        "notes": "Stable founder fact.",
    }


class GoldBenchmarkValidationTests(unittest.TestCase):
    def test_repository_benchmark_is_valid(self):
        benchmark_path = (
            Path(__file__).parents[1] / "benchmarks" / "gold_questions.json"
        )
        records = load_benchmark(benchmark_path)

        self.assertEqual([], validate_benchmark(records))
        self.assertEqual(
            [],
            [
                record["id"]
                for record in records
                if not evaluate_answer(
                    record, record["expected_answer_summary"]
                )["passed_deterministic_checks"]
            ],
        )

    def test_repository_relationship_patterns_handle_reordered_facts(self):
        benchmark_path = (
            Path(__file__).parents[1] / "benchmarks" / "gold_questions.json"
        )
        records = {record["id"]: record for record in load_benchmark(benchmark_path)}
        reordered_answers = {
            "funding-008": (
                "$3.4 billion was the expected gross IPO proceeds for Airbnb."
            ),
            "company-007": (
                "5.5 million hosts and 2.5 billion guest arrivals are reported by "
                "Airbnb."
            ),
            "compound-025": (
                "The IPO prices were $68 for Airbnb, $21 for Dropbox, and $102 for "
                "DoorDash; $250 was Coinbase's reference price, not an offering price."
            ),
        }

        for record_id, answer in reordered_answers.items():
            with self.subTest(record_id=record_id):
                self.assertTrue(
                    evaluate_answer(
                        records[record_id], answer
                    )["passed_deterministic_checks"]
                )

        negated_answers = {
            "company-024": (
                "DoorDash says DashPass is not $9.99 monthly and not $96 annually."
            ),
            "compound-015": (
                "Fred Ehrsam was not added as a Coinbase co-founder; Coinbase "
                "provides developers tools, APIs, and infrastructure."
            ),
            "compound-025": (
                "Airbnb was not $68 but $21. Dropbox was not $21 but $102. "
                "DoorDash was not $102 but $68. Coinbase did not have a $250 "
                "reference price, not an offering price."
            ),
        }
        for record_id, answer in negated_answers.items():
            with self.subTest(record_id=record_id):
                self.assertFalse(
                    evaluate_answer(
                        records[record_id], answer
                    )["passed_deterministic_checks"]
                )

    def test_valid_record_schema(self):
        records = []
        for prefix, category in (
            ("identity", "identity"),
            ("funding", "funding_status"),
            ("company", "company_facts"),
            ("compound", "compound"),
        ):
            for number in range(1, 26):
                record = valid_record()
                record["id"] = f"{prefix}-{number:03d}"
                record["category"] = category
                record["question"] = f"{prefix} benchmark question {number}?"
                records.append(record)

        self.assertEqual([], validate_benchmark(records))

    def test_rejects_uncovered_claim(self):
        record = valid_record()
        record["expected_claims"].append("ExampleCo joined Winter 2020.")

        errors = validate_benchmark([record], expected_count=1)

        self.assertIn("identity-001: claims without sources: [1].", errors)

    def test_load_rejects_non_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            path.write_text(json.dumps({"records": []}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON array"):
                load_benchmark(path)

    def test_answer_accepts_configured_variant(self):
        result = evaluate_answer(
            valid_record(),
            "Jane Doe founded Example Co.",
        )

        self.assertTrue(result["passed_deterministic_checks"])
        self.assertEqual(1.0, result["term_coverage"])

    def test_answer_reports_missing_terms(self):
        result = evaluate_answer(valid_record(), "Jane Doe is the founder.")

        self.assertFalse(result["passed_deterministic_checks"])
        self.assertEqual(["exampleco"], result["missing_terms"])

    def test_ticker_does_not_match_inside_company_name(self):
        record = valid_record()
        record["required_answer_terms"] = ["coin"]

        result = evaluate_answer(record, "Coinbase listed on Nasdaq.")

        self.assertEqual(["coin"], result["missing_terms"])

    def test_series_one_does_not_match_series_two(self):
        record = valid_record()
        record["required_answer_terms"] = ["series i"]

        result = evaluate_answer(record, "The company raised a Series II.")

        self.assertEqual(["series i"], result["missing_terms"])

    def test_answer_pattern_binds_value_to_entity(self):
        record = valid_record()
        record["required_answer_terms"] = ["airbnb", "$68", "dropbox", "$21"]
        record["acceptable_variants"] = {}
        record["required_answer_patterns"] = [
            r"\bairbnb\b.{0,30}\$68\b",
            r"\bdropbox\b.{0,30}\$21\b",
        ]

        result = evaluate_answer(
            record,
            "Airbnb priced its IPO at $68, while Dropbox priced its IPO at $21.",
        )

        self.assertTrue(result["passed_deterministic_checks"])
        self.assertEqual(record["required_answer_patterns"], result["matched_patterns"])
        self.assertEqual([], result["missing_patterns"])

    def test_answer_pattern_can_support_value_first_wording(self):
        record = valid_record()
        record["required_answer_terms"] = ["airbnb", "$68"]
        record["acceptable_variants"] = {}
        record["required_answer_patterns"] = [
            r"(?:\bairbnb\b.{0,30}\$68\b|\$68\b.{0,30}\bairbnb\b)",
        ]

        result = evaluate_answer(record, "The $68 IPO price was Airbnb's.")

        self.assertTrue(result["passed_deterministic_checks"])

    def test_answer_pattern_rejects_swapped_values(self):
        record = valid_record()
        record["required_answer_terms"] = ["airbnb", "$68", "dropbox", "$21"]
        record["acceptable_variants"] = {}
        record["required_answer_patterns"] = [
            r"\bairbnb\b.{0,30}\$68\b",
            r"\bdropbox\b.{0,30}\$21\b",
        ]

        result = evaluate_answer(
            record,
            "Airbnb priced its IPO at $21, while Dropbox priced its IPO at $68.",
        )

        self.assertFalse(result["passed_deterministic_checks"])
        self.assertEqual(record["required_answer_patterns"], result["missing_patterns"])

    def test_rejects_malformed_answer_pattern(self):
        record = valid_record()
        record["required_answer_patterns"] = [r"(unclosed"]

        errors = validate_benchmark([record], expected_count=1)

        self.assertTrue(
            any("invalid required answer pattern 0" in error for error in errors)
        )

    @patch(
        "evaluate_gold_benchmark.requests.post",
        side_effect=requests.Timeout("timed out"),
    )
    def test_request_failure_becomes_failed_result(self, _post):
        result = run_case("https://api.example.com", "secret", valid_record(), 1)

        self.assertIsNone(result["http_status"])
        self.assertIn("Timeout", result["api_error"])
        self.assertFalse(result["passed_deterministic_checks"])

    @patch("evaluate_gold_benchmark.requests.post")
    def test_non_json_response_becomes_failed_result(self, post):
        response = requests.Response()
        response.status_code = 502
        response._content = b"<html>Bad gateway</html>"
        post.return_value = response

        result = run_case("https://api.example.com", "secret", valid_record(), 1)

        self.assertEqual(502, result["http_status"])
        self.assertIn("Invalid JSON response", result["api_error"])
        self.assertFalse(result["passed_deterministic_checks"])


if __name__ == "__main__":
    unittest.main()
