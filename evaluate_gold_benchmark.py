"""Run the source-backed gold benchmark against a deployed StartupLens API."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from gold_benchmark import evaluate_answer, load_benchmark, validate_benchmark


DEFAULT_BENCHMARK = Path("benchmarks/gold_questions.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--api-url", default=os.environ.get("STARTUPLENS_API_URL"))
    parser.add_argument("--api-key", default=os.environ.get("STARTUPLENS_API_KEY"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def run_case(
    api_url: str,
    api_key: str,
    record: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/query",
            headers={"X-API-Key": api_key},
            json={"query": record["question"]},
            timeout=timeout,
        )
    except requests.RequestException as error:
        checks = evaluate_answer(record, "")
        return {
            "id": record["id"],
            "question": record["question"],
            "http_status": None,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "method": None,
            "result_count": None,
            "answer": "",
            "cypher": None,
            "api_error": f"{type(error).__name__}: {error}",
            "expected_answer_summary": record["expected_answer_summary"],
            "expected_claims": record["expected_claims"],
            "sources": record["sources"],
            **checks,
        }
    duration = round(time.perf_counter() - started, 3)
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as error:
        checks = evaluate_answer(record, "")
        return {
            "id": record["id"],
            "question": record["question"],
            "http_status": response.status_code,
            "duration_seconds": duration,
            "method": None,
            "result_count": None,
            "answer": "",
            "cypher": None,
            "api_error": (
                f"Invalid JSON response: {type(error).__name__} "
                f"(HTTP {response.status_code})"
            ),
            "expected_answer_summary": record["expected_answer_summary"],
            "expected_claims": record["expected_claims"],
            "sources": record["sources"],
            **checks,
        }

    answer = payload.get("answer", "") if response.ok else ""
    checks = evaluate_answer(record, answer)
    return {
        "id": record["id"],
        "question": record["question"],
        "http_status": response.status_code,
        "duration_seconds": duration,
        "method": payload.get("method"),
        "result_count": payload.get("result_count"),
        "answer": answer,
        "cypher": payload.get("cypher"),
        "api_error": payload.get("detail") if not response.ok else None,
        "expected_answer_summary": record["expected_answer_summary"],
        "expected_claims": record["expected_claims"],
        "sources": record["sources"],
        **checks,
    }


def write_reports(
    output_dir: Path,
    benchmark_path: Path,
    api_url: str,
    results: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    durations = [result["duration_seconds"] for result in results]
    deterministic_passes = sum(
        result["http_status"] == 200 and result["passed_deterministic_checks"]
        for result in results
    )
    summary = {
        "cases": len(results),
        "http_200": sum(result["http_status"] == 200 for result in results),
        "deterministic_passes": deterministic_passes,
        "average_term_coverage": round(
            statistics.mean(result["term_coverage"] for result in results), 4
        ),
        "average_latency_seconds": round(statistics.mean(durations), 3),
        "manual_fact_review_required": len(results),
    }
    report = {
        "generated_at": generated_at,
        "benchmark": str(benchmark_path),
        "api_url": api_url,
        "authentication": "X-API-Key: [REDACTED]",
        "summary": summary,
        "results": results,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"gold-benchmark-{timestamp}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# StartupLens Source-Backed Anchor Benchmark",
        "",
        f"Generated: {generated_at}",
        "",
        "This five-company anchor suite is a targeted smoke benchmark, not a "
        "representative score for the full YC ecosystem. Deterministic checks verify "
        "HTTP success and required term and relationship-pattern coverage; they do "
        "not replace human comparison against the cited sources.",
        "",
        "## Summary",
        "",
        f"- Cases: **{summary['cases']}**",
        f"- HTTP 200: **{summary['http_200']}/{summary['cases']}**",
        (
            "- Deterministic passes: "
            f"**{summary['deterministic_passes']}/{summary['cases']}**"
        ),
        (
            "- Average required-term coverage: "
            f"**{summary['average_term_coverage']:.1%}**"
        ),
        f"- Average latency: **{summary['average_latency_seconds']:.2f}s**",
        f"- Manual fact reviews pending: **{summary['manual_fact_review_required']}**",
        "",
        "## Results",
        "",
        "| ID | Question | HTTP | Coverage | Missing terms | Missing patterns |",
        "|---|---|---:|---:|---|---|",
    ]
    for result in results:
        missing = ", ".join(result["missing_terms"]) or "-"
        missing_patterns = ", ".join(
            f"`{pattern.replace('|', '&#124;')}`"
            for pattern in result["missing_patterns"]
        ) or "-"
        question = result["question"].replace("|", "\\|")
        lines.append(
            f"| {result['id']} | {question} | {result['http_status']} | "
            f"{result['term_coverage']:.0%} | {missing} | {missing_patterns} |"
        )
    markdown_path = output_dir / f"gold-benchmark-{timestamp}.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    args = parse_args()
    records = load_benchmark(args.benchmark)
    errors = validate_benchmark(records)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    if args.validate_only:
        print(f"Validated {len(records)} source-backed benchmark records.")
        return 0
    if not args.api_url or not args.api_key:
        print("ERROR: Set STARTUPLENS_API_URL and STARTUPLENS_API_KEY.")
        return 2

    results = [
        run_case(args.api_url, args.api_key, record, args.timeout)
        for record in records
    ]
    json_path, markdown_path = write_reports(
        args.output_dir,
        args.benchmark,
        args.api_url,
        results,
    )
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))
    return int(
        not all(
            result["http_status"] == 200 and result["passed_deterministic_checks"]
            for result in results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
