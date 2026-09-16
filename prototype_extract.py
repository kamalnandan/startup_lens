"""Prototype extractor for the assertion-based schema.

Runs extraction over a handful of source documents and writes the result to
JSON for inspection. It deliberately does not write to Neo4j: the point is to
test whether the schema recovers facts the current graph loses, before anything
touches the production graph.

Usage:
    python prototype_extract.py --input-dir <dir> --companies coinbase airbnb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from app_config import get_required_setting
import extraction_schema


AZURE_OPENAI_ENDPOINT = get_required_setting("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = get_required_setting("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = get_required_setting("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = get_required_setting("AZURE_OPENAI_API_KEY")

DEFAULT_COMPANIES = ("coinbase", "airbnb", "dropbox", "doordash", "stripe")
MAX_DOCUMENT_CHARS = 60000


def _chat(prompt: str, timeout: int = 180, temperature: float = 0.0,
          max_tokens: int = 16000, attempts: int = 4) -> dict:
    """Post a prompt and return the parsed JSON reply.

    Retried with backoff because at corpus scale transient failures stop being
    unlucky and start being routine: one dropped connection already cost a
    whole predicate group during a five-document run. A 429 carries a
    Retry-After that is worth obeying; anything else backs off exponentially.
    """
    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        f"/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    )
    last = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json",
                         "api-key": AZURE_OPENAI_API_KEY},
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                },
                timeout=timeout,
            )
            if response.status_code == 429 or response.status_code >= 500:
                wait = float(response.headers.get("Retry-After", 0)) or 2 ** attempt
                last = RuntimeError(f"HTTP {response.status_code}")
                time.sleep(min(wait, 60))
                continue
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return json.loads(payload["choices"][0]["message"]["content"])
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"giving up after {attempts} attempts: {last}")


def extract(company: str, document: str, timeout: int = 180) -> dict:
    """Call the model and return the parsed extraction payload."""
    prompt = extraction_schema.build_extraction_prompt(
        company, document[:MAX_DOCUMENT_CHARS]
    )
    return _chat(prompt, timeout=timeout)


def extract_groups(company: str, window: str) -> tuple:
    """One narrow request per predicate group, combined into one pass.

    Asking for 23 predicates across 29k characters in a single request is a
    broad task, and broad tasks are where the model gets vague: run-to-run
    stability was 26%, whole predicates went missing, and a document with one
    office per city came back with twelve headquarters. Six narrow requests
    against the same document reach 83%, with nothing seen only once.
    """
    accepted, rejected = [], []
    for group in extraction_schema.PREDICATE_GROUPS:
        prompt = extraction_schema.build_targeted_prompt(company, window, group)
        try:
            reply = _chat(prompt)
            ok, bad = extraction_schema.validate_extraction(reply, window)
        except Exception as exc:
            # One group failing costs that group, not the whole company.
            print(f"      {group} failed: {str(exc)[:70]}", file=sys.stderr)
            continue
        accepted.extend(ok)
        rejected.extend(bad)
    return accepted, rejected


def verify(company: str, assertions: list, batch_size: int = 25) -> list:
    """Ask whether each assertion's span really asserts it.

    Verification is batched because one narrow question per fact is reliable
    but a hundred of them in one reply is not: long replies get truncated and
    the tail of the list silently loses its verdicts.
    """
    for start in range(0, len(assertions), batch_size):
        batch = assertions[start:start + batch_size]
        prompt = extraction_schema.build_verification_prompt(company, batch)
        try:
            reply = _chat(prompt)
            verdicts = reply.get("verdicts", [])
        except Exception as exc:
            print(f"    ! verification batch failed: {exc}", file=sys.stderr)
            verdicts = []
        extraction_schema.apply_verdicts(batch, verdicts)
    return assertions


def run(input_dir: Path, companies, output: Path, passes: int = 3,
        min_agreement: int = 2, skip_verify: bool = False,
        sweep: bool = False) -> dict:
    report = {"companies": [], "totals": {"accepted": 0, "rejected": 0,
                                          "contested": 0, "unsupported": 0}}

    for company in companies:
        path = input_dir / f"{company}.txt"
        if not path.exists():
            print(f"  ! missing {path.name}", file=sys.stderr)
            continue

        document = path.read_text(encoding="utf-8", errors="replace")
        window = document[:MAX_DOCUMENT_CHARS]
        started = time.time()
        rejected = []
        try:
            runs = []
            for attempt in range(passes):
                # A pass can fail on its own - a truncated reply, a timeout -
                # and losing the whole document to one bad pass would waste the
                # other passes and leave the company absent from the graph.
                try:
                    if sweep:
                        payload = extract(company, document)
                        accepted, run_rejected = extraction_schema.validate_extraction(
                            payload, window
                        )
                    else:
                        accepted, run_rejected = extract_groups(company, window)
                except Exception as exc:
                    print(f"    pass {attempt + 1} failed: {str(exc)[:90]}",
                          file=sys.stderr)
                    continue
                runs.append(accepted)
                rejected.extend(run_rejected)
                print(f"    pass {attempt + 1}: {len(accepted)} facts")

            if not runs:
                raise RuntimeError("every extraction pass failed")

            # Agreement is meaningless with one surviving pass, so fall back to
            # requiring a single sighting rather than discarding everything.
            required = min(min_agreement, len(runs))
            agreed, contested = extraction_schema.merge_runs(runs, required)
            if not skip_verify:
                verify(company, agreed)
            accepted = agreed + contested
            error = None
        except Exception as exc:
            accepted, contested, error = [], [], str(exc)

        unsupported = sum(1 for a in accepted if a.get("asserted") is False
                          and a.get("agreement", 0) >= min_agreement)
        entry = {
            "company": company,
            "document_chars": len(document),
            "duration_seconds": round(time.time() - started, 1),
            "error": error,
            "passes": passes,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "contested_count": len(contested),
            "unsupported_count": unsupported,
            "accepted": accepted,
            "rejected": rejected,
        }
        report["companies"].append(entry)
        report["totals"]["accepted"] += len(accepted)
        report["totals"]["rejected"] += len(rejected)
        report["totals"]["contested"] += len(contested)
        report["totals"]["unsupported"] += unsupported

        status = error or (
            f"{len(accepted)} facts "
            f"({len(contested)} contested, {unsupported} unsupported)"
        )
        print(f"  {company}: {status}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--companies", nargs="*", default=list(DEFAULT_COMPANIES))
    parser.add_argument("--output", type=Path, default=Path("prototype_extraction.json"))
    parser.add_argument("--passes", type=int, default=3,
                        help="Independent extraction runs per document.")
    parser.add_argument("--min-agreement", type=int, default=2,
                        help="Runs a fact must appear in to be asserted.")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Ask for every predicate in one request "
                             "(the older, less stable approach).")
    args = parser.parse_args()

    report = run(args.input_dir, args.companies, args.output,
                 passes=args.passes, min_agreement=args.min_agreement,
                 skip_verify=args.skip_verify, sweep=args.sweep)
    print(json.dumps({"output": str(args.output), **report["totals"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
