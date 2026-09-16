"""Compare sweep extraction against targeted per-group extraction.

The sweep asks one call to find every kind of fact in a whole document. The
targeted run asks a narrow question per group of related predicates. This
measures the difference on one document, so the approach can be judged before
anything is rebuilt around it.

Usage:
    python compare_extraction.py --input-dir <dir> --company coinbase --passes 3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import extraction_schema
import prototype_extract


def _run_sweep(company: str, window: str, passes: int) -> list:
    runs = []
    for attempt in range(passes):
        try:
            payload = prototype_extract.extract(company, window)
            accepted, _ = extraction_schema.validate_extraction(payload, window)
            runs.append(accepted)
            print(f"    sweep pass {attempt + 1}: {len(accepted)}")
        except Exception as exc:
            print(f"    sweep pass {attempt + 1} failed: {str(exc)[:80]}")
    return runs


def _run_targeted(company: str, window: str, passes: int) -> list:
    runs = []
    for attempt in range(passes):
        combined = []
        for group in extraction_schema.PREDICATE_GROUPS:
            prompt = extraction_schema.build_targeted_prompt(company, window, group)
            try:
                reply = prototype_extract._chat(prompt)
                accepted, _ = extraction_schema.validate_extraction(reply, window)
                combined.extend(accepted)
            except Exception as exc:
                print(f"      {group} failed: {str(exc)[:70]}")
        runs.append(combined)
        print(f"    targeted pass {attempt + 1}: {len(combined)}")
    return runs


def _stability(runs: list) -> dict:
    """How much the runs agree with each other."""
    counts = Counter()
    for run in runs:
        for identity in {extraction_schema.assertion_identity(a) for a in run}:
            counts[identity] += 1
    total = len(counts)
    unanimous = sum(1 for n in counts.values() if n == len(runs))
    once = sum(1 for n in counts.values() if n == 1)
    return {
        "distinct_facts": total,
        "unanimous": unanimous,
        "seen_once": once,
        "stability": round(unanimous / total, 3) if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--company", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    document = (args.input_dir / f"{args.company}.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    window = document[:prototype_extract.MAX_DOCUMENT_CHARS]

    print("  sweep:")
    sweep_runs = _run_sweep(args.company, window, args.passes)
    print("  targeted:")
    targeted_runs = _run_targeted(args.company, window, args.passes)

    result = {
        "company": args.company,
        "passes": args.passes,
        "sweep": _stability(sweep_runs),
        "targeted": _stability(targeted_runs),
    }

    for label, runs in (("sweep", sweep_runs), ("targeted", targeted_runs)):
        agreed, _ = extraction_schema.merge_runs(runs, 2)
        result[label]["agreed"] = len(agreed)
        result[label]["places"] = Counter(
            a.get("role") for a in agreed if a["predicate"] == "located_in"
        )
        result[label]["predicates_found"] = sorted(
            {a["predicate"] for a in agreed}
        )

    print(json.dumps(result, indent=2, default=str))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"summary": result,
                 "sweep_runs": sweep_runs,
                 "targeted_runs": targeted_runs},
                indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
