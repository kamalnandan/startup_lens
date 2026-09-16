"""Extracting many documents without starting over when something breaks.

The prototype ran five documents serially and that was fine. Three hundred is
a different problem: it is long enough that a rate limit, a dropped
connection or a closed laptop will happen somewhere in the middle, and long
enough that losing the work already done actually hurts.

So each company is written to its own file the moment it finishes, and a
company with a file is skipped. Interrupting the run and restarting it costs
the company that was in flight and nothing else. The concurrency is plain
threads, because the work is entirely waiting on HTTP.
"""

import argparse
import json
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import extraction_schema
import prototype_extract

MAX_DOCUMENT_CHARS = prototype_extract.MAX_DOCUMENT_CHARS

_print_lock = threading.Lock()


def say(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def slug_for(company: dict) -> str:
    document = company.get("document") or f'{company["name"]}.txt'
    return pathlib.Path(document).stem


def extract_one(company: dict, input_dir: pathlib.Path,
                output_dir: pathlib.Path, passes: int,
                min_agreement: int, skip_verify: bool) -> dict:
    """Extract one company and write it to its own file."""
    slug = slug_for(company)
    target = output_dir / f"{slug}.json"
    if target.exists():
        return {"company": slug, "status": "skipped"}

    source = input_dir / f"{slug}.txt"
    if not source.exists():
        return {"company": slug, "status": "missing"}

    started = time.time()
    document = source.read_text(encoding="utf-8", errors="replace")
    window = document[:MAX_DOCUMENT_CHARS]

    try:
        runs, rejected = [], []
        for _ in range(passes):
            try:
                accepted, run_rejected = prototype_extract.extract_groups(
                    slug, window)
            except Exception as exc:              # noqa: BLE001
                say(f"    {slug}: pass failed: {str(exc)[:70]}")
                continue
            runs.append(accepted)
            rejected.extend(run_rejected)

        if not runs:
            raise RuntimeError("every extraction pass failed")

        required = min(min_agreement, len(runs))
        agreed, contested = extraction_schema.merge_runs(runs, required)
        if not skip_verify:
            prototype_extract.verify(slug, agreed)
        accepted = agreed + contested
        error = None
    except Exception as exc:                      # noqa: BLE001
        accepted, contested, rejected, error = [], [], [], str(exc)

    entry = {
        "company": slug,
        "display_name": company.get("name"),
        "document_chars": len(document),
        "duration_seconds": round(time.time() - started, 1),
        "error": error,
        "passes": passes,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "contested_count": len(contested),
        "accepted": accepted,
        "rejected": rejected,
    }

    if error:
        # A failure is not written, so a rerun retries it rather than
        # inheriting an empty company that looks finished.
        return {"company": slug, "status": "failed", "error": error}

    target.write_text(json.dumps(entry, indent=1, ensure_ascii=False),
                      encoding="utf-8")
    return {"company": slug, "status": "done", "facts": len(accepted),
            "seconds": entry["duration_seconds"]}


def combine(output_dir: pathlib.Path, combined: pathlib.Path) -> dict:
    """Gather the per-company files into one the loader can read."""
    report = {"companies": [], "totals": {"accepted": 0, "rejected": 0,
                                          "contested": 0, "unsupported": 0}}
    for path in sorted(output_dir.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        report["companies"].append(entry)
        report["totals"]["accepted"] += entry.get("accepted_count", 0)
        report["totals"]["rejected"] += entry.get("rejected_count", 0)
        report["totals"]["contested"] += entry.get("contested_count", 0)
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", required=True, type=pathlib.Path,
                        help="JSON list from select_demo_companies.py")
    parser.add_argument("--input-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--combined", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--min-agreement", type=int, default=1)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.combine_only:
        companies = json.loads(args.companies.read_text(encoding="utf-8"))
        if args.limit:
            companies = companies[:args.limit]

        pending = [c for c in companies
                   if not (args.output_dir / f"{slug_for(c)}.json").exists()]
        say(f"{len(companies)} companies, {len(pending)} still to do, "
            f"{args.workers} workers, {args.passes} pass(es)")

        started = time.time()
        counts = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(extract_one, company, args.input_dir,
                            args.output_dir, args.passes, args.min_agreement,
                            args.skip_verify): company
                for company in pending
            }
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                if result["status"] == "done":
                    detail = f'{result["facts"]} facts in {result["seconds"]}s'
                else:
                    detail = result.get("error", result["status"])[:70]
                rate = done / max(time.time() - started, 1) * 60
                say(f'[{done}/{len(pending)}] {result["company"]}: {detail}'
                    f'   ({rate:.1f}/min)')

        say(json.dumps(counts, indent=1))

    if args.combined:
        report = combine(args.output_dir, args.combined)
        say(f'Combined {len(report["companies"])} companies, '
            f'{report["totals"]["accepted"]} facts -> {args.combined}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
