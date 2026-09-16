"""Estimate what it costs to extract the whole corpus.

The prototype ran on five documents averaging 21,000 characters. The corpus
does not look like that: the median document is under a thousand characters
and the five prototype companies sit in its top one percent. That difference
decides the answer, because the per-request prompt overhead is charged once
per group per pass whether the document is long or short - so on a typical
document the instructions cost more than the text they are about.

Token counts are measured by building the real prompts, not guessed from
character counts.
"""

import argparse
import json
import math
from pathlib import Path

import extraction_schema

# Azure OpenAI standard pricing, US dollars per million tokens. These are the
# one input the estimate cannot measure, so they are stated plainly rather than
# buried, and can be corrected without touching the arithmetic.
PRICING = {
    "gpt-4.1":      {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o":       {"input": 2.50, "output": 10.00},
}


def _encoder():
    import tiktoken
    try:
        return tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def measure_overhead(encode) -> dict:
    """Tokens each group's prompt costs before the document is added."""
    overhead = {}
    for group in extraction_schema.PREDICATE_GROUPS:
        empty = extraction_schema.build_targeted_prompt("Company", "", group)
        overhead[group] = len(encode(empty))
    return overhead


def fact_density(extraction: dict, encode) -> tuple:
    """Facts per thousand document characters, and tokens per fact.

    Taken from the prototype run rather than assumed, so the estimate inherits
    what the extractor actually emitted, including the facts later withheld -
    they were generated, so they were paid for.
    """
    facts = chars = 0
    sample = []
    for company in extraction["companies"]:
        emitted = company.get("accepted", []) + company.get("rejected", [])
        facts += len(emitted) * company.get("passes", 1)
        chars += company["document_chars"]
        sample.extend(emitted[:20])
    per_fact = [len(encode(json.dumps(f, ensure_ascii=False))) for f in sample]
    return facts / max(chars, 1), sum(per_fact) / max(len(per_fact), 1)


def estimate(sizes: list, overhead: dict, facts_per_char: float,
             tokens_per_fact: float, passes: int, price: dict,
             verify_batch: int = 25) -> dict:
    groups = len(overhead)
    group_overhead = sum(overhead.values())

    input_tokens = output_tokens = 0
    verify_input = verify_output = 0

    for chars in sizes:
        doc_tokens = chars / 4  # characters per token, close enough for text
        # Every group re-sends the whole document, every pass.
        input_tokens += passes * (groups * doc_tokens + group_overhead)

        facts = max(facts_per_char * chars, 1)
        output_tokens += passes * facts * tokens_per_fact

        # Verification sees each surviving fact once, batched.
        surviving = max(facts, 1)
        batches = math.ceil(surviving / verify_batch)
        verify_input += batches * 400 + surviving * tokens_per_fact
        verify_output += surviving * 25  # a verdict is short

    total_input = input_tokens + verify_input
    total_output = output_tokens + verify_output
    cost_in = total_input / 1e6 * price["input"]
    cost_out = total_output / 1e6 * price["output"]

    return {
        "documents": len(sizes),
        "passes": passes,
        "requests": len(sizes) * (passes * groups
                                  + math.ceil(1)),
        "extraction_input_tokens": round(input_tokens),
        "extraction_output_tokens": round(output_tokens),
        "verification_input_tokens": round(verify_input),
        "verification_output_tokens": round(verify_output),
        "total_input_tokens": round(total_input),
        "total_output_tokens": round(total_output),
        "input_cost_usd": round(cost_in, 2),
        "output_cost_usd": round(cost_out, 2),
        "total_cost_usd": round(cost_in + cost_out, 2),
        "prompt_overhead_share": round(
            passes * group_overhead * len(sizes) / max(input_tokens, 1), 3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--extraction", required=True, type=Path,
                        help="A prototype run, used to measure fact density.")
    parser.add_argument("--model", default="gpt-4.1", choices=sorted(PRICING))
    parser.add_argument("--passes", nargs="*", type=int, default=[3, 2, 1])
    args = parser.parse_args()

    encode = _encoder().encode
    sizes = [p.stat().st_size for p in args.input_dir.glob("*.txt")]
    extraction = json.loads(args.extraction.read_text(encoding="utf-8"))

    overhead = measure_overhead(encode)
    density, per_fact = fact_density(extraction, encode)
    price = PRICING[args.model]

    print(f"corpus: {len(sizes)} documents, {sum(sizes):,} characters")
    print(f"prompt overhead per pass: {sum(overhead.values()):,} tokens "
          f"across {len(overhead)} groups")
    print(f"measured density: {density * 1000:.1f} facts per 1k chars, "
          f"{per_fact:.0f} tokens per fact")
    print(f"model: {args.model} "
          f"(${price['input']}/M in, ${price['output']}/M out)")
    print()

    for passes in args.passes:
        result = estimate(sizes, overhead, density, per_fact, passes, price)
        print(f"  {passes} pass(es): ${result['total_cost_usd']:>8,.2f}   "
              f"in {result['total_input_tokens'] / 1e6:>6.1f}M  "
              f"out {result['total_output_tokens'] / 1e6:>5.1f}M  "
              f"overhead {result['prompt_overhead_share'] * 100:.0f}% of input")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
