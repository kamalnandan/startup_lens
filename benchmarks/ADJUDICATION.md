# Anchor benchmark adjudication

Automated required-term coverage is a triage signal, not a factual-correctness
score. Review each response against `expected_claims` and the mapped sources before
using benchmark results to prioritize enrichment or query changes.

For every case, record:

- `verdict`: `correct_complete`, `correct_partial`, `incorrect`, `no_data`, or
  `request_error`
- `factual_correctness_score`: integer 1–5
- `completeness_score`: integer 1–5
- zero-based supported, missing, and contradicted claim indices
- `diagnosis`: `pass`, `data_gap`, `stale_or_incorrect_data`,
  `query_generation`, `answer_synthesis`, `benchmark_issue`, or `request_error`
- a concise rationale

## Scoring rubric

| Score | Factual correctness | Completeness |
|---:|---|---|
| 5 | All stated material facts align with the sourced truth set | Fully answers every requested part |
| 4 | Correct with only a harmless imprecision | One minor requested detail is absent |
| 3 | Mixed or uncertain facts | Useful but materially incomplete |
| 2 | At least one major factual error | Most requested facts are absent |
| 1 | Fundamentally wrong or contradictory | Unusable as an answer |

An abstention can have high factual correctness because it states no false fact, but
it must receive low completeness. Report verdict distributions and completeness
alongside factual scores so abstentions do not inflate the headline result.

## Method

1. Compare the answer semantically, accepting harmless paraphrases.
2. Do not require supporting context that the question did not request.
3. Use generated Cypher and result count to distinguish missing data from query
   generation.
4. Treat unsupported false additions as contradictions.
5. Record the reviewer or model identifier, version, date, and the complete
   per-question adjudication in the generated run artifact.

Version 1 covers five prominent anchor companies and contains repeated formulations
of several facts. It measures regression robustness for those anchors; it does not
represent the long tail of the YC ecosystem.
