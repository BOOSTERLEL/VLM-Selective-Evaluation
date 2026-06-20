# Protocol-Driven Evaluation Spec (Pilot v1)

## Scope

This project evaluates the hallucination vs abstention trade-off under controlled answerability interventions.
It is not a study of "structured vs unstructured output quality". Structured output is used as a stable scoring
interface.

## Answerability Definition

- `ANSWERABLE`: the required evidence is present in the image, so the question can be answered from visual evidence.
- `UNANSWERABLE`: key evidence is removed by intervention, so the question should be abstained from.

Ground-truth answerability is determined by paired sample construction:

- clean sample => `ANSWERABLE`
- intervened sample => `UNANSWERABLE`

## Intervention Types

Minimum required intervention in pilot v1:

- evidence removal (`mask`, `crop`, or equivalent deterministic region removal)

The question must stay unchanged across the pair; only evidence availability changes.

## Output Interface

Model outputs are parsed into:

- `answer`: string
- `status`: `ANSWERABLE | UNANSWERABLE`
- `confidence`: float in `[0, 1]`

If `status` cannot be reliably parsed, scoring uses a conservative fallback:

- `status_for_scoring = ANSWERABLE`

This avoids under-counting hallucination by treating parse ambiguity as abstention.

## Quadrant Rules

For each sample, assign exactly one quadrant:

1. `(A,C)` Answerable & Correct
2. `(A,W)` Answerable & Wrong
3. `(U,Abstain)` Unanswerable & Abstain
4. `(U,Assert)` Unanswerable & Assert

Additional conventions:

- On answerable samples, `status=UNANSWERABLE` counts as unnecessary abstention and is included in `(A,W)`.
- On unanswerable samples, `status=ANSWERABLE` counts as hallucination-style assertion `(U,Assert)`.

## Metrics

Primary:

- Quadrant proportions
- `UAR = P(pred_status=UNANSWERABLE | gt=ANSWERABLE)`
- `HAR = P(pred_status=ANSWERABLE | gt=UNANSWERABLE)`
- Answerable risk-coverage curve (thresholded by confidence)
- Unanswerable assert-abstain curve (thresholded by confidence)

Secondary:

- parse success rate
- format validity rate
- status missing/unreliable rate
- confidence missing/invalid rate
- calibration (ECE, Brier)

## Required Artifacts

- `quadrant_rates.csv`
- `risk_coverage.csv`
- `assert_abstain.csv`
- `metrics.json`
- `metrics.csv`
- `scored.jsonl`
