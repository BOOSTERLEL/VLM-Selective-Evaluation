# Issue Checklist Mapping (Implemented)

This file maps `Issue.md` checklist items to concrete implementation locations.

## 0) Repo Structure (Status: Done)

Implemented under `vlm_selective_eval/` package namespace plus `scripts/` and `configs/`:

- protocol docs -> `vlm_selective_eval/protocol/*`
- data generation entry -> `vlm_selective_eval/data_gen/generate_pairs.py`
- inference entry -> `vlm_selective_eval/inference/run_inference.py`
- scoring entries -> `vlm_selective_eval/scoring/*`
- analysis entries -> `vlm_selective_eval/analysis/*_cli.py`
- configs -> `configs/run.yaml`
- scripts -> `scripts/run_all.sh`

## 1) Protocol Spec (Status: Done)

- Protocol definition: `vlm_selective_eval/protocol/protocol.md`
- Output schema: `vlm_selective_eval/protocol/schema.json`

## 2) Paired Data Generation (Status: Done)

- Core implementation: `vlm_selective_eval/datasets/*`, `vlm_selective_eval/pipeline.py`
- Wrapper entrypoint: `vlm_selective_eval/data_gen/generate_pairs.py`

## 3) Model Wrappers + Inference (Status: Done)

- Wrapper interfaces: `vlm_selective_eval/models/base.py`, `vlm_selective_eval/models/*`
- Inference pipeline: `vlm_selective_eval/pipeline.py`
- Wrapper entrypoint: `vlm_selective_eval/inference/run_inference.py`

## 4) Parsing + Parse Quality (Status: Done)

- Parsing: `vlm_selective_eval/parsing/structured.py`, `vlm_selective_eval/parsing/free_text.py`
- Parse quality metrics and conservative status policy:
  - `vlm_selective_eval/evaluation/metrics.py`
  - `parse_fail_status_policy = default_to_ANSWERABLE`

## 5) Scoring + Quadrants (Status: Done)

- Quadrant mapping: `vlm_selective_eval/evaluation/quadrants.py`
- Protocol scoring + per-sample scored rows:
  - `vlm_selective_eval/evaluation/metrics.py`
  - `vlm_selective_eval/pipeline.py` (`scored.jsonl`)

## 6) Metrics + Curves (Status: Done)

- Answerable risk-coverage curve:
  - `answerable_risk_coverage_curve` in `vlm_selective_eval/evaluation/metrics.py`
- Unanswerable assert-abstain curve:
  - `unanswerable_assert_abstain_curve` in `vlm_selective_eval/evaluation/metrics.py`
- Plotting:
  - `vlm_selective_eval/analysis/plots.py` (`assert_abstain_curve.png`)

## 7) Minimal Validity Check (Status: Done)

- Structured vs free-text minimal comparisons are supported by prompt modes and matrix runners:
  - `vlm_selective_eval/prompting.py`
  - `scripts/run_experiment_matrix.py`
  - `scripts/run_protocol_robustness_matrix.py`

## 8) CLI + Config (Status: Done)

- Unified config template: `configs/run.yaml`
- Single-command orchestration: `scripts/run_all.sh`

## 9) Reproducibility + Artifacts (Status: Done)

- Saved artifacts:
  - `quadrant_rates.csv`
  - `risk_coverage.csv`
  - `assert_abstain.csv`
  - `scored.jsonl`
- Matrix summary reproducibility metadata:
  - `scripts/run_experiment_matrix.py`
  - `scripts/run_protocol_robustness_matrix.py`
  - includes seed, git commit, python/platform, selected package versions
