# VLM Selective Eval (Pilot v1)

Inference-only Python 3.11 research codebase for stage-1 structured selective evaluation of vision-language models (VLMs).

## Protocol Guardrails (Read Before Running)

This project is a **protocol study** on hallucination-vs-abstention trade-offs, not a formatting contest.

Do:
- center analysis on 2x2 quadrants (`A&C`, `A&W`, `U&Abstain`, `U&Assert`)
- treat structured output as an evaluation interface for stable auto-scoring
- report both selective curves:
  - answerable `risk_coverage`
  - unanswerable `assert_abstain`

Do not:
- frame results as "structured beats free-form generation quality"
- tune prompts only to boost parse rate while ignoring hallucination (`HAR`)
- treat parse failure as safe abstention in scoring

Conservative scoring policy:
- if `status` is missing/unreliable, scoring defaults to `ANSWERABLE` (`default_to_ANSWERABLE`)
- this prevents artificially low hallucination rates caused by parse failures

## Scope (v1)

Implemented:
- dataset loading and paired sample construction
- prompting (structured + free-text, including tagged free-text robust mode)
- model inference via adapter interface
- structured output parsing (strict JSON + regex fallback)
- evaluation with quadrant mapping + selective metrics
- analysis tables and plots

Not implemented:
- training loop
- optimizer/scheduler
- checkpointed training
- finetuning pipeline

## Repository Architecture

At a high level, the repository follows this execution path:

```text
configs / scripts / CLI
    -> config loading
    -> dataset pair building
    -> model inference
    -> output parsing
    -> evaluation + plots
```

Top-level folders and files:

- `configs/`: YAML experiment templates, including single-run, frozen stage-1 matrix, and robustness matrix configs.
- `scripts/`: batch runners and dataset-prep helpers; use these when you want multi-run orchestration instead of a single CLI call.
- `vlm_selective_eval/`: core package; most actual logic lives here.
- `tests/`: unit and lightweight end-to-end tests for parsing, prompting, datasets, metrics, and pipeline behavior.
- `outputs/`: runtime-generated artifacts; this is the main destination for new experiment outputs.
- `Stage1/`: legacy stage-1 summary snapshots currently kept in the repo workspace.
- `test/`: legacy CSV result snapshots used as quick inspection artifacts in the workspace.
- `Issue.md`: original implementation checklist / task statement that the protocol mapping refers back to.

If you are new to the codebase, the best reading order is:

1. `README.md`
2. `vlm_selective_eval/cli.py`
3. `vlm_selective_eval/pipeline.py`
4. the package submodules below

Core package layout:

- `vlm_selective_eval/config.py`: loads YAML/JSON into dataclass configs shared across the whole pipeline.
- `vlm_selective_eval/pipeline.py`: the main orchestration layer; most CLI and script entrypoints eventually call here.
- `vlm_selective_eval/datasets/`: builds paired answerable / unanswerable samples for `synthetic`, `textvqa`, and `gqa`.
- `vlm_selective_eval/models/`: model adapter layer, including the mock adapter and the Hugging Face VLM adapter.
- `vlm_selective_eval/prompting.py`: prompt construction for structured and free-text variants.
- `vlm_selective_eval/parsing/`: converts raw model text into the evaluation interface (`answer`, `status`, `confidence`).
- `vlm_selective_eval/evaluation/`: quadrant mapping, selective metrics, calibration metrics, and dataset-specific official scoring modules.
  Current official files: `textvqa_official.py`, `stvqa_official.py`, `docvqa_official.py`, `infographicvqa_official.py`.
- `vlm_selective_eval/analysis/`: plots and metrics-export helpers.
- `vlm_selective_eval/data_gen/`, `vlm_selective_eval/inference/`, `vlm_selective_eval/scoring/`: thin module entrypoints around the pipeline for build / infer / score steps.
- `vlm_selective_eval/protocol/`: protocol spec, schema, and checklist-to-code mapping docs.
- `vlm_selective_eval/schemas.py`, `constants.py`, `io_utils.py`: shared types, constants, and JSONL / CSV helpers.

Which entrypoint to use:

- Single run: `configs/run.yaml` + `scripts/run_all.sh` or the `vlm_selective_eval.cli` subcommands.
- Frozen stage-1 experiment matrix: `configs/experiment_matrix.yaml` + `scripts/run_experiment_matrix.py`.
- Prompt/temperature robustness expansion: `configs/protocol_robustness_matrix.yaml` + `scripts/run_protocol_robustness_matrix.py`.

Typical call graph:

```text
cli.py / matrix scripts
    -> load_config(...)
    -> build_pairs_from_config(...)
    -> run_inference_from_config(...)
    -> evaluate_predictions_file(...) or dataset-specific official evaluation helpers
    -> save_all_plots(...)
```

## Structured Output Protocol

Required keys:
- `answer` (string)
- `status` (`ANSWERABLE` or `UNANSWERABLE`)
- `confidence` (float in `[0, 1]`)

Parser behavior:
- strict JSON parsing first
- regex fallback for malformed structured outputs
- explicit `parse_ok` and `format_ok`

Formal spec files:
- `vlm_selective_eval/protocol/protocol.md`
- `vlm_selective_eval/protocol/schema.json`
- `vlm_selective_eval/protocol/issue_mapping.md` (Issue checklist -> implementation mapping)

## Datasets

Supported dataset modes:
- `synthetic`: fully runnable synthetic sanity dataset (no external data required)
- `textvqa_mock`: TextVQA-style OCR adapter with evidence-removal pair construction (mock subset by default, source-file skeleton supported)
- `stvqa`: ST-VQA-style OCR adapter with the same evidence-removal pipeline as TextVQA
- `docvqa`: DocVQA-style document OCR adapter with validation-split source support
- `infographicvqa`: InfographicVQA-style OCR adapter with validation-split source support
- `gqa_mock`: GQA existence/attribute adapter with region masking (mock subset by default, source-file skeleton supported)

Dataset-prep helpers:
- `scripts/build_textvqa_source.py`
- `scripts/build_stvqa_source.py`
- `scripts/build_docvqa_source.py`
- `scripts/build_infographicvqa_source.py`
- `prepare_textvqa.sh`
- `prepare_docvqa.sh`
- `prepare_infographicvqa.sh`

Notes:
- `prepare_textvqa.sh` downloads from public direct links.
- `prepare_docvqa.sh` and `prepare_infographicvqa.sh` normalize files you downloaded from the official challenge pages; because those datasets are login-gated, the scripts accept local files or signed URLs instead of hard-coded public links.

## Install

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

Optional Hugging Face inference adapter:

```bash
pip install -e .[hf]
```

## VRAM-Aware HF Inference

For Hugging Face models, set `model.vram_gb` to `16` or `24` in YAML configs:

```yaml
model:
  adapter: hf
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
  device: cuda
  vram_gb: 16
```

Behavior by tier:
- `24`: keep the current standard evaluation path
- `16`: switch to a lower-VRAM compatibility path (`device_map="auto"`, `use_cache=False`)
- `16` + `InternVL2.5-8B`: also reduce default image tiles from `12` to `4`

If `vram_gb` is omitted on CUDA, the adapter auto-detects GPU memory and maps it into the same `16` / `24` tiers.

## CLI Commands

### 1. Build pairs

```bash
python -m vlm_selective_eval.cli build-pairs --config configs/synthetic_structured.yaml
```

### 2. Run inference

```bash
python -m vlm_selective_eval.cli run-inference \
  --config configs/synthetic_structured.yaml \
  --pairs outputs/synthetic_structured_demo/pairs/synthetic_pairs.jsonl
```

### 3. Evaluate predictions (tables/metrics)

```bash
python -m vlm_selective_eval.cli evaluate \
  --predictions outputs/synthetic_structured_demo/predictions/synthetic_structured_mock-vlm_predictions.jsonl \
  --output-dir outputs/synthetic_structured_demo/evaluation
```

### 3.1 Re-score existing predictions with TextVQA official evaluation

If you already have a `predictions.jsonl` file produced earlier (even if you first used strict single-answer evaluation),
you can directly run official TextVQA multi-answer soft scoring **without rerunning inference**:

```bash
python -m vlm_selective_eval.cli evaluate-textvqa-official \
  --predictions outputs/<your_run>/predictions/<your_predictions>.jsonl \
  --output-dir outputs/<your_run>/evaluation_official
```

Outputs (official + matrix-compatible):
- `textvqa_official_metrics.json`
- `textvqa_official_metrics.csv`
- `metrics.json`
- `metrics.csv`
- `quadrant_rates.csv`
- `risk_coverage.csv`
- `assert_abstain.csv`
- `coverage_risk_curve.csv`
- `calibration_bins.csv`
- `scored.jsonl`

The standard `metrics.*` artifacts use the same indicator schema as `scripts/run_experiment_matrix.py`, so you can compare them directly.

Prerequisite: the prediction rows must include ground-truth answer lists required by TextVQA official scoring.

### 3.2 Re-score existing predictions with ST-VQA official evaluation

If you already have a `predictions.jsonl` file for `stvqa`, you can directly run official
ST-VQA `ANLS` scoring **without rerunning inference**:

```bash
python -m vlm_selective_eval.cli evaluate-stvqa-official \
  --predictions outputs/<your_run>/predictions/<your_predictions>.jsonl \
  --output-dir outputs/<your_run>/evaluation_official
```

Outputs (official + matrix-compatible):
- `stvqa_official_metrics.json`
- `stvqa_official_metrics.csv`
- `metrics.json`
- `metrics.csv`
- `quadrant_rates.csv`
- `risk_coverage.csv`
- `assert_abstain.csv`
- `coverage_risk_curve.csv`
- `calibration_bins.csv`
- `scored.jsonl`

The standard `metrics.*` artifacts stay aligned with the matrix summary schema, while
`stvqa_official_metrics.*` exposes dataset-specific ANLS summary numbers.

Prerequisite: the prediction rows must include ST-VQA answer lists in `metadata.answers`
or `metadata.official_answers`.

### 3.3 Re-score existing predictions with DocVQA official evaluation

Use official evaluation for DocVQA. The strict single-answer path is only a diagnostic fallback and should not be used as the main reported result.

```bash
python -m vlm_selective_eval.cli evaluate-docvqa-official \
  --predictions outputs/<your_run>/predictions/<your_predictions>.jsonl \
  --output-dir outputs/<your_run>/evaluation_official
```

Outputs:
- `docvqa_official_metrics.json`
- `docvqa_official_metrics.csv`
- `metrics.json`
- `metrics.csv`
- `quadrant_rates.csv`
- `risk_coverage.csv`
- `assert_abstain.csv`
- `coverage_risk_curve.csv`
- `calibration_bins.csv`
- `scored.jsonl`

### 3.4 Re-score existing predictions with InfographicVQA official evaluation

Use official evaluation for InfographicVQA. The strict single-answer path is only a diagnostic fallback and should not be used as the main reported result.

```bash
python -m vlm_selective_eval.cli evaluate-infographicvqa-official \
  --predictions outputs/<your_run>/predictions/<your_predictions>.jsonl \
  --output-dir outputs/<your_run>/evaluation_official
```

Outputs:
- `infographicvqa_official_metrics.json`
- `infographicvqa_official_metrics.csv`
- `metrics.json`
- `metrics.csv`
- `quadrant_rates.csv`
- `risk_coverage.csv`
- `assert_abstain.csv`
- `coverage_risk_curve.csv`
- `calibration_bins.csv`
- `scored.jsonl`

### 4. Generate plots

```bash
python -m vlm_selective_eval.cli plot \
  --metrics-json outputs/synthetic_structured_demo/evaluation/metrics.json \
  --output-dir outputs/synthetic_structured_demo/evaluation/plots
```

### Synthetic one-shot demo

```bash
python -m vlm_selective_eval.cli demo --config configs/synthetic_structured.yaml
```

## Required Outputs

Predictions JSONL includes:
- `sample_id`
- `pair_id`
- `task`
- `model_name`
- `prompt_mode`
- `raw_output`
- `parsed_answer`
- `parsed_status`
- `parsed_confidence`
- `parse_ok`
- `format_ok`

Metrics outputs:
- `metrics.json`
- `metrics.csv`
- `quadrant_rates.csv`
- `risk_coverage.csv` (answerable subset)
- `assert_abstain.csv` (unanswerable subset)
- `coverage_risk_curve.csv` (backward-compatible alias of `risk_coverage.csv`)
- `calibration_bins.csv`
- `scored.jsonl`

Plots:
- `quadrant_stacked_bar.png`
- `coverage_risk_curve.png`
- `assert_abstain_curve.png`
- `confidence_calibration.png`

## Evaluation Definitions

Quadrants:
1. Answerable & Correct
2. Answerable & Wrong
3. Unanswerable & Abstain
4. Unanswerable & Assert

Extra metrics:
- unnecessary abstention rate
- hallucinatory assertion rate
- answerable coverage
- answerable risk
- global coverage/risk (diagnostic only)
- parse success rate
- format validity rate
- status missing/unreliable rate
- confidence missing/invalid rate
- calibration stats (ECE, Brier score)

Parse failures are counted explicitly and never silently dropped.
Scoring applies a conservative policy: parse-failed status defaults to `ANSWERABLE`.

## Tests

```bash
pytest
```

Included unit tests:
- strict JSON parsing
- regex fallback parsing
- answer normalization
- quadrant mapping
- synthetic dataset generation
- minimal end-to-end pipeline

## Stage-2 Extensibility

The pairing interfaces are designed to extend to triplets:
- evidence-preserving
- evidence-removal
- evidence-contradiction

## Stage-1 Frozen Matrix Run

Detailed step-by-step guide:
- `guide.md`

Run plan only (do not execute model inference):

```bash
python scripts/run_experiment_matrix.py --matrix configs/experiment_matrix.yaml --dry-run
```

Run the full matrix (2 datasets x 4 models x 2 prompts = 16 runs):

```bash
python scripts/run_experiment_matrix.py --matrix configs/experiment_matrix.yaml
```

Outputs:
- `outputs/stage1_frozen_matrix/generated_configs/*.yaml` (per-run generated configs)
- `outputs/stage1_frozen_matrix/summary/run_plan.csv` (run checklist)
- `outputs/stage1_frozen_matrix/summary/run_metrics_summary.csv` (aggregated metrics)
- `outputs/stage1_frozen_matrix/runs/<run_id>/...` (pairs, predictions, evaluation, plots)

Notes:
- For TextVQA / DocVQA / InfographicVQA official-data experiments, set each dataset `source_path` to a prepared filtered JSON list.
- If `source_path` is `null`, adapters fall back to the built-in mock subset for pipeline sanity checks.
- Matrix runner validates `temperature > 0` for device compatibility.

## Protocol Robustness Mini Matrix

Run plan only:

```bash
python scripts/run_protocol_robustness_matrix.py --matrix configs/protocol_robustness_matrix.yaml --dry-run
```

Run full mini matrix:

```bash
python scripts/run_protocol_robustness_matrix.py --matrix configs/protocol_robustness_matrix.yaml --evaluation-mode official
```

Notes:
- This matrix expands `prompt_variants x temperatures` for protocol robustness checks.
- Temperature is validated as `> 0` (device-compatible; no `temperature=0` runs).
- `--evaluation-mode official` now dispatches to TextVQA soft accuracy or ANLS-based OCR-VQA scoring based on dataset mode.

## Single-Command Run

```bash
bash scripts/run_all.sh configs/run.yaml
```

Issue-aligned wrapper entrypoints are also available:
- `python -m vlm_selective_eval.data_gen.generate_pairs --config configs/run.yaml`
- `python -m vlm_selective_eval.inference.run_inference --config configs/run.yaml --pairs <pairs.jsonl>`
- `python -m vlm_selective_eval.scoring.score --predictions <predictions.jsonl> --output-dir <dir>`
- `python -m vlm_selective_eval.analysis.metrics_cli --predictions <predictions.jsonl> --output-dir <dir>`
- `python -m vlm_selective_eval.analysis.plots_cli --metrics-json <metrics.json> --output-dir <dir>`
