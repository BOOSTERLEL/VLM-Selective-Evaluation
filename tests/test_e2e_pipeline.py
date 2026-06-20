from pathlib import Path

from vlm_selective_eval.pipeline import (
    build_pairs_from_config,
    evaluate_predictions_file,
    run_inference_from_config,
)


def test_minimal_e2e_pipeline(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  name: test_run
  output_dir: "{output_dir}"
  seed: 5
dataset:
  mode: synthetic
  num_pairs: 2
  image_size: 96
  evidence_removal: mask
model:
  adapter: mock
  model_name: mock-vlm
  answer_error_rate: 0.0
  hallucination_rate: 0.0
  malformed_rate: 0.0
  free_text_missing_field_rate: 0.0
prompt:
  mode: structured
  include_system: true
evaluation:
  n_calibration_bins: 5
        """.format(output_dir=str(tmp_path / "outputs").replace("\\", "/")),
        encoding="utf-8",
    )

    pairs_path = build_pairs_from_config(config_path=config_path)
    assert pairs_path.exists()
    preds_path = run_inference_from_config(config_path=config_path, pairs_path=pairs_path)
    assert preds_path.exists()

    eval_dir = tmp_path / "outputs" / "eval"
    metrics = evaluate_predictions_file(predictions_path=preds_path, output_dir=eval_dir, n_calibration_bins=5)
    assert metrics["n_samples"] == 4
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "metrics.csv").exists()
    assert (eval_dir / "quadrant_rates.csv").exists()
    assert (eval_dir / "risk_coverage.csv").exists()
    assert (eval_dir / "assert_abstain.csv").exists()
    assert (eval_dir / "scored.jsonl").exists()
    assert "quadrant_counts" in metrics
