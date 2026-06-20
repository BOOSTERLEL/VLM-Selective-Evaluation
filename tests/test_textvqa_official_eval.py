from pathlib import Path

from vlm_selective_eval.io_utils import write_jsonl
from vlm_selective_eval.pipeline import evaluate_textvqa_official_file


def test_textvqa_official_eval_from_predictions(tmp_path: Path):
    pred_path = tmp_path / "preds.jsonl"
    rows = [
        {
            "sample_id": "1",
            "pair_id": "p1",
            "task": "textvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "stop",
            "parsed_answer": "stop",
            "metadata": {"official_answers": ["stop", "STOP", "stop sign"]},
        },
        {
            "sample_id": "2",
            "pair_id": "p2",
            "task": "textvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "milk",
            "parsed_answer": "juice",
            "metadata": {"official_answers": ["milk", "milk", "milk"]},
        },
        {
            "sample_id": "3",
            "pair_id": "p3",
            "task": "textvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "UNANSWERABLE",
            "ground_truth_answer": "",
            "parsed_answer": "",
            "metadata": {},
        },
    ]
    write_jsonl(pred_path, rows)

    out_dir = tmp_path / "eval"
    metrics = evaluate_textvqa_official_file(predictions_path=pred_path, output_dir=out_dir)
    assert metrics["n_answerable"] == 2
    assert metrics["official_soft_score_mean"] == 1.0 / 3.0
    assert metrics["multi_answer_available_rate"] == 1.0
    assert metrics["avg_reference_answer_count"] == 3.0
    assert metrics["exactly_ten_reference_rate"] == 0.0
    assert metrics["truncated_to_first_10_count"] == 0
    assert (out_dir / "textvqa_official_metrics.json").exists()
    assert (out_dir / "textvqa_official_metrics.csv").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "quadrant_rates.csv").exists()
    assert (out_dir / "risk_coverage.csv").exists()
    assert (out_dir / "coverage_risk_curve.csv").exists()
    assert (out_dir / "assert_abstain.csv").exists()
    assert (out_dir / "calibration_bins.csv").exists()
    assert (out_dir / "scored.jsonl").exists()
    assert "metrics_row" in metrics
