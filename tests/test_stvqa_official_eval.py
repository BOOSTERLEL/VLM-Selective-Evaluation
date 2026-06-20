from pathlib import Path

from pytest import approx

from vlm_selective_eval.evaluation.metrics import evaluate_predictions
from vlm_selective_eval.io_utils import write_jsonl
from vlm_selective_eval.pipeline import evaluate_stvqa_official_file


def test_evaluate_predictions_uses_stvqa_anls_for_stvqa_rows():
    rows = [
        {
            "sample_id": "1",
            "pair_id": "p1",
            "task": "stvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "coca cola",
            "parsed_status": "ANSWERABLE",
            "parsed_answer": "CocaCola",
            "parsed_confidence": 0.9,
            "parse_ok": True,
            "format_ok": True,
            "metadata": {"answers": ["Coca Cola", "Coca Cola Company"]},
        }
    ]

    metrics = evaluate_predictions(rows, n_calibration_bins=5)
    assert metrics["quadrant_counts"]["answerable_correct"] == 1
    assert metrics["answerable_soft_score_mean"] == approx(0.9, abs=1e-6)
    assert metrics["risk"] == approx(0.1, abs=1e-6)


def test_stvqa_official_eval_from_predictions(tmp_path: Path):
    pred_path = tmp_path / "preds.jsonl"
    rows = [
        {
            "sample_id": "1",
            "pair_id": "p1",
            "task": "stvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "coca cola",
            "parsed_status": "ANSWERABLE",
            "parsed_answer": "CocaCola",
            "parsed_confidence": 0.8,
            "parse_ok": True,
            "format_ok": True,
            "metadata": {"official_answers": ["Coca Cola", "Coca Cola Company"]},
        },
        {
            "sample_id": "2",
            "pair_id": "p2",
            "task": "stvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "sale",
            "parsed_status": "ANSWERABLE",
            "parsed_answer": "price",
            "parsed_confidence": 0.7,
            "parse_ok": True,
            "format_ok": True,
            "metadata": {"official_answers": ["sale"]},
        },
        {
            "sample_id": "3",
            "pair_id": "p3",
            "task": "stvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "UNANSWERABLE",
            "ground_truth_answer": "",
            "parsed_status": "UNANSWERABLE",
            "parsed_answer": "",
            "parsed_confidence": 0.1,
            "parse_ok": True,
            "format_ok": True,
            "metadata": {},
        },
    ]
    write_jsonl(pred_path, rows)

    out_dir = tmp_path / "eval"
    metrics = evaluate_stvqa_official_file(predictions_path=pred_path, output_dir=out_dir)
    assert metrics["n_answerable"] == 2
    assert metrics["official_metric_name"] == "anls"
    assert metrics["official_anls_mean"] == approx(0.45, abs=1e-6)
    assert metrics["official_score_mean"] == approx(0.45, abs=1e-6)
    assert metrics["multi_answer_available_rate"] == 1.0
    assert metrics["avg_reference_answer_count"] == 1.5
    assert metrics["anls_threshold"] == 0.5
    assert (out_dir / "stvqa_official_metrics.json").exists()
    assert (out_dir / "stvqa_official_metrics.csv").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "quadrant_rates.csv").exists()
    assert (out_dir / "risk_coverage.csv").exists()
    assert (out_dir / "coverage_risk_curve.csv").exists()
    assert (out_dir / "assert_abstain.csv").exists()
    assert (out_dir / "calibration_bins.csv").exists()
    assert (out_dir / "scored.jsonl").exists()
    assert "metrics_row" in metrics
