from vlm_selective_eval.evaluation.metrics import evaluate_predictions


def test_evaluate_predictions_uses_multi_answer_soft_scoring():
    rows = [
        {
            "sample_id": "1",
            "pair_id": "p1",
            "task": "textvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "stop",
            "parsed_status": "ANSWERABLE",
            "parsed_answer": "stop sign",
            "parsed_confidence": 0.8,
            "parse_ok": True,
            "format_ok": True,
            "metadata": {"answers": ["stop", "stop", "stop sign"]},
        },
        {
            "sample_id": "2",
            "pair_id": "p2",
            "task": "textvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "milk",
            "parsed_status": "ANSWERABLE",
            "parsed_answer": "milk",
            "parsed_confidence": 0.9,
            "parse_ok": True,
            "format_ok": True,
            "metadata": {"answers": ["milk", "milk", "milk"]},
        },
    ]

    metrics = evaluate_predictions(rows, n_calibration_bins=5)
    assert metrics["answerable_soft_score_mean"] == 2.0 / 3.0
    assert metrics["quadrant_counts"]["answerable_correct"] == 2
    assert metrics["risk"] == 1.0 / 3.0


def test_missing_status_defaults_to_answerable_for_conservative_har():
    rows = [
        {
            "sample_id": "u1",
            "pair_id": "p1",
            "task": "textvqa_ocr",
            "model_name": "m",
            "prompt_mode": "free_text",
            "ground_truth_answerability": "UNANSWERABLE",
            "ground_truth_answer": "",
            "parsed_status": None,
            "parsed_answer": "cat",
            "parsed_confidence": None,
            "parse_ok": False,
            "format_ok": False,
            "status_reliable": False,
            "confidence_reliable": False,
            "metadata": {},
        }
    ]

    metrics = evaluate_predictions(rows, n_calibration_bins=5)
    assert metrics["quadrant_counts"]["unanswerable_assert"] == 1
    assert metrics["hallucinatory_assertion_rate"] == 1.0
    assert metrics["parse_fail_status_policy"] == "default_to_ANSWERABLE"
    assert "unanswerable_assert_abstain_curve" in metrics
