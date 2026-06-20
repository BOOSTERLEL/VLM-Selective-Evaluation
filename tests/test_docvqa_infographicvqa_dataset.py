import json
from pathlib import Path

from PIL import Image
from pytest import approx

from scripts.build_docvqa_source import build_source_records as build_docvqa_source_records
from scripts.build_infographicvqa_source import build_source_records as build_infographic_source_records
from vlm_selective_eval.config import DatasetConfig
from vlm_selective_eval.constants import ANSWERABLE, UNANSWERABLE
from vlm_selective_eval.datasets.docvqa import DocVQADatasetAdapter
from vlm_selective_eval.datasets.infographicvqa import InfographicVQADatasetAdapter
from vlm_selective_eval.evaluation.docvqa_official import evaluate_docvqa_official_from_predictions
from vlm_selective_eval.evaluation.infographicvqa_official import (
    evaluate_infographicvqa_official_from_predictions,
)
from vlm_selective_eval.io_utils import write_jsonl
from vlm_selective_eval.pipeline import evaluate_docvqa_official_file


def test_build_docvqa_source_records_with_separate_ocr_payload(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "doc_1.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    annotations = {
        "data": [
            {
                "questionId": 1,
                "image": "doc_1.png",
                "question": "What invoice number is shown?",
                "answers": ["8492", "8492"],
            }
        ]
    }
    ocr_payload = {
        "data": [
            {
                "image": "doc_1.png",
                "pages": [
                    {
                        "width": 100,
                        "height": 80,
                        "words": [
                            {"content": "INV", "polygon": [2, 2, 18, 2, 18, 12, 2, 12]},
                            {"content": "8492", "polygon": [20, 10, 50, 10, 50, 28, 20, 28]},
                        ],
                    }
                ],
            }
        ]
    }

    records, stats = build_docvqa_source_records(
        annotations_payload=annotations,
        image_dir=image_dir,
        ocr_payload=ocr_payload,
    )

    assert len(records) == 1
    assert records[0]["id"] == "docvqa_1"
    assert records[0]["answers"] == ["8492", "8492"]
    assert records[0]["evidence_boxes"] == [[20, 10, 50, 28]]
    assert stats["skipped_no_match"] == 0


def test_build_infographicvqa_source_records_with_embedded_pages(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "info_1.png"
    Image.new("RGB", (120, 90), "white").save(image_path)

    annotations = {
        "data": [
            {
                "question_id": 7,
                "file_name": "info_1.png",
                "question": "What percentage is shown?",
                "answers": ["42%", "42 percent"],
                "pages": [
                    {
                        "width": 120,
                        "height": 90,
                        "words": [
                            {"content": "42%", "polygon": [8, 8, 34, 8, 34, 22, 8, 22]},
                        ],
                    }
                ],
            }
        ]
    }

    records, stats = build_infographic_source_records(
        annotations_payload=annotations,
        image_dir=image_dir,
    )

    assert len(records) == 1
    assert records[0]["id"] == "infographicvqa_7"
    assert records[0]["evidence_boxes"] == [[8, 8, 34, 22]]
    assert stats["skipped_no_match"] == 0


def test_docvqa_and_infographicvqa_adapters_build_pairs(tmp_path: Path):
    doc_image = tmp_path / "doc.png"
    info_image = tmp_path / "info.png"
    Image.new("RGB", (96, 96), "white").save(doc_image)
    Image.new("RGB", (96, 96), "white").save(info_image)

    doc_source = tmp_path / "doc_source.json"
    info_source = tmp_path / "info_source.json"
    doc_source.write_text(
        json.dumps(
            [
                {
                    "id": "doc_sample",
                    "image_path": str(doc_image),
                    "question": "What word is stamped?",
                    "answers": ["paid", "paid"],
                    "evidence_boxes": [[10, 10, 30, 28]],
                }
            ]
        ),
        encoding="utf-8",
    )
    info_source.write_text(
        json.dumps(
            [
                {
                    "id": "info_sample",
                    "image_path": str(info_image),
                    "question": "What year is shown?",
                    "answers": ["2019", "2019"],
                    "evidence_boxes": [[12, 12, 36, 28]],
                }
            ]
        ),
        encoding="utf-8",
    )

    doc_adapter = DocVQADatasetAdapter()
    info_adapter = InfographicVQADatasetAdapter()
    doc_rows = doc_adapter.build_pairs(
        config=DatasetConfig(mode="docvqa", source_path=str(doc_source), num_pairs=1, image_size=96, evidence_removal="mask"),
        output_root=tmp_path,
        seed=7,
    )
    info_rows = info_adapter.build_pairs(
        config=DatasetConfig(mode="infographicvqa", source_path=str(info_source), num_pairs=1, image_size=96, evidence_removal="mask"),
        output_root=tmp_path,
        seed=7,
    )

    assert len(doc_rows) == 2
    assert doc_rows[0].task == "docvqa_ocr"
    assert doc_rows[0].ground_truth_answerability == ANSWERABLE
    assert doc_rows[1].ground_truth_answerability == UNANSWERABLE
    assert doc_rows[0].evidence_metadata["source_kind"] == "docvqa"

    assert len(info_rows) == 2
    assert info_rows[0].task == "infographicvqa_ocr"
    assert info_rows[0].ground_truth_answerability == ANSWERABLE
    assert info_rows[1].ground_truth_answerability == UNANSWERABLE
    assert info_rows[0].evidence_metadata["source_kind"] == "infographicvqa"


def test_docvqa_official_eval_from_predictions(tmp_path: Path):
    pred_path = tmp_path / "preds.jsonl"
    rows = [
        {
            "sample_id": "1",
            "pair_id": "p1",
            "task": "docvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "8492",
            "parsed_status": "ANSWERABLE",
            "parsed_answer": "8492",
            "parsed_confidence": 0.9,
            "parse_ok": True,
            "format_ok": True,
            "metadata": {"official_answers": ["8492", "8492"]},
        },
        {
            "sample_id": "2",
            "pair_id": "p2",
            "task": "docvqa_ocr",
            "model_name": "m",
            "prompt_mode": "structured",
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "invoice",
            "parsed_status": "ANSWERABLE",
            "parsed_answer": "receipt",
            "parsed_confidence": 0.6,
            "parse_ok": True,
            "format_ok": True,
            "metadata": {"official_answers": ["invoice"]},
        },
    ]
    write_jsonl(pred_path, rows)

    out_dir = tmp_path / "eval"
    metrics = evaluate_docvqa_official_file(predictions_path=pred_path, output_dir=out_dir)
    assert metrics["official_anls_mean"] == approx(0.5, abs=1e-6)
    assert (out_dir / "docvqa_official_metrics.json").exists()
    assert (out_dir / "docvqa_official_metrics.csv").exists()
    assert (out_dir / "metrics.json").exists()


def test_dataset_specific_official_modules_exist_and_score():
    doc_rows = [
        {
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "8492",
            "parsed_answer": "8492",
            "metadata": {"official_answers": ["8492", "8492"]},
        }
    ]
    infographic_rows = [
        {
            "ground_truth_answerability": "ANSWERABLE",
            "ground_truth_answer": "42%",
            "parsed_answer": "42%",
            "metadata": {"official_answers": ["42%", "42 percent"]},
        }
    ]

    doc_metrics = evaluate_docvqa_official_from_predictions(doc_rows)
    infographic_metrics = evaluate_infographicvqa_official_from_predictions(infographic_rows)

    assert doc_metrics["official_anls_mean"] == approx(1.0, abs=1e-6)
    assert infographic_metrics["official_anls_mean"] == approx(1.0, abs=1e-6)
