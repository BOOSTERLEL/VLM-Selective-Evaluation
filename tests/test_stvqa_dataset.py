import json
from pathlib import Path

from PIL import Image

from scripts.build_stvqa_source import build_source_records
from vlm_selective_eval.config import DatasetConfig
from vlm_selective_eval.constants import ANSWERABLE, UNANSWERABLE
from vlm_selective_eval.datasets.stvqa import STVQADatasetAdapter


def test_build_stvqa_source_records_with_embedded_ocr(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (100, 80), "white").save(image_path)

    annotations = {
        "data": [
            {
                "question_id": 1,
                "file_name": "sample.jpg",
                "question": "What word is shown?",
                "answers": ["SALE", "sale"],
                "ocr_info": [
                    {"word": "SALE", "bounding_box": {"x": 10, "y": 12, "w": 35, "h": 16}},
                    {"word": "NOW", "bounding_box": {"x": 52, "y": 12, "w": 22, "h": 16}},
                ],
            }
        ]
    }

    records, stats = build_source_records(
        annotations_payload=annotations,
        image_dir=image_dir,
    )

    assert len(records) == 1
    assert records[0]["id"] == "stvqa_1"
    assert records[0]["question"] == "What word is shown?"
    assert records[0]["answers"] == ["SALE", "sale"]
    assert records[0]["evidence_boxes"] == [[10, 12, 45, 28]]
    assert stats["skipped_no_match"] == 0


def test_stvqa_dataset_adapter_builds_pairs_from_source(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (96, 96), "white").save(image_path)
    source_path = tmp_path / "stvqa_source.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "id": "stvqa_demo",
                    "image_path": str(image_path),
                    "question": "What number is shown?",
                    "answers": ["12", "12"],
                    "evidence_boxes": [[20, 20, 50, 44]],
                }
            ]
        ),
        encoding="utf-8",
    )

    adapter = STVQADatasetAdapter()
    config = DatasetConfig(
        mode="stvqa",
        source_path=str(source_path),
        num_pairs=1,
        image_size=96,
        evidence_removal="mask",
    )
    rows = adapter.build_pairs(config=config, output_root=tmp_path, seed=7)

    assert len(rows) == 2
    assert rows[0].task == "stvqa_ocr"
    assert rows[0].ground_truth_answerability == ANSWERABLE
    assert rows[1].ground_truth_answerability == UNANSWERABLE
    assert rows[0].ground_truth_answer == "12"
    assert rows[1].ground_truth_answer == ""
    assert rows[0].evidence_metadata["source_kind"] == "stvqa"
