"""DocVQA-style OCR subset adapter with evidence-removal pair building."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from vlm_selective_eval.config import DatasetConfig
from vlm_selective_eval.constants import ANSWERABLE, UNANSWERABLE
from vlm_selective_eval.schemas import PairSample
from .textvqa import (
    TextVQADatasetAdapter,
    _load_source_records,
    _majority_answer,
    _render_text_image,
    _safe_slug,
)


class DocVQADatasetAdapter(TextVQADatasetAdapter):
    """Adapter for DocVQA-style OCR tasks with removable evidence."""

    def _build_mock_subset(self, config: DatasetConfig, output_root: Path) -> list[dict]:
        image_dir = output_root / "images" / "docvqa_mock"
        image_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            ("INVOICE", "What document title is shown?"),
            ("8492", "What invoice number is visible?"),
            ("PAID", "What payment status appears on the page?"),
        ]
        subset: list[dict] = []
        for idx, (text, question) in enumerate(rows):
            sample_id = f"docvqa_mock_{idx:03d}"
            answer = text.lower()
            answerable = image_dir / f"{sample_id}__answerable__{_safe_slug(answer)}.png"
            box = _render_text_image(answerable, text=text, image_size=config.image_size, text_pos=(26, 36))
            subset.append(
                {
                    "id": sample_id,
                    "image_path": str(answerable),
                    "question": question,
                    "answer": answer,
                    "evidence_boxes": [list(box)],
                }
            )
        return subset

    def build_pairs(
        self,
        config: DatasetConfig,
        output_root: Path,
        seed: int,
    ) -> list[PairSample]:
        del seed
        source_path = Path(config.source_path) if config.source_path else None
        if source_path and source_path.exists():
            records = _load_source_records(source_path)
        else:
            records = self._build_mock_subset(config=config, output_root=output_root)

        if config.num_pairs > 0:
            records = records[: config.num_pairs]

        image_dir = output_root / "images" / "docvqa_pairs"
        image_dir.mkdir(parents=True, exist_ok=True)
        samples: list[PairSample] = []

        for idx, row in enumerate(records):
            pair_id = f"docvqa_pair_{idx:04d}"
            answers = row.get("answers")
            if not isinstance(answers, list):
                answers = []
            answers = [str(x).strip() for x in answers if str(x).strip()]
            if not answers:
                legacy = row.get("official_answers")
                if isinstance(legacy, list):
                    answers = [str(x).strip() for x in legacy if str(x).strip()]
            answer = str(row.get("answer", "")).strip() or _majority_answer(answers)
            if not answers and answer:
                answers = [answer]
            answer_slug = _safe_slug(answer)
            question = str(row["question"])
            image_path = Path(str(row["image_path"]))
            evidence_boxes_raw = row.get("evidence_boxes", [])
            evidence_boxes = [tuple(map(int, box)) for box in evidence_boxes_raw]
            if not image_path.exists():
                continue

            answerable_target = image_dir / f"{pair_id}__answerable__{answer_slug}.png"
            image = Image.open(image_path).convert("RGB")
            image = image.resize((config.image_size, config.image_size))
            image.save(answerable_target)
            hidden = self._apply_removal(image, evidence_boxes, config.evidence_removal)
            unanswerable_target = image_dir / f"{pair_id}__unanswerable__{answer_slug}.png"
            hidden.save(unanswerable_target)

            evidence_metadata = {
                "regions": [list(box) for box in evidence_boxes],
                "removal": config.evidence_removal,
                "source_kind": "docvqa",
            }
            metadata = {
                "original_id": row.get("id"),
                "source_path": str(source_path) if source_path else None,
                "answers": answers,
                "official_answers": answers,
            }

            samples.append(
                PairSample(
                    sample_id=f"{pair_id}_a",
                    pair_id=pair_id,
                    task="docvqa_ocr",
                    image_path=str(answerable_target),
                    question=question,
                    ground_truth_answerability=ANSWERABLE,
                    ground_truth_answer=answer,
                    evidence_metadata=evidence_metadata,
                    metadata=metadata,
                )
            )
            samples.append(
                PairSample(
                    sample_id=f"{pair_id}_u",
                    pair_id=pair_id,
                    task="docvqa_ocr",
                    image_path=str(unanswerable_target),
                    question=question,
                    ground_truth_answerability=UNANSWERABLE,
                    ground_truth_answer="",
                    evidence_metadata=evidence_metadata,
                    metadata=metadata,
                )
            )
        return samples
