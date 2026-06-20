"""Synthetic dataset for sanity checks and unit tests."""

from __future__ import annotations

from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

from vlm_selective_eval.constants import ANSWERABLE, UNANSWERABLE
from vlm_selective_eval.config import DatasetConfig
from vlm_selective_eval.schemas import PairSample
from .base import DatasetAdapter
from .transforms import mask_regions


TOKENS = [
    "APPLE",
    "BLUE",
    "SEVEN",
    "CAT",
    "TRAIN",
    "RIVER",
    "LIME",
    "CLOUD",
    "PAPER",
    "MUSIC",
]


def _text_box(size: int, token: str) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), token, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    x = (size - text_w) // 2
    y = (size - text_h) // 2
    draw.text((x, y), token, fill="black", font=font)
    pad = 4
    evidence = (
        max(0, x - pad),
        max(0, y - pad),
        min(size, x + text_w + pad),
        min(size, y + text_h + pad),
    )
    return image, evidence


class SyntheticDatasetAdapter(DatasetAdapter):
    """Create paired OCR-like synthetic samples."""

    def build_pairs(
        self,
        config: DatasetConfig,
        output_root: Path,
        seed: int,
    ) -> list[PairSample]:
        rng = random.Random(seed)
        image_dir = output_root / "images" / "synthetic"
        image_dir.mkdir(parents=True, exist_ok=True)
        samples: list[PairSample] = []

        for idx in range(config.num_pairs):
            token = TOKENS[idx % len(TOKENS)]
            pair_id = f"synthetic_pair_{idx:04d}"
            image, evidence = _text_box(config.image_size, token)
            answer_slug = token.lower()
            answerable_path = image_dir / f"{pair_id}__answerable__{answer_slug}.png"
            image.save(answerable_path)

            hidden = mask_regions(image, [evidence], fill_color=(120, 120, 120))
            unanswerable_path = image_dir / f"{pair_id}__unanswerable__{answer_slug}.png"
            hidden.save(unanswerable_path)

            question = "What uppercase word is shown in the center of the image?"
            metadata = {"token": token, "seed": seed, "sample_rand": rng.random()}
            evidence_metadata = {"regions": [list(evidence)], "removal": "mask"}

            samples.append(
                PairSample(
                    sample_id=f"{pair_id}_a",
                    pair_id=pair_id,
                    task="synthetic_ocr",
                    image_path=str(answerable_path),
                    question=question,
                    ground_truth_answerability=ANSWERABLE,
                    ground_truth_answer=token,
                    evidence_metadata=evidence_metadata,
                    metadata=metadata,
                )
            )
            samples.append(
                PairSample(
                    sample_id=f"{pair_id}_u",
                    pair_id=pair_id,
                    task="synthetic_ocr",
                    image_path=str(unanswerable_path),
                    question=question,
                    ground_truth_answerability=UNANSWERABLE,
                    ground_truth_answer="",
                    evidence_metadata=evidence_metadata,
                    metadata=metadata,
                )
            )
        return samples
