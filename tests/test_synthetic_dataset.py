from pathlib import Path

from vlm_selective_eval.config import DatasetConfig
from vlm_selective_eval.constants import ANSWERABLE, UNANSWERABLE
from vlm_selective_eval.datasets.synthetic import SyntheticDatasetAdapter


def test_synthetic_pair_generation(tmp_path: Path):
    adapter = SyntheticDatasetAdapter()
    config = DatasetConfig(mode="synthetic", num_pairs=3, image_size=128)
    rows = adapter.build_pairs(config=config, output_root=tmp_path, seed=3)
    assert len(rows) == 6
    by_pair: dict[str, set[str]] = {}
    for row in rows:
        by_pair.setdefault(row.pair_id, set()).add(row.ground_truth_answerability)
        assert Path(row.image_path).exists()
    assert len(by_pair) == 3
    for statuses in by_pair.values():
        assert statuses == {ANSWERABLE, UNANSWERABLE}
