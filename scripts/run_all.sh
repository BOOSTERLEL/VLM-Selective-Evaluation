#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/run.yaml}"

python -m vlm_selective_eval.cli build-pairs --config "$CFG"

OUT_DIR="$(CFG_PATH="$CFG" python - <<'PY'
import yaml
import os
from pathlib import Path
cfg_path = Path(os.environ["CFG_PATH"])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
print(cfg["experiment"]["output_dir"])
PY
)"

PAIRS_PATH="$(ls "$OUT_DIR"/pairs/*_pairs.jsonl | head -n 1)"

python -m vlm_selective_eval.cli run-inference --config "$CFG" --pairs "$PAIRS_PATH"

PRED_PATH="$(ls "$OUT_DIR"/predictions/*_predictions.jsonl | head -n 1)"

python -m vlm_selective_eval.cli evaluate-and-plot --predictions "$PRED_PATH" --output-dir "$OUT_DIR"/evaluation
