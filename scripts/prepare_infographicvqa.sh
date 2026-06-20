#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/featurize/data/infographicvqa}"
VAL_JSON_SRC="${2:-}"
IMAGES_SRC="${3:-}"
OCR_SRC="${4:-}"

mkdir -p "${ROOT}"/downloads "${ROOT}"/images "${ROOT}"/ocr

download_or_copy() {
  local src="$1"
  local dst="$2"
  if [[ -z "${src}" ]]; then
    return 1
  fi
  if [[ "${src}" =~ ^https?:// ]]; then
    if command -v aria2c >/dev/null 2>&1; then
      aria2c -c -x 8 -s 8 -k 1M -o "$(basename "${dst}")" -d "$(dirname "${dst}")" "${src}"
    else
      curl -L "${src}" -o "${dst}"
    fi
  else
    cp "${src}" "${dst}"
  fi
}

extract_archive() {
  local archive="$1"
  local outdir="$2"
  if [[ "${archive}" == *.zip ]]; then
    unzip -q -o "${archive}" -d "${outdir}"
  elif [[ "${archive}" == *.tar.gz || "${archive}" == *.tgz ]]; then
    tar -xzf "${archive}" -C "${outdir}"
  elif [[ "${archive}" == *.tar ]]; then
    tar -xf "${archive}" -C "${outdir}"
  else
    return 0
  fi
}

if [[ -z "${VAL_JSON_SRC}" || -z "${IMAGES_SRC}" || -z "${OCR_SRC}" ]]; then
  cat <<'EOF'
InfographicVQA download is distributed from the official DocVQA challenge pages.

Official pages:
- https://www.docvqa.org/datasets/infographicvqa
- https://www.docvqa.org/challenges/2021

Please download or copy these validation-split assets first:
1. InfographicVQA validation annotation JSON
2. InfographicVQA images archive or directory
3. InfographicVQA OCR archive or directory

Then rerun, for example:
  bash prepare_infographicvqa.sh /home/featurize/data/infographicvqa \
    /path/to/val_v1.0.json \
    /path/to/images.zip \
    /path/to/ocr.zip

You can also pass signed download URLs instead of local paths.
EOF
  exit 1
fi

echo "[1/4] Place validation annotations..."
download_or_copy "${VAL_JSON_SRC}" "${ROOT}/val_v1.0.json"

echo "[2/4] Place image bundle..."
IMAGE_BASENAME="$(basename "${IMAGES_SRC}")"
download_or_copy "${IMAGES_SRC}" "${ROOT}/downloads/${IMAGE_BASENAME}"
if [[ -f "${ROOT}/downloads/${IMAGE_BASENAME}" ]]; then
  extract_archive "${ROOT}/downloads/${IMAGE_BASENAME}" "${ROOT}/images"
elif [[ -d "${IMAGES_SRC}" ]]; then
  cp -R "${IMAGES_SRC}/." "${ROOT}/images/"
fi

echo "[3/4] Place OCR bundle..."
OCR_BASENAME="$(basename "${OCR_SRC}")"
download_or_copy "${OCR_SRC}" "${ROOT}/downloads/${OCR_BASENAME}"
if [[ -f "${ROOT}/downloads/${OCR_BASENAME}" ]]; then
  extract_archive "${ROOT}/downloads/${OCR_BASENAME}" "${ROOT}/ocr"
elif [[ -d "${OCR_SRC}" ]]; then
  cp -R "${OCR_SRC}/." "${ROOT}/ocr/"
fi

echo "[4/4] Build source JSON..."
python scripts/build_infographicvqa_source.py \
  --annotations "${ROOT}/val_v1.0.json" \
  --ocr-dir "${ROOT}/ocr" \
  --image-dir "${ROOT}/images" \
  --output "${ROOT}/infographicvqa_val_source.json"

echo "Done: ${ROOT}"
