#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/featurize/data/textvqa}"
mkdir -p "${ROOT}"
cd "${ROOT}"

echo "[1/4] Download TextVQA questions..."
aria2c -c -x 8 -s 8 -k 1M \
  -o TextVQA_0.5.1_train.json \
  "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json"

aria2c -c -x 8 -s 8 -k 1M \
  -o TextVQA_0.5.1_val.json \
  "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"

# test 可选；本阶段通常不需要，因为没有公开答案
aria2c -c -x 8 -s 8 -k 1M \
  -o TextVQA_0.5.1_test.json \
  "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json"

echo "[2/4] Download TextVQA OCR files..."
aria2c -c -x 8 -s 8 -k 1M \
  -o TextVQA_Rosetta_OCR_v0.2_train.json \
  "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_train.json"

aria2c -c -x 8 -s 8 -k 1M \
  -o TextVQA_Rosetta_OCR_v0.2_val.json \
  "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_val.json"

aria2c -c -x 8 -s 8 -k 1M \
  -o TextVQA_Rosetta_OCR_v0.2_test.json \
  "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_test.json"

echo "[3/4] Download train/val images..."
aria2c -c -x 8 -s 8 -k 1M \
  -o train_val_images.zip \
  "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"

if [ ! -d train_images ]; then
  unzip -q train_val_images.zip
fi

echo "[4/4] Done."
echo "Files under: ${ROOT}"