#!/usr/bin/env bash

set -euo pipefail

# Build a Kaggle-ready zip from src/kaggle/
# Usage: scripts/build_kaggle_zip.sh [OUTPUT_ZIP]
# Default OUTPUT_ZIP: <repo_root>/kaggle_files.zip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src/kaggle"
OUT_ZIP="${1:-${ROOT_DIR}/kaggle_files.zip}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Error: src directory not found at ${SRC_DIR}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

echo "Rebuilding zip from: ${SRC_DIR}"
echo "Output zip: ${OUT_ZIP}"

rm -f "${OUT_ZIP}"

# Exclude caches and OS cruft
zip -r "${OUT_ZIP}" src/kaggle \
  -x "**/__pycache__/**" \
     "**/*.pyc" \
     "**/.DS_Store" \
     "**/.ipynb_checkpoints/**" | cat

echo "Done. Wrote ${OUT_ZIP}"


