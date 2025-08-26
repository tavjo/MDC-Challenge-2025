#!/usr/bin/env bash
# make_wheels.sh — Build a Linux wheelhouse for CPython 3.11 (Kaggle-friendly)
# Usage:
#   ./make_wheels.sh [requirements-file]
#
# Key behavior:
#   • Prefetch PyTorch (torch/vision/audio) from download.pytorch.org (CUDA 12.x by default)
#   • Download all deps for BOTH: manylinux_2_28_x86_64 THEN manylinux2014_x86_64
#   • Mix wheels in one folder; pip will choose the right ones on Kaggle install
#
# Env overrides:
#   REQ_FILE=requirements-inference.txt
#   WHEEL_DIR=./wheels/cp311
#   PYVER=311
#   PLATS="manylinux_2_28_x86_64 manylinux2014_x86_64"
#
#   # ---- PyTorch (vLLM 0.9.x pins torch==2.7.*) ----
#   TORCH_VERSION=2.7.0
#   TORCHVISION_VERSION=0.22.0
#   TORCHAUDIO_VERSION=2.7.0
#   TORCH_CUDA=cu128                 # candidates tried in order below
#   TORCH_CUDA_CANDIDATES="cu128 cu126 cu124 cu121 cpu"
#   TORCH_INDEX_URL=                 # override full index if you want
#
#   # ---- ONNX Runtime GPU channel override (optional) ----
#   ORT_CUDA=11                      # set to 11 ONLY if Kaggle image is CUDA 11.x
#   ORT_INDEX_URL=

set -Eeuo pipefail

REQ_FILE="${1:-${REQ_FILE:-requirements-inference.txt}}"
: "${PYVER:=311}"
: "${PLATS:=manylinux_2_28_x86_64 manylinux2014_x86_64}"
: "${WHEEL_DIR:=./wheels/cp${PYVER}}"

[[ -f "$REQ_FILE" ]] || { echo "ERROR: requirements file not found: $REQ_FILE" >&2; exit 1; }

# --- Effective requirements (optionally rewrite duckdb pin on cp311) ---
: "${DUCKDB_CP311_FALLBACK:=1.2.0}"   # 1.2.0 has cp311 manylinux wheels
REQ_EFFECTIVE="$REQ_FILE"
if [[ "$PYVER" == "311" && -n "$DUCKDB_CP311_FALLBACK" ]]; then
  if grep -Eq '^[[:space:]]*duckdb==1\.3\.2([[:space:]]|$)' "$REQ_FILE"; then
    REQ_EFFECTIVE="$(mktemp)"
    # replace the entire matching line with fallback pin (pass value explicitly into awk)
    awk -v duckver="$DUCKDB_CP311_FALLBACK" 'BEGIN{done=0}
         /^[[:space:]]*duckdb==1\.3\.2([[:space:]]|$)/ && !done {print "duckdb==" duckver; done=1; next}
         {print $0}' "$REQ_FILE" > "$REQ_EFFECTIVE"
    trap '[[ -n "${REQ_EFFECTIVE:-}" && "$REQ_EFFECTIVE" != "$REQ_FILE" ]] && rm -f "$REQ_EFFECTIVE"' EXIT
    echo "INFO: cp311 detected; using duckdb==$DUCKDB_CP311_FALLBACK (1.3.2 has no cp311 manylinux wheel)."
  fi
fi

# Isolated helper venv
VENV_DIR=".whlvenv"
[[ -d "$VENV_DIR" ]] || python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel

mkdir -p "$WHEEL_DIR"
WHEEL_DIR="$(cd "$WHEEL_DIR" && pwd)"
echo "==> Wheelhouse: $WHEEL_DIR"
echo "==> Requirements: $REQ_FILE"
echo "==> Python: cp${PYVER}"
echo "==> Platforms: $PLATS"

# --------------------------------------------
# 0) Prefetch PyTorch (torch/vision/audio) GPU
#    Use platform="$PLAT" (manylinux_2_28), not linux_x86_64.
#    Use --no-deps so we don’t try to resolve deps on the PT index.
# --------------------------------------------
: "${TORCH_VERSION:=2.7.0}"
: "${TORCHVISION_VERSION:=0.22.0}"
: "${TORCHAUDIO_VERSION:=2.7.0}"
: "${TORCH_CUDA:=cu128}"
: "${TORCH_CUDA_CANDIDATES:=cu128 cu126 cu124 cu121 cpu}"

prefetch_torch_ok=""
for plat in $PLATS; do
  # pick index url from TORCH_CUDA if not explicitly set
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    _TORCH_IDX="$TORCH_INDEX_URL"
    _CUDA_FLAVOR="${TORCH_CUDA}"
    _CANDS="$TORCH_CUDA"
  else
    _CANDS="$TORCH_CUDA_CANDIDATES"
  fi
  for flavor in $_CANDS; do
    case "$flavor" in
      cpu)   _TORCH_IDX="https://download.pytorch.org/whl/cpu"   ;;
      cu121) _TORCH_IDX="https://download.pytorch.org/whl/cu121" ;;
      cu124) _TORCH_IDX="https://download.pytorch.org/whl/cu124" ;;
      cu126) _TORCH_IDX="https://download.pytorch.org/whl/cu126" ;;
      cu128) _TORCH_IDX="https://download.pytorch.org/whl/cu128" ;;
      *) echo "ERROR: Unknown TORCH_CUDA flavor '$flavor'." >&2; exit 2 ;;
    esac
    echo ">>> Prefetch PyTorch ($flavor) from ${_TORCH_IDX} for platform ${plat}"
    set +e
    pip download "torch==${TORCH_VERSION}" \
                 "torchvision==${TORCHVISION_VERSION}" \
                 "torchaudio==${TORCHAUDIO_VERSION}" \
      --no-deps \
      --only-binary=:all: \
      --implementation cp \
      --platform "$plat" \
      --python-version "$PYVER" \
      --abi "cp${PYVER}" \
      --dest "$WHEEL_DIR" \
      --index-url "$_TORCH_IDX"
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      prefetch_torch_ok="yes"
      break
    else
      echo "WARN: PyTorch prefetch failed for $flavor on $plat; trying next candidate/index..."
    fi
  done
  [[ -n "$prefetch_torch_ok" ]] && break
done

if [[ -z "$prefetch_torch_ok" ]]; then
  echo "ERROR: Could not prefetch torch/vision/audio (tried: $TORCH_CUDA_CANDIDATES) for platforms: $PLATS" >&2
  exit 3
fi

# -------------------------------------------------------
# 1) Helper: download for one platform, but don't hard-fail
# -------------------------------------------------------
download_for_platform() {
  local plat="$1"
  echo "==> Downloading for platform: ${plat}"
  set +e
  pip download -r "$REQ_EFFECTIVE" \
    --only-binary=:all: --implementation cp \
    --platform "$plat" --python-version "$PYVER" --abi "cp${PYVER}" \
    --dest "$WHEEL_DIR" --find-links "$WHEEL_DIR"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "WARN: Some wheels for '$plat' were unavailable. Continuing; next pass may fetch the rest."
  fi
}

# ------------------------------------------------
# 2) Multi-pass download across both Linux targets
# ------------------------------------------------
for plat in $PLATS; do
  download_for_platform "$plat"
done

# --- Force specific wheels that are known to exist but may be skipped by resolver ---
: "${FORCE_WHEELS:=tiktoken==0.8.0}"
if [[ -n "$FORCE_WHEELS" ]]; then
  IFS=',' read -ra __FORCE <<< "$FORCE_WHEELS"
  for plat in $PLATS; do
    for spec in "${__FORCE[@]}"; do
      spec="$(echo "$spec" | xargs)"; [[ -z "$spec" ]] && continue
      echo "==> Forcing download: $spec ($plat)"
      set +e
      pip download "$spec" \
  --only-binary=:all: \
  --implementation cp \
        --platform "$plat" \
  --python-version "$PYVER" \
  --abi "cp${PYVER}" \
  --dest "$WHEEL_DIR"
      rc=$?
      set -e
      if [[ $rc -ne 0 ]]; then
        echo "WARN: Force-download failed for $spec on $plat; continuing."
      fi
    done
  done
fi

# -----------------------------------------------------------------
# 3) Optional: adjust onnxruntime-gpu channel (CUDA 11 vs default)
# -----------------------------------------------------------------
if grep -Eiq '^[[:space:]]*onnxruntime-gpu([[:space:]]|==|$)' "$REQ_FILE"; then
  ORT_PKG_LINE="$(grep -Eih '^[[:space:]]*onnxruntime-gpu([[:space:]]|==|$).*' "$REQ_FILE" | head -n1 | sed 's/^[[:space:]]*//')"
  if [[ -n "${ORT_CUDA:-}" || -n "${ORT_INDEX_URL:-}" ]]; then
    echo "==> Adjusting onnxruntime-gpu via alternate index (CUDA ${ORT_CUDA:-12.x} or custom)."
    find "$WHEEL_DIR" -maxdepth 1 -type f -name 'onnxruntime_gpu-*.whl' -delete || true
    if [[ -n "${ORT_INDEX_URL:-}" ]]; then
      IDX="$ORT_INDEX_URL"
    elif [[ "${ORT_CUDA:-}" == "11" ]]; then
      IDX="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/"
    else
      IDX=""
    fi
    if [[ -n "$IDX" ]]; then
      pip download $ORT_PKG_LINE \
        --only-binary=:all: --implementation cp \
        --platform "$(echo "$PLATS" | awk '{print $1}')" \
        --python-version "$PYVER" --abi "cp${PYVER}" \
        --dest "$WHEEL_DIR" --index-url "$IDX"
    fi
  fi
fi

# Guardrail: no sdists
if find "$WHEEL_DIR" -maxdepth 1 -type f ! -name '*.whl' | grep -q . ; then
  echo "ERROR: Non-wheel distributions detected in $WHEEL_DIR. Refusing to proceed." >&2
  find "$WHEEL_DIR" -maxdepth 1 -type f ! -name '*.whl' -print >&2
  exit 4
fi

# Dry-run resolver (simulated offline) — only when host matches target (Linux x86_64, cp${PYVER})
  echo "==> Verifying resolution with pip (dry-run)..."
HOST_OS="$(uname -s)" || HOST_OS="unknown"
HOST_ARCH="$(uname -m)" || HOST_ARCH="unknown"
HOST_PYTAG="$(python -c 'import sys;print(f"cp{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null || echo unknown)"
if [[ "$HOST_OS" == "Linux" && "$HOST_ARCH" == "x86_64" && "$HOST_PYTAG" == "cp${PYVER}" ]]; then
  set +e
  pip install --no-index --find-links="$WHEEL_DIR" -r "$REQ_EFFECTIVE" \
    --dry-run --report ./wheelhouse_resolve.json >/dev/null
  if [[ $? -ne 0 ]]; then
    echo "WARNING: Dry-run reported issues. Inspect wheelhouse_resolve.json for details."
  fi
  set -e
else
  echo "INFO: Skipping dry-run on host ($HOST_OS/$HOST_ARCH,$HOST_PYTAG); target is Linux/x86_64/cp${PYVER}."
fi

echo "==> Done. Wheels are in: $WHEEL_DIR"
