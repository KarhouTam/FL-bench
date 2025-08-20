#!/usr/bin/env bash
# Purpose: Download OrganAMNIST from Zenodo and convert to xdata.npy/ydata.npy
#          Then, run generate_data.py to get a partition.
# Faster tips: If a remote server is used to run your code,
#       you'd better uploading the npz file manually, and then run this.

[ -n "${BASH_VERSION:-}" ] || { echo "Please run with: bash $0"; exit 1; }

set -Eeuo pipefail
IFS=$'\n\t'
trap 'printf "ERROR: line %d exited with status %d\n" "$LINENO" "$?" >&2' ERR

# ---- Config -----------------------------------------------------------------
readonly URL="https://zenodo.org/records/10519652/files/organamnist.npz?download=1"
readonly FILENAME="organamnist.npz"

# Resolve directories (default to ../medmnistA/raw; allow override via RAW_DIR)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR_DEFAULT="${SCRIPT_DIR}/../medmnistA/raw"
RAW_DIR="${RAW_DIR:-$RAW_DIR_DEFAULT}"

mkdir -p -- "$RAW_DIR"
cd -- "$RAW_DIR"

# ---- Dependency checks ------------------------------------------------------
for bin in python3; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    printf "Missing dependency: %s\n" "$bin" >&2
    exit 1
  fi
done

# ---- Download ---------------------------------------------------------------
if [[ ! -s "$FILENAME" ]]; then
  printf ">>> Downloading %s to %s\n" "$FILENAME" "$RAW_DIR"
  if command -v wget >/dev/null 2>&1; then
    wget -c --tries=5 -O "$FILENAME" "$URL"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL -C - --retry 5 --retry-delay 5 -o "$FILENAME" "$URL"
  else
    printf "Neither curl nor wget is available.\n" >&2
    exit 1
  fi
else
  printf ">>> Found existing %s, skip downloading.\n" "$FILENAME"
fi

# ---- Preprocess: NPZ -> xdata.npy / ydata.npy -------------------------------
python3 - <<'PY'
import os, numpy as np, sys
raw_dir = os.getcwd()
npz_path = os.path.join(raw_dir, "organamnist.npz")
print(f">>> Loading {npz_path}")

try:
    data = np.load(npz_path)
except Exception as e:
    print(f"Failed to load NPZ: {e}", file=sys.stderr)
    raise

# NpzFile behaves like a dict but does not implement .get(); use keys in .files
keys = set(list(getattr(data, "files", [])))
def pick(k):
    return data[k] if k in keys else None

x_train, y_train = pick("train_images"), pick("train_labels")
x_val,   y_val   = pick("val_images"),   pick("val_labels")
x_test,  y_test  = pick("test_images"),  pick("test_labels")

def cat_nonempty(*arrs):
    arrs = [a for a in arrs if a is not None and hasattr(a, "size") and a.size > 0]
    return np.concatenate(arrs, axis=0) if arrs else np.empty((0,), dtype=np.uint8)

x_all = cat_nonempty(x_train, x_val, x_test)
y_all = cat_nonempty(y_train, y_val, y_test)

# Squeeze grayscale channel if present, e.g., (N,H,W,1) -> (N,H,W)
if x_all.ndim == 4 and x_all.shape[-1] == 1:
    x_all = x_all.squeeze(-1)

# Labels: squeeze/flatten, make zero-based when appropriate, cast to int64
y_all = y_all.squeeze()
if y_all.ndim > 1:
    y_all = y_all.reshape(-1)
if y_all.size and y_all.min() >= 1 and y_all.max() <= 1000:
    y_all = y_all - 1
y_all = y_all.astype(np.int64, copy=False)

# Pixels to uint8 in [0,255]
if x_all.dtype != np.uint8:
    x_tmp = x_all.astype(np.float32, copy=False)
    if x_tmp.size and x_tmp.max() <= 1.0:
        x_tmp = x_tmp * 255.0
    x_all = np.clip(x_tmp, 0, 255).astype(np.uint8, copy=False)

np.save(os.path.join(raw_dir, "xdata.npy"), x_all)
np.save(os.path.join(raw_dir, "ydata.npy"), y_all)
print(f">>> Saved: xdata.npy {x_all.shape} {x_all.dtype}, ydata.npy {y_all.shape} {y_all.dtype}")
PY

printf "Done. Output dir: %s\n" "$RAW_DIR"


# NOTICE: The script below is the previous downloading approach.
#         But could lead to some issues.
#         (Refer to: https://github.com/microsoft/PersonalizedFL/issues/6 to see the details. )


##!/bin/bash
#
#if [ ! -d "../medmnistA/raw" ]; then
#    mkdir -p ../medmnistA/raw
#fi
#
#cd ../medmnistA/raw
#
#wget https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnistA.tar.gz
#
#tar -xzvf medmnistA.tar.gz
#
#mv medmnistA/* ./
#
#rm -rf medmnistA
#
#cd ../..
