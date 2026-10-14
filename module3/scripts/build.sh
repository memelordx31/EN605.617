#!/usr/bin/env bash
# scripts/build.sh - reads ../assignment_config.yaml and builds the CUDA source it points to.
# Lives in scripts/ ; the project root (config.yaml, Makefile, src/) is one level up.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${1:-$ROOT_DIR/assignment_config.yaml}"
GET="python3 $ROOT_DIR/get_config.py $CONFIG_FILE"

SRC_DIR=$($GET project.src_dir)
SRC_FILE=$($GET project.src_file)
BUILD_DIR=$($GET project.build_dir)
BIN_NAME=$($GET project.binary_name)
NVCC=$($GET build.compiler)
ARCH=$($GET build.arch)
FLAGS=$($GET build.flags)

SRC_PATH="$ROOT_DIR/$SRC_DIR/$SRC_FILE"
if [[ ! -f "$SRC_PATH" ]]; then
    echo "Error: source file not found at $SRC_PATH (check project.src_dir / project.src_file in $CONFIG_FILE)" >&2
    exit 1
fi

echo "Building: $SRC_PATH"
echo "  compiler : $NVCC"
echo "  arch     : $ARCH"
echo "  flags    : $FLAGS"
echo "  output   : $BUILD_DIR/$BIN_NAME"

make -C "$ROOT_DIR" \
    SRC_DIR="$SRC_DIR" \
    SRC_FILE="$SRC_FILE" \
    BUILD_DIR="$BUILD_DIR" \
    BIN_NAME="$BIN_NAME" \
    NVCC="$NVCC" \
    ARCH="$ARCH" \
    FLAGS="$FLAGS"

echo "Build complete: $ROOT_DIR/$BUILD_DIR/$BIN_NAME"
