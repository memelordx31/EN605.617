#!/usr/bin/env bash
# scripts/run.sh - reads ../config.yaml and runs the built binary with its run parameters.
# Lives in scripts/ ; the project root (config.yaml, build/) is one level up.
# Any CLI args you pass override the corresponding config.yaml run.* values:
#   ./scripts/run.sh                  -> uses totalThreads/blockSize/iterations from config.yaml
#   ./scripts/run.sh 512 256          -> overrides totalThreads and blockSize only
#   ./scripts/run.sh 512 256 5        -> overrides all three
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/assignment_config.yaml"
GET="python3 $ROOT_DIR/get_config.py $CONFIG_FILE"

BUILD_DIR=$($GET project.build_dir)
BIN_NAME=$($GET project.binary_name)
BIN_PATH="$ROOT_DIR/$BUILD_DIR/$BIN_NAME"

TOTAL_THREADS=${1:-$($GET run.totalThreads)}
BLOCK_SIZE=${2:-$($GET run.blockSize)}
ITERATIONS=${3:-$($GET run.iterations)}

if [[ ! -x "$BIN_PATH" ]]; then
    echo "Error: binary not found at $BIN_PATH. Run ./scripts/build.sh first." >&2
    exit 1
fi

echo "Running: $BIN_PATH $TOTAL_THREADS $BLOCK_SIZE $ITERATIONS"
"$BIN_PATH" "$TOTAL_THREADS" "$BLOCK_SIZE" "$ITERATIONS"
