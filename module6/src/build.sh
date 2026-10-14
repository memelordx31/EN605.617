#!/usr/bin/env bash
#
# build.sh - compiles auth_streams.cu with nvcc.
#
# Usage: ./build.sh [-O LEVEL] [--arch SM] [--debug] [--output NAME]
#   -O LEVEL       host/device optimization level (default: 2)
#   --arch SM      GPU architecture, e.g. sm_75, sm_86, native (default: sm_75)
#   --debug        add -g -G (device debug info; disables device optimizations)
#   --output NAME  output binary name (default: auth_streams)
#
# Values may also be given as --flag=value. Exits non-zero on failure.

set -uo pipefail

SRC="auth_streams.cu"
OUT="auth_streams"
OPT="2"
ARCH="sm_75"
DEBUG_FLAGS=()

usage() {
    sed -n '4,10p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        -O|--O|--opt)   OPT="${2:?missing value for $1}"; shift 2 ;;
        -O*)            OPT="${1#-O}"; shift ;;
        --arch)         ARCH="${2:?missing value for --arch}"; shift 2 ;;
        --arch=*)       ARCH="${1#*=}"; shift ;;
        --output|-o)    OUT="${2:?missing value for $1}"; shift 2 ;;
        --output=*)     OUT="${1#*=}"; shift ;;
        --debug)
            # tolerate "--debug true/false" as produced by the YAML dict form
            if [ "${2:-}" = "false" ]; then shift 2; continue; fi
            DEBUG_FLAGS=(-g -G)
            if [ "${2:-}" = "true" ]; then shift 2; else shift; fi ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "build.sh: unknown argument '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")" || exit 1

if ! command -v nvcc >/dev/null 2>&1; then
    echo "build.sh: nvcc not found on PATH (is the CUDA toolkit installed/loaded?)" >&2
    exit 1
fi
if [ ! -f "$SRC" ]; then
    echo "build.sh: source file '$SRC' not found in $(pwd)" >&2
    exit 1
fi

CMD=(nvcc "-O${OPT}" "-arch=${ARCH}" "${DEBUG_FLAGS[@]}" "$SRC" -o "$OUT")
echo "nvcc version: $(nvcc --version | tail -n 1)"
echo "Compiling: ${CMD[*]}"
"${CMD[@]}" || { echo "build.sh: compilation failed" >&2; exit 1; }

echo "Build succeeded: $(pwd)/${OUT}"
