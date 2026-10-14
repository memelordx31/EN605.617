#!/usr/bin/env bash
#
# run.sh - runs auth_scan_memtypes against the LANL cyber1 auth log.
#
# Usage: ./run.sh [--file PATH] [--elements N] [--threads N] [--blocks N] [--runs N]
#
# All flags are forwarded to the binary (see auth_scan_memtypes.cu).
# Values may also be given as --flag=value.
#
# Dataset resolution (when --file is not given, default ../auth.txt):
#   1. ../auth.txt is used directly if it exists.
#   2. Otherwise, if ../auth.txt.gz exists (the form LANL distributes),
#      the first --elements lines are extracted to auth_head.txt and used.
#   3. Otherwise a synthetic sample in the LANL auth.txt format is generated
#      (sample_auth.txt) so the program can still be exercised. A warning is
#      printed; timings on synthetic data are not representative of the
#      real dataset. Set AUTH_NO_SAMPLE=1 to make a missing dataset an error.

set -uo pipefail

BIN="./auth_scan_memtypes"
DATA_FILE=""
ELEMENTS="131072"
FWD=()

while [ $# -gt 0 ]; do
    case "$1" in
        --file)       DATA_FILE="${2:?missing value for --file}"; shift 2 ;;
        --file=*)     DATA_FILE="${1#*=}"; shift ;;
        --elements)   ELEMENTS="${2:?missing value for --elements}"; shift 2 ;;
        --elements=*) ELEMENTS="${1#*=}"; shift ;;
        --threads|--blocks|--runs)
                      FWD+=("$1" "${2:?missing value for $1}"); shift 2 ;;
        --threads=*|--blocks=*|--runs=*)
                      FWD+=("${1%%=*}" "${1#*=}"); shift ;;
        -h|--help)    sed -n '4,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "run.sh: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")" || exit 1

if [ ! -x "$BIN" ]; then
    echo "run.sh: '$BIN' not found or not executable in $(pwd) - run build.sh first" >&2
    exit 1
fi

generate_sample() {
    # $1 = output path, $2 = number of lines
    # LANL auth.txt format:
    # time,src_user@domain,dst_user@domain,src_computer,dst_computer,
    # auth_type,logon_type,auth_orientation,success/failure
    awk -v n="$2" 'BEGIN {
        srand(617);
        split("Kerberos,NTLM,Negotiate,?", at, ",");
        split("Network,Interactive,Batch,Service,Unlock,RemoteInteractive", lt, ",");
        split("LogOn,LogOff,TGS,TGT,AuthMap", ori, ",");
        for (i = 1; i <= n; i++) {
            u  = "U" int(rand() * 12000) "@DOM1";
            sc = "C" int(rand() * 18000);
            dc = (rand() < 0.3) ? sc : "C" int(rand() * 18000);
            printf "%d,%s,%s,%s,%s,%s,%s,%s,%s\n", int(i / 20) + 1, u, u, sc, dc,
                   at[int(rand() * 4) + 1], lt[int(rand() * 6) + 1],
                   ori[int(rand() * 5) + 1], (rand() < 0.02) ? "Fail" : "Success";
        }
    }' > "$1"
}

if [ -z "$DATA_FILE" ]; then
    if [ -f ../auth.txt ]; then
        DATA_FILE="../auth.txt"
    elif [ -f ../auth.txt.gz ]; then
        DATA_FILE="auth_head.txt"
        echo "Extracting first ${ELEMENTS} records from ../auth.txt.gz -> ${DATA_FILE}"
        gzip -dc ../auth.txt.gz | head -n "$ELEMENTS" > "$DATA_FILE"
    elif [ "${AUTH_NO_SAMPLE:-0}" = "1" ]; then
        echo "run.sh: ../auth.txt not found and AUTH_NO_SAMPLE=1" >&2
        exit 1
    else
        DATA_FILE="sample_auth.txt"
        echo "WARNING: LANL dataset not found at ../auth.txt or ../auth.txt.gz"
        echo "WARNING: (https://csr.lanl.gov/data/cyber1/). Using a synthetic"
        echo "WARNING: sample in the same format: ${DATA_FILE} (${ELEMENTS} records)."
        generate_sample "$DATA_FILE" "$ELEMENTS"
    fi
elif [ ! -f "$DATA_FILE" ]; then
    echo "run.sh: data file '$DATA_FILE' does not exist" >&2
    exit 1
fi

CMD=("$BIN" --file "$DATA_FILE" --elements "$ELEMENTS" "${FWD[@]}")
echo "Running: ${CMD[*]}"
"${CMD[@]}"
