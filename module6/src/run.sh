#!/usr/bin/env bash
#
# run.sh - runs auth_streams on the LANL cyber1 auth + red-team logs.
#
# Usage: ./run.sh [--fileA PATH] [--fileB PATH] [--sizeA N] [--sizeB N]
#                 [--threads N] [--blocks N]
#
# All flags are forwarded to the binary (see auth_streams.cu).
# Values may also be given as --flag=value.
#
# Dataset resolution, done separately for A (auth.txt) and B (redteam.txt)
# when --fileA / --fileB is not given:
#   1. ../<name>.txt is used directly if it exists.
#   2. Otherwise, if ../<name>.txt.gz exists (the form LANL distributes),
#      the first sizeA / sizeB lines are extracted to <name>_head.txt.
#   3. Otherwise a synthetic sample in the matching LANL format is
#      generated (sample_<name>.txt) so the program can still be exercised.
#      A warning is printed; results on synthetic data are not
#      representative. Set AUTH_NO_SAMPLE=1 to make a missing dataset an error.

set -uo pipefail

BIN="./auth_streams"
FILE_A=""
FILE_B=""
SIZE_A="1048576"   # program defaults
SIZE_B="65536"
FWD=()

while [ $# -gt 0 ]; do
    case "$1" in
        --fileA)   FILE_A="${2:?missing value for --fileA}"; shift 2 ;;
        --fileA=*) FILE_A="${1#*=}"; shift ;;
        --fileB)   FILE_B="${2:?missing value for --fileB}"; shift 2 ;;
        --fileB=*) FILE_B="${1#*=}"; shift ;;
        --sizeA)   SIZE_A="${2:?missing value for --sizeA}"; shift 2 ;;
        --sizeA=*) SIZE_A="${1#*=}"; shift ;;
        --sizeB)   SIZE_B="${2:?missing value for --sizeB}"; shift 2 ;;
        --sizeB=*) SIZE_B="${1#*=}"; shift ;;
        --threads|--blocks)
                   FWD+=("$1" "${2:?missing value for $1}"); shift 2 ;;
        --threads=*|--blocks=*)
                   FWD+=("${1%%=*}" "${1#*=}"); shift ;;
        -h|--help) sed -n '4,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "run.sh: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")" || exit 1

if [ ! -x "$BIN" ]; then
    echo "run.sh: '$BIN' not found or not executable in $(pwd) - run build.sh first" >&2
    exit 1
fi

# auth.txt format:
#   time,src_user@domain,dst_user@domain,src_computer,dst_computer,
#   auth_type,logon_type,auth_orientation,success/failure
generate_auth_sample() {
    awk -v n="$2" -v seed="$RANDOM" 'BEGIN {
        srand(seed);
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

# redteam.txt format:
#   time,user@domain,source_computer,destination_computer
generate_redteam_sample() {
    awk -v n="$2" -v seed="$RANDOM" 'BEGIN {
        srand(seed);
        t = 150000;
        for (i = 1; i <= n; i++) {
            t += int(rand() * 3000);
            printf "%d,U%d@DOM1,C%d,C%d\n", t, int(rand() * 1000),
                   int(rand() * 18000), int(rand() * 18000);
        }
    }' > "$1"
}

# resolve_dataset NAME SIZE GENERATOR -> sets RESOLVED to the path to use
resolve_dataset() {
    local name="$1" size="$2" gen="$3"
    RESOLVED=""
    if [ -f "../${name}.txt" ]; then
        RESOLVED="../${name}.txt"
    elif [ -f "../${name}.txt.gz" ]; then
        RESOLVED="${name}_head.txt"
        echo "Extracting first ${size} records from ../${name}.txt.gz -> ${RESOLVED}"
        gzip -dc "../${name}.txt.gz" | head -n "$size" > "$RESOLVED"
    elif [ "${AUTH_NO_SAMPLE:-0}" = "1" ]; then
        echo "run.sh: ../${name}.txt not found and AUTH_NO_SAMPLE=1" >&2
        return 1
    else
        RESOLVED="sample_${name}.txt"
        echo "WARNING: LANL ${name} dataset not found at ../${name}.txt or ../${name}.txt.gz"
        echo "WARNING: (https://csr.lanl.gov/data/cyber1/). Using a synthetic sample in"
        echo "WARNING: the same format: ${RESOLVED} (${size} records)."
        "$gen" "$RESOLVED" "$size"
    fi
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "run.sh: data file '$1' does not exist" >&2
        exit 1
    fi
}

if [ -z "$FILE_A" ]; then
    resolve_dataset auth "$SIZE_A" generate_auth_sample || exit 1
    FILE_A="$RESOLVED"
else
    check_file "$FILE_A"
fi

if [ -z "$FILE_B" ]; then
    resolve_dataset redteam "$SIZE_B" generate_redteam_sample || exit 1
    FILE_B="$RESOLVED"
else
    check_file "$FILE_B"
fi

CMD=("$BIN" --fileA "$FILE_A" --fileB "$FILE_B" --sizeA "$SIZE_A" --sizeB "$SIZE_B" "${FWD[@]}")
echo "Running: ${CMD[*]}"
"${CMD[@]}"
