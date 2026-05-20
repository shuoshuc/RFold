#!/bin/bash
set -e

# run_astra.sh — invoke the rfold-astra container for one fluid-model
# astra-sim job. Drops the upstream start-fluid.sh's host mounts for
# examples/, tools/, and configs/ because the rfold-astra image already
# bakes them in.
#
# Usage: run_astra.sh <JOB_SHAPE> --input-dir DIR --output-dir DIR
#   JOB_SHAPE: torus shape NxNx... (e.g. 2x2x1)

usage() {
    echo "Usage: $0 <JOB_SHAPE> --input-dir DIR --output-dir DIR" >&2
    exit 2
}

if [[ $# -lt 1 ]]; then usage; fi
JOB_SHAPE="$1"; shift

INPUT_DIR=""
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)  INPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "run_astra.sh: unknown arg: $1" >&2; usage ;;
    esac
done

if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]]; then usage; fi

if [[ ! "${JOB_SHAPE}" =~ ^[0-9]+(x[0-9]+)*$ ]]; then
    echo "run_astra.sh: JOB_SHAPE '${JOB_SHAPE}' must be NxNx... (e.g. 2x2x1)" >&2
    exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "run_astra.sh: input dir '${INPUT_DIR}' missing" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"
INPUT_DIR=$(realpath "${INPUT_DIR}")
OUTPUT_DIR=$(realpath "${OUTPUT_DIR}")

sudo docker run --rm --ipc=host --ulimit nofile=65536:65536 \
    -v "${INPUT_DIR}":/app/inputs:ro \
    -v "${OUTPUT_DIR}":/app/output \
    rfold-astra \
    bash /app/astra-sim-artifacts/examples/fluid-model/run.sh "${JOB_SHAPE}"
