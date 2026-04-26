#!/bin/sh
set -eu

: "${HF_HOME:=/models/huggingface}"

mkdir -p "$HF_HOME"

if ! touch "$HF_HOME/.write-test" 2>/dev/null; then
    echo "ERROR: Hugging Face cache directory is not writable: $HF_HOME" >&2
    echo "Check the ./models/* bind mount directories and permissions on the NAS." >&2
    exit 1
fi

rm -f "$HF_HOME/.write-test"

exec "$@"
