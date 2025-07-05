#!/bin/bash

set -e

echo "Env vars:"
env | grep -v -E "SECRET|KEY"

if [ "${1}" = "superlink" ]; then
    echo "Running superlink..."
    uv run flower-superlink --insecure
elif [ "${1}" = "supernode" ]; then
    echo "Running supernode..."
    uv run flower-supernode --insecure --superlink="${SUPERLINK}" --node-config="data-path='${NODE_DATA_PATH}' partition-id=${PARTITION_ID} num-partitions=${NUM_PARTITIONS}"
else
    exec "$@"
fi
