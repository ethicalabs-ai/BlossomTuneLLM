#!/bin/bash

uv run flower-supernode --superlink="${SUPERLINK}" --node-config="data-path='${NODE_DATA_PATH}'"
