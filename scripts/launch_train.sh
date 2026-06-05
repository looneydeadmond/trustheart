#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment/main.yaml}"
python -m trustheart.console.run train --config "${CONFIG}"
