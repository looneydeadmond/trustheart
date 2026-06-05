#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment/main.yaml}"
CKPT="${2:-runs/main/model.pt}"
python -m trustheart.console.run evaluate --config "${CONFIG}" --checkpoint "${CKPT}"
python -m trustheart.console.run detect --config "${CONFIG}" --checkpoint "${CKPT}"
python -m trustheart.console.run explain --config "${CONFIG}" --checkpoint "${CKPT}"
