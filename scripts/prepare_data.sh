#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data/mitdb}"
mkdir -p "${DEST}"
python -c "import wfdb; wfdb.dl_database('mitdb', '${DEST}')"
