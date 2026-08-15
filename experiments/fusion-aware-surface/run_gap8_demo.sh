#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv-gap8-demo"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --disable-pip-version-check --quiet \
  -r "${SCRIPT_DIR}/demo_requirements.txt"
"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/demo_gap8.py" \
  --out "${1:-${SCRIPT_DIR}/gap8-demo-output}"
