#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src

rm -rf test_coverage

set +e
uv run pytest test/ -v \
  --cov=src \
  --cov-report=html:test_coverage \
  --cov-report=term-missing
pytest_status=$?
set -e

if [[ -f test_coverage/index.html ]]; then
  uv run python -c "
import webbrowser
from pathlib import Path
webbrowser.open(Path('test_coverage/index.html').resolve().as_uri())
"
fi

exit "$pytest_status"
