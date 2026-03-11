#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python -m py_compile app.py build_sec_sic_cache.py build_distribution_cache.py tests/test_extraction_logic.py
.venv/bin/python -m unittest discover -v -s tests -p "test_*.py"
