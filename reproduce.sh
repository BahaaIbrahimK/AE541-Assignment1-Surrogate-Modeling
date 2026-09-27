#!/usr/bin/env bash
# Rebuild everything from scratch on Linux or macOS:  bash reproduce.sh
set -euo pipefail

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# regenerates every figure and table in code/outputs (about 2 minutes)
.venv/bin/python code/202510630_Bahaa_Karawia.py

# the reports need a LaTeX distribution; skip them quietly if there is none
if command -v latexmk > /dev/null; then
    (cd report_AIAA && latexmk -pdf 202510630_Bahaa_Karawia.tex \
        && cp 202510630_Bahaa_Karawia.pdf ../submission/)
    (cd report_normal && latexmk -pdf 202510630_Bahaa_Karawia_KFUPM.tex)
else
    echo "latexmk not found - results regenerated, reports not rebuilt."
fi
