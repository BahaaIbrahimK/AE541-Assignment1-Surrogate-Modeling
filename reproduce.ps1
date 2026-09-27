# Rebuild everything from scratch on Windows (PowerShell):
#   powershell -ExecutionPolicy Bypass -File reproduce.ps1
$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt

# regenerates every figure and table in code/outputs (about 2 minutes)
.\.venv\Scripts\python code\202510630_Bahaa_Karawia.py

# the reports need a LaTeX distribution; skip them quietly if there is none
if (Get-Command latexmk -ErrorAction SilentlyContinue) {
    Push-Location report_AIAA
    latexmk -pdf 202510630_Bahaa_Karawia.tex
    Copy-Item 202510630_Bahaa_Karawia.pdf ..\submission\ -Force
    Pop-Location
    Push-Location report_normal
    latexmk -pdf 202510630_Bahaa_Karawia_KFUPM.tex
    Pop-Location
} else {
    Write-Host "latexmk not found - results regenerated, reports not rebuilt."
}
