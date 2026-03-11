$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$drive = $repoRoot.Substring(0, 1).ToLower()
$rest = $repoRoot.Substring(2) -replace "\\", "/"
$wslRepo = "/mnt/$drive$rest"

Write-Host "Running validation in WSL for $wslRepo"
wsl bash -lc "cd '$wslRepo' && .venv/bin/python -m py_compile app.py build_sec_sic_cache.py build_distribution_cache.py tests/test_extraction_logic.py && .venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'"
