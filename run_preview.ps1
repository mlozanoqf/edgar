param(
    [int]$Port = 8501,
    [switch]$RefreshDeps
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPath = Join-Path $repoRoot ".venv-win"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

function Get-UsableWindowsPython {
    $candidates = @()

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $candidates += [pscustomobject]@{ Launcher = $pythonCmd.Source; Prefix = @() }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $candidates += [pscustomobject]@{ Launcher = $pyCmd.Source; Prefix = @("-3") }
    }

    foreach ($candidate in $candidates) {
        if ($candidate.Launcher -like "*WindowsApps*") {
            continue
        }
        try {
            $null = & $candidate.Launcher @($candidate.Prefix) -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
            continue
        }
    }

    return $null
}

function Get-WslRepoPath([string]$path) {
    $drive = $path.Substring(0, 1).ToLower()
    $rest = $path.Substring(2) -replace "\\", "/"
    return "/mnt/$drive$rest"
}

function Start-WslPreview {
    $wslCmd = Get-Command wsl -ErrorAction SilentlyContinue
    if (-not $wslCmd) {
        throw "No usable Windows Python was found, and WSL is not available. Install Python for Windows or run the app from WSL/Bash."
    }

    $wslRepo = Get-WslRepoPath $repoRoot
    $wslPython = Join-Path $repoRoot ".venv\bin\python"
    if (-not (Test-Path $wslPython)) {
        throw "No usable Windows Python was found, and the WSL virtual environment `.venv/bin/python` is missing."
    }

    if ($RefreshDeps) {
        Write-Host "Refreshing dependencies in the WSL environment at $wslRepo"
        wsl bash -lc "cd '$wslRepo' && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt"
        if ($LASTEXITCODE -ne 0) {
            throw "WSL dependency refresh failed."
        }
    }

    Write-Host "No usable Windows Python detected; using the existing WSL environment instead."
    Write-Host "Starting Streamlit preview at http://localhost:$Port"
    wsl bash -lc "cd '$wslRepo' && .venv/bin/python -m streamlit run app.py --server.port $Port"
}

$windowsPython = Get-UsableWindowsPython
if (-not $windowsPython) {
    Start-WslPreview
    exit $LASTEXITCODE
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating Windows virtual environment at $venvPath"
    & $windowsPython.Launcher @($windowsPython.Prefix) -m venv $venvPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        Write-Host "Windows virtual environment creation failed; falling back to WSL."
        Start-WslPreview
        exit $LASTEXITCODE
    }
    $RefreshDeps = $true
}

if ($RefreshDeps) {
    Write-Host "Installing dependencies in $venvPath"
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
}

Write-Host "Starting Streamlit preview at http://localhost:$Port"
& $venvPython -m streamlit run app.py --server.port $Port
