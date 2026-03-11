# EDGAR Explorer

Simple Streamlit app to extract company filings and financial facts from SEC EDGAR APIs.

## Features

- Ticker to CIK lookup from SEC official mapping
- Recent filings table from `submissions`
- Core financial metrics from companyfacts (XBRL)
- Account-distribution analysis from a local financial cache (distribution_cache.csv.gz)
- CSV export for filings and metrics

## Local run

### Fastest preview in WSL / Bash

If you already have the Linux virtual environment in this repo, you do not need to activate it again. Run:

```bash
cd /mnt/c/Users/DELL/Desktop/MartinGit/edgar
.venv/bin/python -m streamlit run app.py
```

Open the local preview in your browser at `http://localhost:8501`.

If `.venv` does not exist yet, create it once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

If `python3 -m venv .venv` fails in WSL/Debian-based Linux because `ensurepip` is missing, install the required packages first:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

### Fastest preview in Windows PowerShell

This repository currently contains a Linux/WSL `.venv`. The PowerShell helper first looks for a real Windows Python; if it does not find one, or only finds the Microsoft Store alias, it automatically falls back to the existing WSL environment. When Windows Python is available, it creates and reuses `.venv-win`:

```powershell
cd "C:\Users\DELL\Desktop\MartinGit\edgar"
powershell -ExecutionPolicy Bypass -File .\run_preview.ps1
```

Optional flags:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_preview.ps1 -RefreshDeps
powershell -ExecutionPolicy Bypass -File .\run_preview.ps1 -Port 8502
```

Notes:

- `RStudio` only helps as a terminal host; the app still runs through Streamlit.
- Streamlit Community Cloud is still useful for the deployed version after commit/push, but the fastest edit loop is local preview.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (`main` branch).
2. Open `https://streamlit.io/cloud`.
3. Click `New app`.
4. Select repo: `mlozanoqf/edgar`.
5. Main file path: `app.py`.
6. Deploy.

## SEC SIC Industry Cache

- The app's `By industry` mode reads `sec_sic_lookup.csv`.
- A GitHub Action (`.github/workflows/update_sec_sic_cache.yml`) updates that file automatically once per month and can also be run manually from the Actions tab.
- Set a repository secret named `SEC_USER_AGENT` (example: `EDGAR Explorer (your-email@domain.com)`) so the workflow can download SEC bulk metadata.

## Distribution Cache

- The `Account Distribution` module reads `distribution_cache.csv.gz`.
- The cache is built by `build_distribution_cache.py` and now targets annual `Simplified Statements` rows for `FY 2025` within the same SEC SIC operating-company universe used by industry mode.
- A GitHub Action (`.github/workflows/update_distribution_cache.yml`) updates that file automatically once per month and can also be run manually from the Actions tab.
- You can also build it locally with:

```bash
cd /mnt/c/Users/DELL/Desktop/MartinGit/edgar
SEC_USER_AGENT="EDGAR Explorer (your-email@domain.com)" python build_distribution_cache.py
```

- Optional environment variables:
  - `DISTRIBUTION_TARGET_FY` (default `2025`)
  - `DISTRIBUTION_TICKER_LIMIT` (useful for smoke tests)
  - `SEC_REQUEST_SLEEP_SECONDS` (default `0.12`)`r`n  - `DISTRIBUTION_TICKER_MAP_MAX_ATTEMPTS` (default `5`)`r`n  - `DISTRIBUTION_TICKER_MAP_RETRY_SLEEP_SECONDS` (default `5`)

- While the builder is running locally, it can checkpoint progress in temporary files:
  - `distribution_cache.partial.csv`
  - `distribution_cache.checkpoint.csv`
  - `distribution_cache.state.json`
- Those files are only for incremental / resumable local builds and are ignored by Git. A successful build still writes the final cache to `distribution_cache.csv.gz`.
## Continuity and Maintenance

### What persists if you close Codex or shut down the computer

- Saved file edits remain in the local repository on disk.
- Any running preview, validation job, or cache-build process stops if the computer shuts down or the active session is closed.
- `SESSION_NOTES.md` is the canonical continuity log for resuming work in a later Codex session.

### What the caches do

- `sec_sic_lookup.csv` is a metadata cache used to support `By industry` selection.
- `distribution_cache.csv.gz` is a derived financial cache used only by the `Account Distribution` module.
- The SEC remains the source of truth. The distribution cache is a precomputed analytical artifact built from SEC data with the project's strict extraction logic.
- `Cross-Section`, `Time Series`, `Panel`, and `Debt Distribution` still use live SEC extraction.
- `Account Distribution` reads the local distribution cache instead of calling the SEC API on every interaction.

### What is automated

- `.github/workflows/update_sec_sic_cache.yml` can refresh `sec_sic_lookup.csv` automatically in GitHub once per month.
- `.github/workflows/update_distribution_cache.yml` can refresh `distribution_cache.csv.gz` automatically in GitHub once per month.
- Those workflows can also be triggered manually from the GitHub Actions tab.
- In the deployed app, updated cache files become available after the workflow commits them to the repository and the app redeploys.

### What still requires attention

- Local preview uses the local files in this repo. If GitHub updates a cache file remotely, the local preview will not see that update until the local repo is refreshed.
- The distribution-cache builder now supports incremental / resumable execution through local checkpoint files. If you want to discard partial progress and start over, set `DISTRIBUTION_RESET_PROGRESS=1`.
- The GitHub workflow still runs on ephemeral runners, so a failed remote run is retried from scratch on the next workflow execution.
- If a GitHub workflow fails, the last committed cache file remains in use until the workflow is rerun successfully.
## SEC usage notes

- Use a clear `User-Agent` with contact info (email) in requests.
- Respect SEC fair access/rate limits.

Reference:
- https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.sec.gov/os/accessing-edgar-data

## Validation

### WSL / Bash

```bash
cd /mnt/c/Users/DELL/Desktop/MartinGit/edgar
bash run_validation.sh
```

### Windows PowerShell

```powershell
cd "C:\Users\DELL\Desktop\MartinGit\edgar"
powershell -ExecutionPolicy Bypass -File .\run_validation.ps1
```

This runs:

- `python -m py_compile` on the app, both cache builders, and the test suite
- `python -m unittest discover -s tests -p "test_*.py" -v`










