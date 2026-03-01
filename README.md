# EDGAR Explorer

Simple Streamlit app to extract company filings and financial facts from SEC EDGAR APIs.

## Features

- Ticker to CIK lookup from SEC official mapping
- Recent filings table from `submissions`
- Core financial metrics from `companyfacts` (XBRL)
- CSV export for filings and metrics

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local preview in your browser at `http://localhost:8501`.

If `python3 -m venv .venv` fails in WSL/Debian-based Linux because `ensurepip` is missing, install the required packages first:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

If you prefer to run the app from Windows PowerShell instead of WSL, install Python for Windows first and then use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

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

## SEC usage notes

- Use a clear `User-Agent` with contact info (email) in requests.
- Respect SEC fair access/rate limits.

Reference:
- https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.sec.gov/os/accessing-edgar-data
