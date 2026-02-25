# EDGAR Explorer

Simple Streamlit app to extract company filings and financial facts from SEC EDGAR APIs.

## Features

- Ticker to CIK lookup from SEC official mapping
- Recent filings table from `submissions`
- Core financial metrics from `companyfacts` (XBRL)
- CSV export for filings and metrics

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (`main` branch).
2. Open `https://streamlit.io/cloud`.
3. Click `New app`.
4. Select repo: `mlozanoqf/edgar`.
5. Main file path: `app.py`.
6. Deploy.

## SEC usage notes

- Use a clear `User-Agent` with contact info (email) in requests.
- Respect SEC fair access/rate limits.

Reference:
- https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.sec.gov/os/accessing-edgar-data
