import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests


BULK_SUBMISSIONS_ZIP_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
OUTPUT_PATH = Path(__file__).with_name("sec_sic_lookup.csv")


def download_file(url: str, user_agent: str, target_path: str) -> None:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    with requests.get(url, headers=headers, timeout=120, stream=True) as response:
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> int:
    user_agent = os.getenv("SEC_USER_AGENT") or os.getenv("SEC_CONTACT_EMAIL")
    if not user_agent:
        print("Set SEC_USER_AGENT or SEC_CONTACT_EMAIL before running this script.", file=sys.stderr)
        return 1
    if "@" in user_agent and " " not in user_agent:
        user_agent = f"EDGAR Explorer ({user_agent})"

    temp_path = None
    rows = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            temp_path = tmp.name
        print("Downloading SEC submissions.zip ...")
        download_file(BULK_SUBMISSIONS_ZIP_URL, user_agent, temp_path)
        print("Parsing SIC metadata ...")
        with zipfile.ZipFile(temp_path, "r") as zf:
            for i, name in enumerate(zf.namelist(), start=1):
                if not name.lower().endswith(".json"):
                    continue
                with zf.open(name) as f:
                    try:
                        payload = json.load(f)
                    except Exception:
                        continue
                cik_val = payload.get("cik")
                if cik_val in [None, ""]:
                    continue
                try:
                    cik_int = int(cik_val)
                except Exception:
                    continue
                rows.append(
                    {
                        "cik_str": cik_int,
                        "sic": payload.get("sic"),
                        "sicDescription": payload.get("sicDescription"),
                    }
                )
                if i % 5000 == 0:
                    print(f"Processed {i:,} archive entries ...")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    if not rows:
        print("No SIC rows were extracted.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows).drop_duplicates(subset=["cik_str"], keep="first")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
