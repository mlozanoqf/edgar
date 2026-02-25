import os

import pandas as pd
from streamlit.errors import StreamlitSecretNotFoundError
import requests
import streamlit as st

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

FACT_TAGS = {
    "Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "Net Income": ["NetIncomeLoss"],
    "Assets": ["Assets"],
    "Liabilities": ["Liabilities"],
    "Operating Cash Flow": ["NetCashProvidedByUsedInOperatingActivities"],
}


def sec_get(url: str, user_agent: str) -> dict:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3600)
def get_ticker_map(user_agent: str) -> pd.DataFrame:
    raw = requests.get(TICKERS_URL, headers={"User-Agent": user_agent}, timeout=30)
    raw.raise_for_status()
    data = raw.json()
    df = pd.DataFrame.from_dict(data, orient="index")
    df["ticker"] = df["ticker"].str.upper()
    df["cik_str"] = df["cik_str"].astype(int)
    return df


def normalize_cik(cik: int) -> str:
    return str(cik).zfill(10)


def parse_recent_filings(submissions: dict) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    keys = [
        "filingDate",
        "reportDate",
        "form",
        "accessionNumber",
        "primaryDocument",
    ]
    length = len(recent.get("filingDate", []))

    rows = []
    for i in range(length):
        row = {k: (recent.get(k, [None] * length)[i]) for k in keys}
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("filingDate", ascending=False)
    return df


def pick_latest_fact(company_facts: dict, tags: list[str]) -> tuple[pd.DataFrame, str | None]:
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})

    for tag in tags:
        if tag not in us_gaap:
            continue

        units = us_gaap[tag].get("units", {})
        unit_key = "USD" if "USD" in units else next(iter(units), None)
        if not unit_key:
            continue

        df = pd.DataFrame(units[unit_key])
        if df.empty:
            continue

        if "end" in df.columns:
            df = df.sort_values("end", ascending=False)

        cols = [c for c in ["end", "fy", "fp", "form", "val", "filed"] if c in df.columns]
        return df[cols].head(10), tag

    return pd.DataFrame(), None


def default_contact_email() -> str:
    try:
        secret_email = st.secrets.get("SEC_CONTACT_EMAIL", "")
    except StreamlitSecretNotFoundError:
        secret_email = ""

    env_email = os.getenv("SEC_CONTACT_EMAIL", "")
    if secret_email or env_email:
        return secret_email or env_email

    try:
        secret_ua = st.secrets.get("SEC_USER_AGENT", "")
    except StreamlitSecretNotFoundError:
        secret_ua = ""

    env_ua = os.getenv("SEC_USER_AGENT", "")
    fallback = secret_ua or env_ua
    if "@" in fallback:
        return fallback.split()[-1].strip("()")

    return ""


def build_user_agent(contact_email: str) -> str:
    return f"EDGAR Explorer ({contact_email})"


def show_sec_limit_message(exc: requests.HTTPError) -> None:
    response = exc.response
    status_code = response.status_code if response is not None else None

    if status_code == 429:
        st.warning(
            "SEC rate limit reached (HTTP 429). Please wait and try again. "
            "EDGAR allows up to 10 requests per second."
        )
    elif status_code == 403:
        st.warning(
            "SEC rejected the request (HTTP 403). Verify the contact email is valid and try again."
        )
    else:
        st.error(f"SEC request failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="EDGAR Explorer", layout="wide")
    st.title("EDGAR Explorer")
    st.caption("SEC data extractor for filings and core financial facts")

    with st.sidebar:
        st.subheader("Configuration")
        contact_email = st.text_input(
            "Contact email (required)",
            value=default_contact_email(),
            placeholder="your-email@domain.com",
            help="SEC asks for identifiable requests with a contact email.",
        ).strip()
        ticker = st.text_input("Ticker", value="AAPL").upper().strip()
        run = st.button("Load Data", type="primary")

    st.info(
        "Free-tier usage notes: SEC EDGAR requires an identifiable User-Agent and enforces fair-access limits "
        "(up to 10 requests/sec). Streamlit Community Cloud also has resource limits and may sleep after "
        "inactivity. If requests fail, it may be due to these limits, so wait and retry."
    )

    if not run:
        st.stop()

    if not contact_email or "@" not in contact_email:
        st.error("Please enter a valid contact email to continue.")
        st.stop()

    user_agent = build_user_agent(contact_email)

    try:
        tickers = get_ticker_map(user_agent)
    except Exception as exc:
        st.error(f"Failed loading ticker map: {exc}")
        st.stop()

    match = tickers[tickers["ticker"] == ticker]
    if match.empty:
        st.error(f"Ticker '{ticker}' not found in SEC ticker list.")
        st.stop()

    row = match.iloc[0]
    cik = normalize_cik(int(row["cik_str"]))
    title = row.get("title", ticker)

    st.success(f"{ticker} | {title} | CIK {cik}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recent Filings")
        try:
            submissions = sec_get(SUBMISSIONS_URL.format(cik=cik), user_agent)
            filings_df = parse_recent_filings(submissions)
            if filings_df.empty:
                st.warning("No recent filings found.")
            else:
                st.dataframe(filings_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download filings CSV",
                    filings_df.to_csv(index=False),
                    file_name=f"{ticker}_recent_filings.csv",
                    mime="text/csv",
                )
        except requests.HTTPError as exc:
            show_sec_limit_message(exc)
        except Exception as exc:
            st.error(f"Unexpected error while loading submissions: {exc}")

    with col2:
        st.subheader("Core Financial Facts")
        try:
            facts = sec_get(COMPANYFACTS_URL.format(cik=cik), user_agent)
            metrics = []
            for metric, tags in FACT_TAGS.items():
                fact_df, selected_tag = pick_latest_fact(facts, tags)
                if fact_df.empty:
                    metrics.append({"Metric": metric, "Latest Value": None, "Tag": None, "Period End": None})
                    continue

                latest = fact_df.iloc[0]
                metrics.append(
                    {
                        "Metric": metric,
                        "Latest Value": latest.get("val"),
                        "Tag": selected_tag,
                        "Period End": latest.get("end"),
                    }
                )

                with st.expander(f"{metric} ({selected_tag})"):
                    st.dataframe(fact_df, use_container_width=True, hide_index=True)

            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download metrics CSV",
                metrics_df.to_csv(index=False),
                file_name=f"{ticker}_metrics.csv",
                mime="text/csv",
            )
        except requests.HTTPError as exc:
            show_sec_limit_message(exc)
        except Exception as exc:
            st.error(f"Unexpected error while loading company facts: {exc}")


if __name__ == "__main__":
    main()
