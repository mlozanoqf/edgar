import os

import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

FACT_TAGS = {
    "Assets": ["Assets"],
    "Liabilities": ["Liabilities"],
    "Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "Net Income": ["NetIncomeLoss"],
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
    response = requests.get(TICKERS_URL, headers={"User-Agent": user_agent}, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame.from_dict(data, orient="index")
    df["ticker"] = df["ticker"].str.upper()
    df["cik_str"] = df["cik_str"].astype(int)
    return df[["ticker", "cik_str", "title"]].sort_values("ticker")


@st.cache_data(ttl=3600)
def get_company_facts(cik: str, user_agent: str) -> dict:
    return sec_get(COMPANYFACTS_URL.format(cik=cik), user_agent)


def normalize_cik(cik: int) -> str:
    return str(cik).zfill(10)


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


def parse_error_message(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)

    if response.status_code == 429:
        return "SEC rate limit reached (429). Wait a bit and retry."
    if response.status_code == 403:
        return "SEC rejected the request (403). Verify contact email format."
    return f"SEC request failed ({response.status_code})."


def extract_metric_rows(company_facts: dict, ticker: str) -> pd.DataFrame:
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    rows = []

    for metric, tags in FACT_TAGS.items():
        selected_tag = None
        selected_df = pd.DataFrame()

        for tag in tags:
            tag_data = us_gaap.get(tag)
            if not tag_data:
                continue

            units = tag_data.get("units", {})
            unit_key = "USD" if "USD" in units else next(iter(units), None)
            if not unit_key:
                continue

            candidate = pd.DataFrame(units[unit_key])
            if candidate.empty:
                continue

            selected_tag = tag
            selected_df = candidate
            break

        if selected_df.empty:
            continue

        needed_cols = ["end", "fy", "fp", "form", "val", "filed"]
        for col in needed_cols:
            if col not in selected_df.columns:
                selected_df[col] = None

        selected_df = selected_df[needed_cols].copy()
        selected_df["metric"] = metric
        selected_df["tag"] = selected_tag
        selected_df["ticker"] = ticker

        rows.append(selected_df)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce")
    df["fy"] = df["fy"].astype("Int64")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")

    return df.dropna(subset=["val"])


def get_company_metric_df(ticker: str, ticker_map: pd.DataFrame, user_agent: str) -> pd.DataFrame:
    row = ticker_map[ticker_map["ticker"] == ticker]
    if row.empty:
        raise ValueError(f"Ticker {ticker} not found")

    cik = normalize_cik(int(row.iloc[0]["cik_str"]))
    facts = get_company_facts(cik, user_agent)
    return extract_metric_rows(facts, ticker)


def load_dataset(selected_tickers: list[str], ticker_map: pd.DataFrame, user_agent: str) -> tuple[pd.DataFrame, list[str]]:
    data_frames = []
    errors = []

    for ticker in selected_tickers:
        try:
            df = get_company_metric_df(ticker, ticker_map, user_agent)
            if not df.empty:
                data_frames.append(df)
            else:
                errors.append(f"{ticker}: no metric data found")
        except requests.HTTPError as exc:
            errors.append(f"{ticker}: {parse_error_message(exc)}")
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    if not data_frames:
        return pd.DataFrame(), errors

    return pd.concat(data_frames, ignore_index=True), errors


def render_cross_section(ticker_options: list[str], ticker_map: pd.DataFrame, user_agent: str) -> None:
    st.subheader("Cross-Section")
    st.caption("Select one year or quarter and compare multiple companies across selected accounts.")

    companies = st.multiselect(
        "Companies",
        ticker_options,
        default=[t for t in ["AAPL", "MSFT", "GOOGL"] if t in ticker_options],
        key="cs_companies",
    )
    metrics = st.multiselect(
        "Accounts",
        list(FACT_TAGS.keys()),
        default=list(FACT_TAGS.keys()),
        key="cs_metrics",
    )
    period_type = st.selectbox("Period type", ["Annual (FY)", "Quarterly (Q1-Q4)"], key="cs_period_type")

    if st.button("Run Cross-Section", key="run_cs"):
        if not companies:
            st.warning("Select at least one company.")
            return
        if not metrics:
            st.warning("Select at least one account.")
            return

        data, errors = load_dataset(companies, ticker_map, user_agent)
        if errors:
            st.warning(" | ".join(errors))
        if data.empty:
            st.error("No data available for the selected filters.")
            return

        data = data[data["metric"].isin(metrics)].copy()

        if period_type.startswith("Annual"):
            annual = data[data["fp"] == "FY"].copy()
            annual = annual.dropna(subset=["fy"])
            if annual.empty:
                st.error("No annual observations (FY) found.")
                return

            years = sorted(annual["fy"].dropna().astype(int).unique())
            selected_year = st.selectbox("Year", years, index=len(years) - 1, key="cs_year")

            filtered = annual[annual["fy"] == selected_year].copy()
        else:
            quarter = data[data["fp"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
            quarter = quarter.dropna(subset=["fy"])
            if quarter.empty:
                st.error("No quarterly observations (Q1-Q4) found.")
                return

            years = sorted(quarter["fy"].dropna().astype(int).unique())
            selected_year = st.selectbox("Year", years, index=len(years) - 1, key="cs_q_year")
            selected_quarter = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"], index=3, key="cs_quarter")

            filtered = quarter[(quarter["fy"] == selected_year) & (quarter["fp"] == selected_quarter)].copy()

        if filtered.empty:
            st.error("No rows found for that period.")
            return

        filtered = filtered.sort_values(["ticker", "metric", "filed"], ascending=[True, True, False])
        latest = filtered.drop_duplicates(subset=["ticker", "metric"], keep="first")

        pivot = latest.pivot(index="ticker", columns="metric", values="val").reset_index()
        st.dataframe(pivot, use_container_width=True, hide_index=True)
        st.download_button(
            "Download cross-section CSV",
            pivot.to_csv(index=False),
            file_name="cross_section.csv",
            mime="text/csv",
        )


def render_time_series(ticker_options: list[str], ticker_map: pd.DataFrame, user_agent: str) -> None:
    st.subheader("Time Series")
    st.caption("Select one company and track selected accounts through time.")

    ticker = st.selectbox("Company", ticker_options, index=ticker_options.index("AAPL") if "AAPL" in ticker_options else 0, key="ts_ticker")
    metrics = st.multiselect(
        "Accounts",
        list(FACT_TAGS.keys()),
        default=["Revenue", "Net Income", "Assets"],
        key="ts_metrics",
    )
    freq = st.selectbox("Frequency", ["Quarterly", "Annual"], key="ts_freq")

    if st.button("Run Time Series", key="run_ts"):
        if not metrics:
            st.warning("Select at least one account.")
            return

        try:
            data = get_company_metric_df(ticker, ticker_map, user_agent)
        except requests.HTTPError as exc:
            st.error(parse_error_message(exc))
            return
        except Exception as exc:
            st.error(str(exc))
            return

        if data.empty:
            st.error("No data available for this company.")
            return

        data = data[data["metric"].isin(metrics)].copy()

        if freq == "Annual":
            data = data[data["fp"] == "FY"]
        else:
            data = data[data["fp"].isin(["Q1", "Q2", "Q3", "Q4"])]

        data = data.dropna(subset=["end", "val", "fy"])
        if data.empty:
            st.error("No rows available for selected frequency.")
            return

        years = sorted(data["fy"].dropna().astype(int).unique())
        min_year, max_year = years[0], years[-1]
        year_range = st.slider("Year range", min_year, max_year, (min_year, max_year), key="ts_year_range")

        data = data[(data["fy"] >= year_range[0]) & (data["fy"] <= year_range[1])]
        if data.empty:
            st.error("No rows left after year filter.")
            return

        data = data.sort_values(["end", "metric", "filed"], ascending=[True, True, False])
        latest = data.drop_duplicates(subset=["end", "metric"], keep="first")

        chart_df = latest.pivot(index="end", columns="metric", values="val").sort_index()
        st.line_chart(chart_df)

        out = latest[["ticker", "metric", "fy", "fp", "end", "val", "form", "filed"]].sort_values("end")
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button(
            "Download time-series CSV",
            out.to_csv(index=False),
            file_name=f"time_series_{ticker}.csv",
            mime="text/csv",
        )


def render_panel(ticker_options: list[str], ticker_map: pd.DataFrame, user_agent: str) -> None:
    st.subheader("Panel")
    st.caption("Combine multiple companies over time in a long panel dataset.")

    companies = st.multiselect(
        "Companies",
        ticker_options,
        default=[t for t in ["AAPL", "MSFT", "AMZN"] if t in ticker_options],
        key="panel_companies",
    )
    metrics = st.multiselect(
        "Accounts",
        list(FACT_TAGS.keys()),
        default=["Revenue", "Net Income", "Assets", "Liabilities", "Equity"],
        key="panel_metrics",
    )
    freq = st.selectbox("Frequency", ["Quarterly", "Annual"], key="panel_freq")

    if st.button("Run Panel", key="run_panel"):
        if not companies:
            st.warning("Select at least one company.")
            return
        if not metrics:
            st.warning("Select at least one account.")
            return

        data, errors = load_dataset(companies, ticker_map, user_agent)
        if errors:
            st.warning(" | ".join(errors))
        if data.empty:
            st.error("No panel data available.")
            return

        data = data[data["metric"].isin(metrics)].copy()

        if freq == "Annual":
            data = data[data["fp"] == "FY"]
        else:
            data = data[data["fp"].isin(["Q1", "Q2", "Q3", "Q4"])]

        data = data.dropna(subset=["fy", "end", "val"])
        if data.empty:
            st.error("No rows available for selected frequency.")
            return

        years = sorted(data["fy"].dropna().astype(int).unique())
        min_year, max_year = years[0], years[-1]
        year_range = st.slider("Year range", min_year, max_year, (min_year, max_year), key="panel_year_range")

        data = data[(data["fy"] >= year_range[0]) & (data["fy"] <= year_range[1])]
        if data.empty:
            st.error("No rows left after year filter.")
            return

        data = data.sort_values(["ticker", "metric", "end", "filed"], ascending=[True, True, True, False])
        latest = data.drop_duplicates(subset=["ticker", "metric", "end"], keep="first")

        out = latest[["ticker", "metric", "fy", "fp", "end", "val", "form", "filed"]]
        st.dataframe(out.sort_values(["ticker", "end", "metric"]), use_container_width=True, hide_index=True)
        st.download_button(
            "Download panel CSV",
            out.to_csv(index=False),
            file_name="panel_data.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="EDGAR Explorer", layout="wide")
    st.title("EDGAR Explorer")
    st.caption("Financial data extraction from SEC EDGAR in Cross-Section, Time-Series, and Panel formats.")

    with st.sidebar:
        st.subheader("Configuration")
        contact_email = st.text_input(
            "Contact email (required)",
            value=default_contact_email(),
            placeholder="your-email@domain.com",
            help="SEC asks for identifiable requests with a contact email.",
        ).strip()

    st.info(
        "Free-tier usage notes: SEC EDGAR requires identifiable requests and enforces fair-access limits "
        "(up to 10 requests/sec). Streamlit Community Cloud has resource limits and may sleep after inactivity. "
        "If requests fail, wait and retry."
    )

    if not contact_email or "@" not in contact_email:
        st.warning("Please enter a valid contact email to use the app.")
        st.stop()

    user_agent = build_user_agent(contact_email)

    try:
        ticker_map = get_ticker_map(user_agent)
    except requests.HTTPError as exc:
        st.error(parse_error_message(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed loading SEC ticker list: {exc}")
        st.stop()

    ticker_options = ticker_map["ticker"].tolist()

    tab1, tab2, tab3 = st.tabs(["Cross-Section", "Time Series", "Panel"])

    with tab1:
        render_cross_section(ticker_options, ticker_map, user_agent)

    with tab2:
        render_time_series(ticker_options, ticker_map, user_agent)

    with tab3:
        render_panel(ticker_options, ticker_map, user_agent)


if __name__ == "__main__":
    main()
