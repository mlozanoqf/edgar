import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
SEC_SIC_CACHE_PATH = Path(__file__).with_name("sec_sic_lookup.csv")

FACT_TAGS = {
    "Assets": ["Assets"],
    "Liabilities": ["Liabilities"],
    "Equity": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity",
    ],
    "Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "Net Income": ["NetIncomeLoss"],
}

DEBT_TOTAL_TAGS = {
    "Total debt (direct)": [
        "LongTermDebtAndFinanceLeaseLiabilities",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndOperatingLeaseLiabilities",
        "LongTermDebt",
        "Debt",
        "DebtAndCapitalLeaseObligations",
    ],
    "Current debt": [
        "LongTermDebtAndFinanceLeaseLiabilitiesCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "CurrentPortionOfLongTermDebtAndCapitalLeaseObligations",
        "CurrentPortionOfLongTermDebt",
        "CurrentPortionOfLongTermDebtAndNotesPayable",
        "LongTermDebtCurrent",
        "DebtCurrent",
        "ShortTermDebt",
        "ShortTermBorrowings",
        "CommercialPaper",
    ],
    "Noncurrent debt": [
        "LongTermDebtAndFinanceLeaseLiabilitiesNoncurrent",
        "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
        "LongTermDebtAndOperatingLeaseLiabilitiesNoncurrent",
        "LongTermDebtNoncurrent",
    ],
}

MATURITY_CONCEPT_ORDER = {
    "next_12_months": 1,
    "year_2": 2,
    "year_3": 3,
    "year_4": 4,
    "year_5": 5,
    "after_year_5": 6,
    "years_2_to_5_agg": 7,
    "other": 99,
}

USD_MM_DIVISOR = 1_000_000
USD_MM_LABEL = "USD mm (1 USD mm = 1,000,000 USD)"
CROSS_SECTION_KEY_COLS = ["ticker", "fy", "fp", "end", "filed", "form"]
FLOW_METRICS = {"Revenue", "Net Income"}
STOCK_METRICS = {"Assets", "Liabilities", "Equity"}
METHOD_LABELS = {
    "baseline_schedule_only": "Schedule-based",
    "short_term_anchored": "Current-debt adjusted",
}
METHOD_HELP = {
    "Schedule-based": "Uses company-reported maturities as-is.",
    "Current-debt adjusted": "Replaces <=1Y schedule amount with balance-sheet current debt.",
}


def sec_get(url: str, user_agent: str) -> dict:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def sec_get_text(url: str, user_agent: str) -> str:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


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


@st.cache_data(ttl=3600)
def get_submissions(cik: str, user_agent: str) -> dict:
    return sec_get(SUBMISSIONS_URL.format(cik=cik), user_agent)


def load_local_sec_sic_lookup() -> pd.DataFrame:
    if not SEC_SIC_CACHE_PATH.exists():
        return pd.DataFrame(columns=["cik_str", "sic", "sicDescription"])
    try:
        df = pd.read_csv(SEC_SIC_CACHE_PATH)
    except Exception:
        return pd.DataFrame(columns=["cik_str", "sic", "sicDescription"])
    needed_cols = ["cik_str", "sic", "sicDescription"]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = None
    df["cik_str"] = pd.to_numeric(df["cik_str"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cik_str"]).copy()
    df["cik_str"] = df["cik_str"].astype(int)
    return df[needed_cols].drop_duplicates(subset=["cik_str"], keep="first")


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


def enrich_ticker_map_with_sec_metadata(ticker_map: pd.DataFrame) -> pd.DataFrame:
    sic_lookup = load_local_sec_sic_lookup()
    if sic_lookup.empty:
        return pd.DataFrame()

    enriched = ticker_map.merge(sic_lookup, on="cik_str", how="left")
    enriched["issuer_category"] = enriched["sicDescription"].map(
        lambda v: "Operating company (SEC SIC-based)" if pd.notna(v) and str(v).strip() else "Not SIC-classified / review"
    )
    enriched["is_operating_company"] = enriched["issuer_category"] == "Operating company (SEC SIC-based)"
    enriched["industry_group"] = enriched["sicDescription"].map(classify_sic_industry_group)
    enriched.loc[~enriched["is_operating_company"], "industry_group"] = "Unclassified (missing SIC)"
    return enriched


def classify_sic_industry_group(sic_description: object) -> str:
    txt = str(sic_description or "").upper()
    if not txt or txt == "NAN":
        return "Unclassified (missing SIC)"
    if any(k in txt for k in ["SOFTWARE", "SEMICONDUCTOR", "COMPUTER", "ELECTRONIC", "DATA PROCESSING", "INTERNET"]):
        return "Technology"
    if any(k in txt for k in ["PHARM", "BIO", "MEDIC", "HEALTH", "DIAGNOSTIC"]):
        return "Healthcare / Life Sciences"
    if any(k in txt for k in ["BANK", "FINANCE", "INSUR", "CREDIT", "SAVINGS", "INVESTMENT", "ASSET MANAGEMENT"]):
        return "Financials / Insurance"
    if any(k in txt for k in ["OIL", "GAS", "PETROLEUM", "ELECTRIC", "POWER", "UTILITY", "PIPELINE", "ENERGY"]):
        return "Energy / Utilities"
    if any(k in txt for k in ["RETAIL", "APPAREL", "FOOD", "BEVERAGE", "RESTAURANT", "DEPARTMENT STORE", "MERCHANDISE"]):
        return "Consumer / Retail"
    if any(k in txt for k in ["AIR", "TRUCK", "TRANSPORT", "LOGISTICS", "AEROSPACE", "MACHINERY", "MANUFACTURING", "INDUSTRIAL"]):
        return "Industrials / Transportation"
    if any(k in txt for k in ["REAL ESTATE", "REIT", "CONSTRUCTION", "BUILDING", "PROPERTY"]):
        return "Real Estate / Construction"
    if any(k in txt for k in ["CHEMICAL", "MINING", "METAL", "STEEL", "LUMBER", "MATERIAL"]):
        return "Materials / Chemicals"
    if any(k in txt for k in ["COMMUNICATION", "TELEPHONE", "TELECOM", "BROADCAST", "MEDIA", "CABLE"]):
        return "Telecom / Media"
    return "Other operating"


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

        needed_cols = ["start", "end", "fy", "fp", "form", "val", "filed", "frame", "segment"]
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
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.dropna(subset=["val"])
    return df[df["segment"].isna()].copy()


def extract_tag_rows(company_facts: dict, tag: str) -> pd.DataFrame:
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    tag_data = us_gaap.get(tag)
    if not tag_data:
        return pd.DataFrame()

    units = tag_data.get("units", {})
    unit_key = "USD" if "USD" in units else next(iter(units), None)
    if not unit_key:
        return pd.DataFrame()

    df = pd.DataFrame(units[unit_key])
    if df.empty:
        return pd.DataFrame()

    needed_cols = ["start", "end", "fy", "fp", "form", "val", "filed", "frame", "segment"]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = None

    df = df[needed_cols].copy()
    df["tag"] = tag
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.dropna(subset=["val"])
    return df[df["segment"].isna()].copy()


def build_metric_rows_for_metric(company_facts: dict, ticker: str, metric: str) -> pd.DataFrame:
    tags = FACT_TAGS.get(metric, [])
    rows = []

    for tag_rank, tag in enumerate(tags):
        tag_rows = extract_tag_rows(company_facts, tag)
        if tag_rows.empty:
            continue
        tag_rows = tag_rows.copy()
        tag_rows["metric"] = metric
        tag_rows["ticker"] = ticker
        tag_rows["tag_rank"] = tag_rank
        rows.append(tag_rows)

    if not rows:
        return pd.DataFrame()

    cols = ["ticker", "metric", "tag", "tag_rank", "start", "end", "fy", "fp", "form", "val", "filed", "frame"]
    return pd.concat(rows, ignore_index=True)[cols]


def latest_value_for_year(rows: pd.DataFrame, fiscal_year: int, fy_only: bool = False) -> pd.Series | None:
    if rows.empty:
        return None

    year_rows = rows[rows["fy"] == fiscal_year].copy()
    if year_rows.empty:
        return None

    annual = year_rows[year_rows["fp"] == "FY"].copy()
    if fy_only:
        selected = annual
    else:
        selected = annual if not annual.empty else year_rows
    selected = selected.sort_values(["filed", "end"], ascending=[False, False])
    if selected.empty:
        return None
    return selected.iloc[0]


def pick_component_for_year(company_facts: dict, tags: list[str], fiscal_year: int, fy_only: bool = False) -> dict:
    for tag in tags:
        rows = extract_tag_rows(company_facts, tag)
        best = latest_value_for_year(rows, fiscal_year, fy_only=fy_only)
        if best is not None:
            return {
                "tag": tag,
                "val": float(best["val"]),
                "fp": best["fp"],
                "end": best["end"],
                "filed": best["filed"],
            }
    return {"tag": None, "val": None, "fp": None, "end": None, "filed": None}


def maturity_concept_from_tag(tag: str) -> str:
    if "MaturitiesRepaymentsOfPrincipalInNextTwelveMonths" in tag:
        return "next_12_months"
    if "MaturitiesRepaymentsOfPrincipalInYearTwo" in tag:
        return "year_2"
    if "MaturitiesRepaymentsOfPrincipalInYearThree" in tag:
        return "year_3"
    if "MaturitiesRepaymentsOfPrincipalInYearFour" in tag:
        return "year_4"
    if "MaturitiesRepaymentsOfPrincipalInYearFive" in tag:
        return "year_5"
    if "MaturitiesRepaymentsOfPrincipalAfterYearFive" in tag:
        return "after_year_5"
    if "MaturitiesRepaymentsOfPrincipalInYearsTwoThroughFive" in tag:
        return "years_2_to_5_agg"
    return "other"


def is_maturity_tag(tag: str) -> bool:
    if "MaturitiesRepaymentsOfPrincipal" not in tag:
        return False
    return bool(re.search(r"(Debt|FinanceLeaseLiabilities)", tag))


def maturity_item_label(company_facts: dict, tag: str) -> str:
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    tag_data = us_gaap.get(tag, {})
    label = tag_data.get("label")
    if label:
        return str(label)

    raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", tag)
    return raw.strip()


def maturity_sort_key(row: pd.Series) -> tuple:
    concept = str(row.get("concept", "other"))
    order = MATURITY_CONCEPT_ORDER.get(concept, 99)
    return (order, str(row.get("item", "")))


def maturity_axis_label_info(item_label: str, concept: str, fiscal_year: int, tag: str | None = None) -> tuple[str, str]:
    item_text = str(item_label)
    probe_text = f"{item_text} {tag or ''}"
    item_lower = probe_text.lower()
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", item_text)]

    # If filing label explicitly says "thereafter/after", force an open-ended label.
    if any(k in item_lower for k in ["thereafter", "after", "beyond", "onward", "plus", "orlater", "orlater"]):
        if years:
            return f">{max(years)}*", "open_ended"
        return f">{fiscal_year + 5}*", "open_ended"

    # Some extension tags encode "year five and thereafter" without clear concept mapping.
    if any(k in item_lower for k in ["yearfiveandthereafter", "year5andthereafter", "afteryearfive", "afterfiveyears"]):
        if years:
            return f">{max(years)}*", "open_ended"
        return f">{fiscal_year + 5}*", "open_ended"

    # If label carries multiple years, preserve that range instead of collapsing to a single year.
    if len(set(years)) >= 2:
        return f"{min(years)}-{max(years)}", "range"

    # Explicit year reported by the company should be preserved as-is.
    if len(years) == 1:
        return str(years[0]), "explicit_year"

    # Concept-based labels take priority for aggregated maturity buckets.
    if concept == "after_year_5":
        return f">{fiscal_year + 5}", "open_ended"
    if concept == "years_2_to_5_agg":
        return f"{fiscal_year + 2}-{fiscal_year + 5}", "range"
    if concept == "next_12_months":
        return str(fiscal_year + 1), "relative_year"
    if concept == "year_2":
        return str(fiscal_year + 2), "relative_year"
    if concept == "year_3":
        return str(fiscal_year + 3), "relative_year"
    if concept == "year_4":
        return str(fiscal_year + 4), "relative_year"
    if concept == "year_5":
        return str(fiscal_year + 5), "relative_year"

    return "Other", "other"


def maturity_label_sort_value(label: str, fiscal_year: int) -> int:
    txt = str(label).strip()
    if txt.startswith(">"):
        num = re.sub(r"[^\d]", "", txt)
        return int(num) + 1 if num else fiscal_year + 100
    if "-" in txt:
        left = txt.split("-", 1)[0]
        if left.isdigit():
            return int(left)
    if txt.isdigit():
        return int(txt)
    return fiscal_year + 200


def build_company_reported_maturity_rows(company_facts: dict, fiscal_year: int, fy_only: bool = True) -> pd.DataFrame:
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    tags = [tag for tag in us_gaap.keys() if is_maturity_tag(tag)]
    rows = []

    for tag in tags:
        tag_rows = extract_tag_rows(company_facts, tag)
        best = latest_value_for_year(tag_rows, fiscal_year, fy_only=fy_only)
        if best is None:
            continue
        concept = maturity_concept_from_tag(tag)
        rows.append(
            {
                "item": maturity_item_label(company_facts, tag),
                "tag": tag,
                "concept": concept,
                "amount": float(best["val"]),
                "fp": best["fp"],
                "end": best["end"],
                "filed": best["filed"],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["include_in_sum"] = False
    df["exclusion_reason"] = "not selected"

    detailed_present = any(df["concept"].isin(["year_2", "year_3", "year_4", "year_5"]))
    candidate_df = df.copy()
    if detailed_present:
        candidate_df = candidate_df[candidate_df["concept"] != "years_2_to_5_agg"].copy()

    candidate_df = candidate_df[candidate_df["concept"] != "other"].copy()
    if not candidate_df.empty:
        for concept in candidate_df["concept"].unique().tolist():
            concept_rows = candidate_df[candidate_df["concept"] == concept].copy()
            if concept_rows.empty:
                continue
            concept_rows["score"] = 0
            concept_rows.loc[concept_rows["tag"].str.contains("AndFinanceLeaseLiabilities", na=False), "score"] += 2
            concept_rows.loc[concept_rows["fp"] == "FY", "score"] += 1
            concept_rows = concept_rows.sort_values(["score", "filed", "end"], ascending=[False, False, False])
            chosen_idx = concept_rows.index[0]
            df.loc[chosen_idx, "include_in_sum"] = True
            df.loc[chosen_idx, "exclusion_reason"] = ""

    if detailed_present:
        df.loc[df["concept"] == "years_2_to_5_agg", "exclusion_reason"] = "Excluded because detailed Y2-Y5 exists"

    duplicate_mask = (df["concept"] != "other") & (~df["include_in_sum"]) & (df["exclusion_reason"] == "not selected")
    df.loc[duplicate_mask, "exclusion_reason"] = "Duplicate concept (alternative tag)"
    df = df.sort_values(by=["concept", "item"], key=lambda c: c.map(lambda x: MATURITY_CONCEPT_ORDER.get(str(x), 99)) if c.name == "concept" else c)
    return df.reset_index(drop=True)


def split_tag(tag: str) -> tuple[str | None, str]:
    if "}" in tag:
        ns, local = tag[1:].split("}", 1)
        return ns, local
    if ":" in tag:
        prefix, local = tag.split(":", 1)
        return prefix, local
    return None, tag


def is_maturity_concept_name(name: str) -> bool:
    lower = name.lower()
    has_maturity = ("maturit" in lower) or ("thereafter" in lower) or ("repaymentsofprincipal" in lower)
    has_scope_hint = any(
        k in lower
        for k in ["debt", "lease", "borrow", "notepayable", "notespayable", "principal"]
    )
    return has_maturity and has_scope_hint


def numeric_or_none(raw: str | None) -> float | None:
    if raw is None:
        return None
    txt = str(raw).strip().replace(",", "")
    if txt in ["", "-", "—", "–"]:
        return None
    try:
        return float(txt)
    except Exception:
        return None


def concept_from_name_or_label(name: str, label: str, fiscal_year: int) -> str:
    probe = f"{name} {label}".lower()
    if "nexttwelvemonths" in probe or "next 12 months" in probe:
        return "next_12_months"
    if "yeartwo" in probe or "year two" in probe or "year2" in probe:
        return "year_2"
    if "yearthree" in probe or "year three" in probe or "year3" in probe:
        return "year_3"
    if "yearfour" in probe or "year four" in probe or "year4" in probe:
        return "year_4"
    if "yearfive" in probe or "year five" in probe or "year5" in probe:
        return "year_5"
    if "afteryearfive" in probe or "after year five" in probe or "thereafter" in probe:
        return "after_year_5"
    if "twothroughfive" in probe or "two through five" in probe:
        return "years_2_to_5_agg"

    years = re.findall(r"\b(?:19|20)\d{2}\b", label)
    if years:
        y = int(years[0])
        delta = y - fiscal_year
        if delta == 1:
            return "next_12_months"
        if delta == 2:
            return "year_2"
        if delta == 3:
            return "year_3"
        if delta == 4:
            return "year_4"
        if delta == 5:
            return "year_5"
        if delta > 5:
            return "after_year_5"
    return "other"


def get_filing_candidates_for_year(submissions: dict, fiscal_year: int) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    forms = recent.get("form", [])
    accession = recent.get("accessionNumber", [])
    report_dates = recent.get("reportDate", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    n = min(len(forms), len(accession), len(report_dates), len(filing_dates), len(primary_docs))
    rows = []
    for i in range(n):
        form = str(forms[i] or "")
        report_date = str(report_dates[i] or "")
        filing_date = str(filing_dates[i] or "")
        report_year = int(report_date[:4]) if re.match(r"^\d{4}", report_date) else None
        filing_year = int(filing_date[:4]) if re.match(r"^\d{4}", filing_date) else None
        if report_year != fiscal_year and filing_year != fiscal_year:
            continue
        if form not in ["10-K", "20-F", "40-F", "10-K/A", "20-F/A", "40-F/A"]:
            continue
        rows.append(
            {
                "form": form,
                "accession": accession[i],
                "report_date": report_date,
                "filing_date": filing_date,
                "primary_document": primary_docs[i],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["form_rank"] = df["form"].map({"10-K": 1, "20-F": 1, "40-F": 1, "10-K/A": 2, "20-F/A": 2, "40-F/A": 2}).fillna(9)
    return df.sort_values(["form_rank", "filing_date"], ascending=[True, False]).reset_index(drop=True)


def get_recent_filing_candidates(
    submissions: dict,
    fiscal_year: int,
    allowed_forms: list[str],
    target_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    forms = recent.get("form", [])
    accession = recent.get("accessionNumber", [])
    report_dates = recent.get("reportDate", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    n = min(len(forms), len(accession), len(report_dates), len(filing_dates), len(primary_docs))
    rows = []
    for i in range(n):
        form = str(forms[i] or "")
        if form not in allowed_forms:
            continue
        report_date = str(report_dates[i] or "")
        filing_date = str(filing_dates[i] or "")
        report_year = int(report_date[:4]) if re.match(r"^\d{4}", report_date) else None
        filing_year = int(filing_date[:4]) if re.match(r"^\d{4}", filing_date) else None
        if report_year != fiscal_year and filing_year != fiscal_year:
            continue
        rows.append(
            {
                "form": form,
                "accession": accession[i],
                "report_date": report_date,
                "filing_date": filing_date,
                "primary_document": primary_docs[i],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if target_end is not None and pd.notna(target_end):
        target_str = pd.Timestamp(target_end).strftime("%Y-%m-%d")
        df["report_match"] = df["report_date"] == target_str
    else:
        df["report_match"] = False
    return df.sort_values(["report_match", "filing_date"], ascending=[False, False]).reset_index(drop=True)


@st.cache_data(ttl=3600)
def get_recent_filing_index(cik: str, user_agent: str) -> pd.DataFrame:
    submissions = get_submissions(cik, user_agent)
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    forms = recent.get("form", [])
    accession = recent.get("accessionNumber", [])
    report_dates = recent.get("reportDate", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    n = min(len(forms), len(accession), len(report_dates), len(filing_dates), len(primary_docs))
    rows = []
    cik_int = str(int(cik))
    for i in range(n):
        form = str(forms[i] or "")
        acc = str(accession[i] or "")
        report_date = str(report_dates[i] or "")
        filing_date = str(filing_dates[i] or "")
        primary_document = str(primary_docs[i] or "")
        if not acc or not primary_document:
            continue
        rows.append(
            {
                "form": form,
                "accession": acc,
                "report_date": report_date,
                "filing_date": filing_date,
                "primary_document": primary_document,
                "filing_url": f"{ARCHIVES_BASE}/{cik_int}/{accession_no_dashes(acc)}/{primary_document}",
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def resolve_filing_url_for_row(
    cik: str,
    user_agent: str,
    form: object,
    filed: object,
    end: object | None = None,
    accession: str | None = None,
) -> str | None:
    filing_index = get_recent_filing_index(cik, user_agent)
    if filing_index.empty:
        return None

    if accession:
        exact_acc = filing_index[filing_index["accession"] == str(accession)].copy()
        if not exact_acc.empty:
            return str(exact_acc.iloc[0]["filing_url"])

    candidates = filing_index.copy()
    if form is not None and not pd.isna(form):
        candidates = candidates[candidates["form"] == str(form)]
    if filed is not None and not pd.isna(filed):
        filed_str = pd.Timestamp(filed).strftime("%Y-%m-%d")
        candidates = candidates[candidates["filing_date"] == filed_str]
    if candidates.empty:
        return None
    if end is not None and not pd.isna(end):
        end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
        exact_report = candidates[candidates["report_date"] == end_str].copy()
        if not exact_report.empty:
            return str(exact_report.iloc[0]["filing_url"])
    return str(candidates.iloc[0]["filing_url"])


def accession_no_dashes(accession: str) -> str:
    return str(accession).replace("-", "")


@st.cache_data(ttl=3600)
def get_filing_index_json(cik: str, accession: str, user_agent: str) -> dict:
    cik_int = str(int(cik))
    acc_no_dash = accession_no_dashes(accession)
    url = f"{ARCHIVES_BASE}/{cik_int}/{acc_no_dash}/index.json"
    return sec_get(url, user_agent)


def pick_xbrl_instance_filename(index_json: dict) -> str | None:
    items = index_json.get("directory", {}).get("item", [])
    if not items:
        return None
    names = [str(it.get("name", "")) for it in items]
    xml_names = [n for n in names if n.lower().endswith(".xml")]
    if not xml_names:
        return None

    def score(name: str) -> int:
        lower = name.lower()
        if "filingsummary" in lower:
            return -10
        if lower.endswith(("_cal.xml", "_def.xml", "_lab.xml", "_pre.xml")):
            return -5
        if re.match(r"^r\d+\.xml$", lower):
            return -5
        if "_htm.xml" in lower:
            return 5
        if "instance" in lower:
            return 4
        if lower.count("-") >= 2:
            return 3
        return 1

    ranked = sorted(xml_names, key=lambda n: (score(n), n), reverse=True)
    return ranked[0] if ranked else None


@st.cache_data(ttl=3600)
def get_filing_xbrl_text(cik: str, accession: str, user_agent: str) -> str | None:
    index_json = get_filing_index_json(cik, accession, user_agent)
    filename = pick_xbrl_instance_filename(index_json)
    if not filename:
        return None
    cik_int = str(int(cik))
    acc_no_dash = accession_no_dashes(accession)
    url = f"{ARCHIVES_BASE}/{cik_int}/{acc_no_dash}/{filename}"
    return sec_get_text(url, user_agent)


def parse_filing_maturity_rows(xbrl_text: str, fiscal_year: int, filing_date: str) -> pd.DataFrame:
    try:
        root = ET.fromstring(xbrl_text)
    except Exception:
        return pd.DataFrame()

    contexts: dict[str, dict] = {}
    for elem in root.iter():
        _, local = split_tag(elem.tag)
        if local != "context":
            continue
        ctx_id = elem.attrib.get("id")
        if not ctx_id:
            continue
        end_date = None
        instant = None
        for child in elem.iter():
            _, c_local = split_tag(child.tag)
            if c_local == "endDate":
                end_date = (child.text or "").strip()
            elif c_local == "instant":
                instant = (child.text or "").strip()
        period_date = end_date or instant
        contexts[ctx_id] = {"period_date": period_date}

    rows = []
    for elem in root.iter():
        if list(elem):
            continue
        _, local = split_tag(elem.tag)
        if not is_maturity_concept_name(local):
            continue
        context_ref = elem.attrib.get("contextRef")
        if not context_ref or context_ref not in contexts:
            continue
        period_date = contexts[context_ref].get("period_date")
        if not period_date or not re.match(r"^\d{4}", str(period_date)):
            continue
        year = int(str(period_date)[:4])
        if year != fiscal_year:
            continue
        amount = numeric_or_none(elem.text)
        if amount is None:
            continue
        label = re.sub(r"([a-z])([A-Z])", r"\1 \2", local).strip()
        concept = concept_from_name_or_label(local, label, fiscal_year)
        rows.append(
            {
                "item": label,
                "tag": local,
                "concept": concept,
                "amount": float(amount),
                "fp": "FY",
                "end": pd.to_datetime(period_date, errors="coerce"),
                "filed": pd.to_datetime(filing_date, errors="coerce"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["tag", "amount", "end"])
    df["include_in_sum"] = True
    df["exclusion_reason"] = ""

    detailed_present = any(df["concept"].isin(["year_2", "year_3", "year_4", "year_5"]))
    if detailed_present:
        agg_mask = df["concept"] == "years_2_to_5_agg"
        df.loc[agg_mask, "include_in_sum"] = False
        df.loc[agg_mask, "exclusion_reason"] = "Excluded because detailed Y2-Y5 exists"

    df = df.sort_values(by=["concept", "item"], key=lambda c: c.map(lambda x: MATURITY_CONCEPT_ORDER.get(str(x), 99)) if c.name == "concept" else c)
    return df.reset_index(drop=True)


def parse_filing_total_rows(xbrl_text: str, fiscal_year: int, filing_date: str) -> pd.DataFrame:
    try:
        root = ET.fromstring(xbrl_text)
    except Exception:
        return pd.DataFrame()

    needed_tags = {t for tags in DEBT_TOTAL_TAGS.values() for t in tags}
    contexts: dict[str, dict] = {}
    for elem in root.iter():
        _, local = split_tag(elem.tag)
        if local != "context":
            continue
        ctx_id = elem.attrib.get("id")
        if not ctx_id:
            continue
        end_date = None
        instant = None
        for child in elem.iter():
            _, c_local = split_tag(child.tag)
            if c_local == "endDate":
                end_date = (child.text or "").strip()
            elif c_local == "instant":
                instant = (child.text or "").strip()
        period_date = end_date or instant
        contexts[ctx_id] = {"period_date": period_date}

    rows = []
    for elem in root.iter():
        if list(elem):
            continue
        _, local = split_tag(elem.tag)
        if local not in needed_tags:
            continue
        context_ref = elem.attrib.get("contextRef")
        if not context_ref or context_ref not in contexts:
            continue
        period_date = contexts[context_ref].get("period_date")
        if not period_date or not re.match(r"^\d{4}", str(period_date)):
            continue
        year = int(str(period_date)[:4])
        if year != fiscal_year:
            continue
        amount = numeric_or_none(elem.text)
        if amount is None:
            continue
        rows.append(
            {
                "tag": local,
                "val": float(amount),
                "fp": "FY",
                "end": pd.to_datetime(period_date, errors="coerce"),
                "filed": pd.to_datetime(filing_date, errors="coerce"),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["tag", "end", "filed"], ascending=[True, False, False]).reset_index(drop=True)


def parse_filing_flow_context_pairs(
    xbrl_text: str,
    filing_date: str,
    period_type: str,
    selected_quarter: str | None,
    target_end: pd.Timestamp | None,
) -> pd.DataFrame:
    try:
        root = ET.fromstring(xbrl_text)
    except Exception:
        return pd.DataFrame()

    contexts: dict[str, dict] = {}
    for elem in root.iter():
        _, local = split_tag(elem.tag)
        if local != "context":
            continue
        ctx_id = elem.attrib.get("id")
        if not ctx_id:
            continue
        start_date = None
        end_date = None
        instant = None
        has_dimensions = False
        for child in elem.iter():
            _, c_local = split_tag(child.tag)
            if c_local == "startDate":
                start_date = (child.text or "").strip()
            elif c_local == "endDate":
                end_date = (child.text or "").strip()
            elif c_local == "instant":
                instant = (child.text or "").strip()
            elif c_local in ["explicitMember", "typedMember"]:
                has_dimensions = True
        contexts[ctx_id] = {
            "start_date": start_date,
            "end_date": end_date,
            "instant": instant,
            "has_dimensions": has_dimensions,
        }

    flow_rows = []
    flow_tags = {metric: tags for metric, tags in FACT_TAGS.items() if metric in FLOW_METRICS}
    tag_to_metric: dict[str, tuple[str, int]] = {}
    for metric, tags in flow_tags.items():
        for tag_rank, tag in enumerate(tags):
            tag_to_metric[tag] = (metric, tag_rank)

    for elem in root.iter():
        if list(elem):
            continue
        _, local = split_tag(elem.tag)
        if local not in tag_to_metric:
            continue
        context_ref = elem.attrib.get("contextRef")
        if not context_ref or context_ref not in contexts:
            continue
        ctx = contexts[context_ref]
        if ctx["has_dimensions"]:
            continue
        if not ctx["end_date"] or not ctx["start_date"]:
            continue
        amount = numeric_or_none(elem.text)
        if amount is None:
            continue
        metric, tag_rank = tag_to_metric[local]
        flow_rows.append(
            {
                "context_ref": context_ref,
                "metric": metric,
                "tag": local,
                "tag_rank": tag_rank,
                "val": float(amount),
                "start": pd.to_datetime(ctx["start_date"], errors="coerce"),
                "end": pd.to_datetime(ctx["end_date"], errors="coerce"),
                "filed": pd.to_datetime(filing_date, errors="coerce"),
            }
        )

    if not flow_rows:
        return pd.DataFrame()

    flow_df = pd.DataFrame(flow_rows)
    flow_df["period_length_days"] = (flow_df["end"] - flow_df["start"]).dt.days
    flow_df["period_length_days"] = flow_df["period_length_days"].where(flow_df["period_length_days"] >= 0)
    flow_df = flow_df.sort_values(
        ["context_ref", "metric", "period_length_days", "tag_rank"],
        ascending=[True, True, False, True],
    )
    best_by_metric = flow_df.drop_duplicates(subset=["context_ref", "metric"], keep="first")

    value_pairs = best_by_metric.pivot(index="context_ref", columns="metric", values="val").reset_index()
    tag_pairs = best_by_metric.pivot(index="context_ref", columns="metric", values="tag").reset_index()
    start_pairs = best_by_metric.pivot(index="context_ref", columns="metric", values="start").reset_index()
    end_pairs = best_by_metric.pivot(index="context_ref", columns="metric", values="end").reset_index()
    dur_pairs = best_by_metric.pivot(index="context_ref", columns="metric", values="period_length_days").reset_index()

    if "Revenue" not in value_pairs.columns or "Net Income" not in value_pairs.columns:
        return pd.DataFrame()

    paired = value_pairs.merge(tag_pairs, on="context_ref", suffixes=("", " tag")).merge(
        start_pairs, on="context_ref", suffixes=("", " start")
    ).merge(end_pairs, on="context_ref", suffixes=("", " end")).merge(
        dur_pairs, on="context_ref", suffixes=("", " duration")
    )
    paired = paired.rename(
        columns={
            "Revenue tag": "Revenue tag",
            "Net Income tag": "Net Income tag",
            "Revenue start": "Revenue start",
            "Net Income start": "Net Income start",
            "Revenue end": "Revenue end",
            "Net Income end": "Net Income end",
            "Revenue duration": "Revenue duration",
            "Net Income duration": "Net Income duration",
        }
    )
    paired["same_end"] = paired["Revenue end"] == paired["Net Income end"]
    paired["same_start"] = paired["Revenue start"] == paired["Net Income start"]
    paired = paired[paired["same_end"] & paired["same_start"]].copy()
    if paired.empty:
        return pd.DataFrame()

    paired["common_start"] = paired["Revenue start"]
    paired["common_end"] = paired["Revenue end"]
    paired["period_length_days"] = paired["Revenue duration"]
    paired["target_end_match"] = False
    if target_end is not None and pd.notna(target_end):
        paired["target_end_match"] = paired["common_end"] == pd.Timestamp(target_end)

    expected_days = None
    if period_type.startswith("Quarterly"):
        quarter_days = {"Q1": 95, "Q2": 185, "Q3": 275, "Q4": 365}
        expected_days = quarter_days.get(selected_quarter or "")
    elif period_type.startswith("Annual"):
        expected_days = 365

    if expected_days is not None:
        paired["duration_distance"] = (paired["period_length_days"] - expected_days).abs()
    else:
        paired["duration_distance"] = float("inf")

    paired["filed"] = pd.to_datetime(filing_date, errors="coerce")
    return paired.sort_values(
        ["target_end_match", "duration_distance", "filed", "common_end"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)


def get_strict_flow_pair_for_period(
    cik: str,
    fiscal_year: int,
    period_type: str,
    selected_quarter: str | None,
    target_end: pd.Timestamp | None,
    user_agent: str,
) -> dict | None:
    try:
        submissions = get_submissions(cik, user_agent)
    except Exception:
        return None

    if period_type.startswith("Quarterly") and selected_quarter in ["Q1", "Q2", "Q3"]:
        allowed_forms = ["10-Q", "10-Q/A"]
    else:
        allowed_forms = ["10-K", "20-F", "40-F", "10-K/A", "20-F/A", "40-F/A"]

    candidates = get_recent_filing_candidates(submissions, fiscal_year, allowed_forms, target_end=target_end)
    if candidates.empty:
        return None

    for _, cand in candidates.iterrows():
        accession = str(cand["accession"])
        filing_date = str(cand["filing_date"])
        xbrl_text = get_filing_xbrl_text(cik, accession, user_agent)
        if not xbrl_text:
            continue
        paired = parse_filing_flow_context_pairs(xbrl_text, filing_date, period_type, selected_quarter, target_end)
        if paired.empty:
            continue
        best = paired.iloc[0]
        return {
            "Revenue": float(best["Revenue"]),
            "Revenue tag": best["Revenue tag"],
            "Net Income": float(best["Net Income"]),
            "Net Income tag": best["Net Income tag"],
            "flow_context_ref": best["context_ref"],
            "flow_common_start": best["common_start"],
            "flow_common_end": best["common_end"],
            "flow_period_length_days": best["period_length_days"],
            "flow_filing_accession": accession,
            "flow_filing_date": pd.to_datetime(filing_date, errors="coerce"),
            "flow_context_source": "filing_xbrl_same_context",
        }
    return None


def build_total_components_from_filing_rows(total_rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for label, tags in DEBT_TOTAL_TAGS.items():
        chosen = pd.DataFrame()
        chosen_tag = None
        for tag in tags:
            subset = total_rows[total_rows["tag"] == tag].copy() if not total_rows.empty else pd.DataFrame()
            if subset.empty:
                continue
            subset = subset.sort_values(["filed", "end"], ascending=[False, False])
            chosen = subset
            chosen_tag = tag
            break

        if chosen.empty:
            out.append(
                {
                    "component": label,
                    "value": None,
                    "tag": None,
                    "fp": None,
                    "end": None,
                    "filed": None,
                    "source": "filing_xbrl",
                }
            )
        else:
            best = chosen.iloc[0]
            out.append(
                {
                    "component": label,
                    "value": float(best["val"]),
                    "tag": chosen_tag,
                    "fp": best["fp"],
                    "end": best["end"],
                    "filed": best["filed"],
                    "source": "filing_xbrl",
                }
            )
    return pd.DataFrame(out)


def build_total_components_from_companyfacts(company_facts: dict, fiscal_year: int, fy_only: bool) -> pd.DataFrame:
    total_components = []
    for label, tags in DEBT_TOTAL_TAGS.items():
        picked = pick_component_for_year(company_facts, tags, fiscal_year, fy_only=fy_only)
        total_components.append(
            {
                "component": label,
                "value": picked["val"],
                "tag": picked["tag"],
                "fp": picked["fp"],
                "end": picked["end"],
                "filed": picked["filed"],
                "source": "companyfacts",
            }
        )
    return pd.DataFrame(total_components)


def combine_total_components(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component in ["Total debt (direct)", "Current debt", "Noncurrent debt"]:
        p = primary_df[primary_df["component"] == component]
        s = secondary_df[secondary_df["component"] == component]
        p_row = p.iloc[0].to_dict() if not p.empty else {}
        s_row = s.iloc[0].to_dict() if not s.empty else {}
        p_val = p_row.get("value")
        if p_val is not None and pd.notna(p_val):
            rows.append(p_row)
        elif s_row:
            rows.append(s_row)
        else:
            rows.append(
                {
                    "component": component,
                    "value": None,
                    "tag": None,
                    "fp": None,
                    "end": None,
                    "filed": None,
                    "source": "none",
                }
            )
    return pd.DataFrame(rows)


def build_maturity_rows_from_filing(cik: str, fiscal_year: int, user_agent: str) -> tuple[pd.DataFrame, str | None, pd.DataFrame]:
    try:
        submissions = get_submissions(cik, user_agent)
    except Exception:
        return pd.DataFrame(), None, pd.DataFrame()

    candidates = get_filing_candidates_for_year(submissions, fiscal_year)
    if candidates.empty:
        return pd.DataFrame(), None, pd.DataFrame()

    for _, cand in candidates.iterrows():
        accession = str(cand["accession"])
        filing_date = str(cand["filing_date"])
        xbrl_text = get_filing_xbrl_text(cik, accession, user_agent)
        if not xbrl_text:
            continue
        df = parse_filing_maturity_rows(xbrl_text, fiscal_year, filing_date)
        total_rows = parse_filing_total_rows(xbrl_text, fiscal_year, filing_date)
        if not df.empty:
            return df, accession, total_rows
    return pd.DataFrame(), None, pd.DataFrame()


def available_years_for_tag_groups(company_facts: dict, tag_groups: list[list[str]], fy_only: bool = False) -> list[int]:
    years: set[int] = set()
    for tag_group in tag_groups:
        for tag in tag_group:
            rows = extract_tag_rows(company_facts, tag)
            if rows.empty:
                continue
            if fy_only:
                rows = rows[rows["fp"] == "FY"].copy()
                if rows.empty:
                    continue
            valid = rows["fy"].dropna().astype(int).tolist()
            years.update(valid)
    return sorted(years)


def available_debt_years(company_facts: dict, fy_only: bool = False) -> list[int]:
    maturity_tags = [tag for tag in company_facts.get("facts", {}).get("us-gaap", {}).keys() if is_maturity_tag(tag)]
    groups = list(DEBT_TOTAL_TAGS.values()) + [[tag] for tag in maturity_tags]
    return available_years_for_tag_groups(company_facts, groups, fy_only=fy_only)


def available_bucket_years(company_facts: dict, fy_only: bool = False) -> list[int]:
    maturity_tags = [tag for tag in company_facts.get("facts", {}).get("us-gaap", {}).keys() if is_maturity_tag(tag)]
    return available_years_for_tag_groups(company_facts, [[tag] for tag in maturity_tags], fy_only=fy_only)


def quality_label(gap_pct: float | None, maturity_items_used: int) -> str:
    if maturity_items_used == 0:
        return "Insufficient maturity items"
    if gap_pct is None:
        return "No target debt"
    if gap_pct <= 0.05:
        return "OK"
    if gap_pct <= 0.15:
        return "Usable"
    return "Weak"


def to_usd_mm(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value) / USD_MM_DIVISOR


def format_usd_mm(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{to_usd_mm(value):,.1f}"


def status_style(check_status: str) -> tuple[str, str]:
    if check_status == "Passed":
        return ("#1b5e20", "#e8f5e9")
    if check_status == "Warning":
        return ("#8a6d1f", "#fff8e1")
    return ("#455a64", "#eceff1")


def tag_scope(tag: str | None) -> str | None:
    if not tag or pd.isna(tag):
        return None
    return "debt+lease" if "FinanceLease" in str(tag) else "debt-only"


def calc_gap(total_debt: float | None, maturity_sum: float) -> tuple[float | None, float | None]:
    if total_debt is not None and pd.notna(total_debt) and total_debt != 0:
        gap = float(total_debt) - maturity_sum
        return gap, abs(gap) / abs(float(total_debt))
    if total_debt is not None and pd.notna(total_debt):
        gap = float(total_debt) - maturity_sum
        return gap, None
    return None, None


def app_signature() -> str:
    file_path = Path(__file__)
    modified = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%B %d, %Y, %H:%M:%S")
    try:
        version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        version = "n/a"
    return f"Dr. Martin Lozano · Modified: {modified} · Version {version} · https://mlozanoqf.github.io/"


def render_sidebar_guide() -> None:
    st.markdown("---")
    st.subheader("Quick Guide")
    st.caption("General")
    st.caption("All monetary values are shown in USD mm.")
    st.caption("Check status: Passed, Warning, or Insufficient data.")

    module = st.selectbox(
        "Module guide",
        ["Cross-Section", "Time Series", "Panel", "Debt Distribution"],
        index=0,
        key="sidebar_module_guide",
    )

    if module == "Cross-Section":
        st.caption("Compare multiple firms at one selected year or quarter.")
        st.caption("Use this for same-period benchmarking across companies.")
    elif module == "Time Series":
        st.caption("Track one company through time for selected accounts.")
        st.caption("Use year range + frequency filters before interpreting trends.")
    elif module == "Panel":
        st.caption("Build long-format multi-company datasets over time.")
        st.caption("Best when you need export-ready data for modeling.")
    else:
        st.caption("Shows company-reported maturity items from SEC facts.")
        st.caption("Reconciliation compares maturity sum vs debt target.")
        st.caption("Method labels: Schedule-based and Current-debt adjusted.")


def render_debt_distribution(ticker_options: list[str], ticker_map: pd.DataFrame, user_agent: str) -> None:
    st.subheader("Debt Distribution")
    st.caption("Company-reported maturities with reconciliation to total debt.")
    st.caption(f"Units: {USD_MM_LABEL}")
    st.caption("Source order: filing XBRL (year-specific) -> companyfacts (same year only).")
    st.markdown("### A. Inputs")

    name_map = dict(zip(ticker_map["ticker"], ticker_map["title"]))

    ticker = st.selectbox(
        "Company",
        ticker_options,
        index=ticker_options.index("AAPL") if "AAPL" in ticker_options else 0,
        format_func=lambda t: ticker_display_label(str(t), name_map),
        key="dd_ticker",
    )

    if st.button("Run Debt Distribution", key="run_dd"):
        try:
            row = ticker_map[ticker_map["ticker"] == ticker]
            if row.empty:
                st.error(f"Ticker {ticker} not found.")
                return

            cik = normalize_cik(int(row.iloc[0]["cik_str"]))
            facts = get_company_facts(cik, user_agent)
        except requests.HTTPError as exc:
            st.error(parse_error_message(exc))
            return
        except Exception as exc:
            st.error(str(exc))
            return

        years = available_debt_years(facts, fy_only=True)
        fallback_to_non_fy = False
        if not years:
            years = available_debt_years(facts, fy_only=False)
            fallback_to_non_fy = True
        if not years:
            st.error("No debt-related tags were found for this company in companyfacts.")
            return

        default_year = years[-1]
        selected_year = st.selectbox("Fiscal year", years, index=years.index(default_year), key="dd_year")
        if fallback_to_non_fy:
            st.warning("FY observations were not found consistently; using any available period.")
        st.caption(f"Selected company: {ticker} - {name_map.get(ticker, 'Unknown company')}")

        reported_df, filing_accession, filing_total_rows = build_maturity_rows_from_filing(cik, selected_year, user_agent)
        maturity_source = "filing_xbrl"
        maturity_filing_link = None
        if reported_df.empty:
            reported_df = build_company_reported_maturity_rows(
                facts,
                selected_year,
                fy_only=not fallback_to_non_fy,
            )
            maturity_source = "companyfacts_same_year"

        if reported_df.empty:
            st.warning(f"No company-reported maturity items were found for fiscal year {selected_year}.")
            st.info(
                "Year-mixing is disabled: the app will not reconcile debt target from one year with maturity "
                "distribution from another year. Select another fiscal year or use a company with maturity detail."
            )
            return

        if maturity_source == "filing_xbrl":
            st.caption(f"Maturity source: Filing XBRL ({filing_accession})")
            maturity_filing_link = resolve_filing_url_for_row(
                cik=cik,
                user_agent=user_agent,
                form=None,
                filed=None,
                end=None,
                accession=filing_accession,
            )
            if maturity_filing_link:
                st.markdown(f"[Open maturity filing]({maturity_filing_link})")
        else:
            st.caption("Maturity source: SEC companyfacts (same fiscal year fallback)")

        totals_companyfacts = build_total_components_from_companyfacts(
            facts,
            selected_year,
            fy_only=not fallback_to_non_fy,
        )
        totals_filing = build_total_components_from_filing_rows(filing_total_rows)
        if maturity_source == "filing_xbrl":
            totals_df = combine_total_components(totals_filing, totals_companyfacts)
        else:
            totals_df = combine_total_components(totals_companyfacts, totals_filing)

        direct_total = totals_df.loc[totals_df["component"] == "Total debt (direct)", "value"].iloc[0]
        current_debt = totals_df.loc[totals_df["component"] == "Current debt", "value"].iloc[0]
        noncurrent_debt = totals_df.loc[totals_df["component"] == "Noncurrent debt", "value"].iloc[0]

        total_debt = direct_total
        debt_target_source = "direct total tag"
        if pd.isna(total_debt):
            if pd.notna(current_debt) and pd.notna(noncurrent_debt):
                total_debt = current_debt + noncurrent_debt
                debt_target_source = "current + noncurrent"
            else:
                total_debt = None
                debt_target_source = "not available"

        if total_debt is None or pd.isna(total_debt):
            st.warning(
                "Maturity distribution is available, but total debt target could not be built for this year/source mix. "
                "Reconciliation metrics will remain unavailable."
            )

        maturity_sum_baseline = float(reported_df.loc[reported_df["include_in_sum"], "amount"].sum())
        maturity_items_used = int(reported_df["include_in_sum"].sum())
        baseline_gap, baseline_gap_pct = calc_gap(total_debt, maturity_sum_baseline)

        short_term_schedule = float(
            reported_df.loc[
                reported_df["include_in_sum"] & (reported_df["concept"] == "next_12_months"),
                "amount",
            ].sum()
        )
        included_concepts = set(reported_df.loc[reported_df["include_in_sum"], "concept"].astype(str).tolist())
        has_short_bucket = "next_12_months" in included_concepts
        has_mid_bucket = bool(included_concepts.intersection({"year_2", "year_3", "year_4", "year_5", "years_2_to_5_agg"}))
        has_long_bucket = "after_year_5" in included_concepts
        coverage_penalty = (0.04 if not has_short_bucket else 0.0) + (0.04 if not has_long_bucket else 0.0)

        has_schedule_short_term = short_term_schedule != 0
        has_current_debt = current_debt is not None and pd.notna(current_debt)

        maturity_sum_anchored = None
        anchored_gap = None
        anchored_gap_pct = None
        if has_current_debt:
            maturity_sum_anchored = maturity_sum_baseline - short_term_schedule + float(current_debt)
            anchored_gap, anchored_gap_pct = calc_gap(total_debt, maturity_sum_anchored)

        direct_tag = totals_df.loc[totals_df["component"] == "Total debt (direct)", "tag"].iloc[0]
        current_tag = totals_df.loc[totals_df["component"] == "Current debt", "tag"].iloc[0]
        noncurrent_tag = totals_df.loc[totals_df["component"] == "Noncurrent debt", "tag"].iloc[0]
        total_source_components = totals_df.loc[totals_df["value"].notna(), "source"].astype(str).unique().tolist()
        total_source = ",".join(total_source_components) if total_source_components else "none"

        if debt_target_source == "direct total tag":
            target_scope = tag_scope(direct_tag)
        elif debt_target_source == "current + noncurrent":
            scopes = [tag_scope(current_tag), tag_scope(noncurrent_tag)]
            scopes = [s for s in scopes if s]
            target_scope = scopes[0] if scopes else None
        else:
            target_scope = None

        maturity_tags_used = reported_df.loc[reported_df["include_in_sum"], "tag"].dropna().tolist()
        maturity_scopes = [tag_scope(t) for t in maturity_tags_used if tag_scope(t)]
        maturity_scope = None
        if maturity_scopes:
            debt_lease_count = sum(1 for s in maturity_scopes if s == "debt+lease")
            maturity_scope = "debt+lease" if debt_lease_count >= (len(maturity_scopes) / 2) else "debt-only"

        scope_mismatch = bool(target_scope and maturity_scope and target_scope != maturity_scope)
        source_mismatch = bool(
            maturity_source == "filing_xbrl" and ("filing_xbrl" not in total_source_components)
        ) or bool(
            maturity_source == "companyfacts_same_year" and ("companyfacts" not in total_source_components)
        )
        short_term_mismatch = False
        if has_current_debt and has_schedule_short_term:
            denom = max(abs(float(current_debt)), abs(short_term_schedule), 1.0)
            short_term_mismatch = abs(float(current_debt) - short_term_schedule) / denom > 0.25

        methods = []
        methods.append(
                {
                    "method": "baseline_schedule_only",
                    "maturity_sum": maturity_sum_baseline,
                    "gap": baseline_gap,
                    "gap_pct": baseline_gap_pct,
                    "gap_pct_value": float("inf") if baseline_gap_pct is None else baseline_gap_pct,
                    "penalty": (0.02 if scope_mismatch else 0.0) + (0.02 if source_mismatch else 0.0) + coverage_penalty,
                    "short_term_source": "schedule <=1Y",
                }
            )
        if maturity_sum_anchored is not None:
            methods.append(
                {
                    "method": "short_term_anchored",
                    "maturity_sum": maturity_sum_anchored,
                    "gap": anchored_gap,
                    "gap_pct": anchored_gap_pct,
                    "gap_pct_value": float("inf") if anchored_gap_pct is None else anchored_gap_pct,
                    "penalty": (0.02 if scope_mismatch else 0.0) + (0.03 if short_term_mismatch else 0.0) + (0.02 if source_mismatch else 0.0) + coverage_penalty,
                    "short_term_source": "current debt (balance sheet)",
                }
            )

        for m in methods:
            m["score"] = m["gap_pct_value"] + m["penalty"]
        methods_df = pd.DataFrame(methods).sort_values("score", ascending=True)
        chosen = methods_df.iloc[0].to_dict()
        alt_method = methods_df.iloc[1].to_dict() if len(methods_df) > 1 else None

        maturity_sum = float(chosen["maturity_sum"])
        gap = chosen["gap"]
        gap_pct = chosen["gap_pct"]
        chosen_method = str(chosen["method"])
        short_term_source_used = chosen["short_term_source"]
        chosen_method_label = METHOD_LABELS.get(chosen_method, chosen_method)

        # Block B: Debt Distribution (main result)
        st.markdown("### B. Debt Distribution")
        dist_df = reported_df[["item", "amount", "tag", "include_in_sum", "exclusion_reason"]].rename(
            columns={
                "item": "Maturity item (company-reported)",
                "amount": "amount_usd_raw",
                "tag": "Source tag",
                "include_in_sum": "Included in maturity sum",
                "exclusion_reason": "Note",
            }
        )
        dist_df.insert(0, "Period", f"FY {selected_year}")
        dist_df["amount_usd_mm"] = dist_df["amount_usd_raw"] / USD_MM_DIVISOR
        dist_df["Included in maturity sum"] = dist_df["Included in maturity sum"].map(lambda v: "Yes" if bool(v) else "No")
        dist_df.loc[dist_df["Included in maturity sum"] == "Yes", "Note"] = ""
        total_row = pd.DataFrame(
            [
                {
                    "Period": f"FY {selected_year}",
                    "Maturity item (company-reported)": "Total maturities (selected items)",
                    "amount_usd_raw": maturity_sum_baseline,
                    "amount_usd_mm": maturity_sum_baseline / USD_MM_DIVISOR,
                    "Source tag": "computed",
                    "Included in maturity sum": "Yes",
                    "Note": "",
                }
            ]
        )
        dist_df = pd.concat([dist_df, total_row], ignore_index=True)

        if total_debt is not None and pd.notna(total_debt) and total_debt != 0:
            dist_df["pct_of_total_debt_raw"] = dist_df["amount_usd_raw"] / float(total_debt)
            dist_df["% of total debt"] = dist_df["pct_of_total_debt_raw"].map(lambda x: f"{x:.1%}")
        else:
            dist_df["pct_of_total_debt_raw"] = None
            dist_df["% of total debt"] = "n/a"

        dist_display = dist_df.rename(columns={"amount_usd_mm": "Amount (USD mm)"})
        st.dataframe(
            dist_display[
                [
                    "Period",
                    "Maturity item (company-reported)",
                    "Amount (USD mm)",
                    "% of total debt",
                    "Included in maturity sum",
                    "Source tag",
                    "Note",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        chart_df = reported_df[reported_df["include_in_sum"]].copy()
        label_info = chart_df.apply(
            lambda r: maturity_axis_label_info(str(r["item"]), str(r["concept"]), selected_year, str(r.get("tag", ""))),
            axis=1,
        )
        chart_df["Maturity"] = label_info.map(lambda x: x[0])
        chart_df["label_type"] = label_info.map(lambda x: x[1])
        chart_df = chart_df.groupby("Maturity", as_index=False)["amount"].sum()
        chart_df["amount_usd_mm"] = chart_df["amount"] / USD_MM_DIVISOR
        if not chart_df.empty:
            chart_df["sort_val"] = chart_df["Maturity"].map(lambda x: maturity_label_sort_value(str(x), selected_year))
            chart_df = chart_df.sort_values(["sort_val", "Maturity"]).drop(columns=["sort_val"])
            st.bar_chart(chart_df.set_index("Maturity")[["amount_usd_mm"]])
            if chart_df["Maturity"].astype(str).str.contains(r"\*").any():
                st.caption("* Open-ended aggregated maturity bucket (e.g., thereafter/after-year range).")

        # Block C: Reconciliation check (independent diagnostic)
        st.markdown("### C. Reconciliation Check")
        quality = quality_label(gap_pct, maturity_items_used)
        if quality == "OK":
            check_status = "Passed"
        elif quality in ["Usable", "Weak"]:
            check_status = "Warning"
        else:
            check_status = "Insufficient data"

        overview = pd.DataFrame(
            [
                {
                    "Total Debt Target (USD mm)": format_usd_mm(total_debt),
                    "Maturity Sum (USD mm)": format_usd_mm(maturity_sum),
                    "Gap (USD mm)": format_usd_mm(gap),
                    "Gap %": "n/a" if gap_pct is None else f"{gap_pct:.1%}",
                    "Chosen Method": chosen_method_label,
                    "Check Status": check_status,
                }
            ]
        )
        st.dataframe(overview, use_container_width=True, hide_index=True)
        status_color, status_bg = status_style(check_status)
        st.markdown(
            f"<span style='color:{status_color};background:{status_bg};padding:0.2rem 0.45rem;border-radius:0.35rem;font-weight:700;'>{check_status}</span>",
            unsafe_allow_html=True,
        )
        if alt_method is not None:
            alt_method_name = METHOD_LABELS.get(str(alt_method["method"]), str(alt_method["method"]))
            alt_gap_text = "n/a" if pd.isna(alt_method["gap_pct"]) else f"{float(alt_method['gap_pct']):.1%}"
            st.caption(
                "Alternative method: "
                f"{alt_method_name} (gap {alt_gap_text}, "
                f"score {alt_method['score']:.3f})."
            )
        if short_term_mismatch:
            st.warning("Short-term mismatch detected: schedule <=1Y differs materially from current debt.")
        if not has_short_bucket or not has_long_bucket:
            st.warning(
                "Coverage warning: "
                f"<=1Y present={has_short_bucket}, >{selected_year + 5} present={has_long_bucket}. "
                "Missing edge buckets can inflate reconciliation gap."
            )
        st.caption(f"Amounts shown as {USD_MM_LABEL}.")

        summary = pd.DataFrame(
            [
                {"item": "Ticker", "value": ticker},
                {"item": "Company", "value": name_map.get(ticker, None)},
                {"item": "Fiscal year", "value": selected_year},
                {"item": "Total debt target (USD mm)", "value": to_usd_mm(total_debt)},
                {"item": "Debt target source", "value": debt_target_source},
                {"item": "Maturity source", "value": maturity_source},
                {"item": "Total debt source", "value": total_source},
                {"item": "Chosen reconciliation method", "value": chosen_method_label},
                {"item": "Short-term source used", "value": short_term_source_used},
                {"item": "Scope mismatch flag", "value": scope_mismatch},
                {"item": "Source mismatch flag", "value": source_mismatch},
                {"item": "Short-term mismatch flag", "value": short_term_mismatch},
                {"item": "Has <=1Y maturity bucket", "value": has_short_bucket},
                {"item": f"Has >{selected_year + 5} maturity bucket", "value": has_long_bucket},
                {"item": "Has mid-term maturity bucket", "value": has_mid_bucket},
                {"item": "Company-reported maturity items found", "value": int(len(reported_df))},
                {"item": "Items included in maturity sum", "value": maturity_items_used},
                {"item": "Maturity sum (USD mm)", "value": to_usd_mm(maturity_sum)},
                {"item": "Gap (target - maturity sum) USD mm", "value": to_usd_mm(gap)},
                {"item": "Gap %", "value": "n/a" if gap_pct is None else f"{gap_pct:.1%}"},
                {"item": "Quality", "value": quality},
                {"item": "Check status", "value": check_status},
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.download_button(
            "Download debt distribution CSV",
            dist_df.to_csv(index=False),
            file_name=f"debt_distribution_{ticker}_{selected_year}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download reconciliation CSV",
            summary.to_csv(index=False),
            file_name=f"reconciliation_{ticker}_{selected_year}.csv",
            mime="text/csv",
        )

        with st.expander("Show technical details (raw components and tags)"):
            st.markdown("Debt target components")
            st.dataframe(totals_df, use_container_width=True, hide_index=True)
            st.markdown("Company-reported maturity extraction")
            st.dataframe(reported_df, use_container_width=True, hide_index=True)
            st.markdown("Chart label diagnostics")
            chart_diag = reported_df[reported_df["include_in_sum"]].copy()
            chart_diag_info = chart_diag.apply(
                lambda r: maturity_axis_label_info(str(r["item"]), str(r["concept"]), selected_year, str(r.get("tag", ""))),
                axis=1,
            )
            chart_diag["chart_label"] = chart_diag_info.map(lambda x: x[0])
            chart_diag["label_type"] = chart_diag_info.map(lambda x: x[1])
            st.dataframe(chart_diag[["item", "tag", "concept", "chart_label", "label_type", "amount"]], use_container_width=True, hide_index=True)
            st.markdown("Reconciliation methods comparison")
            methods_display = methods_df.copy()
            methods_display["method"] = methods_display["method"].map(lambda x: METHOD_LABELS.get(str(x), str(x)))
            methods_display["gap_pct"] = methods_display["gap_pct"].map(lambda x: "n/a" if pd.isna(x) else f"{float(x):.1%}")
            st.dataframe(methods_display, use_container_width=True, hide_index=True)
            st.markdown(
                f"{METHOD_LABELS['baseline_schedule_only']}: {METHOD_HELP['Schedule-based']} "
                f"{METHOD_LABELS['short_term_anchored']}: {METHOD_HELP['Current-debt adjusted']}"
            )


def get_company_metric_df(ticker: str, ticker_map: pd.DataFrame, user_agent: str) -> pd.DataFrame:
    row = ticker_map[ticker_map["ticker"] == ticker]
    if row.empty:
        raise ValueError(f"Ticker {ticker} not found")

    cik = normalize_cik(int(row.iloc[0]["cik_str"]))
    facts = get_company_facts(cik, user_agent)
    return extract_metric_rows(facts, ticker)


def get_company_cross_section_df(ticker: str, ticker_map: pd.DataFrame, user_agent: str, metrics: list[str]) -> pd.DataFrame:
    row = ticker_map[ticker_map["ticker"] == ticker]
    if row.empty:
        raise ValueError(f"Ticker {ticker} not found")

    cik = normalize_cik(int(row.iloc[0]["cik_str"]))
    facts = get_company_facts(cik, user_agent)

    rows = []
    for metric in metrics:
        metric_rows = build_metric_rows_for_metric(facts, ticker, metric)
        if not metric_rows.empty:
            rows.append(metric_rows)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


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


def load_cross_section_dataset(
    selected_tickers: list[str], ticker_map: pd.DataFrame, user_agent: str, metrics: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    data_frames = []
    errors = []

    for ticker in selected_tickers:
        try:
            df = get_company_cross_section_df(ticker, ticker_map, user_agent, metrics)
            if not df.empty:
                data_frames.append(df)
            else:
                errors.append(f"{ticker}: no aligned metric data found")
        except requests.HTTPError as exc:
            errors.append(f"{ticker}: {parse_error_message(exc)}")
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    if not data_frames:
        return pd.DataFrame(), errors

    return pd.concat(data_frames, ignore_index=True), errors


def cross_section_metric_display_name(metric: str, period_type: str) -> str:
    if period_type.startswith("Quarterly") and metric in FLOW_METRICS:
        return f"{metric} (YTD)"
    return metric


def format_period_label(fy: object, fp: object) -> str:
    if pd.isna(fy) and pd.isna(fp):
        return "n/a"
    fy_txt = ""
    if not pd.isna(fy):
        try:
            fy_txt = str(int(fy))
        except Exception:
            fy_txt = str(fy)
    fp_txt = "" if pd.isna(fp) else str(fp)
    if fp_txt == "FY":
        return f"FY {fy_txt}".strip()
    if fp_txt and fy_txt:
        return f"{fp_txt} {fy_txt}"
    return fp_txt or fy_txt or "n/a"


def ticker_display_label(ticker: str, name_map: dict[str, str]) -> str:
    return f"{ticker} - {name_map.get(ticker, 'Unknown company')}"


def build_link_column_config(columns: list[str]) -> dict[str, object]:
    config: dict[str, object] = {}
    for col in columns:
        config[col] = st.column_config.LinkColumn(col, display_text="Open filing")
    return config


def sec_sic_cache_status() -> dict[str, object]:
    if not SEC_SIC_CACHE_PATH.exists():
        return {"status": "missing", "modified": None, "age_days": None}
    modified = datetime.fromtimestamp(SEC_SIC_CACHE_PATH.stat().st_mtime)
    age_days = (datetime.now() - modified).days
    status = "fresh" if age_days <= 30 else "stale"
    return {"status": status, "modified": modified, "age_days": age_days}


def balance_check_status(gap_pct: float | None) -> str:
    if gap_pct is None or pd.isna(gap_pct):
        return "n/a"
    gap = abs(float(gap_pct))
    if gap <= 0.01:
        return "Passed"
    if gap <= 0.05:
        return "Warning"
    return "Review"


def render_cross_section_chart_block(
    pivot_mm: pd.DataFrame,
    metrics_subset: list[str],
    period_type: str,
    selected_period_label: str,
    title: str,
    name_map: dict[str, str],
) -> None:
    if not metrics_subset:
        return

    chart_rows = []
    for _, row in pivot_mm.iterrows():
        ticker = row["ticker"]
        for metric in metrics_subset:
            display_col = f"{cross_section_metric_display_name(metric, period_type)} ({selected_period_label}, USD mm)"
            value = row.get(display_col)
            if pd.isna(value):
                continue
            chart_rows.append(
                {
                    "Series": f"{ticker} | {cross_section_metric_display_name(metric, period_type)}",
                    "value_usd_mm": value,
                    "ticker": ticker,
                    "company_name": name_map.get(ticker, ticker),
                }
            )

    if not chart_rows:
        return

    st.caption(f"{title} ({selected_period_label})")
    chart_df = pd.DataFrame(chart_rows)
    try:
        import altair as alt

        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Series:N", sort=None, title=None),
                y=alt.Y("value_usd_mm:Q", title=f"Value ({USD_MM_LABEL})"),
                color=alt.Color(
                    "company_name:N",
                    title=None,
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("company_name:N", title="Company"),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("value_usd_mm:Q", title="Value (USD mm)", format=",.2f"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    except ImportError:
        st.caption("Company-color grouping is unavailable until `altair` is installed; using fallback chart.")
        st.bar_chart(chart_df.set_index("Series")[["value_usd_mm"]])


def select_cross_section_snapshots(data: pd.DataFrame, metrics: list[str], period_type: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    work = data.dropna(subset=["fy", "end", "val"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["form"] = work["form"].fillna("")
    work["is_flow_metric"] = work["metric"].isin(FLOW_METRICS)
    work["has_start"] = work["start"].notna()
    work["period_length_days"] = (work["end"] - work["start"]).dt.days
    work["period_length_days"] = work["period_length_days"].where(work["period_length_days"] >= 0)
    use_ytd_for_flows = period_type.startswith("Quarterly")
    work["flow_duration_priority"] = -1
    if use_ytd_for_flows:
        flow_mask = work["is_flow_metric"]
        work.loc[flow_mask, "flow_duration_priority"] = work.loc[flow_mask, "period_length_days"].fillna(-1)

    work = work.sort_values(
        ["ticker", "fy", "fp", "end", "filed", "form", "metric", "has_start", "flow_duration_priority", "tag_rank"],
        ascending=[True, True, True, False, False, False, True, False, False, True],
    )
    best_rows = work.drop_duplicates(subset=CROSS_SECTION_KEY_COLS + ["metric"], keep="first")

    value_snapshots = best_rows.pivot(
        index=CROSS_SECTION_KEY_COLS,
        columns="metric",
        values="val",
    ).reset_index()
    if value_snapshots.empty:
        return pd.DataFrame()

    tag_snapshots = best_rows.pivot(
        index=CROSS_SECTION_KEY_COLS,
        columns="metric",
        values="tag",
    ).reset_index()
    tag_snapshots = tag_snapshots.rename(columns={metric: f"{metric} tag" for metric in metrics if metric in tag_snapshots.columns})

    snapshots = value_snapshots.merge(tag_snapshots, on=CROSS_SECTION_KEY_COLS, how="left")
    metric_values = snapshots.reindex(columns=metrics)
    snapshots["available_metric_count"] = metric_values.notna().sum(axis=1)
    snapshots["has_all_selected_metrics"] = snapshots["available_metric_count"] == len(metrics)

    balance_values = snapshots.reindex(columns=["Assets", "Liabilities", "Equity"])
    snapshots["has_balance_triplet"] = balance_values.notna().all(axis=1)
    snapshots["Liabilities + Equity"] = balance_values["Liabilities"] + balance_values["Equity"]
    snapshots["Balance gap"] = balance_values["Assets"] - snapshots["Liabilities + Equity"]
    snapshots["Balance gap %"] = snapshots["Balance gap"] / balance_values["Assets"].abs()
    snapshots["balance_gap_abs"] = snapshots["Balance gap"].abs().fillna(float("inf"))
    snapshots["flow_basis"] = "As reported"
    if use_ytd_for_flows:
        snapshots["flow_basis"] = "YTD"

    ranked = snapshots.sort_values(
        [
            "ticker",
            "has_all_selected_metrics",
            "available_metric_count",
            "has_balance_triplet",
            "balance_gap_abs",
            "filed",
            "end",
        ],
        ascending=[True, False, False, False, True, False, False],
    )
    return ranked.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)


def apply_strict_flow_pair_overrides(
    snapshots: pd.DataFrame,
    ticker_map: pd.DataFrame,
    user_agent: str,
    period_type: str,
    selected_quarter: str | None,
    metrics: list[str],
) -> pd.DataFrame:
    if snapshots.empty or not {"Revenue", "Net Income"}.issubset(set(metrics)):
        return snapshots

    out = snapshots.copy()
    out["flow_context_source"] = "not_applied"
    out["flow_context_ref"] = None
    out["flow_filing_accession"] = None
    out["flow_period_length_days"] = None
    out["flow_common_end"] = pd.NaT
    out["flow_filing_date"] = pd.NaT

    for idx, row in out.iterrows():
        ticker = str(row["ticker"])
        ticker_row = ticker_map[ticker_map["ticker"] == ticker]
        if ticker_row.empty:
            continue
        cik = normalize_cik(int(ticker_row.iloc[0]["cik_str"]))
        strict_pair = get_strict_flow_pair_for_period(
            cik=cik,
            fiscal_year=int(row["fy"]),
            period_type=period_type,
            selected_quarter=selected_quarter,
            target_end=row["end"],
            user_agent=user_agent,
        )
        if not strict_pair:
            continue
        out.at[idx, "Revenue"] = strict_pair["Revenue"]
        out.at[idx, "Revenue tag"] = strict_pair["Revenue tag"]
        out.at[idx, "Net Income"] = strict_pair["Net Income"]
        out.at[idx, "Net Income tag"] = strict_pair["Net Income tag"]
        out.at[idx, "flow_context_source"] = strict_pair["flow_context_source"]
        out.at[idx, "flow_context_ref"] = strict_pair["flow_context_ref"]
        out.at[idx, "flow_filing_accession"] = strict_pair["flow_filing_accession"]
        out.at[idx, "flow_period_length_days"] = strict_pair["flow_period_length_days"]
        out.at[idx, "flow_common_end"] = strict_pair["flow_common_end"]
        out.at[idx, "flow_filing_date"] = strict_pair["flow_filing_date"]
        out.at[idx, "flow_basis"] = "YTD (same XBRL context)" if period_type.startswith("Quarterly") else "As reported (same XBRL context)"

    return out


def apply_strict_flow_pair_to_long_rows(
    rows: pd.DataFrame,
    ticker: str,
    ticker_map: pd.DataFrame,
    user_agent: str,
    period_type: str,
) -> pd.DataFrame:
    if rows.empty or not {"Revenue", "Net Income"}.issubset(set(rows["metric"].astype(str).unique().tolist())):
        return rows

    ticker_row = ticker_map[ticker_map["ticker"] == ticker]
    if ticker_row.empty:
        return rows

    cik = normalize_cik(int(ticker_row.iloc[0]["cik_str"]))
    out = rows.copy()
    out["flow_context_source"] = "not_applied"
    out["flow_context_ref"] = None
    out["flow_filing_accession"] = None

    period_keys = out[["fy", "fp", "end"]].drop_duplicates()
    for _, period in period_keys.iterrows():
        fy = period["fy"]
        fp = period["fp"]
        end = period["end"]
        if pd.isna(fy) or pd.isna(fp) or pd.isna(end):
            continue

        period_mask = (out["fy"] == fy) & (out["fp"] == fp) & (out["end"] == end)
        period_rows = out.loc[period_mask].copy()
        metric_set = set(period_rows["metric"].astype(str).tolist())
        if not {"Revenue", "Net Income"}.issubset(metric_set):
            continue

        strict_pair = get_strict_flow_pair_for_period(
            cik=cik,
            fiscal_year=int(fy),
            period_type=period_type,
            selected_quarter=None if str(fp) == "FY" else str(fp),
            target_end=end,
            user_agent=user_agent,
        )
        if not strict_pair:
            continue

        revenue_mask = period_mask & (out["metric"] == "Revenue")
        net_income_mask = period_mask & (out["metric"] == "Net Income")

        out.loc[revenue_mask, "val"] = strict_pair["Revenue"]
        out.loc[revenue_mask, "tag"] = strict_pair["Revenue tag"]
        out.loc[net_income_mask, "val"] = strict_pair["Net Income"]
        out.loc[net_income_mask, "tag"] = strict_pair["Net Income tag"]
        out.loc[period_mask, "flow_context_source"] = strict_pair["flow_context_source"]
        out.loc[period_mask, "flow_context_ref"] = strict_pair["flow_context_ref"]
        out.loc[period_mask, "flow_filing_accession"] = strict_pair["flow_filing_accession"]

    return out


def render_cross_section(ticker_options: list[str], ticker_map: pd.DataFrame, user_agent: str) -> None:
    st.subheader("Cross-Section")
    st.caption("Select one year or quarter and compare multiple companies across selected accounts.")
    st.caption(f"Units: {USD_MM_LABEL}")
    name_map = dict(zip(ticker_map["ticker"], ticker_map["title"]))

    companies = st.multiselect(
        "Companies",
        ticker_options,
        default=[t for t in ["AAPL", "MSFT", "GOOGL"] if t in ticker_options],
        format_func=lambda t: ticker_display_label(str(t), name_map),
        key="cs_companies",
    )
    metrics = st.multiselect(
        "Accounts",
        list(FACT_TAGS.keys()),
        default=list(FACT_TAGS.keys()),
        key="cs_metrics",
    )
    period_type = st.selectbox("Period type", ["Annual (FY)", "Quarterly (Q1-Q4)"], key="cs_period_type")

    if not companies:
        st.warning("Select at least one company.")
        return
    if not metrics:
        st.warning("Select at least one account.")
        return

    data, errors = load_cross_section_dataset(companies, ticker_map, user_agent, metrics)
    if errors:
        st.warning(" | ".join(errors))
    if data.empty:
        st.error("No data available for the selected filters.")
        return

    selected_quarter = None
    selected_period_label = None
    if period_type.startswith("Annual"):
        annual = data[data["fp"] == "FY"].copy()
        annual = annual.dropna(subset=["fy"])
        if annual.empty:
            st.error("No annual observations (FY) found.")
            return

        years = sorted(annual["fy"].dropna().astype(int).unique())
        selected_year = st.selectbox("Year", years, index=len(years) - 1, key="cs_year")
        selected_period_label = format_period_label(selected_year, "FY")

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
        selected_period_label = format_period_label(selected_year, selected_quarter)

        filtered = quarter[(quarter["fy"] == selected_year) & (quarter["fp"] == selected_quarter)].copy()

    if st.button("Run Cross-Section", key="run_cs"):
        st.session_state["cs_results_visible"] = True

    if not st.session_state.get("cs_results_visible", False):
        return

    if filtered.empty:
        st.error("No rows found for that period.")
        return

    selected_snapshots = select_cross_section_snapshots(filtered, metrics, period_type)
    if selected_snapshots.empty:
        st.error("No aligned filing snapshot found for the selected period.")
        return
    selected_snapshots = apply_strict_flow_pair_overrides(
        selected_snapshots,
        ticker_map=ticker_map,
        user_agent=user_agent,
        period_type=period_type,
        selected_quarter=selected_quarter,
        metrics=metrics,
    )
    selected_snapshots["primary_filing_link"] = selected_snapshots.apply(
        lambda r: resolve_filing_url_for_row(
            cik=normalize_cik(int(ticker_map.loc[ticker_map["ticker"] == r["ticker"], "cik_str"].iloc[0])),
            user_agent=user_agent,
            form=r["form"],
            filed=r["filed"],
            end=r["end"],
        ),
        axis=1,
    )
    if "flow_filing_accession" in selected_snapshots.columns:
        selected_snapshots["flow_filing_link"] = selected_snapshots.apply(
            lambda r: resolve_filing_url_for_row(
                cik=normalize_cik(int(ticker_map.loc[ticker_map["ticker"] == r["ticker"], "cik_str"].iloc[0])),
                user_agent=user_agent,
                form=r["form"],
                filed=r.get("flow_filing_date", r["filed"]),
                end=r.get("flow_common_end", r["end"]),
                accession=r.get("flow_filing_accession"),
            )
            if pd.notna(r.get("flow_filing_accession"))
            else None,
            axis=1,
        )

    snapshot_cols = ["ticker"] + [metric for metric in metrics if metric in selected_snapshots.columns]
    pivot_usd = selected_snapshots[snapshot_cols].copy()
    pivot_mm = pivot_usd.copy()
    metric_cols = [c for c in pivot_mm.columns if c != "ticker"]
    for col in metric_cols:
        pivot_mm[col] = pivot_mm[col] / USD_MM_DIVISOR
    pivot_mm = pivot_mm.rename(
        columns={
            col: f"{cross_section_metric_display_name(col, period_type)} ({selected_period_label}, USD mm)"
            for col in metric_cols
        }
    )
    st.caption(f"All monetary values are shown in {USD_MM_LABEL}.")
    if period_type.startswith("Quarterly") and any(metric in FLOW_METRICS for metric in metrics):
        st.caption(
            "Quarterly flow metrics (`Revenue`, `Net Income`) are shown as year-to-date (YTD) values from the same aligned income-statement period."
        )
    if {"Revenue", "Net Income"}.issubset(set(metrics)):
        exact_context_count = int((selected_snapshots["flow_context_source"] == "filing_xbrl_same_context").sum()) if "flow_context_source" in selected_snapshots.columns else 0
        if exact_context_count > 0:
            st.caption(
                f"Strict income-statement pairing: {exact_context_count} company(ies) use `Revenue` and `Net Income` from the same filing XBRL context."
            )
        if exact_context_count < len(selected_snapshots):
            st.warning(
                "Some companies could not be paired from the same filing XBRL context for `Revenue` and `Net Income`; those cases fall back to the aligned companyfacts snapshot."
            )
    st.dataframe(pivot_mm, use_container_width=True, hide_index=True)
    if metric_cols:
        stock_metrics_selected = [m for m in metric_cols if m in STOCK_METRICS]
        flow_metrics_selected = [m for m in metric_cols if m in FLOW_METRICS]
        other_metrics_selected = [m for m in metric_cols if m not in STOCK_METRICS and m not in FLOW_METRICS]

        if stock_metrics_selected and flow_metrics_selected:
            render_cross_section_chart_block(
                pivot_mm,
                stock_metrics_selected,
                period_type,
                selected_period_label,
                "Chart view: balance-sheet accounts",
                name_map,
            )
            render_cross_section_chart_block(
                pivot_mm,
                flow_metrics_selected,
                period_type,
                selected_period_label,
                "Chart view: flow accounts",
                name_map,
            )
        else:
            primary_metrics = stock_metrics_selected or flow_metrics_selected or metric_cols
            render_cross_section_chart_block(
                pivot_mm,
                primary_metrics,
                period_type,
                selected_period_label,
                "Chart view",
                name_map,
            )

        if other_metrics_selected:
            render_cross_section_chart_block(
                pivot_mm,
                other_metrics_selected,
                period_type,
                selected_period_label,
                "Chart view: other accounts",
                name_map,
            )

    incomplete = selected_snapshots[selected_snapshots["available_metric_count"] < len(metrics)][
        ["ticker", "available_metric_count"]
    ].copy()
    if not incomplete.empty:
        incomplete["note"] = incomplete["available_metric_count"].map(
            lambda n: f"{int(n)}/{len(metrics)} selected accounts available in one aligned filing snapshot"
        )
        st.warning(
            "Some companies do not have all selected accounts in a single aligned filing snapshot for this period."
        )
        st.dataframe(incomplete[["ticker", "note"]], use_container_width=True, hide_index=True)

    st.caption("Source alignment: one filing snapshot per company, matched by fiscal period, end date, filing date, and form. Segmented facts are excluded.")
    snapshot_meta_cols = ["ticker", "end", "filed", "form", "available_metric_count"]
    if period_type.startswith("Quarterly") and any(metric in FLOW_METRICS for metric in metrics):
        snapshot_meta_cols.append("flow_basis")
    if {"Revenue", "Net Income"}.issubset(set(metrics)):
        snapshot_meta_cols.extend(["flow_context_source", "flow_context_ref"])
        if "flow_filing_link" in selected_snapshots.columns:
            snapshot_meta_cols.append("flow_filing_link")
    snapshot_meta_cols.append("primary_filing_link")
    snapshot_meta = selected_snapshots[snapshot_meta_cols].copy()
    snapshot_meta = snapshot_meta.rename(columns={"available_metric_count": "accounts_in_selected_snapshot"})
    snapshot_meta.insert(1, "selected_period", selected_period_label)
    link_cols = [c for c in ["primary_filing_link", "flow_filing_link"] if c in snapshot_meta.columns]
    st.dataframe(
        snapshot_meta,
        use_container_width=True,
        hide_index=True,
        column_config=build_link_column_config(link_cols) if link_cols else None,
    )

    has_balance_inputs = all(col in pivot_usd.columns for col in ["Assets", "Liabilities", "Equity"])
    if has_balance_inputs:
        balance_df = selected_snapshots[
            ["ticker", "Assets", "Liabilities", "Equity", "Liabilities + Equity", "Balance gap", "Balance gap %"]
        ].copy()
        balance_df["Balance check"] = balance_df["Balance gap %"].map(balance_check_status)

        balance_display = balance_df.copy()
        for col in ["Assets", "Liabilities", "Equity", "Liabilities + Equity", "Balance gap"]:
            balance_display[col] = balance_display[col] / USD_MM_DIVISOR
        balance_display["Balance gap %"] = balance_display["Balance gap %"].map(
            lambda x: "n/a" if pd.isna(x) else f"{float(x):.2%}"
        )
        balance_display = balance_display.rename(
            columns={
                "Assets": "Assets (USD mm)",
                "Liabilities": "Liabilities (USD mm)",
                "Equity": "Equity (USD mm)",
                "Liabilities + Equity": "Liabilities + Equity (USD mm)",
                "Balance gap": "Balance gap (USD mm)",
            }
        )

        status_counts = balance_df["Balance check"].value_counts().to_dict()
        st.caption(
            "Balance check summary (diagnostic only): "
            f"Passed {status_counts.get('Passed', 0)} | "
            f"Warning {status_counts.get('Warning', 0)} | "
            f"Review {status_counts.get('Review', 0)}"
        )
        badge_specs = {
            "Passed": ("#1b5e20", "#e8f5e9"),
            "Warning": ("#8a6d1f", "#fff8e1"),
            "Review": ("#b71c1c", "#ffebee"),
        }
        active_badges = []
        for label in ["Passed", "Warning", "Review"]:
            if status_counts.get(label, 0) > 0:
                fg, bg = badge_specs[label]
                active_badges.append(
                    f"<span style='color:{fg};background:{bg};padding:0.15rem 0.4rem;border-radius:0.35rem;font-weight:700;'>{label}</span>"
                )
        if active_badges:
            st.markdown(" ".join(active_badges), unsafe_allow_html=True)
        with st.expander("Show balance diagnostics"):
            st.caption(
                "Diagnostic only: this is a consistency check, not a primary output table. "
                "Residual differences can reflect issuer-specific equity presentation."
            )
            st.dataframe(balance_display, use_container_width=True, hide_index=True)

    pivot_csv = selected_snapshots[["ticker", "fy", "fp", "end", "filed", "form"]].copy()
    pivot_csv.insert(1, "period_label", selected_snapshots.apply(lambda r: format_period_label(r["fy"], r["fp"]), axis=1))
    if period_type.startswith("Quarterly") and any(metric in FLOW_METRICS for metric in metrics):
        pivot_csv["flow_basis"] = selected_snapshots["flow_basis"]
    if {"Revenue", "Net Income"}.issubset(set(metrics)) and "flow_context_source" in selected_snapshots.columns:
        pivot_csv["flow_context_source"] = selected_snapshots["flow_context_source"]
        pivot_csv["flow_context_ref"] = selected_snapshots["flow_context_ref"]
        pivot_csv["flow_filing_accession"] = selected_snapshots["flow_filing_accession"]
        if "flow_filing_link" in selected_snapshots.columns:
            pivot_csv["flow_filing_link"] = selected_snapshots["flow_filing_link"]
    pivot_csv["primary_filing_link"] = selected_snapshots["primary_filing_link"]
    for metric in metrics:
        if metric in selected_snapshots.columns:
            pivot_csv[metric] = selected_snapshots[metric]
        tag_col = f"{metric} tag"
        if tag_col in selected_snapshots.columns:
            pivot_csv[tag_col] = selected_snapshots[tag_col]
    for col in metric_cols:
        pivot_csv[f"{col}_usd_mm"] = pivot_csv[col] / USD_MM_DIVISOR
    if has_balance_inputs:
        pivot_csv["liabilities_plus_equity_usd"] = selected_snapshots["Liabilities + Equity"]
        pivot_csv["balance_gap_usd"] = selected_snapshots["Balance gap"]
        pivot_csv["balance_gap_pct"] = selected_snapshots["Balance gap %"]
    st.download_button(
        "Download cross-section CSV",
        pivot_csv.to_csv(index=False),
        file_name="cross_section.csv",
        mime="text/csv",
    )


def render_time_series(ticker_options: list[str], ticker_map: pd.DataFrame, user_agent: str) -> None:
    st.subheader("Time Series")
    st.caption("Select one company and track selected accounts through time.")
    st.caption(f"Units: {USD_MM_LABEL}")
    name_map = dict(zip(ticker_map["ticker"], ticker_map["title"]))

    ticker = st.selectbox(
        "Company",
        ticker_options,
        index=ticker_options.index("AAPL") if "AAPL" in ticker_options else 0,
        format_func=lambda t: ticker_display_label(str(t), name_map),
        key="ts_ticker",
    )
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
        latest = data.drop_duplicates(subset=["end", "metric"], keep="first").copy()
        if {"Revenue", "Net Income"}.issubset(set(metrics)):
            period_type_label = "Annual (FY)" if freq == "Annual" else "Quarterly (Q1-Q4)"
            latest = apply_strict_flow_pair_to_long_rows(
                latest,
                ticker=ticker,
                ticker_map=ticker_map,
                user_agent=user_agent,
                period_type=period_type_label,
            )

        latest["val_usd_mm"] = latest["val"] / USD_MM_DIVISOR
        chart_df = latest.pivot(index="end", columns="metric", values="val_usd_mm").sort_index()
        if freq == "Quarterly":
            chart_df = chart_df.rename(columns={m: cross_section_metric_display_name(m, "Quarterly (Q1-Q4)") for m in chart_df.columns})
        st.line_chart(chart_df)
        if {"Revenue", "Net Income"}.issubset(set(metrics)) and "flow_context_source" in latest.columns:
            strict_periods = (
                latest[latest["flow_context_source"] == "filing_xbrl_same_context"][["fy", "fp", "end"]]
                .drop_duplicates()
            )
            if not strict_periods.empty:
                st.caption(
                    f"Strict income-statement pairing applied in {len(strict_periods)} period(s): `Revenue` and `Net Income` come from the same filing XBRL context."
                )
            if len(strict_periods) < len(latest[["fy", "fp", "end"]].drop_duplicates()):
                st.warning(
                    "Some periods could not be paired from the same filing XBRL context for `Revenue` and `Net Income`; those periods use the aligned companyfacts rows."
                )

        out_cols = ["ticker", "metric", "fy", "fp", "end", "val", "val_usd_mm", "form", "filed"]
        if "flow_context_source" in latest.columns:
            out_cols.extend(["flow_context_source", "flow_context_ref", "flow_filing_accession"])
        out = latest[out_cols].sort_values("end")
        if freq == "Quarterly":
            out["metric"] = out["metric"].map(lambda m: cross_section_metric_display_name(str(m), "Quarterly (Q1-Q4)"))
        cik = normalize_cik(int(ticker_map.loc[ticker_map["ticker"] == ticker, "cik_str"].iloc[0]))
        out["primary_filing_link"] = out.apply(
            lambda r: resolve_filing_url_for_row(
                cik=cik,
                user_agent=user_agent,
                form=r["form"],
                filed=r["filed"],
                end=r["end"],
            ),
            axis=1,
        )
        if "flow_filing_accession" in out.columns:
            out["flow_filing_link"] = out.apply(
                lambda r: resolve_filing_url_for_row(
                    cik=cik,
                    user_agent=user_agent,
                    form=r["form"],
                    filed=r["filed"],
                    end=r["end"],
                    accession=r["flow_filing_accession"],
                )
                if pd.notna(r["flow_filing_accession"])
                else None,
                axis=1,
            )
        out_display = out.rename(columns={"val_usd_mm": "value_usd_mm", "val": "value_usd_raw"})
        out_display.insert(4, "period_label", out_display.apply(lambda r: format_period_label(r["fy"], r["fp"]), axis=1))
        ts_link_cols = [c for c in ["primary_filing_link", "flow_filing_link"] if c in out_display.columns]
        st.dataframe(
            out_display,
            use_container_width=True,
            hide_index=True,
            column_config=build_link_column_config(ts_link_cols) if ts_link_cols else None,
        )
        st.download_button(
            "Download time-series CSV",
            out_display.to_csv(index=False),
            file_name=f"time_series_{ticker}.csv",
            mime="text/csv",
        )


def render_panel(ticker_options: list[str], ticker_map: pd.DataFrame, user_agent: str) -> None:
    st.subheader("Panel")
    st.caption("Combine multiple companies over time in a long panel dataset.")
    st.caption(f"Units: {USD_MM_LABEL}")
    name_map = dict(zip(ticker_map["ticker"], ticker_map["title"]))

    companies = st.multiselect(
        "Companies",
        ticker_options,
        default=[t for t in ["AAPL", "MSFT", "AMZN"] if t in ticker_options],
        format_func=lambda t: ticker_display_label(str(t), name_map),
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
        latest = data.drop_duplicates(subset=["ticker", "metric", "end"], keep="first").copy()
        if {"Revenue", "Net Income"}.issubset(set(metrics)):
            period_type_label = "Annual (FY)" if freq == "Annual" else "Quarterly (Q1-Q4)"
            updated_frames = []
            for company in companies:
                company_rows = latest[latest["ticker"] == company].copy()
                if company_rows.empty:
                    continue
                company_rows = apply_strict_flow_pair_to_long_rows(
                    company_rows,
                    ticker=company,
                    ticker_map=ticker_map,
                    user_agent=user_agent,
                    period_type=period_type_label,
                )
                updated_frames.append(company_rows)
            if updated_frames:
                latest = pd.concat(updated_frames, ignore_index=True)

        latest["val_usd_mm"] = latest["val"] / USD_MM_DIVISOR
        if {"Revenue", "Net Income"}.issubset(set(metrics)) and "flow_context_source" in latest.columns:
            strict_periods = (
                latest[latest["flow_context_source"] == "filing_xbrl_same_context"][["ticker", "fy", "fp", "end"]]
                .drop_duplicates()
            )
            if not strict_periods.empty:
                st.caption(
                    f"Strict income-statement pairing applied in {len(strict_periods)} company-period observations: `Revenue` and `Net Income` come from the same filing XBRL context."
                )
            if len(strict_periods) < len(latest[["ticker", "fy", "fp", "end"]].drop_duplicates()):
                st.warning(
                    "Some company-period observations could not be paired from the same filing XBRL context for `Revenue` and `Net Income`; those cases use the aligned companyfacts rows."
                )

        out_cols = ["ticker", "metric", "fy", "fp", "end", "val", "val_usd_mm", "form", "filed"]
        if "flow_context_source" in latest.columns:
            out_cols.extend(["flow_context_source", "flow_context_ref", "flow_filing_accession"])
        out = latest[out_cols]
        if freq == "Quarterly":
            out["metric"] = out["metric"].map(lambda m: cross_section_metric_display_name(str(m), "Quarterly (Q1-Q4)"))
        out["primary_filing_link"] = out.apply(
            lambda r: resolve_filing_url_for_row(
                cik=normalize_cik(int(ticker_map.loc[ticker_map["ticker"] == r["ticker"], "cik_str"].iloc[0])),
                user_agent=user_agent,
                form=r["form"],
                filed=r["filed"],
                end=r["end"],
            ),
            axis=1,
        )
        if "flow_filing_accession" in out.columns:
            out["flow_filing_link"] = out.apply(
                lambda r: resolve_filing_url_for_row(
                    cik=normalize_cik(int(ticker_map.loc[ticker_map["ticker"] == r["ticker"], "cik_str"].iloc[0])),
                    user_agent=user_agent,
                    form=r["form"],
                    filed=r["filed"],
                    end=r["end"],
                    accession=r["flow_filing_accession"],
                )
                if pd.notna(r["flow_filing_accession"])
                else None,
                axis=1,
            )
        out_display = out.rename(columns={"val_usd_mm": "value_usd_mm", "val": "value_usd_raw"})
        out_display.insert(4, "period_label", out_display.apply(lambda r: format_period_label(r["fy"], r["fp"]), axis=1))
        panel_link_cols = [c for c in ["primary_filing_link", "flow_filing_link"] if c in out_display.columns]
        st.dataframe(
            out_display.sort_values(["ticker", "end", "metric"]),
            use_container_width=True,
            hide_index=True,
            column_config=build_link_column_config(panel_link_cols) if panel_link_cols else None,
        )
        st.download_button(
            "Download panel CSV",
            out_display.to_csv(index=False),
            file_name="panel_data.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="EDGAR Explorer", layout="wide")
    st.title("EDGAR Explorer")
    st.caption("Financial data extraction from SEC EDGAR in Cross-Section, Time-Series, Panel, and Debt Distribution formats.")
    st.caption(f"Unit standard across the app: {USD_MM_LABEL}.")

    with st.sidebar:
        st.subheader("Configuration")
        contact_email = st.text_input(
            "Contact email (required)",
            value=default_contact_email(),
            placeholder="your-email@domain.com",
            help="SEC asks for identifiable requests with a contact email.",
        ).strip()
        render_sidebar_guide()

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

    with st.sidebar:
        st.markdown("---")
        st.subheader("Issuer Filter")
        selection_mode = st.radio(
            "Selection mode",
            ["By company", "By industry"],
            index=0,
            key="selection_mode",
            help="Choose companies directly, or load the SEC SIC index and narrow by industry first.",
        )
        sic_cache = sec_sic_cache_status()
        if sic_cache["status"] == "missing":
            st.caption("SEC SIC cache: missing (`sec_sic_lookup.csv`).")
        else:
            modified_txt = sic_cache["modified"].strftime("%Y-%m-%d %H:%M")
            st.caption(
                f"SEC SIC cache: {sic_cache['status']} "
                f"(updated {modified_txt}, age {sic_cache['age_days']} days)."
            )
    filtered_ticker_map = ticker_map.copy()

    selected_industry_groups = None
    if selection_mode == "By industry":
        with st.sidebar:
            if st.button("Load Local SEC SIC Cache", key="load_sic_index"):
                st.session_state["sic_index_enabled"] = True
            st.caption(
                "Industry-first selection uses a local SEC SIC cache file (`sec_sic_lookup.csv`)."
            )
        if not st.session_state.get("sic_index_enabled", False):
            st.info("Choose `By industry`, then click `Load Local SEC SIC Cache` in the sidebar to enable SEC industry filtering.")
            st.stop()
        with st.spinner("Loading local SEC SIC cache..."):
            sic_enriched = enrich_ticker_map_with_sec_metadata(ticker_map)
        if sic_enriched.empty:
            st.error(
                "The local SEC SIC cache (`sec_sic_lookup.csv`) is missing or invalid, so industry-first selection is unavailable right now."
            )
            st.stop()
        cache_state = sec_sic_cache_status()
        if cache_state["status"] == "stale":
            st.warning(
                f"SEC SIC cache is stale ({cache_state['age_days']} days old). Industry filtering still works, but classifications may be outdated."
            )
        filtered_ticker_map = sic_enriched[
            (sic_enriched["issuer_category"] == "Operating company (SEC SIC-based)")
            & sic_enriched["sicDescription"].notna()
        ].copy()
        if filtered_ticker_map.empty:
            st.error("No operating companies with SEC SIC metadata are available for industry-first selection.")
            st.stop()

        industry_options = sorted(
            filtered_ticker_map["industry_group"].dropna().unique().tolist()
        )
        if not industry_options:
            st.error("No industry groups are available for the current issuer category filter.")
            st.stop()
        st.markdown("**Industry Filter**")
        selected_industry_groups = st.multiselect(
            "Industry group (SEC SIC-based)",
            industry_options,
            default=industry_options,
            key="industry_groups",
            help="Industry grouping derived from SEC `sicDescription`, not from issuer-name guesses.",
        )

        filtered_ticker_map = filtered_ticker_map[filtered_ticker_map["industry_group"].isin(selected_industry_groups)].copy()
        if filtered_ticker_map.empty:
            st.error("No issuers match the selected industry-group filter.")
            st.stop()

    if selection_mode == "By company":
        st.caption(
            f"Issuer universe loaded: {len(filtered_ticker_map):,} SEC tickers in direct company-selection mode."
        )
    else:
        st.caption(
            f"Issuer universe loaded: {len(filtered_ticker_map):,} SEC SIC-classified operating companies "
            f"after industry filtering."
        )
    if selection_mode == "By industry" and selected_industry_groups is not None:
        st.caption(
            f"Industry filter active: {len(selected_industry_groups)} group(s), {len(filtered_ticker_map):,} issuers available."
        )

    ticker_options = filtered_ticker_map["ticker"].tolist()

    tab1, tab2, tab3, tab4 = st.tabs(["Cross-Section", "Time Series", "Panel", "Debt Distribution"])

    with tab1:
        render_cross_section(ticker_options, filtered_ticker_map, user_agent)

    with tab2:
        render_time_series(ticker_options, filtered_ticker_map, user_agent)

    with tab3:
        render_panel(ticker_options, filtered_ticker_map, user_agent)

    with tab4:
        render_debt_distribution(ticker_options, filtered_ticker_map, user_agent)

    st.caption(app_signature())


if __name__ == "__main__":
    main()
