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

    needed_cols = ["end", "fy", "fp", "form", "val", "filed", "frame"]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = None

    df = df[needed_cols].copy()
    df["tag"] = tag
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    return df.dropna(subset=["val"])


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
        format_func=lambda t: f"{t} - {name_map.get(t, 'Unknown company')}",
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
        dist_df["amount_usd_mm"] = dist_df["amount_usd_raw"] / USD_MM_DIVISOR
        dist_df["Included in maturity sum"] = dist_df["Included in maturity sum"].map(lambda v: "Yes" if bool(v) else "No")
        dist_df.loc[dist_df["Included in maturity sum"] == "Yes", "Note"] = ""
        total_row = pd.DataFrame(
            [
                {
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
    st.caption(f"Units: {USD_MM_LABEL}")

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

        pivot_usd = latest.pivot(index="ticker", columns="metric", values="val").reset_index()
        pivot_mm = pivot_usd.copy()
        metric_cols = [c for c in pivot_mm.columns if c != "ticker"]
        for col in metric_cols:
            pivot_mm[col] = pivot_mm[col] / USD_MM_DIVISOR
        st.dataframe(pivot_mm, use_container_width=True, hide_index=True)

        pivot_csv = pivot_usd.copy()
        for col in metric_cols:
            pivot_csv[f"{col}_usd_mm"] = pivot_csv[col] / USD_MM_DIVISOR
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
        latest = data.drop_duplicates(subset=["end", "metric"], keep="first").copy()

        latest["val_usd_mm"] = latest["val"] / USD_MM_DIVISOR
        chart_df = latest.pivot(index="end", columns="metric", values="val_usd_mm").sort_index()
        st.line_chart(chart_df)

        out = latest[["ticker", "metric", "fy", "fp", "end", "val", "val_usd_mm", "form", "filed"]].sort_values("end")
        out_display = out.rename(columns={"val_usd_mm": "value_usd_mm", "val": "value_usd_raw"})
        st.dataframe(out_display, use_container_width=True, hide_index=True)
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
        latest = data.drop_duplicates(subset=["ticker", "metric", "end"], keep="first").copy()

        latest["val_usd_mm"] = latest["val"] / USD_MM_DIVISOR
        out = latest[["ticker", "metric", "fy", "fp", "end", "val", "val_usd_mm", "form", "filed"]]
        out_display = out.rename(columns={"val_usd_mm": "value_usd_mm", "val": "value_usd_raw"})
        st.dataframe(out_display.sort_values(["ticker", "end", "metric"]), use_container_width=True, hide_index=True)
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

    ticker_options = ticker_map["ticker"].tolist()

    tab1, tab2, tab3, tab4 = st.tabs(["Cross-Section", "Time Series", "Panel", "Debt Distribution"])

    with tab1:
        render_cross_section(ticker_options, ticker_map, user_agent)

    with tab2:
        render_time_series(ticker_options, ticker_map, user_agent)

    with tab3:
        render_panel(ticker_options, ticker_map, user_agent)

    with tab4:
        render_debt_distribution(ticker_options, ticker_map, user_agent)

    st.caption(app_signature())


if __name__ == "__main__":
    main()
