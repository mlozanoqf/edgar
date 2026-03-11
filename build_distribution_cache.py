import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import app


OUTPUT_PATH = Path(__file__).with_name("distribution_cache.csv.gz")
PARTIAL_OUTPUT_PATH = Path(__file__).with_name("distribution_cache.partial.csv")
CHECKPOINT_PATH = Path(__file__).with_name("distribution_cache.checkpoint.csv")
STATE_PATH = Path(__file__).with_name("distribution_cache.state.json")
TARGET_FISCAL_YEAR = int(os.getenv("DISTRIBUTION_TARGET_FY", "2025"))
TICKER_LIMIT = int(os.getenv("DISTRIBUTION_TICKER_LIMIT", "0"))
REQUEST_SLEEP_SECONDS = float(os.getenv("SEC_REQUEST_SLEEP_SECONDS", "0.12"))
TICKER_MAP_MAX_ATTEMPTS = int(os.getenv("DISTRIBUTION_TICKER_MAP_MAX_ATTEMPTS", "5"))
TICKER_MAP_RETRY_SLEEP_SECONDS = float(os.getenv("DISTRIBUTION_TICKER_MAP_RETRY_SLEEP_SECONDS", "5"))
RESET_PROGRESS = str(os.getenv("DISTRIBUTION_RESET_PROGRESS", "")).strip().lower() in {"1", "true", "yes", "y"}

OUTPUT_COLUMNS = [
    "ticker",
    "company_title",
    "cik_str",
    "account_mode",
    "metric",
    "fy",
    "fp",
    "end",
    "value_usd_raw",
    "value_usd_mm",
    "tag",
    "source_type",
    "quality_flag",
    "form",
    "filed",
    "aligned_snapshot_metric_count",
    "aligned_snapshot_has_all_selected_metrics",
    "cache_built_at",
]
CHECKPOINT_COLUMNS = [
    "ticker",
    "cik_str",
    "company_title",
    "status",
    "row_count",
    "error_message",
    "processed_at",
]
SUCCESS_STATUSES = {"ok_with_rows", "ok_no_rows"}


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def build_run_config() -> dict[str, object]:
    return {
        "target_fiscal_year": TARGET_FISCAL_YEAR,
        "ticker_limit": TICKER_LIMIT,
        "account_mode": "Simplified Statements",
        "period_type": "Annual (FY)",
        "output_version": 2,
    }


def run_config_matches(existing: dict[str, object], current: dict[str, object]) -> bool:
    keys = ["target_fiscal_year", "ticker_limit", "account_mode", "period_type", "output_version"]
    return all(existing.get(key) == current.get(key) for key in keys)


def build_user_agent() -> str | None:
    user_agent = os.getenv("SEC_USER_AGENT") or os.getenv("SEC_CONTACT_EMAIL")
    if not user_agent:
        return None
    if "@" in user_agent and " " not in user_agent:
        return f"EDGAR Explorer ({user_agent})"
    return user_agent


def normalize_error_message(exc: Exception) -> str:
    return str(exc).replace("\n", " ").replace("\r", " ").strip()[:500]


def fetch_ticker_map(user_agent: str) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, TICKER_MAP_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(app.TICKERS_URL, headers={"User-Agent": user_agent}, timeout=30)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame.from_dict(data, orient="index")
            df["ticker"] = df["ticker"].astype(str).str.upper()
            df["cik_str"] = df["cik_str"].astype(int)
            return df[["ticker", "cik_str", "title"]].sort_values("ticker").reset_index(drop=True)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= TICKER_MAP_MAX_ATTEMPTS:
                break
            wait_seconds = TICKER_MAP_RETRY_SLEEP_SECONDS * attempt
            print(
                f"Ticker map fetch attempt {attempt}/{TICKER_MAP_MAX_ATTEMPTS} failed: {normalize_error_message(exc)}. "
                f"Retrying in {wait_seconds:.1f}s ...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Ticker map fetch failed before any request was attempted.")


def extract_simplified_rows(company_facts: dict, ticker: str) -> pd.DataFrame:
    rows = []
    for metric in app.SIMPLIFIED_ACCOUNT_OPTIONS:
        metric_rows = app.build_metric_rows_for_metric(company_facts, ticker, metric)
        if not metric_rows.empty:
            rows.append(metric_rows)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_company_distribution_rows(ticker_row: pd.Series, user_agent: str, cache_built_at: str) -> pd.DataFrame:
    ticker = str(ticker_row["ticker"])
    cik = app.normalize_cik(int(ticker_row["cik_str"]))
    company_facts = app.sec_get(app.COMPANYFACTS_URL.format(cik=cik), user_agent)
    metric_df = extract_simplified_rows(company_facts, ticker)
    if metric_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    metric_df = metric_df[metric_df["fp"] == "FY"].dropna(subset=["fy", "end", "val"]).copy()
    if metric_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    selected = app.select_aligned_long_rows(
        metric_df,
        metrics=app.SIMPLIFIED_ACCOUNT_OPTIONS,
        period_type="Annual (FY)",
        entity_keys=[],
        account_mode="Simplified Statements",
    )
    if selected.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    selected = selected[selected["fy"] == TARGET_FISCAL_YEAR].copy()
    if selected.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    selected["company_title"] = str(ticker_row["title"])
    selected["cik_str"] = int(ticker_row["cik_str"])
    selected["account_mode"] = "Simplified Statements"
    selected["value_usd_raw"] = pd.to_numeric(selected["val"], errors="coerce")
    selected["value_usd_mm"] = selected["value_usd_raw"] / app.USD_MM_DIVISOR
    selected["cache_built_at"] = cache_built_at
    return selected[OUTPUT_COLUMNS].copy()


def remove_progress_files() -> None:
    for path in [PARTIAL_OUTPUT_PATH, CHECKPOINT_PATH, STATE_PATH]:
        if path.exists():
            path.unlink()


def save_state(state: dict[str, object]) -> None:
    state["last_updated_at"] = utc_now_str()
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_or_initialize_state(run_config: dict[str, object], total_tickers: int) -> dict[str, object] | None:
    if RESET_PROGRESS:
        remove_progress_files()

    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Failed reading {STATE_PATH.name}: {exc}", file=sys.stderr)
            return None
        existing_config = state.get("run_config", {})
        if existing_config and not run_config_matches(existing_config, run_config):
            print(
                "Existing distribution-cache progress was built with different settings. "
                "Set DISTRIBUTION_RESET_PROGRESS=1 before rerunning.",
                file=sys.stderr,
            )
            return None
    else:
        if PARTIAL_OUTPUT_PATH.exists() or CHECKPOINT_PATH.exists():
            print(
                "Found partial distribution-cache progress without a state file; attempting best-effort resume.",
                file=sys.stderr,
            )
        state = {"started_at": utc_now_str()}

    state["run_config"] = run_config
    state["total_tickers"] = int(total_tickers)
    save_state(state)
    return state


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    try:
        checkpoint = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    except Exception:
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    for col in CHECKPOINT_COLUMNS:
        if col not in checkpoint.columns:
            checkpoint[col] = pd.NA
    return checkpoint[CHECKPOINT_COLUMNS].copy()


def latest_checkpoint_rows(checkpoint: pd.DataFrame) -> pd.DataFrame:
    if checkpoint.empty:
        return checkpoint
    latest = checkpoint.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="last")
    return latest.reset_index(drop=True)


def checkpoint_row_counts_as_success(row: pd.Series) -> bool:
    status = str(row.get("status", ""))
    if status in SUCCESS_STATUSES:
        return True
    error_message = str(row.get("error_message", ""))
    return status == "http_error" and "404" in error_message


def successful_tickers_from_checkpoint(checkpoint: pd.DataFrame) -> set[str]:
    latest = latest_checkpoint_rows(checkpoint)
    if latest.empty:
        return set()
    success_mask = latest.apply(checkpoint_row_counts_as_success, axis=1)
    success_rows = latest[success_mask]
    return set(success_rows["ticker"].astype(str))


def append_dataframe(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def append_checkpoint_row(path: Path, row: dict[str, object]) -> None:
    checkpoint_row = pd.DataFrame([[row.get(col) for col in CHECKPOINT_COLUMNS]], columns=CHECKPOINT_COLUMNS)
    append_dataframe(path, checkpoint_row)


def finalize_output(
    sic_enriched: pd.DataFrame,
    partial_path: Path = PARTIAL_OUTPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> int:
    if not partial_path.exists():
        return 0
    try:
        out = pd.read_csv(partial_path, parse_dates=["end", "filed"])
    except pd.errors.EmptyDataError:
        return 0
    if out.empty:
        return 0

    dedupe_keys = ["ticker", "metric", "fy", "fp", "end", "account_mode"]
    out = out.drop_duplicates(subset=dedupe_keys, keep="last")

    if not sic_enriched.empty:
        keep_cols = ["ticker", "sic", "sicDescription", "issuer_category", "industry_group"]
        keep_cols = [col for col in keep_cols if col in sic_enriched.columns]
        out = out.merge(sic_enriched[keep_cols], on="ticker", how="left")

    out = out.sort_values(["ticker", "fy", "metric", "filed", "end"]).reset_index(drop=True)
    out.to_csv(output_path, index=False, compression="gzip")
    return int(len(out))


def main() -> int:
    user_agent = build_user_agent()
    if not user_agent:
        print("Set SEC_USER_AGENT or SEC_CONTACT_EMAIL before running this script.", file=sys.stderr)
        return 1

    try:
        raw_ticker_map = fetch_ticker_map(user_agent)
    except Exception as exc:
        print(f"Failed loading SEC ticker map: {exc}", file=sys.stderr)
        return 1

    full_ticker_map = app.filter_operating_company_ticker_map(raw_ticker_map)
    if full_ticker_map.empty:
        print("The SEC SIC cache is missing or did not produce an operating-company universe for the distribution cache.", file=sys.stderr)
        return 1

    if TICKER_LIMIT > 0:
        full_ticker_map = full_ticker_map.head(TICKER_LIMIT).copy()

    run_config = build_run_config()
    state = load_or_initialize_state(run_config, total_tickers=len(full_ticker_map))
    if state is None:
        return 1

    checkpoint = load_checkpoint()
    completed_tickers = successful_tickers_from_checkpoint(checkpoint)
    pending_ticker_map = full_ticker_map[~full_ticker_map["ticker"].astype(str).isin(completed_tickers)].copy()

    sic_enriched = full_ticker_map.copy()

    if completed_tickers:
        print(
            f"Resuming distribution cache build for FY {TARGET_FISCAL_YEAR}: "
            f"{len(completed_tickers):,} tickers already completed, {len(pending_ticker_map):,} remaining."
        )
    else:
        print(f"Starting distribution cache build for FY {TARGET_FISCAL_YEAR} across {len(full_ticker_map):,} tickers.")

    cache_built_at = utc_now_str()
    total = len(full_ticker_map)
    attempted_this_run = 0
    starting_completed_count = len(completed_tickers)
    for offset, (_, ticker_row) in enumerate(pending_ticker_map.iterrows(), start=1):
        ticker = str(ticker_row["ticker"])
        global_position = starting_completed_count + offset
        status = "ok_no_rows"
        row_count = 0
        error_message = ""

        try:
            company_rows = build_company_distribution_rows(ticker_row, user_agent, cache_built_at)
            row_count = int(len(company_rows))
            if row_count > 0:
                append_dataframe(PARTIAL_OUTPUT_PATH, company_rows)
                status = "ok_with_rows"
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            if response is not None and response.status_code == 404:
                status = "ok_no_rows"
                error_message = "companyfacts not available (404)"
            else:
                status = "http_error"
                error_message = normalize_error_message(exc)
                print(f"[{global_position}/{total}] {ticker}: SEC error {error_message}", file=sys.stderr)
        except Exception as exc:
            status = "error"
            error_message = normalize_error_message(exc)
            print(f"[{global_position}/{total}] {ticker}: {error_message}", file=sys.stderr)

        checkpoint_row = {
            "ticker": ticker,
            "cik_str": int(ticker_row["cik_str"]),
            "company_title": str(ticker_row["title"]),
            "status": status,
            "row_count": row_count,
            "error_message": error_message,
            "processed_at": utc_now_str(),
        }
        append_checkpoint_row(CHECKPOINT_PATH, checkpoint_row)

        if status in SUCCESS_STATUSES:
            completed_tickers.add(ticker)

        attempted_this_run += 1
        state["successful_ticker_count"] = int(len(completed_tickers))
        state["attempted_this_run"] = int(attempted_this_run)
        state["last_processed_ticker"] = ticker
        save_state(state)

        if global_position % 100 == 0 or global_position == total:
            print(f"Processed {global_position:,}/{total:,} tickers ...")
        time.sleep(REQUEST_SLEEP_SECONDS)

    row_count = finalize_output(sic_enriched)
    if row_count == 0:
        print("No distribution rows were extracted.", file=sys.stderr)
        return 1

    print(f"Wrote {row_count:,} rows to {OUTPUT_PATH}")
    remove_progress_files()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


