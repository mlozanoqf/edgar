import tempfile
import unittest
from pathlib import Path

import pandas as pd

import app
import build_distribution_cache as distribution_builder


class ExtractionLogicTests(unittest.TestCase):
    def test_finalize_maturity_rows_prefers_debt_only_and_excludes_aggregate(self):
        df = pd.DataFrame(
            [
                {
                    "item": "Next twelve months debt-only",
                    "tag": "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
                    "concept": "next_12_months",
                    "amount": 100.0,
                    "fp": "FY",
                    "end": pd.Timestamp("2024-12-31"),
                    "filed": pd.Timestamp("2025-02-01"),
                },
                {
                    "item": "Next twelve months debt+lease",
                    "tag": "LongTermDebtAndFinanceLeaseLiabilitiesMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
                    "concept": "next_12_months",
                    "amount": 110.0,
                    "fp": "FY",
                    "end": pd.Timestamp("2024-12-31"),
                    "filed": pd.Timestamp("2025-02-02"),
                },
                {
                    "item": "Years two through five aggregate",
                    "tag": "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearsTwoThroughFive",
                    "concept": "years_2_to_5_agg",
                    "amount": 300.0,
                    "fp": "FY",
                    "end": pd.Timestamp("2024-12-31"),
                    "filed": pd.Timestamp("2025-02-01"),
                },
                {
                    "item": "Year two detail",
                    "tag": "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",
                    "concept": "year_2",
                    "amount": 90.0,
                    "fp": "FY",
                    "end": pd.Timestamp("2024-12-31"),
                    "filed": pd.Timestamp("2025-02-01"),
                },
            ]
        )

        result = app.finalize_maturity_rows(df)

        selected = result[result["include_in_sum"]]
        self.assertEqual(selected["concept"].tolist(), ["next_12_months", "year_2"])
        self.assertEqual(selected["tag"].tolist()[0], "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths")
        aggregate_note = result.loc[result["concept"] == "years_2_to_5_agg", "exclusion_reason"].iloc[0]
        self.assertEqual(aggregate_note, "Excluded because detailed Y2-Y5 exists")

    def test_parse_filing_maturity_rows_ignores_dimensional_contexts_and_duplicate_tags(self):
        xbrl = """
        <xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
                    xmlns:us-gaap="http://fasb.org/us-gaap/2024-01-31"
                    xmlns:xbrldi="http://xbrl.org/2006/xbrldi">
          <xbrli:context id="c_main">
            <xbrli:entity><xbrli:identifier scheme="test">0001</xbrli:identifier></xbrli:entity>
            <xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period>
          </xbrli:context>
          <xbrli:context id="c_dim">
            <xbrli:entity><xbrli:identifier scheme="test">0001</xbrli:identifier></xbrli:entity>
            <xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period>
            <xbrli:scenario>
              <xbrldi:explicitMember dimension="test:DebtInstrumentAxis">test:DebtInstrumentMember</xbrldi:explicitMember>
            </xbrli:scenario>
          </xbrli:context>
          <us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths contextRef="c_main">100</us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths>
          <us-gaap:LongTermDebtAndFinanceLeaseLiabilitiesMaturitiesRepaymentsOfPrincipalInNextTwelveMonths contextRef="c_main">110</us-gaap:LongTermDebtAndFinanceLeaseLiabilitiesMaturitiesRepaymentsOfPrincipalInNextTwelveMonths>
          <us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo contextRef="c_main">90</us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo>
          <us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearsTwoThroughFive contextRef="c_main">300</us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearsTwoThroughFive>
          <us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths contextRef="c_dim">999</us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths>
        </xbrli:xbrl>
        """

        result = app.parse_filing_maturity_rows(xbrl, 2024, "2025-02-15")

        selected = result[result["include_in_sum"]]
        self.assertEqual(selected["amount"].sum(), 190.0)
        self.assertNotIn(999.0, selected["amount"].tolist())
        self.assertIn("Duplicate concept (alternative tag)", result["exclusion_reason"].tolist())
        self.assertIn("Excluded because detailed Y2-Y5 exists", result["exclusion_reason"].tolist())

    def test_parse_filing_total_rows_ignores_dimensional_contexts(self):
        xbrl = """
        <xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
                    xmlns:us-gaap="http://fasb.org/us-gaap/2024-01-31"
                    xmlns:xbrldi="http://xbrl.org/2006/xbrldi">
          <xbrli:context id="c_main">
            <xbrli:entity><xbrli:identifier scheme="test">0001</xbrli:identifier></xbrli:entity>
            <xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period>
          </xbrli:context>
          <xbrli:context id="c_dim">
            <xbrli:entity><xbrli:identifier scheme="test">0001</xbrli:identifier></xbrli:entity>
            <xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period>
            <xbrli:scenario>
              <xbrldi:explicitMember dimension="test:DebtInstrumentAxis">test:DebtInstrumentMember</xbrldi:explicitMember>
            </xbrli:scenario>
          </xbrli:context>
          <us-gaap:DebtCurrent contextRef="c_main">50</us-gaap:DebtCurrent>
          <us-gaap:DebtCurrent contextRef="c_dim">999</us-gaap:DebtCurrent>
        </xbrli:xbrl>
        """

        result = app.parse_filing_total_rows(xbrl, 2024, "2025-02-15")

        self.assertEqual(result["val"].tolist(), [50.0])

    def test_select_aligned_long_rows_prefers_complete_snapshot_over_newer_partial_snapshot(self):
        df = pd.DataFrame(
            [
                self._row("Assets", 100.0, "2024-03-31", "2024-04-20", "10-Q", "Direct"),
                self._row("Liabilities", 60.0, "2024-03-31", "2024-04-20", "10-Q", "Direct"),
                self._row("Equity", 40.0, "2024-03-31", "2024-04-20", "10-Q", "Direct"),
                self._row("Assets", 105.0, "2024-03-31", "2024-05-01", "10-Q/A", "Direct"),
                self._row("Equity", 45.0, "2024-03-31", "2024-05-01", "10-Q/A", "Direct"),
            ]
        )

        result = app.select_aligned_long_rows(
            df,
            metrics=["Assets", "Liabilities", "Equity"],
            period_type="Quarterly (Q1-Q4)",
            entity_keys=[],
        )

        self.assertCountEqual(result["metric"].tolist(), ["Assets", "Liabilities", "Equity"])
        self.assertEqual(set(result["filed"].dt.strftime("%Y-%m-%d")), {"2024-04-20"})
        self.assertTrue(result["aligned_snapshot_has_all_selected_metrics"].all())

    def test_select_aligned_long_rows_prefers_longer_quarterly_flow_within_snapshot(self):
        df = pd.DataFrame(
            [
                self._row("Revenue", 30.0, "2024-03-31", "2024-04-20", "10-Q", "Direct", start="2024-03-16"),
                self._row("Revenue", 120.0, "2024-03-31", "2024-04-20", "10-Q", "Direct", start="2024-01-01"),
                self._row("Net Income", 20.0, "2024-03-31", "2024-04-20", "10-Q", "Direct", start="2024-01-01"),
            ]
        )

        result = app.select_aligned_long_rows(
            df,
            metrics=["Revenue", "Net Income"],
            period_type="Quarterly (Q1-Q4)",
            entity_keys=[],
        )

        revenue_value = result.loc[result["metric"] == "Revenue", "val"].iloc[0]
        self.assertEqual(revenue_value, 120.0)

    def test_select_aligned_long_rows_prefers_lower_quality_snapshot_when_coverage_matches(self):
        df = pd.DataFrame(
            [
                self._row("Assets", 100.0, "2024-12-31", "2025-02-01", "10-K", "Direct"),
                self._row("Current Assets", 40.0, "2024-12-31", "2025-02-01", "10-K", "Direct"),
                self._row("Noncurrent Assets", 60.0, "2024-12-31", "2025-02-01", "10-K", "Direct"),
                self._row("Assets", 100.0, "2024-12-31", "2025-02-10", "10-K/A", "Direct"),
                self._row("Current Assets", 40.0, "2024-12-31", "2025-02-10", "10-K/A", "Direct"),
                self._row("Noncurrent Assets", 60.0, "2024-12-31", "2025-02-10", "10-K/A", "Derived from aligned components"),
            ]
        )

        result = app.select_aligned_long_rows(
            df,
            metrics=["Assets", "Current Assets", "Noncurrent Assets"],
            period_type="Annual (FY)",
            entity_keys=[],
        )

        self.assertEqual(set(result["filed"].dt.strftime("%Y-%m-%d")), {"2025-02-01"})

    def test_select_aligned_long_rows_specific_mode_prefers_recency_over_statement_fit(self):
        df = pd.DataFrame(
            [
                self._row("Assets", 100.0, "2024-12-31", "2025-02-01", "10-K", "Direct"),
                self._row("Liabilities", 60.0, "2024-12-31", "2025-02-01", "10-K", "Direct"),
                self._row("Equity", 40.0, "2024-12-31", "2025-02-01", "10-K", "Direct"),
                self._row("Assets", 100.0, "2024-12-31", "2025-02-10", "10-K/A", "Direct"),
                self._row("Liabilities", 60.0, "2024-12-31", "2025-02-10", "10-K/A", "Direct"),
                self._row("Equity", 35.0, "2024-12-31", "2025-02-10", "10-K/A", "Direct"),
            ]
        )

        simplified = app.select_aligned_long_rows(
            df,
            metrics=["Assets", "Liabilities", "Equity"],
            period_type="Annual (FY)",
            entity_keys=[],
            account_mode="Simplified Statements",
        )
        specific = app.select_aligned_long_rows(
            df,
            metrics=["Assets", "Liabilities", "Equity"],
            period_type="Annual (FY)",
            entity_keys=[],
            account_mode="Specific Account Explorer",
        )

        self.assertEqual(set(simplified["filed"].dt.strftime("%Y-%m-%d")), {"2025-02-01"})
        self.assertEqual(set(specific["filed"].dt.strftime("%Y-%m-%d")), {"2025-02-10"})

    def test_select_cross_section_snapshots_specific_mode_prefers_recency_over_statement_fit(self):
        df = pd.DataFrame(
            [
                self._row("Assets", 100.0, "2024-12-31", "2025-02-01", "10-K", "Direct"),
                self._row("Liabilities", 60.0, "2024-12-31", "2025-02-01", "10-K", "Direct"),
                self._row("Equity", 40.0, "2024-12-31", "2025-02-01", "10-K", "Direct"),
                self._row("Assets", 100.0, "2024-12-31", "2025-02-10", "10-K/A", "Direct"),
                self._row("Liabilities", 60.0, "2024-12-31", "2025-02-10", "10-K/A", "Direct"),
                self._row("Equity", 35.0, "2024-12-31", "2025-02-10", "10-K/A", "Direct"),
            ]
        )

        simplified = app.select_cross_section_snapshots(
            df,
            metrics=["Assets", "Liabilities", "Equity"],
            period_type="Annual (FY)",
            account_mode="Simplified Statements",
        )
        specific = app.select_cross_section_snapshots(
            df,
            metrics=["Assets", "Liabilities", "Equity"],
            period_type="Annual (FY)",
            account_mode="Specific Account Explorer",
        )

        self.assertEqual(simplified.iloc[0]["filed"].strftime("%Y-%m-%d"), "2025-02-01")
        self.assertEqual(specific.iloc[0]["filed"].strftime("%Y-%m-%d"), "2025-02-10")

    def test_compute_concentration_metrics_ignores_nonpositive_values(self):
        result = app.compute_concentration_metrics(pd.Series([100.0, 50.0, 0.0, -10.0]), top_n=1)

        self.assertEqual(result["positive_company_count"], 2)
        self.assertAlmostEqual(result["positive_total_value"], 150.0)
        self.assertAlmostEqual(result["top_n_share"], 100.0 / 150.0)
        self.assertAlmostEqual(result["hhi"], (100.0 / 150.0) ** 2 + (50.0 / 150.0) ** 2)

    def test_compute_distribution_percentiles_returns_expected_quantiles(self):
        result = app.compute_distribution_percentiles(pd.Series([0.0, 100.0, 200.0, 300.0, 400.0]))

        self.assertAlmostEqual(result["p10"], 40.0)
        self.assertAlmostEqual(result["median"], 200.0)
        self.assertAlmostEqual(result["p90"], 360.0)

    def test_build_derived_noncurrent_liabilities_rows_uses_aligned_components(self):
        company_facts = {
            "facts": {
                "us-gaap": {
                    "Liabilities": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2024-12-31",
                                    "fy": 2024,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "val": 200,
                                    "filed": "2025-02-01",
                                }
                            ]
                        }
                    },
                    "LiabilitiesCurrent": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2024-12-31",
                                    "fy": 2024,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "val": 80,
                                    "filed": "2025-02-01",
                                }
                            ]
                        }
                    },
                }
            }
        }

        result = app.build_derived_noncurrent_liabilities_rows(company_facts, "TEST")

        self.assertEqual(result.iloc[0]["val"], 120)
        self.assertEqual(result.iloc[0]["tag"], "DERIVED: Liabilities - Current Liabilities")
        self.assertEqual(result.iloc[0]["source_type"], "derived")

    def test_tag_scope_recognizes_capital_and_operating_leases(self):
        self.assertEqual(app.tag_scope("LongTermDebtAndCapitalLeaseObligations"), "debt+lease")
        self.assertEqual(app.tag_scope("LongTermDebtAndOperatingLeaseLiabilities"), "debt+lease")
        self.assertEqual(app.tag_scope("LongTermDebt"), "debt-only")

    def test_select_operating_company_ticker_map_keeps_only_sic_classified_operating_companies(self):
        enriched = pd.DataFrame(
            [
                {"ticker": "AAA", "issuer_category": "Operating company (SEC SIC-based)", "sicDescription": "TECHNOLOGY", "industry_group": "Technology"},
                {"ticker": "BBB", "issuer_category": "Not SIC-classified / review", "sicDescription": pd.NA, "industry_group": "Unclassified (missing SIC)"},
                {"ticker": "CCC", "issuer_category": "Operating company (SEC SIC-based)", "sicDescription": pd.NA, "industry_group": "Unclassified (missing SIC)"},
            ]
        )

        result = app.select_operating_company_ticker_map(enriched)

        self.assertEqual(result["ticker"].tolist(), ["AAA"])

    @staticmethod
    def _row(metric, val, end, filed, form, quality_flag, start=None):
        return {
            "ticker": "TEST",
            "metric": metric,
            "tag": f"{metric.replace(' ', '')}Tag",
            "tag_rank": 0,
            "source_type": "derived" if quality_flag == "Derived from aligned components" else "direct",
            "quality_flag": quality_flag,
            "start": pd.Timestamp(start) if start else pd.NaT,
            "end": pd.Timestamp(end),
            "fy": 2024,
            "fp": "Q1" if str(end).startswith("2024-03-31") else "FY",
            "form": form,
            "val": val,
            "filed": pd.Timestamp(filed),
            "frame": None,
        }


class DistributionCacheBuilderTests(unittest.TestCase):
    def test_run_config_matches_requires_same_core_settings(self):
        current = {
            "target_fiscal_year": 2025,
            "ticker_limit": 0,
            "account_mode": "Simplified Statements",
            "period_type": "Annual (FY)",
            "output_version": 2,
        }

        self.assertTrue(distribution_builder.run_config_matches(current, dict(current)))

        changed = dict(current)
        changed["target_fiscal_year"] = 2024
        self.assertFalse(distribution_builder.run_config_matches(current, changed))

    def test_successful_tickers_from_checkpoint_uses_latest_status_per_ticker(self):
        checkpoint = pd.DataFrame(
            [
                {"ticker": "AAA", "status": "http_error"},
                {"ticker": "AAA", "status": "ok_with_rows"},
                {"ticker": "BBB", "status": "ok_no_rows"},
                {"ticker": "DDD", "status": "http_error", "error_message": "404 Client Error: Not Found for url: https://data.sec.gov/api/xbrl/companyfacts/..."},
                {"ticker": "CCC", "status": "ok_with_rows"},
                {"ticker": "CCC", "status": "error"},
            ]
        )

        self.assertEqual(distribution_builder.successful_tickers_from_checkpoint(checkpoint), {"AAA", "BBB", "DDD"})

    def test_finalize_output_deduplicates_rerun_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            partial_path = Path(tmpdir) / "distribution_cache.partial.csv"
            output_path = Path(tmpdir) / "distribution_cache.csv.gz"
            partial = pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "company_title": "Alpha Corp",
                        "cik_str": 1,
                        "account_mode": "Simplified Statements",
                        "metric": "Cash",
                        "fy": 2025,
                        "fp": "FY",
                        "end": "2025-12-31",
                        "value_usd_raw": 100.0,
                        "value_usd_mm": 0.1,
                        "tag": "CashTag",
                        "source_type": "direct",
                        "quality_flag": "Direct",
                        "form": "10-K",
                        "filed": "2026-02-01",
                        "aligned_snapshot_metric_count": 10,
                        "aligned_snapshot_has_all_selected_metrics": True,
                        "cache_built_at": "2026-03-10 10:00:00",
                    },
                    {
                        "ticker": "AAA",
                        "company_title": "Alpha Corp",
                        "cik_str": 1,
                        "account_mode": "Simplified Statements",
                        "metric": "Cash",
                        "fy": 2025,
                        "fp": "FY",
                        "end": "2025-12-31",
                        "value_usd_raw": 110.0,
                        "value_usd_mm": 0.11,
                        "tag": "CashTag",
                        "source_type": "direct",
                        "quality_flag": "Direct",
                        "form": "10-K/A",
                        "filed": "2026-02-05",
                        "aligned_snapshot_metric_count": 10,
                        "aligned_snapshot_has_all_selected_metrics": True,
                        "cache_built_at": "2026-03-10 10:05:00",
                    },
                ]
            )
            partial.to_csv(partial_path, index=False)

            row_count = distribution_builder.finalize_output(
                pd.DataFrame(),
                partial_path=partial_path,
                output_path=output_path,
            )

            self.assertEqual(row_count, 1)
            finalized = pd.read_csv(output_path, compression="gzip")
            self.assertEqual(len(finalized), 1)
            self.assertEqual(float(finalized.iloc[0]["value_usd_raw"]), 110.0)


if __name__ == "__main__":
    unittest.main()




