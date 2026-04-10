"""Week 25 tests for decision package generation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import sys
    import scripts.generate_decision_package as gdp
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    gdp = None
    _IMPORT_ERROR = exc


class TestDecisionPackageGeneration(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def _sample_df(self, coverage_shift: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "corridor_id": ["C1", "C2", "C3"],
                "rank": [1, 2, 3],
                "daily_ridership": [5000.0, 4000.0, 3000.0],
                "debt_coverage_ratio": [1.1 + coverage_shift, 0.9 + coverage_shift, 0.7 + coverage_shift],
                "project_npv_dynamic_musd": [10.0, -20.0, -60.0],
                "tif_revenue_cumulative": [300_000_000.0, 200_000_000.0, 100_000_000.0],
                "financially_viable": [True, False, False],
                "financial_viability_probability": [0.6, 0.3, 0.1],
            }
        )

    def test_build_decision_package_outputs_required_files(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            zoning = td_path / "zoning.csv"
            no_zoning = td_path / "no_zoning.csv"
            out_root = td_path / "decision_package"

            self._sample_df(coverage_shift=0.0).to_csv(zoning, index=False)
            self._sample_df(coverage_shift=-0.1).to_csv(no_zoning, index=False)

            package_dir, _meta = gdp.build_decision_package(
                zoning_path=zoning,
                no_zoning_path=no_zoning,
                output_root=out_root,
                top_n=2,
                model_options={
                    "decision_package_maps": True,
                    "technical_appendix_auto": True,
                },
            )

            expected_files = [
                package_dir / "FINAL_DECISION_PACKAGE.md",
                package_dir / "TECHNICAL_APPENDIX.md",
                package_dir / "run_metadata.json",
                package_dir / "tables" / "scenario_summary.csv",
                package_dir / "tables" / "corridor_scenario_delta.csv",
                package_dir / "tables" / "top_recommendations.csv",
                package_dir / "plots" / "debt_coverage_distribution.svg",
                package_dir / "plots" / "ridership_vs_coverage_zoning.svg",
                package_dir / "plots" / "top_corridor_delta.svg",
            ]
            for p in expected_files:
                self.assertTrue(p.exists(), f"Missing output artifact: {p}")


class TestMultiScenarioSummary(unittest.TestCase):
    """Component 6 Enhancement 1: N-scenario comparison."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_three_scenario_summary(self):
        dfs = {}
        for name, shift in [("current", 0.0), ("tif", 0.1), ("no_zoning", -0.05)]:
            dfs[name] = pd.DataFrame({
                "corridor_id": ["C1", "C2"],
                "debt_coverage_ratio": [0.8 + shift, 0.6 + shift],
                "daily_ridership": [5000, 4000],
                "project_npv_dynamic_musd": [10, -20],
                "tif_revenue_cumulative": [3e8, 2e8],
                "financially_viable": [1, 0],
            })
        result = gdp.build_scenario_summary_multi(dfs)
        # 3 scenarios + 3 pairwise deltas = 6 rows
        base_rows = result[~result["scenario"].str.startswith("delta_")]
        delta_rows = result[result["scenario"].str.startswith("delta_")]
        self.assertEqual(len(base_rows), 3)
        self.assertEqual(len(delta_rows), 3)


class TestUncertaintyDashboard(unittest.TestCase):
    """Component 6 Enhancement 4."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_dashboard_with_percentiles(self):
        wide_df = pd.DataFrame({
            "corridor_id": ["C1", "C2"],
            "debt_coverage_ratio_p10": [0.2, 0.15],
            "debt_coverage_ratio_p50": [0.35, 0.28],
            "debt_coverage_ratio_p90": [0.52, 0.42],
            "viability_probability": [0.05, 0.02],
        })
        result = gdp.build_uncertainty_dashboard(wide_df)
        self.assertEqual(len(result), 2)
        self.assertIn("dcr_p10", result.columns)
        self.assertIn("dcr_p50", result.columns)
        self.assertIn("dcr_p90", result.columns)
        self.assertIn("p_viable", result.columns)

    def test_dashboard_with_robust_ranking(self):
        wide_df = pd.DataFrame({
            "corridor_id": ["C1", "C2"],
            "debt_coverage_ratio_p10": [0.2, 0.15],
            "debt_coverage_ratio_p50": [0.35, 0.28],
            "debt_coverage_ratio_p90": [0.52, 0.42],
            "viability_probability": [0.05, 0.02],
        })
        robust_df = pd.DataFrame({
            "corridor_id": ["C1", "C2"],
            "p_top_k": [0.6, 0.4],
            "expected_shortfall_dcr": [0.18, 0.12],
            "robust_rank": [1, 2],
        })
        result = gdp.build_uncertainty_dashboard(wide_df, robust_df)
        self.assertIn("p_top5", result.columns)
        self.assertIn("downside_dcr", result.columns)
        self.assertIn("robust_rank", result.columns)
        # C1 should be rank 1
        self.assertEqual(int(result.iloc[0]["robust_rank"]), 1)

    def test_dashboard_empty_input(self):
        result = gdp.build_uncertainty_dashboard(pd.DataFrame())
        self.assertEqual(len(result), 0)


class TestTechnicalAppendix(unittest.TestCase):
    """Component 6 Enhancement 5."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_appendix_generates_file(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "TECHNICAL_APPENDIX.md"
            gdp.generate_technical_appendix(out)
            self.assertTrue(out.exists())
            content = out.read_text(encoding="utf-8")
            self.assertIn("Technical Appendix", content)
            self.assertIn("Calibrated Parameters", content)
            self.assertIn("Model Description", content)

    def test_appendix_includes_financial_params(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "TECHNICAL_APPENDIX.md"
            gdp.generate_technical_appendix(out)
            content = out.read_text(encoding="utf-8")
            # Should have at least some constants from financial_params
            self.assertIn("financial_params", content)


class TestExecutiveSummary(unittest.TestCase):
    """Component 6 Enhancement 6."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_summary_includes_lead_recommendation(self):
        recs = pd.DataFrame({
            "corridor_id": ["C1", "C2"],
            "daily_ridership": [5000, 4000],
            "debt_coverage_ratio": [0.35, 0.28],
        })
        scenario = pd.DataFrame({
            "scenario": ["current", "tif"],
            "corridor_count": [20, 20],
            "mean_daily_ridership": [3500, 4200],
            "mean_debt_coverage_ratio": [0.25, 0.32],
        })
        result = gdp.build_executive_summary(recs, scenario)
        self.assertIn("C1", result)
        self.assertIn("Lead recommendation", result)
        self.assertIn("Next Steps", result)

    def test_summary_with_equity(self):
        recs = pd.DataFrame({
            "corridor_id": ["C1"],
            "daily_ridership": [5000],
            "debt_coverage_ratio": [0.35],
        })
        scenario = pd.DataFrame({"scenario": ["current"], "corridor_count": [1],
                                  "mean_daily_ridership": [5000], "mean_debt_coverage_ratio": [0.35]})
        equity = pd.DataFrame({
            "corridor_id": ["C1"],
            "daily_ridership": [5000],
            "debt_coverage_ratio": [0.35],
            "equity_score": [0.72],
        })
        result = gdp.build_executive_summary(recs, scenario, equity_ranking=equity)
        self.assertIn("equity score", result.lower())


class TestRobustCorridorRanking(unittest.TestCase):
    """Component 3C: Robust ranking from Monte Carlo draws."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")
        try:
            import scripts.apm_corridor_evaluation_integrated as acei
            self.acei = acei
        except Exception as exc:
            self.skipTest(f"Cannot import evaluation module: {exc}")

    def test_robust_ranking_basic(self):
        """3 corridors, 10 draws — verify ranking structure."""
        rows = []
        rng = np.random.default_rng(42)
        for d in range(10):
            for cid in ["C1", "C2", "C3"]:
                base = {"C1": 0.4, "C2": 0.3, "C3": 0.2}[cid]
                rows.append({
                    "corridor_id": cid,
                    "draw": d,
                    "debt_coverage_ratio": base + rng.normal(0, 0.05),
                })
        draws_df = pd.DataFrame(rows)
        result = self.acei.robust_corridor_ranking(draws_df)
        self.assertEqual(len(result), 3)
        self.assertIn("robust_rank", result.columns)
        self.assertIn("p_top_k", result.columns)
        self.assertIn("expected_shortfall_dcr", result.columns)
        self.assertIn("max_regret_dcr", result.columns)
        # C1 has highest base DCR, should generally rank 1
        top = result.iloc[0]
        self.assertEqual(top["corridor_id"], "C1")

    def test_robust_ranking_empty_draws(self):
        result = self.acei.robust_corridor_ranking(pd.DataFrame())
        self.assertEqual(len(result), 0)

    def test_robust_ranking_p_top_k_range(self):
        """P(top-k) should be in [0, 1]."""
        rows = []
        for d in range(20):
            for cid in ["A", "B"]:
                rows.append({"corridor_id": cid, "draw": d,
                             "debt_coverage_ratio": 0.3 if cid == "A" else 0.2})
        result = self.acei.robust_corridor_ranking(pd.DataFrame(rows), top_k=1)
        for p in result["p_top_k"]:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


class TestEquityRanking(unittest.TestCase):
    """Component 5 Addition 4: Equity-weighted corridor ranking."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"Skipping due to import error: {_IMPORT_ERROR}")

    def test_low_dcr_high_equity_ranks_higher(self):
        """Corridor with low DCR but high equity should outrank mid-DCR zero-equity."""
        df = pd.DataFrame({
            "corridor_id": ["HighEquity", "MidDCR"],
            "debt_coverage_ratio": [0.20, 0.40],
            "equity_ratio": [2.0, 0.5],
            "se01_riders_share": [0.60, 0.10],
            "cumulative_displaced": [0, 50],
            "dac_benefit_share": [0.80, 0.10],
        })
        result = gdp.build_equity_ranking(df, financial_weight=0.50, equity_weight=0.50)
        self.assertEqual(result.iloc[0]["corridor_id"], "HighEquity")

    def test_missing_equity_cols_uses_defaults(self):
        """When equity columns are absent, result still works (all get 0.5 norm)."""
        df = pd.DataFrame({
            "corridor_id": ["C1", "C2"],
            "debt_coverage_ratio": [0.5, 0.3],
        })
        result = gdp.build_equity_ranking(df)
        self.assertEqual(len(result), 2)
        self.assertIn("composite_rank", result.columns)


if __name__ == "__main__":
    unittest.main()
