import numpy as np
import pandas as pd
import pytest

from lmsstat.stat._tests import (
    t_test,
    u_test,
    _decide_utest_method,
    anova_test,
    kruskal_test,
    norm_test,
)
from lmsstat.stat._utils import preprocess_data


# ── t_test ──────────────────────────────────────────────────────────────

class TestTTest:
    def test_returns_df(self, preprocessed_two_groups):
        _, _, gs, names = preprocessed_two_groups
        result = t_test(gs, names)
        assert isinstance(result, pd.DataFrame)

    def test_shape_two_groups(self, preprocessed_two_groups):
        _, _, gs, names = preprocessed_two_groups
        result = t_test(gs, names)
        assert result.shape == (5, 1)

    def test_shape_three_groups(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = t_test(gs, names)
        assert result.shape == (5, 3)

    def test_column_name_ttest(self, preprocessed_two_groups):
        _, _, gs, names = preprocessed_two_groups
        result = t_test(gs, names)
        assert "_ttest" in result.columns[0]

    def test_values_in_01(self, preprocessed_two_groups):
        _, _, gs, names = preprocessed_two_groups
        result = t_test(gs, names)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0

    def test_significant_for_separated_means(self, preprocessed_two_groups):
        _, _, gs, names = preprocessed_two_groups
        result = t_test(gs, names)
        # Groups separated by 3 std devs should be significant
        assert (result.values < 0.05).any()

    def test_not_significant_for_same_dist(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(20):
            g = "A" if i < 10 else "B"
            row = {"Sample": f"S{i}", "Group": g, "Met_0": rng.normal(0, 1)}
            rows.append(row)
        df = pd.DataFrame(rows)
        _, gs, names = preprocess_data(df)
        result = t_test(gs, names)
        # Same distribution → likely not significant (p > 0.05)
        assert result.values[0, 0] > 0.01


# ── u_test ──────────────────────────────────────────────────────────────

class TestUTest:
    def test_returns_df(self, preprocessed_two_groups):
        _, _, gs, names = preprocessed_two_groups
        result = u_test(gs, names)
        assert isinstance(result, pd.DataFrame)

    def test_shape(self, preprocessed_two_groups):
        _, _, gs, names = preprocessed_two_groups
        result = u_test(gs, names)
        assert result.shape == (5, 1)

    def test_column_name_utest(self, preprocessed_two_groups):
        _, _, gs, names = preprocessed_two_groups
        result = u_test(gs, names)
        assert "_utest" in result.columns[0]

    def test_values_in_01(self, preprocessed_two_groups):
        _, _, gs, names = preprocessed_two_groups
        result = u_test(gs, names)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0


# ── _decide_utest_method ────────────────────────────────────────────────

class TestDecideUtestMethod:
    def test_exact_for_small_no_ties(self, small_exact_data):
        _, gs, names = preprocess_data(small_exact_data)
        method = _decide_utest_method(gs, names)
        assert method == "exact"

    def test_asymptotic_for_large_n(self):
        # Need n > _UTEST_EXACT_MAX_N (10) per group
        rng = np.random.default_rng(77)
        rows = []
        for i in range(24):
            g = "A" if i < 12 else "B"
            row = {"Sample": f"S{i}", "Group": g}
            for j in range(3):
                row[f"Met_{j}"] = rng.uniform(0, 100)
            rows.append(row)
        df = pd.DataFrame(rows)
        _, gs, names = preprocess_data(df)
        method = _decide_utest_method(gs, names)
        assert method == "asymptotic"

    def test_asymptotic_for_ties(self):
        # Create data with ties
        df = pd.DataFrame({
            "Sample": [f"S{i}" for i in range(12)],
            "Group": ["A"] * 6 + ["B"] * 6,
            "Met_0": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        _, gs, names = preprocess_data(df)
        method = _decide_utest_method(gs, names)
        assert method == "asymptotic"


# ── anova_test ──────────────────────────────────────────────────────────

class TestAnovaTest:
    def test_shape(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = anova_test(gs, names)
        assert result.shape == (5, 1)

    def test_column_name(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = anova_test(gs, names)
        assert result.columns[0] == "p-value_ANOVA"

    def test_values_in_01(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = anova_test(gs, names)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0

    def test_significant_for_separated_groups(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = anova_test(gs, names)
        assert (result.values < 0.05).any()

    def test_constant_column_pvalue_one(self, data_with_constant_col):
        _, gs, names = preprocess_data(data_with_constant_col)
        result = anova_test(gs, names)
        # Constant column → no variance → p should be 1.0
        idx = names.index("Met_4")
        assert result.iloc[idx, 0] == 1.0

    def test_unequal_var_runs(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = anova_test(gs, names, use_var="unequal")
        assert result.shape == (5, 1)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0

    def test_equal_vs_unequal_differ(self, heteroscedastic_data):
        _, gs, names = preprocess_data(heteroscedastic_data)
        eq = anova_test(gs, names, use_var="equal")
        un = anova_test(gs, names, use_var="unequal")
        assert not np.allclose(eq.values, un.values)

    def test_invalid_use_var_raises(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        with pytest.raises(ValueError):
            anova_test(gs, names, use_var="bogus")


# ── kruskal_test ────────────────────────────────────────────────────────

class TestKruskalTest:
    def test_shape(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = kruskal_test(gs, names)
        assert result.shape == (5, 1)

    def test_column_name(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = kruskal_test(gs, names)
        assert result.columns[0] == "p-value_KW"

    def test_values_in_01(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = kruskal_test(gs, names)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0

    def test_significant_for_separated_groups(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = kruskal_test(gs, names)
        assert (result.values < 0.05).any()


# ── norm_test ───────────────────────────────────────────────────────────

class TestNormTest:
    def test_shapiro_shape(self, two_group_data):
        result = norm_test(two_group_data, method="shapiro")
        assert result.shape == (2, 5)

    def test_shapiro_index(self, two_group_data):
        result = norm_test(two_group_data, method="shapiro")
        assert list(result.index) == ["W-statistic", "p-value"]

    def test_normaltest_shape(self, two_group_data):
        result = norm_test(two_group_data, method="normaltest")
        assert result.shape == (2, 5)

    def test_normaltest_index(self, two_group_data):
        result = norm_test(two_group_data, method="normaltest")
        assert list(result.index) == ["χ²", "p-value"]

    def test_pvalues_in_01(self, two_group_data):
        result = norm_test(two_group_data, method="shapiro")
        pvals = result.loc["p-value"]
        assert (pvals >= 0.0).all()
        assert (pvals <= 1.0).all()

    def test_invalid_method_raises(self, two_group_data):
        with pytest.raises(ValueError):
            norm_test(two_group_data, method="invalid")

    def test_few_samples_fallback(self):
        df = pd.DataFrame({
            "Sample": ["S0", "S1"],
            "Group": ["A", "B"],
            "Met_0": [1.0, 2.0],
        })
        result = norm_test(df, method="shapiro")
        # Too few samples → fallback p=1.0
        assert result.loc["p-value", "Met_0"] == 1.0
