import numpy as np
import pandas as pd
import pytest

from lmsstat.stat._utils import (
    _sanitize_pvalues_array,
    _sanitize_pvalues_df,
    ensure_sample_group_columns,
    preprocess_data,
    p_adjust,
    correlation,
    scaling,
)


# ── _sanitize_pvalues_array ─────────────────────────────────────────────

class TestSanitizePvaluesArray:
    def test_nan_replaced_by_one(self):
        assert _sanitize_pvalues_array(np.array([np.nan]))[0] == 1.0

    def test_inf_replaced_by_one(self):
        assert _sanitize_pvalues_array(np.array([np.inf]))[0] == 1.0

    def test_neginf_replaced_by_one(self):
        assert _sanitize_pvalues_array(np.array([-np.inf]))[0] == 1.0

    def test_negative_clipped_to_zero(self):
        assert _sanitize_pvalues_array(np.array([-0.5]))[0] == 0.0

    def test_normal_unchanged(self):
        arr = np.array([0.01, 0.5, 1.0])
        np.testing.assert_array_equal(_sanitize_pvalues_array(arr), arr)

    def test_empty_array(self):
        result = _sanitize_pvalues_array(np.array([]))
        assert result.shape == (0,)


# ── _sanitize_pvalues_df ────────────────────────────────────────────────

class TestSanitizePvaluesDf:
    def test_nan_inf_replaced(self):
        df = pd.DataFrame({"a": [np.nan, np.inf, -np.inf]})
        result = _sanitize_pvalues_df(df)
        assert (result["a"] == 1.0).all()

    def test_non_numeric_coerced_to_one(self):
        df = pd.DataFrame({"a": ["foo", "bar"]})
        result = _sanitize_pvalues_df(df)
        assert (result["a"] == 1.0).all()

    def test_clip_to_01(self):
        df = pd.DataFrame({"a": [-0.5, 0.5, 1.5]})
        result = _sanitize_pvalues_df(df)
        assert result["a"].min() >= 0.0
        assert result["a"].max() <= 1.0


# ── ensure_sample_group_columns ─────────────────────────────────────────

class TestEnsureSampleGroupColumns:
    def test_correct_rename(self):
        df = pd.DataFrame({"id": [1], "cls": ["A"], "x": [1.0]})
        out = ensure_sample_group_columns(df)
        assert list(out.columns[:2]) == ["Sample", "Group"]

    def test_already_correct(self):
        df = pd.DataFrame({"Sample": [1], "Group": ["A"], "x": [1.0]})
        out = ensure_sample_group_columns(df)
        assert list(out.columns[:2]) == ["Sample", "Group"]

    def test_non_df_raises(self):
        with pytest.raises(TypeError):
            ensure_sample_group_columns("not a dataframe")

    def test_too_few_cols_raises(self):
        with pytest.raises(ValueError):
            ensure_sample_group_columns(pd.DataFrame({"a": [1]}))

    def test_duplicate_names_raises(self):
        df = pd.DataFrame({"Sample": [1], "Group": ["A"], "Sample": [2.0]})
        # pandas silently merges duplicate column names during construction,
        # so we must construct a DF that actually causes the duplication after rename
        df2 = pd.DataFrame({0: [1], 1: ["A"], "Sample": [2.0]})
        df2.columns = ["col1", "col2", "Sample"]
        # After rename, col1→Sample, col2→Group, Sample→Sample → duplicate "Sample"
        with pytest.raises(ValueError, match="duplicated"):
            ensure_sample_group_columns(df2)


# ── preprocess_data ─────────────────────────────────────────────────────

class TestPreprocessData:
    def test_returns_three_tuple(self, two_group_data):
        result = preprocess_data(two_group_data)
        assert len(result) == 3

    def test_raw_shape(self, two_group_data):
        raw, _, _ = preprocess_data(two_group_data)
        assert raw.shape == (20, 5)

    def test_group_keys(self, two_group_data):
        _, gs, _ = preprocess_data(two_group_data)
        assert set(gs.groups.keys()) == {"A", "B"}

    def test_metabolite_names(self, two_group_data):
        _, _, names = preprocess_data(two_group_data)
        assert names == [f"Met_{i}" for i in range(5)]

    def test_sorted_by_group(self, two_group_data):
        _, gs, _ = preprocess_data(two_group_data)
        groups = gs.obj["Group"].tolist()
        assert groups == sorted(groups)

    def test_string_in_numeric_becomes_nan(self):
        df = pd.DataFrame({
            "Sample": ["S0", "S1"],
            "Group": ["A", "B"],
            "x": [1.0, "oops"],
        })
        raw, _, _ = preprocess_data(df)
        assert np.isnan(raw["x"].iloc[1])


# ── p_adjust ────────────────────────────────────────────────────────────

class TestPAdjust:
    def test_returns_df(self, preprocessed_two_groups):
        _, raw, gs, names = preprocessed_two_groups
        from lmsstat.stat._tests import t_test
        mat = t_test(gs, names)
        result = p_adjust(mat)
        assert isinstance(result, pd.DataFrame)

    def test_values_in_01(self, preprocessed_two_groups):
        _, raw, gs, names = preprocessed_two_groups
        from lmsstat.stat._tests import t_test
        result = p_adjust(t_test(gs, names))
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0

    def test_adjusted_ge_raw(self, preprocessed_two_groups):
        _, raw, gs, names = preprocessed_two_groups
        from lmsstat.stat._tests import t_test
        raw_p = t_test(gs, names)
        adj_p = p_adjust(raw_p)
        # FDR-adjusted p-values should be >= raw (may not hold exactly for
        # all methods, but BH typically preserves this for well-behaved inputs)
        diff = (adj_p.values - raw_p.values).flatten()
        assert np.all(diff >= -1e-10)

    def test_preserves_shape(self, preprocessed_two_groups):
        _, raw, gs, names = preprocessed_two_groups
        from lmsstat.stat._tests import t_test
        mat = t_test(gs, names)
        result = p_adjust(mat)
        assert result.shape == mat.shape


# ── correlation ─────────────────────────────────────────────────────────

class TestCorrelation:
    def test_sample_axis_shape(self, two_group_data):
        result = correlation(two_group_data, axis="sample")
        assert result.shape == (20, 20)

    def test_metabolite_axis_shape(self, two_group_data):
        result = correlation(two_group_data, axis="metabolite")
        assert result.shape == (5, 5)

    def test_diagonal_is_one(self, two_group_data):
        result = correlation(two_group_data, axis="metabolite")
        np.testing.assert_allclose(np.diag(result.values), 1.0, atol=1e-10)

    def test_symmetric(self, two_group_data):
        result = correlation(two_group_data, axis="metabolite")
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-10)

    def test_invalid_axis_raises(self, two_group_data):
        with pytest.raises(ValueError):
            correlation(two_group_data, axis="invalid")

    def test_spearman(self, two_group_data):
        result = correlation(two_group_data, axis="metabolite", method="spearman")
        assert result.shape == (5, 5)


# ── scaling ─────────────────────────────────────────────────────────────

class TestScaling:
    def test_auto_mean_zero(self, two_group_data):
        result = scaling(two_group_data, method="auto")
        numeric = result.drop(columns=["Sample", "Group"])
        np.testing.assert_allclose(numeric.mean().values, 0.0, atol=1e-10)

    def test_auto_std_one(self, two_group_data):
        result = scaling(two_group_data, method="auto")
        numeric = result.drop(columns=["Sample", "Group"])
        np.testing.assert_allclose(numeric.std(ddof=1).values, 1.0, atol=1e-10)

    def test_pareto_mean_zero(self, two_group_data):
        result = scaling(two_group_data, method="pareto")
        numeric = result.drop(columns=["Sample", "Group"])
        np.testing.assert_allclose(numeric.mean().values, 0.0, atol=1e-10)

    def test_preserves_sample_group(self, two_group_data):
        result = scaling(two_group_data, method="auto")
        assert "Sample" in result.columns
        assert "Group" in result.columns

    def test_preserves_shape(self, two_group_data):
        result = scaling(two_group_data, method="auto")
        assert result.shape == two_group_data.shape

    def test_invalid_method_raises(self, two_group_data):
        with pytest.raises(ValueError):
            scaling(two_group_data, method="invalid")

    def test_constant_column_no_nan_inf(self, data_with_constant_col):
        result = scaling(data_with_constant_col, method="auto")
        numeric = result.drop(columns=["Sample", "Group"])
        assert not numeric.isnull().any().any()
        assert np.all(np.isfinite(numeric.values))
