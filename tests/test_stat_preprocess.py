import numpy as np
import pandas as pd
import pytest

from lmsstat.stat._preprocess import impute_missing, log_transform, normalize


@pytest.fixture()
def simple_data():
    """Small, fully-known wide table for exact-value assertions."""
    return pd.DataFrame(
        {
            "Sample": ["S0", "S1", "S2", "S3"],
            "Group": ["A", "A", "B", "B"],
            "Met_0": [1.0, 2.0, 3.0, 4.0],
            "Met_1": [10.0, 20.0, 30.0, 40.0],
            "Met_2": [100.0, 200.0, 300.0, 400.0],
        }
    )


@pytest.fixture()
def positive_with_nan():
    """Strictly positive feature values with a couple of NaNs (valid for all three)."""
    return pd.DataFrame(
        {
            "Sample": ["S0", "S1", "S2", "S3"],
            "Group": ["A", "A", "B", "B"],
            "Met_0": [1.0, np.nan, 3.0, 4.0],
            "Met_1": [10.0, 20.0, np.nan, 40.0],
            "Met_2": [100.0, 200.0, 300.0, 400.0],
        }
    )


@pytest.fixture()
def single_group_data():
    return pd.DataFrame(
        {
            "Sample": ["S0", "S1", "S2"],
            "Group": ["A", "A", "A"],
            "Met_0": [1.0, np.nan, 3.0],
            "Met_1": [4.0, 5.0, 6.0],
        }
    )


# ── shared contract ──────────────────────────────────────────────────────

class TestContract:
    @pytest.mark.parametrize(
        "fn",
        [
            lambda d: impute_missing(d, method="min"),
            lambda d: log_transform(d),
            lambda d: normalize(d, method="median"),
        ],
    )
    def test_preserves_sample_group_and_order(self, simple_data, fn):
        out = fn(simple_data)
        assert list(out.columns) == list(simple_data.columns)
        assert out["Sample"].tolist() == simple_data["Sample"].tolist()
        assert out["Group"].tolist() == simple_data["Group"].tolist()

    @pytest.mark.parametrize(
        "fn",
        [
            lambda d: impute_missing(d, method="min"),
            lambda d: log_transform(d),
            lambda d: normalize(d, method="median"),
        ],
    )
    def test_does_not_mutate_input(self, positive_with_nan, fn):
        before = positive_with_nan.copy(deep=True)
        _ = fn(positive_with_nan)
        pd.testing.assert_frame_equal(positive_with_nan, before)

    @pytest.mark.parametrize(
        "fn",
        [
            lambda d: impute_missing(d, method="min"),
            lambda d: log_transform(d),
            lambda d: normalize(d, method="median"),
        ],
    )
    def test_non_numeric_feature_coerced_to_nan(self, fn):
        # A fully non-numeric feature column becomes all-NaN (per _split contract);
        # none of the three functions can manufacture a value for it.
        df = pd.DataFrame(
            {"Sample": ["a", "b"], "Group": ["A", "B"],
             "Met_0": ["x", "y"], "Met_1": [1.0, 2.0]}
        )
        out = fn(df)
        assert list(out.columns) == list(df.columns)
        assert out["Sample"].tolist() == ["a", "b"]
        assert out["Group"].tolist() == ["A", "B"]
        assert out["Met_0"].isna().all()

    @pytest.mark.parametrize(
        "fn",
        [
            lambda d: impute_missing(d, method="min"),
            lambda d: log_transform(d),
            lambda d: normalize(d, method="median"),
        ],
    )
    def test_preserves_input_index(self, fn):
        df = pd.DataFrame(
            {"Sample": ["a", "b", "c"], "Group": ["A", "A", "B"],
             "Met_0": [1.0, 2.0, 3.0], "Met_1": [4.0, 5.0, 6.0]},
            index=["r1", "r2", "r3"],
        )
        out = fn(df)
        assert list(out.index) == ["r1", "r2", "r3"]


# ── impute_missing ─────────────────────────────────────────────────────────

class TestImputeMissing:
    def test_min_fills_all_nans(self, data_with_nans):
        out = impute_missing(data_with_nans, method="min")
        assert out.iloc[:, 2:].isna().sum().sum() == 0

    def test_half_min_fills_all_nans(self, data_with_nans):
        out = impute_missing(data_with_nans, method="half_min")
        assert out.iloc[:, 2:].isna().sum().sum() == 0

    def test_knn_fills_all_nans(self, data_with_nans):
        out = impute_missing(data_with_nans, method="knn")
        assert out.iloc[:, 2:].isna().sum().sum() == 0

    def test_min_uses_column_minimum(self):
        df = pd.DataFrame(
            {"Sample": ["a", "b", "c"], "Group": ["A", "A", "B"],
             "Met_0": [2.0, 5.0, np.nan]}
        )
        out = impute_missing(df, method="min")
        assert out.loc[2, "Met_0"] == 2.0

    def test_half_min_uses_half_column_minimum(self):
        df = pd.DataFrame(
            {"Sample": ["a", "b", "c"], "Group": ["A", "A", "B"],
             "Met_0": [2.0, 5.0, np.nan]}
        )
        out = impute_missing(df, method="half_min")
        assert out.loc[2, "Met_0"] == 1.0

    def test_does_not_change_observed_values(self, simple_data):
        out = impute_missing(simple_data, method="half_min")
        pd.testing.assert_frame_equal(
            out[simple_data.columns], simple_data, check_dtype=False
        )

    def test_all_nan_column_stays_nan(self):
        df = pd.DataFrame(
            {"Sample": ["a", "b"], "Group": ["A", "B"],
             "Met_0": [np.nan, np.nan], "Met_1": [1.0, 2.0]}
        )
        out = impute_missing(df, method="half_min")
        assert out["Met_0"].isna().all()

    def test_knn_all_nan_column_stays_nan(self):
        df = pd.DataFrame(
            {"Sample": ["a", "b", "c"], "Group": ["A", "A", "B"],
             "Met_0": [np.nan, np.nan, np.nan],
             "Met_1": [1.0, np.nan, 3.0],
             "Met_2": [4.0, 5.0, 6.0]}
        )
        out = impute_missing(df, method="knn")
        # Column order preserved, all-NaN feature stays NaN, others imputed.
        assert list(out.columns) == list(df.columns)
        assert out["Met_0"].isna().all()
        assert out["Met_1"].isna().sum() == 0

    def test_single_group(self, single_group_data):
        out = impute_missing(single_group_data, method="min")
        assert out["Met_0"].isna().sum() == 0

    def test_knn_single_sample_does_not_error(self):
        # n_neighbors = max(1, min(5, 0)) = 1; must not crash on one sample.
        df = pd.DataFrame(
            {"Sample": ["a"], "Group": ["A"], "Met_0": [5.0], "Met_1": [7.0]}
        )
        out = impute_missing(df, method="knn")
        assert out.shape == df.shape
        assert np.isfinite(out.iloc[:, 2:].to_numpy(dtype=float)).all()

    def test_knn_all_nan_row(self):
        # A fully-missing sample is filled from observed columns (sklearn falls
        # back to column means) — codify "no crash, shape preserved, finite".
        df = pd.DataFrame(
            {"Sample": ["a", "b", "c"], "Group": ["A", "A", "B"],
             "Met_0": [np.nan, 2.0, 3.0], "Met_1": [np.nan, 5.0, 6.0]}
        )
        out = impute_missing(df, method="knn")
        assert out.shape == df.shape
        assert np.isfinite(out.iloc[:, 2:].to_numpy(dtype=float)).all()

    def test_invalid_method_raises(self, simple_data):
        with pytest.raises(ValueError):
            impute_missing(simple_data, method="bogus")


# ── log_transform ──────────────────────────────────────────────────────────

class TestLogTransform:
    def test_log2_known_values(self):
        df = pd.DataFrame(
            {"Sample": ["a", "b"], "Group": ["A", "B"], "Met_0": [3.0, 7.0]}
        )
        out = log_transform(df, base=2, offset=1.0)
        # log2(3+1)=2, log2(7+1)=3
        assert out.loc[0, "Met_0"] == pytest.approx(2.0)
        assert out.loc[1, "Met_0"] == pytest.approx(3.0)

    def test_offset_handles_zero(self):
        df = pd.DataFrame(
            {"Sample": ["a"], "Group": ["A"], "Met_0": [0.0]}
        )
        out = log_transform(df, base=2, offset=1.0)
        assert out.loc[0, "Met_0"] == pytest.approx(0.0)

    def test_base10(self):
        df = pd.DataFrame(
            {"Sample": ["a"], "Group": ["A"], "Met_0": [99.0]}
        )
        out = log_transform(df, base=10, offset=1.0)
        assert out.loc[0, "Met_0"] == pytest.approx(2.0)

    def test_base_e(self):
        df = pd.DataFrame(
            {"Sample": ["a"], "Group": ["A"], "Met_0": [np.e - 1.0]}
        )
        out = log_transform(df, base=np.e, offset=1.0)
        assert out.loc[0, "Met_0"] == pytest.approx(1.0)

    def test_preserves_nan(self):
        df = pd.DataFrame(
            {"Sample": ["a", "b"], "Group": ["A", "B"], "Met_0": [np.nan, 3.0]}
        )
        out = log_transform(df, base=2, offset=1.0)
        assert np.isnan(out.loc[0, "Met_0"])

    def test_nonpositive_after_offset_raises(self):
        df = pd.DataFrame(
            {"Sample": ["a"], "Group": ["A"], "Met_0": [-5.0]}
        )
        with pytest.raises(ValueError):
            log_transform(df, base=2, offset=1.0)

    def test_invalid_base_raises(self, simple_data):
        with pytest.raises(ValueError):
            log_transform(simple_data, base=1)

    @pytest.mark.parametrize("bad_offset", [np.nan, np.inf, -np.inf])
    def test_non_finite_offset_raises(self, simple_data, bad_offset):
        with pytest.raises(ValueError):
            log_transform(simple_data, offset=bad_offset)


# ── normalize ──────────────────────────────────────────────────────────────

class TestNormalize:
    def test_total_area_rows_sum_to_one(self, simple_data):
        out = normalize(simple_data, method="total_area")
        row_sums = out.iloc[:, 2:].sum(axis=1)
        assert np.allclose(row_sums.to_numpy(), 1.0)

    def test_median_rows_have_unit_median(self, simple_data):
        out = normalize(simple_data, method="median")
        row_meds = out.iloc[:, 2:].median(axis=1)
        assert np.allclose(row_meds.to_numpy(), 1.0)

    def test_pqn_runs_and_keeps_shape(self, simple_data):
        out = normalize(simple_data, method="pqn")
        assert out.shape == simple_data.shape
        assert np.isfinite(out.iloc[:, 2:].to_numpy(dtype=float)).all()

    def test_total_area_zero_sum_row_does_not_crash(self):
        df = pd.DataFrame(
            {"Sample": ["a", "b"], "Group": ["A", "B"],
             "Met_0": [0.0, 2.0], "Met_1": [0.0, 4.0]}
        )
        out = normalize(df, method="total_area")
        # zero-sum row left unchanged (no inf/nan introduced)
        assert np.isfinite(out.iloc[:, 2:].to_numpy(dtype=float)).all()

    def test_handles_nan(self):
        df = pd.DataFrame(
            {"Sample": ["a", "b"], "Group": ["A", "B"],
             "Met_0": [1.0, np.nan], "Met_1": [3.0, 4.0]}
        )
        out = normalize(df, method="median")
        # NaN stays NaN; observed entries finite
        assert np.isnan(out.loc[1, "Met_0"])

    @pytest.mark.parametrize("method", ["median", "total_area", "pqn"])
    def test_all_nan_row_preserved(self, method):
        df = pd.DataFrame(
            {"Sample": ["a", "b"], "Group": ["A", "B"],
             "Met_0": [np.nan, 2.0], "Met_1": [np.nan, 4.0]}
        )
        out = normalize(df, method=method)
        assert out.loc[0, ["Met_0", "Met_1"]].isna().all()
        assert not np.isinf(out.iloc[:, 2:].to_numpy(dtype=float)).any()

    @pytest.mark.parametrize("method", ["median", "total_area", "pqn"])
    def test_all_nan_feature_column_preserved(self, method):
        df = pd.DataFrame(
            {"Sample": ["a", "b"], "Group": ["A", "B"],
             "Met_0": [np.nan, np.nan], "Met_1": [1.0, 2.0]}
        )
        out = normalize(df, method=method)
        assert out["Met_0"].isna().all()
        assert not np.isinf(out.iloc[:, 2:].to_numpy(dtype=float)).any()

    def test_invalid_method_raises(self, simple_data):
        with pytest.raises(ValueError):
            normalize(simple_data, method="bogus")
