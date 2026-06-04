import numpy as np
import pandas as pd
import pytest

from lmsstat.stat._effect import effect_size_table
from lmsstat.stat import allstats

EXPECTED_COLS = [
    "feature", "group1", "group2", "mean1", "mean2",
    "fold_change", "log2fc", "cohens_d", "p_value", "p_adj",
]


@pytest.fixture()
def two_group_known():
    # A: Met_0 mean=2 (sd=1), Met_1 constant=10
    # B: Met_0 mean=5 (sd=1), Met_1 constant=10
    return pd.DataFrame(
        {
            "Sample": [f"S{i}" for i in range(6)],
            "Group": ["A", "A", "A", "B", "B", "B"],
            "Met_0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Met_1": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )


class TestEffectSizeTable:
    def test_columns_and_order(self, two_group_known):
        out = effect_size_table(two_group_known)
        assert list(out.columns) == EXPECTED_COLS

    def test_one_row_per_feature(self, two_group_known):
        out = effect_size_table(two_group_known)
        assert out["feature"].tolist() == ["Met_0", "Met_1"]

    def test_raises_on_three_groups(self, three_group_data):
        with pytest.raises(ValueError):
            effect_size_table(three_group_data)

    def test_group_labels_sorted_default(self, two_group_known):
        out = effect_size_table(two_group_known)
        assert (out["group1"] == "A").all()
        assert (out["group2"] == "B").all()

    def test_means(self, two_group_known):
        out = effect_size_table(two_group_known).set_index("feature")
        assert out.loc["Met_0", "mean1"] == pytest.approx(2.0)
        assert out.loc["Met_0", "mean2"] == pytest.approx(5.0)

    def test_fold_change_and_log2fc(self, two_group_known):
        out = effect_size_table(two_group_known).set_index("feature")
        assert out.loc["Met_0", "fold_change"] == pytest.approx(2.5)
        assert out.loc["Met_0", "log2fc"] == pytest.approx(np.log2(2.5))

    def test_cohens_d(self, two_group_known):
        out = effect_size_table(two_group_known).set_index("feature")
        # s_pooled = sqrt((2*1 + 2*1)/4) = 1 ; d = (5-2)/1 = 3
        assert out.loc["Met_0", "cohens_d"] == pytest.approx(3.0)

    def test_constant_feature(self, two_group_known):
        out = effect_size_table(two_group_known).set_index("feature")
        assert out.loc["Met_1", "fold_change"] == pytest.approx(1.0)
        assert out.loc["Met_1", "log2fc"] == pytest.approx(0.0)
        assert out.loc["Met_1", "cohens_d"] == pytest.approx(0.0)

    def test_group_order_flips_sign(self, two_group_known):
        base = effect_size_table(two_group_known).set_index("feature")
        flip = effect_size_table(two_group_known, group_order=["B", "A"]).set_index("feature")
        assert flip.loc["Met_0", "cohens_d"] == pytest.approx(-base.loc["Met_0", "cohens_d"])
        assert flip.loc["Met_0", "log2fc"] == pytest.approx(-base.loc["Met_0", "log2fc"])
        assert flip.loc["Met_0", "group1"] == "B"
        assert flip.loc["Met_0", "group2"] == "A"

    def test_invalid_group_order_raises(self, two_group_known):
        with pytest.raises(ValueError):
            effect_size_table(two_group_known, group_order=["A", "C"])
        with pytest.raises(ValueError):
            effect_size_table(two_group_known, group_order=["A"])

    def test_invalid_p_source_raises(self, two_group_known):
        with pytest.raises(ValueError):
            effect_size_table(two_group_known, p_source="anova")

    def test_p_value_matches_allstats_ttest(self, two_group_known):
        st = allstats(two_group_known, p_adj=False)
        out = effect_size_table(two_group_known).set_index("feature")
        col = [c for c in st.columns if c.endswith("_ttest")][0]
        assert out.loc["Met_0", "p_value"] == pytest.approx(st.loc["Met_0", col])

    def test_p_source_utest_in_range(self, two_group_known):
        out = effect_size_table(two_group_known, p_source="u-test")
        assert (out["p_value"] >= 0).all() and (out["p_value"] <= 1).all()

    def test_p_adj_ge_p_value(self, two_group_known):
        out = effect_size_table(two_group_known)
        assert np.all(out["p_adj"].to_numpy() + 1e-12 >= out["p_value"].to_numpy())

    def test_stats_res_path_matches_internal(self, two_group_known):
        st = allstats(two_group_known, p_adj=False)
        a = effect_size_table(two_group_known)
        b = effect_size_table(two_group_known, stats_res=st)
        pd.testing.assert_frame_equal(a, b)

    def test_does_not_mutate_input(self, two_group_known):
        before = two_group_known.copy(deep=True)
        _ = effect_size_table(two_group_known)
        pd.testing.assert_frame_equal(two_group_known, before)

    def test_nonpositive_mean_nan_fold_change(self):
        df = pd.DataFrame(
            {"Sample": ["a", "b", "c", "d"], "Group": ["A", "A", "B", "B"],
             "Met_0": [-1.0, -2.0, 3.0, 4.0]}  # group A mean = -1.5 (<= 0)
        )
        out = effect_size_table(df).set_index("feature")
        assert np.isnan(out.loc["Met_0", "fold_change"])
        assert np.isnan(out.loc["Met_0", "log2fc"])

    def test_nan_tolerant(self):
        df = pd.DataFrame(
            {"Sample": ["a", "b", "c", "d"], "Group": ["A", "A", "B", "B"],
             "Met_0": [1.0, np.nan, 4.0, 6.0]}  # A mean=1, B mean=5
        )
        out = effect_size_table(df).set_index("feature")
        assert out.loc["Met_0", "mean1"] == pytest.approx(1.0)
        assert out.loc["Met_0", "mean2"] == pytest.approx(5.0)

    def test_cohens_d_with_singleton_group(self):
        # Group A has a single finite value (n1 == 1): its variance is undefined,
        # but Cohen's d should still be computable from group B's spread.
        df = pd.DataFrame(
            {"Sample": ["a", "b", "c", "d"], "Group": ["A", "A", "B", "B"],
             "Met_0": [2.0, np.nan, 4.0, 6.0]}  # A: one finite (2.0); B: [4, 6]
        )
        out = effect_size_table(df).set_index("feature")
        d = out.loc["Met_0", "cohens_d"]
        assert np.isfinite(d)
        # ss_A = 0 (n1 == 1); ss_B = (2-1)*var_B = 2; df_pool = 1+2-2 = 1
        # s_pooled = sqrt((0 + 2) / 1) = sqrt(2); d = (5 - 2) / sqrt(2)
        assert d == pytest.approx(3.0 / np.sqrt(2.0))

    def test_stats_res_missing_feature_raises(self, two_group_known):
        st = allstats(two_group_known, p_adj=False)
        st_partial = st.drop(index="Met_1")  # omit a feature present in data
        with pytest.raises(ValueError):
            effect_size_table(two_group_known, stats_res=st_partial)
