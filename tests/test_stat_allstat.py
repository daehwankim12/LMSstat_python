import numpy as np
import pandas as pd
import pytest

from lmsstat.stat._allstat import allstats


class TestAllstats:
    def test_two_groups_shape(self, two_group_data):
        result = allstats(two_group_data)
        # 2 groups → ttest + utest = 2 columns
        assert result.shape == (5, 2)

    def test_three_groups_shape(self, three_group_data):
        result = allstats(three_group_data)
        # 3 groups: C(3,2)=3 ttest + 3 utest + 1 ANOVA + 3 scheffe + 1 KW + 3 dunn = 14
        assert result.shape == (5, 14)

    def test_values_in_01(self, two_group_data):
        result = allstats(two_group_data)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0

    def test_three_groups_values_in_01(self, three_group_data):
        result = allstats(three_group_data)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0

    def test_p_adj_inflates(self, two_group_data):
        raw = allstats(two_group_data, p_adj=False)
        adj = allstats(two_group_data, p_adj=True)
        # Adjusted should be >= raw (for well-behaved BH)
        diff = (adj.values - raw.values).flatten()
        assert np.all(diff >= -1e-10)

    def test_one_group_raises(self):
        df = pd.DataFrame({
            "Sample": ["S0", "S1"],
            "Group": ["A", "A"],
            "Met_0": [1.0, 2.0],
        })
        with pytest.raises(ValueError):
            allstats(df)

    def test_index_is_metabolite_names(self, two_group_data):
        result = allstats(two_group_data)
        assert list(result.index) == [f"Met_{i}" for i in range(5)]

    def test_three_groups_all_test_types(self, three_group_data):
        result = allstats(three_group_data)
        cols = result.columns.tolist()
        assert any("_ttest" in c for c in cols)
        assert any("_utest" in c for c in cols)
        assert any("ANOVA" in c for c in cols)
        assert any("_scheffe" in c for c in cols)
        assert any("KW" in c for c in cols)
        assert any("_dunn" in c for c in cols)


class TestAllstatsPosthocOption:
    def test_default_posthoc_is_scheffe_only(self, three_group_data):
        result = allstats(three_group_data)
        cols = result.columns.tolist()
        assert any("_scheffe" in c for c in cols)
        assert not any("_games_howell" in c for c in cols)

    def test_posthoc_games_howell(self, three_group_data):
        result = allstats(three_group_data, posthoc="games_howell")
        cols = result.columns.tolist()
        assert any("_games_howell" in c for c in cols)
        assert not any("_scheffe" in c for c in cols)
        # same total column count as the scheffe default (swap, not add)
        assert result.shape == (5, 14)

    def test_posthoc_both(self, three_group_data):
        result = allstats(three_group_data, posthoc="both")
        cols = result.columns.tolist()
        assert any("_scheffe" in c for c in cols)
        assert any("_games_howell" in c for c in cols)
        # 3 ttest + 3 utest + 1 ANOVA + 3 scheffe + 3 games_howell + 1 KW + 3 dunn = 17
        assert result.shape == (5, 17)

    def test_invalid_posthoc_raises(self, three_group_data):
        with pytest.raises(ValueError):
            allstats(three_group_data, posthoc="tukey")

    def test_posthoc_noop_for_two_groups(self, two_group_data):
        # No post-hoc columns for two groups regardless of the option.
        result = allstats(two_group_data, posthoc="games_howell")
        assert result.shape == (5, 2)

    def test_values_in_01_with_both(self, three_group_data):
        result = allstats(three_group_data, posthoc="both")
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0


class TestAllstatsAnovaUseVar:
    def test_default_runs_equal_var(self, three_group_data):
        result = allstats(three_group_data)
        assert any("ANOVA" in c for c in result.columns)

    def test_unequal_var_runs(self, three_group_data):
        result = allstats(three_group_data, anova_use_var="unequal")
        assert any("ANOVA" in c for c in result.columns)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0

    def test_equal_vs_unequal_differ(self, three_group_data):
        eq = allstats(three_group_data, anova_use_var="equal", p_adj=False)
        un = allstats(three_group_data, anova_use_var="unequal", p_adj=False)
        assert not np.allclose(eq["p-value_ANOVA"].values, un["p-value_ANOVA"].values)

    def test_invalid_use_var_raises(self, three_group_data):
        with pytest.raises(ValueError):
            allstats(three_group_data, anova_use_var="bogus")

    def test_invalid_use_var_raises_for_two_groups(self, two_group_data):
        # Validated up front, even though two-group data never runs ANOVA.
        with pytest.raises(ValueError):
            allstats(two_group_data, anova_use_var="bogus")
