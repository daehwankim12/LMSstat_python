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
