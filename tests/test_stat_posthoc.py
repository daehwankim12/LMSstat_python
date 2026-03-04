import numpy as np
import pandas as pd
import pytest

from lmsstat.stat._posthoc import (
    preprocess_groups,
    scheffe_test,
    dunn_test,
    games_howell_test,
)
from lmsstat.stat._utils import preprocess_data


# ── preprocess_groups ───────────────────────────────────────────────────

class TestPreprocessGroups:
    def test_returns_dict_and_int(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result, max_len = preprocess_groups(gs, names)
        assert isinstance(result, dict)
        assert isinstance(max_len, (int, np.integer))

    def test_keys_match_groups(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result, _ = preprocess_groups(gs, names)
        assert set(result.keys()) == set(gs.groups.keys())

    def test_pads_to_max_length(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result, max_len = preprocess_groups(gs, names)
        for group_data in result.values():
            for arr in group_data.values():
                assert len(arr) == max_len

    def test_metabolite_names_filter(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        subset = names[:2]
        result, _ = preprocess_groups(gs, subset)
        for group_data in result.values():
            assert set(group_data.keys()) == set(subset)

    def test_none_uses_all_except_sample_group(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result, _ = preprocess_groups(gs, metabolite_names=None)
        for group_data in result.values():
            assert set(group_data.keys()) == set(names)


# ── scheffe_test ────────────────────────────────────────────────────────

class TestScheffeTest:
    def test_shape(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = scheffe_test(gs, names)
        # 3 groups → C(3,2)=3 pairs
        assert result.shape == (5, 3)

    def test_column_names(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = scheffe_test(gs, names)
        for col in result.columns:
            assert "_scheffe" in col

    def test_values_in_01(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = scheffe_test(gs, names)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0

    def test_index_matches_metabolites(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = scheffe_test(gs, names)
        assert list(result.index) == names


# ── dunn_test ───────────────────────────────────────────────────────────

class TestDunnTest:
    def test_shape(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = dunn_test(gs, names)
        assert result.shape == (5, 3)

    def test_column_names(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = dunn_test(gs, names)
        for col in result.columns:
            assert "_dunn" in col

    def test_values_in_01(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = dunn_test(gs, names)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0


# ── games_howell_test ───────────────────────────────────────────────────

class TestGamesHowellTest:
    def test_shape(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = games_howell_test(gs, names)
        assert result.shape == (5, 3)

    def test_column_names(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = games_howell_test(gs, names)
        for col in result.columns:
            assert "_games_howell" in col

    def test_values_in_01(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = games_howell_test(gs, names)
        assert result.min().min() >= 0.0
        assert result.max().max() <= 1.0

    def test_index_matches_metabolites(self, preprocessed_three_groups):
        _, _, gs, names = preprocessed_three_groups
        result = games_howell_test(gs, names)
        assert list(result.index) == names
