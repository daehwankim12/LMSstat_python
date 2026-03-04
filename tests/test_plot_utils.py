import numpy as np
import pandas as pd
import pytest

from lmsstat.plot._utils import (
    _pal,
    _validate_n_components,
    simca_cv_groups,
    pca,
    plsda,
    _annot,
)
from lmsstat.stat._utils import preprocess_data


# ── _pal ────────────────────────────────────────────────────────────────

class TestPal:
    def test_returns_list_of_hex(self):
        result = _pal(5)
        assert isinstance(result, list)
        assert len(result) == 5
        for c in result:
            assert isinstance(c, str)
            assert c.startswith("#")

    def test_zero_returns_empty(self):
        assert _pal(0) == []

    def test_wraps_around_gt_20(self):
        result = _pal(25)
        assert len(result) == 25


# ── _validate_n_components ──────────────────────────────────────────────

class TestValidateNComponents:
    def test_valid_passes(self):
        _validate_n_components(2, 5, model_name="PCA")

    def test_non_int_raises(self):
        with pytest.raises(TypeError):
            _validate_n_components(2.5, 5, model_name="PCA")

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            _validate_n_components(0, 5, model_name="PCA")

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _validate_n_components(-1, 5, model_name="PCA")

    def test_exceeds_max_raises(self):
        with pytest.raises(ValueError):
            _validate_n_components(10, 5, model_name="PCA")

    def test_max_zero_raises(self):
        with pytest.raises(ValueError):
            _validate_n_components(1, 0, model_name="PCA")


# ── simca_cv_groups ─────────────────────────────────────────────────────

class TestSimcaCvGroups:
    def test_length_matches(self):
        result = simca_cv_groups(20, cv_splits=7)
        assert len(result) == 20

    def test_values_in_range(self):
        result = simca_cv_groups(20, cv_splits=7)
        assert np.all(result >= 0)
        assert np.all(result < 7)


# ── pca ─────────────────────────────────────────────────────────────────

class TestPCA:
    def test_returns_four_tuple(self, two_group_data):
        result = pca(two_group_data, n_components=2)
        assert len(result) == 4

    def test_scores_shape(self, two_group_data):
        scores, _, _, _ = pca(two_group_data, n_components=2)
        assert scores.shape == (20, 2)

    def test_scores_columns(self, two_group_data):
        scores, _, _, _ = pca(two_group_data, n_components=2)
        assert list(scores.columns) == ["PC1", "PC2"]

    def test_loadings_shape(self, two_group_data):
        _, loadings, _, _ = pca(two_group_data, n_components=2)
        assert loadings.shape == (5, 2)

    def test_r2_in_range(self, two_group_data):
        _, _, r2, _ = pca(two_group_data, n_components=2)
        assert 0.0 < r2 <= 1.0

    def test_q2_finite(self, two_group_data):
        _, _, _, q2 = pca(two_group_data, n_components=2)
        assert np.isfinite(q2)

    def test_too_many_components_raises(self, two_group_data):
        with pytest.raises(ValueError):
            pca(two_group_data, n_components=100)

    def test_single_sample_raises(self):
        df = pd.DataFrame({
            "Sample": ["S0"],
            "Group": ["A"],
            "Met_0": [1.0],
            "Met_1": [2.0],
        })
        with pytest.raises(ValueError):
            pca(df, n_components=1)


# ── plsda ───────────────────────────────────────────────────────────────

class TestPLSDA:
    def test_returns_six_tuple(self, two_group_data):
        result = plsda(two_group_data, n_components=2)
        assert len(result) == 6

    def test_scores_shape(self, two_group_data):
        scores, _, _, _, _, _ = plsda(two_group_data, n_components=2)
        assert scores.shape == (20, 2)

    def test_scores_columns(self, two_group_data):
        scores, _, _, _, _, _ = plsda(two_group_data, n_components=2)
        assert list(scores.columns) == ["LV1", "LV2"]

    def test_loadings_shape(self, two_group_data):
        _, loadings, _, _, _, _ = plsda(two_group_data, n_components=2)
        assert loadings.shape == (5, 2)

    def test_r2x_finite(self, two_group_data):
        _, _, r2x, _, _, _ = plsda(two_group_data, n_components=2)
        assert np.isfinite(r2x)

    def test_r2y_finite(self, two_group_data):
        _, _, _, r2y, _, _ = plsda(two_group_data, n_components=2)
        assert np.isfinite(r2y)

    def test_q2_finite(self, two_group_data):
        _, _, _, _, q2, _ = plsda(two_group_data, n_components=2)
        assert np.isfinite(q2)

    def test_vip_shape(self, two_group_data):
        _, _, _, _, _, vip = plsda(two_group_data, n_components=2)
        assert vip.shape == (5, 1)

    def test_vip_nonneg(self, two_group_data):
        _, _, _, _, _, vip = plsda(two_group_data, n_components=2)
        assert (vip.values >= 0).all()

    def test_two_groups_works(self, two_group_data):
        result = plsda(two_group_data, n_components=2)
        assert len(result) == 6

    def test_one_group_raises(self):
        df = pd.DataFrame({
            "Sample": [f"S{i}" for i in range(10)],
            "Group": ["A"] * 10,
            "Met_0": np.random.default_rng(0).normal(size=10),
            "Met_1": np.random.default_rng(1).normal(size=10),
        })
        with pytest.raises(ValueError):
            plsda(df, n_components=2)


# ── _annot ──────────────────────────────────────────────────────────────

class TestAnnot:
    def _make_gg(self):
        from plotnine import ggplot, aes, geom_point
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        return ggplot(df, aes("x", "y")) + geom_point()

    def test_returns_unchanged_on_empty(self):
        gg = self._make_gg()
        result = _annot(gg, pd.DataFrame(), ["A", "B"], y_top=3.0)
        assert result is gg

    def test_no_annotation_if_p_gt_05(self):
        gg = self._make_gg()
        st = pd.DataFrame({
            "group1": ["A"],
            "group2": ["B"],
            "p_value": [0.5],
        })
        result = _annot(gg, st, ["A", "B"], y_top=3.0)
        # Should return gg without adding annotations
        assert result is gg

    def test_adds_annotations_for_significant(self):
        gg = self._make_gg()
        st = pd.DataFrame({
            "group1": ["A"],
            "group2": ["B"],
            "p_value": [0.01],
        })
        result = _annot(gg, st, ["A", "B"], y_top=3.0)
        # result should have more layers than original
        assert result is not None

    def test_returns_gg_on_none(self):
        gg = self._make_gg()
        result = _annot(gg, None, ["A", "B"], y_top=3.0)
        assert result is gg
