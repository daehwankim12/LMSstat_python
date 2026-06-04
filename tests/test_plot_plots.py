import numpy as np
import pandas as pd
import pytest

from lmsstat.plot._plots import (
    _safe_filename,
    _parse_pair,
    plot_pca,
    plot_plsda,
    plot_heatmap,
    plot_box,
    plot_bar,
    plot_volcano,
    plot_correlation,
    plot_vip,
    FOLDER,
)
from lmsstat.stat._allstat import allstats
from lmsstat.stat._effect import effect_size_table


pytestmark = pytest.mark.slow


# ── _safe_filename ──────────────────────────────────────────────────────

class TestSafeFilename:
    def test_normal_pass(self):
        assert _safe_filename("hello", ".png") == "hello.png"

    def test_invalid_chars_replaced(self):
        result = _safe_filename('a/b*c?d"e', ".png")
        assert "/" not in result.replace(".png", "")
        assert "*" not in result
        assert "?" not in result

    def test_long_name_truncated(self):
        long = "x" * 300
        result = _safe_filename(long, ".png")
        assert len(result) <= 255

    def test_empty_becomes_plot(self):
        result = _safe_filename("", ".png")
        assert result.startswith("plot")

    def test_newlines_replaced(self):
        result = _safe_filename("line1\nline2", ".png")
        assert "\n" not in result


# ── _parse_pair ─────────────────────────────────────────────────────────

class TestParsePair:
    def test_standard_ttest(self):
        result = _parse_pair("(A, B)_ttest")
        assert result == ("A", "B")

    def test_with_groups_hint(self):
        result = _parse_pair("(A, B)_ttest", groups=["A", "B"])
        assert result == ("A", "B")

    def test_reversed_order(self):
        result = _parse_pair("(B, A)_ttest", groups=["A", "B"])
        assert result == ("B", "A")

    def test_no_groups_fallback(self):
        result = _parse_pair("(X, Y)_utest")
        assert result == ("X", "Y")

    def test_no_match_returns_none(self):
        result = _parse_pair("not_a_pair")
        assert result is None

    def test_scheffe_suffix(self):
        result = _parse_pair("(X, Y)_scheffe")
        assert result == ("X", "Y")


# ── plot_pca ────────────────────────────────────────────────────────────

class TestPlotPCA:
    def test_returns_three_tuple(self, two_group_data, tmp_path):
        save = tmp_path / "pca.png"
        result = plot_pca(two_group_data, save_path=str(save))
        assert len(result) == 3

    def test_saves_png(self, two_group_data, tmp_path):
        save = tmp_path / "pca.png"
        plot_pca(two_group_data, save_path=str(save))
        assert save.exists()

    def test_png_header(self, two_group_data, tmp_path):
        save = tmp_path / "pca.png"
        plot_pca(two_group_data, save_path=str(save))
        with open(save, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    def test_r2_q2_finite(self, two_group_data, tmp_path):
        save = tmp_path / "pca.png"
        _, r2, q2 = plot_pca(two_group_data, save_path=str(save))
        assert isinstance(r2, float)
        assert isinstance(q2, float)
        assert np.isfinite(r2)
        assert np.isfinite(q2)

    def test_n_components_1_raises(self, two_group_data, tmp_path):
        with pytest.raises(ValueError):
            plot_pca(two_group_data, n_components=1, save_path=str(tmp_path / "x.png"))


# ── plot_plsda ──────────────────────────────────────────────────────────

class TestPlotPLSDA:
    def test_returns_five_tuple(self, two_group_data, tmp_path):
        save = tmp_path / "plsda.png"
        result = plot_plsda(two_group_data, save_path=str(save))
        assert len(result) == 5

    def test_saves_png(self, two_group_data, tmp_path):
        save = tmp_path / "plsda.png"
        plot_plsda(two_group_data, save_path=str(save))
        assert save.exists()

    def test_vip_is_dataframe(self, two_group_data, tmp_path):
        save = tmp_path / "plsda.png"
        _, _, _, _, vip = plot_plsda(two_group_data, save_path=str(save))
        assert isinstance(vip, pd.DataFrame)

    def test_n_components_1_raises(self, two_group_data, tmp_path):
        with pytest.raises(ValueError):
            plot_plsda(two_group_data, n_components=1, save_path=str(tmp_path / "x.png"))


# ── plot_heatmap ────────────────────────────────────────────────────────

class TestPlotHeatmap:
    def test_saves_file(self, two_group_data, tmp_path):
        out = tmp_path / "heatmap.png"
        result = plot_heatmap(two_group_data, out_path=str(out))
        assert out.exists()
        assert result is None

    def test_no_cluster(self, two_group_data, tmp_path):
        out = tmp_path / "heatmap_nc.png"
        plot_heatmap(two_group_data, out_path=str(out),
                     row_cluster=False, col_cluster=False)
        assert out.exists()


# ── plot_box ────────────────────────────────────────────────────────────

class TestPlotBox:
    def test_creates_files(self, two_group_data, tmp_path, monkeypatch):
        stats = allstats(two_group_data, p_adj=False)
        outdir = str(tmp_path / "boxplot")
        monkeypatch.setitem(FOLDER, "box", outdir)
        result = plot_box(two_group_data, stats, max_workers=1)
        assert result is None
        from pathlib import Path
        created = list(Path(outdir).glob("*.png"))
        assert len(created) > 0

    def test_games_howell_test_type(self, three_group_data, tmp_path, monkeypatch):
        stats = allstats(three_group_data, p_adj=False, posthoc="games_howell")
        outdir = str(tmp_path / "box_gh")
        monkeypatch.setitem(FOLDER, "box", outdir)
        result = plot_box(three_group_data, stats, test_type="games_howell", max_workers=1)
        assert result is None
        from pathlib import Path
        created = list(Path(outdir).glob("*.png"))
        assert len(created) > 0

    def test_invalid_test_type_raises(self, two_group_data):
        stats = allstats(two_group_data, p_adj=False)
        with pytest.raises(ValueError):
            plot_box(two_group_data, stats, test_type="tukey", max_workers=1)

    def test_significant_only_filters(self, two_group_data, tmp_path, monkeypatch):
        stats = allstats(two_group_data, p_adj=False)
        outdir_all = str(tmp_path / "box_all")
        monkeypatch.setitem(FOLDER, "box", outdir_all)
        plot_box(two_group_data, stats, max_workers=1, significant_only=False)

        outdir_sig = str(tmp_path / "box_sig")
        monkeypatch.setitem(FOLDER, "box", outdir_sig)
        plot_box(two_group_data, stats, max_workers=1, significant_only=True)

        from pathlib import Path
        all_count = len(list(Path(outdir_all).glob("*.png")))
        sig_count = len(list(Path(outdir_sig).glob("*.png")))
        assert sig_count <= all_count


# ── plot_bar ────────────────────────────────────────────────────────────

class TestPlotBar:
    def test_creates_files(self, two_group_data, tmp_path, monkeypatch):
        stats = allstats(two_group_data, p_adj=False)
        outdir = str(tmp_path / "barplot")
        monkeypatch.setitem(FOLDER, "bar", outdir)
        result = plot_bar(two_group_data, stats, max_workers=1)
        assert result is None
        from pathlib import Path
        created = list(Path(outdir).glob("*.png"))
        assert len(created) > 0

    def test_games_howell_test_type(self, three_group_data, tmp_path, monkeypatch):
        stats = allstats(three_group_data, p_adj=False, posthoc="games_howell")
        outdir = str(tmp_path / "bar_gh")
        monkeypatch.setitem(FOLDER, "bar", outdir)
        result = plot_bar(three_group_data, stats, test_type="games_howell", max_workers=1)
        assert result is None
        from pathlib import Path
        created = list(Path(outdir).glob("*.png"))
        assert len(created) > 0


# ── plot_volcano ────────────────────────────────────────────────────────

class TestPlotVolcano:
    def test_returns_object_and_saves_png(self, two_group_data, tmp_path):
        et = effect_size_table(two_group_data)
        save = tmp_path / "volcano.png"
        g = plot_volcano(et, save_path=str(save))
        assert g is not None
        assert save.exists()
        with open(save, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    def test_use_adjusted_false_runs(self, two_group_data, tmp_path):
        et = effect_size_table(two_group_data)
        save = tmp_path / "volcano_raw.png"
        plot_volcano(et, use_adjusted=False, save_path=str(save))
        assert save.exists()

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError):
            plot_volcano(pd.DataFrame({"x": [1.0], "y": [2.0]}))

    def test_invalid_log2fc_threshold_raises(self, two_group_data):
        et = effect_size_table(two_group_data)
        with pytest.raises(ValueError):
            plot_volcano(et, log2fc_threshold=-1.0)

    def test_invalid_p_threshold_raises(self, two_group_data):
        et = effect_size_table(two_group_data)
        with pytest.raises(ValueError):
            plot_volcano(et, p_threshold=2.0)

    def test_zero_pvalue_no_inf(self, two_group_data, tmp_path):
        et = effect_size_table(two_group_data)
        et.loc[et.index[0], "p_adj"] = 0.0  # force a zero p-value
        save = tmp_path / "volcano_zero.png"
        # Should floor p for plotting and not raise on -log10(0).
        plot_volcano(et, save_path=str(save))
        assert save.exists()

    def test_drops_nonfinite_pvalue_rows(self, two_group_data, tmp_path):
        et = effect_size_table(two_group_data)
        et.loc[et.index[0], "p_adj"] = np.nan  # invalid p → must be dropped, not plotted at y=0
        save = tmp_path / "volcano_partial.png"
        g = plot_volcano(et, save_path=str(save))
        # The plotted data must exclude the NaN-p feature.
        assert len(g.data) == len(et) - 1
        assert save.exists()

    def test_all_nonplottable_raises(self, two_group_data):
        et = effect_size_table(two_group_data)
        et["p_adj"] = np.nan
        with pytest.raises(ValueError):
            plot_volcano(et)


# ── plot_correlation ──────────────────────────────────────────────────────

class TestPlotCorrelation:
    def test_cluster_returns_clustergrid_and_saves(self, two_group_data, tmp_path):
        from seaborn.matrix import ClusterGrid
        out = tmp_path / "corr.png"
        cg = plot_correlation(two_group_data, out_path=str(out))
        assert isinstance(cg, ClusterGrid)
        assert out.exists()
        with open(out, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    def test_no_cluster_returns_axes_and_saves(self, two_group_data, tmp_path):
        import matplotlib.axes
        out = tmp_path / "corr_nc.png"
        ax = plot_correlation(two_group_data, cluster=False, out_path=str(out))
        assert isinstance(ax, matplotlib.axes.Axes)
        assert out.exists()

    def test_axis_metabolite_shape(self, two_group_data, tmp_path):
        # 5 metabolite columns → 5×5 correlation matrix
        cg = plot_correlation(two_group_data, axis="metabolite", out_path=str(tmp_path / "m.png"))
        assert cg.data2d.shape == (5, 5)

    def test_axis_sample_shape(self, two_group_data, tmp_path):
        # 20 samples → 20×20 correlation matrix
        cg = plot_correlation(two_group_data, axis="sample", out_path=str(tmp_path / "s.png"))
        assert cg.data2d.shape == (20, 20)

    def test_invalid_axis_raises(self, two_group_data, tmp_path):
        with pytest.raises(ValueError):
            plot_correlation(two_group_data, axis="bogus", out_path=str(tmp_path / "x.png"))

    def test_invalid_method_raises(self, two_group_data, tmp_path):
        with pytest.raises(ValueError):
            plot_correlation(two_group_data, method="bogus", out_path=str(tmp_path / "x.png"))

    def test_annot_small_matrix(self, tmp_path):
        df = pd.DataFrame(
            {"Sample": ["a", "b", "c"], "Group": ["A", "A", "B"],
             "M0": [1.0, 2.0, 3.0], "M1": [3.0, 2.0, 1.5]}
        )
        out = tmp_path / "annot.png"
        plot_correlation(df, annot=True, out_path=str(out))
        assert out.exists()

    def test_cluster_drops_constant_feature(self, tmp_path):
        df = pd.DataFrame(
            {"Sample": [f"S{i}" for i in range(6)], "Group": ["A"] * 3 + ["B"] * 3,
             "M0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
             "M1": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
             "M2": [5.0] * 6}  # constant → NaN correlations, must be dropped
        )
        cg = plot_correlation(df, out_path=str(tmp_path / "c.png"))
        assert "M2" not in list(cg.data2d.index)
        assert cg.data2d.shape == (2, 2)

    def test_cluster_too_few_finite_raises(self, tmp_path):
        df = pd.DataFrame(
            {"Sample": ["a", "b", "c"], "Group": ["A", "A", "B"],
             "M0": [1.0, 2.0, 3.0], "M1": [5.0] * 3, "M2": [7.0] * 3}  # two constants
        )
        with pytest.raises(ValueError):
            plot_correlation(df, out_path=str(tmp_path / "x.png"))

    def test_sample_axis_labels_are_sample_ids(self, two_group_data, tmp_path):
        cg = plot_correlation(two_group_data, axis="sample", out_path=str(tmp_path / "s.png"))
        sample_ids = set(two_group_data.iloc[:, 0].astype(str))
        assert set(map(str, cg.data2d.index)) == sample_ids

    def test_out_path_none_no_save(self, two_group_data, tmp_path, monkeypatch):
        from seaborn.matrix import ClusterGrid
        monkeypatch.chdir(tmp_path)
        cg = plot_correlation(two_group_data, out_path=None)
        assert isinstance(cg, ClusterGrid)
        assert not (tmp_path / "correlation_plot.png").exists()


# ── plot_vip ──────────────────────────────────────────────────────────────

def _vip_df(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"VIP": rng.uniform(0.0, 3.0, n)},
        index=[f"M{i:02d}" for i in range(n)],
    )


class TestPlotVip:
    def test_returns_object_and_saves_png(self, tmp_path):
        save = tmp_path / "vip.png"
        g = plot_vip(_vip_df(), save_path=str(save))
        assert g is not None
        assert save.exists()
        with open(save, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    def test_top_n_limits_rows(self, tmp_path):
        g = plot_vip(_vip_df(n=30), top_n=10, save_path=str(tmp_path / "v.png"))
        assert len(g.data) == 10

    def test_top_n_larger_than_n_shows_all(self, tmp_path):
        g = plot_vip(_vip_df(n=8), top_n=50, save_path=str(tmp_path / "v.png"))
        assert len(g.data) == 8

    def test_selects_highest_vip(self, tmp_path):
        df = pd.DataFrame({"VIP": [0.2, 2.5, 1.1, 0.9]}, index=["a", "b", "c", "d"])
        g = plot_vip(df, top_n=2, save_path=str(tmp_path / "v.png"))
        assert set(g.data["feature"]) == {"b", "c"}

    def test_missing_vip_column_raises(self, tmp_path):
        df = pd.DataFrame({"score": [1.0, 2.0]}, index=["a", "b"])
        with pytest.raises(ValueError):
            plot_vip(df, save_path=str(tmp_path / "v.png"))

    def test_invalid_top_n_raises(self, tmp_path):
        for bad in (0, -3, 2.5):
            with pytest.raises((ValueError, TypeError)):
                plot_vip(_vip_df(), top_n=bad, save_path=str(tmp_path / "v.png"))

    def test_non_finite_vip_dropped(self, tmp_path):
        df = pd.DataFrame({"VIP": [2.0, np.nan, np.inf, 1.0]}, index=["a", "b", "c", "d"])
        g = plot_vip(df, save_path=str(tmp_path / "v.png"))
        assert set(g.data["feature"]) == {"a", "d"}

    def test_all_non_finite_raises(self, tmp_path):
        df = pd.DataFrame({"VIP": [np.nan, np.inf]}, index=["a", "b"])
        with pytest.raises(ValueError):
            plot_vip(df, save_path=str(tmp_path / "v.png"))

    def test_invalid_threshold_raises(self, tmp_path):
        for bad in (-1.0, np.nan, np.inf):
            with pytest.raises(ValueError):
                plot_vip(_vip_df(), vip_threshold=bad, save_path=str(tmp_path / "v.png"))

    def test_input_not_mutated(self, tmp_path):
        df = _vip_df()
        before = df.copy(deep=True)
        _ = plot_vip(df, save_path=str(tmp_path / "v.png"))
        pd.testing.assert_frame_equal(df, before)

    def test_consumes_plsda_vip_table(self, two_group_data, tmp_path):
        from lmsstat.plot._utils import plsda
        *_, vip_df = plsda(two_group_data, n_components=2)
        g = plot_vip(vip_df, save_path=str(tmp_path / "v.png"))
        assert g is not None
