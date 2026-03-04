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
    FOLDER,
)
from lmsstat.stat._allstat import allstats


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
