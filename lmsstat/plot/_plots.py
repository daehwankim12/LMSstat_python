from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotnine import (
    aes, ggplot, geom_boxplot, geom_point, position_jitter,
    scale_fill_manual, scale_x_discrete, scale_y_continuous,
    theme_classic, theme, geom_bar, geom_errorbar, labs,
    scale_color_manual, theme_minimal, element_text, stat_ellipse,
    geom_text, scale_x_continuous
)

from ._utils import _annot, _pal, scaling, pca, plsda

# silence plotnine’s “saving with transparency” warnings, etc.
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------- #
#                                helpers                                 #
# ---------------------------------------------------------------------- #
def _plot_box(df: pd.DataFrame, st: pd.DataFrame, order: list[str]):
    """Internal: draw a box-and-jitter plot with significance brackets."""
    df["Group"] = pd.Categorical(df["Group"], categories=order, ordered=True)
    y_raw_max = df.Value.max()
    y_max_bracket = y_raw_max * 1.15

    g = (
            ggplot(df, aes("Group", "Value", fill="Group"))
            + geom_boxplot(
        width=.35, colour="black", size=.7,
        outlier_shape=None, alpha=.85, show_legend=False
    )
            + geom_point(
        position=position_jitter(width=.15),
        size=1.2, alpha=.5, colour="black", show_legend=False
    )
            + scale_fill_manual(values=_pal(len(order)))
            + scale_x_discrete(limits=order)
            + scale_y_continuous(
        limits=(None, y_max_bracket),
        labels=lambda l: [f"{v:.1e}" for v in l]
    )
            + theme_classic(base_size=13)
            + theme(legend_position="none")
    )

    return _annot(g, st, order, y_top=y_raw_max)


def _plot_bar(df: pd.DataFrame, st: pd.DataFrame, order: list[str]):
    """Internal: draw a mean ± SE bar plot with significance brackets."""
    df["Group"] = pd.Categorical(df["Group"], categories=order, ordered=True)

    summ = (
        df.groupby("Group", observed=False)["Value"]
        .agg(["mean", "std", "count"])
        .reindex(order)
        .reset_index()
    )
    summ["se"] = summ["std"] / np.sqrt(summ["count"])

    y_top_raw = (summ["mean"] + summ["se"]).max()
    y_limit = y_top_raw * 1.15

    g = (
            ggplot(summ, aes("Group", "mean", fill="Group"))
            + geom_bar(stat="identity", colour="black", width=.6, size=.4)
            + geom_errorbar(
        aes(ymin="mean-se", ymax="mean+se"),
        width=.15, colour="black"
    )
            + scale_fill_manual(values=_pal(len(order)))
            + scale_x_discrete(limits=order)
            + scale_y_continuous(
        limits=(0, y_limit),
        expand=(0, 0),
        labels=lambda l: [f"{v:.1e}" for v in l]
    )
            + labs(y="Mean ± SE")
            + theme_classic(base_size=13)
            + theme(legend_position="none")
    )

    return _annot(g, st, order, y_top=y_top_raw)


# kind → drawing function, and default folders for auto-save
_KIND = {"box": _plot_box, "bar": _plot_bar}
FOLDER = {k: f"{k}plot" for k in _KIND}


# ---------------------------------------------------------------------- #
#                              core workers                              #
# ---------------------------------------------------------------------- #
def _worker(task):
    """Multiprocessing helper that actually draws and saves the plot."""
    metab, df_dict, st_rows, kind, order = task
    df = pd.DataFrame(df_dict)
    st = pd.DataFrame(st_rows, columns=["group1", "group2", "p_value"])

    g = (
            _KIND[kind](df, st, order)
            + labs(title=metab, x="Group")
            + theme(plot_title=element_text(weight="bold"))
    )

    Path(FOLDER[kind]).mkdir(exist_ok=True)

    import hashlib
    def _safe_filename(raw: str, tail: str) -> str:
        """
        Sanitize a string so it can safely be used as a file name.
        Keeps at least 10 characters and appends a short hash if truncated.
        """
        name = re.sub(r'[\\/*?:"<>|]', "_", raw).replace("\n", "_").strip()
        room = 255 - len(tail)
        if len(name) > room:
            prefix = name[: max(10, room - 8)]  # keep ≥10 characters
            suffix = hashlib.blake2b(name.encode(), digest_size=3).hexdigest()
            name = f"{prefix}_{suffix}"
        return f"{name}{tail}"

    fname = _safe_filename(metab, f"_{kind}.png")

    fig = g.draw()
    fig.set_size_inches(6, 6)
    fig.savefig(Path(FOLDER[kind], fname), dpi=600,
                pil_kwargs={"compress_level": 3})
    import matplotlib.pyplot as plt
    plt.close(fig)  # free memory
    del g, fig, df, st  # help GC


# ---------------------------------------------------------------------- #
#                             public helpers                             #
# ---------------------------------------------------------------------- #
def _plot_multi(
        data: pd.DataFrame,
        stats_res: pd.DataFrame,
        *,
        kind: str = "box",
        test_type: str = "t-test",
        order: list[str] | None = None
):
    """
    Draw multiple box/bar plots in parallel with significance annotations.

    Parameters
    ----------
    data : DataFrame
        Wide table. Col0 = Sample, Col1 = Group, remaining = variables.
    stats_res : DataFrame
        Statistics table. Index = variables, columns include p-value columns.
    kind : {"box", "bar"}
        Type of plot for each variable.
    test_type : {"t-test", "u-test", "scheffe", "dunn"}
        Which p-value column to pick in `stats_res`.
    order : list[str] | None
        Order of categorical groups on the x-axis.
    """
    if kind not in _KIND:
        raise ValueError("kind must be 'box' or 'bar'.")

    if test_type.lower() not in ("t-test", "u-test", "scheffe", "dunn"):
        raise ValueError(
            "test_type must be 't-test', 'u-test', 'scheffe', or 'dunn'."
        )

    # match the exact column name pattern in stats_res
    if test_type.lower() == "t-test":
        test_type = "ttest"
    elif test_type.lower() == "u-test":
        test_type = "utest"

    order = order or list(data.Group.unique())
    pcols = [c for c in stats_res.columns if test_type.lower() in c.lower()]

    tasks = []
    for metab in data.columns[2:]:
        df_dict = {"Group": data.Group.values, "Value": data[metab].values}

        rows = [
            {
                "group1": c.split("_")[0].strip("()").split(", ")[0],
                "group2": c.split("_")[0].strip("()").split(", ")[1],
                "p_value": float(stats_res.at[metab, c]),
            }
            for c in pcols
        ]
        tasks.append((metab, df_dict, rows, kind, order))

    with ProcessPoolExecutor() as ex:
        list(ex.map(_worker, tasks))


# high-level wrappers
def plot_bar(data, stats_res, *, test_type="t-test", order=None):
    _plot_multi(data, stats_res, kind="bar", test_type=test_type, order=order)
    return None


def plot_box(data, stats_res, *, test_type="t-test", order=None):
    _plot_multi(data, stats_res, kind="box", test_type=test_type, order=order)
    return None


# ---------------------------------------------------------------------- #
#                           ordination plots                             #
# ---------------------------------------------------------------------- #
def plot_pca(
        data: pd.DataFrame,
        *,
        save_path: str | Path = "pca_plot.png",
        size: float = 60,
        ellipse: bool = True,
        dpi: int = 600,
        names: bool = False,
        labsize: float = 3,
        show: bool = False
):
    """
    Draw PC1–PC2 score plot and return R²X / Q².

    Returns
    -------
    ggplot object, R²X (float), Q² (float)
    """
    pc_scores, _, r2, q2 = pca(data)

    df = pc_scores.copy()
    df["Group"] = data.iloc[:, 1].astype(str).values
    if names:
        df["Sample"] = data.iloc[:, 0].astype(str).values

    groups = sorted(df["Group"].unique())
    cmap = dict(zip(groups, _pal(len(groups))))

    x_pad = (df.PC1.max() - df.PC1.min()) * 0.05
    y_pad = (df.PC2.max() - df.PC2.min()) * 0.05

    g = (
            ggplot(df, aes("PC1", "PC2", color="Group"))
            + geom_point(size=size / 20)
            + scale_color_manual(values=cmap)
            + scale_x_continuous(expand=(0, x_pad))
            + scale_y_continuous(expand=(0, y_pad))
            + labs(x="PC1", y="PC2")
            + theme_minimal(base_size=11)
            + theme(
        legend_position="right",
        plot_title=element_text(weight="bold", ha="center"),
    )
    )

    if ellipse:
        g += stat_ellipse(level=0.95, type="norm", show_legend=False)
    if names:
        g += geom_text(aes(label="Sample"), size=labsize, adjust_text={})

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True)
        fig = g.draw()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    if show:
        print(g)

    return g, r2, q2


# ---------------------------------------------------------------------- #
#                               heat-map                                 #
# ---------------------------------------------------------------------- #
def plot_heatmap(
        data: pd.DataFrame,
        *,
        scale: bool = True,
        scale_method: str = "auto",
        row_cluster: bool = True,
        col_cluster: bool = True,
        metric: str = "euclidean",
        linkage_method: str = "average",
        row_order=None,
        col_order=None,
        show_dendrogram: bool = True,
        cmap: str = "viridis",
        figsize: tuple = (10, 8),
        dpi: int = 600,
        label_size: int = 8,
        col_angle: int = 90,
        row_angle: int = 0,
        out_path: str | Path = "heatmap.png",
):
    """
    Draw a clustered (or ordered) heat map with plenty of layout options.
    """
    if scale:
        data = scaling(data, scale_method)

    data = data.rename(columns={data.columns[0]: "Sample",
                                data.columns[1]: "Group"})
    mat = data.set_index("Sample").drop(columns=["Group"])

    if not row_cluster and row_order is not None:
        mat = mat.loc[row_order]
    if not col_cluster and col_order is not None:
        mat = mat[col_order]

    # --- dynamic figure size based on label length ----------------------
    col_labels = mat.columns
    row_labels = mat.index

    max_col_len = max(len(str(label)) for label in col_labels)
    max_row_len = max(len(str(label)) for label in row_labels)

    n_cols = len(mat.columns)
    n_rows = len(mat.index)

    width, height = figsize
    if col_angle >= 45:
        width = max(width, figsize[0] * (1 + 0.01 * n_cols + 0.005 * max_col_len))
    else:
        width = max(width, figsize[0] * (1 + 0.02 * n_cols + 0.01 * max_col_len))
    height = max(height, figsize[1] * (1 + 0.01 * n_rows + 0.005 * max_row_len))

    adjusted_figsize = (width, height)

    dendrogram_ratio = (0.15, 0.2) if show_dendrogram else (0, 0)
    cg = sns.clustermap(
        mat,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        metric=metric,
        method=linkage_method,
        cmap=cmap,
        figsize=adjusted_figsize,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=dendrogram_ratio,
        colors_ratio=0.03,
        cbar_kws={"label": "value"},
    )

    # move colorbar a bit to the right
    if hasattr(cg, "cax"):
        pos = cg.cax.get_position()
        cg.cax.set_position([pos.x0 + 0.02, pos.y0, pos.width, pos.height])

    # stretch column dendrogram to match heat-map width
    if show_dendrogram and col_cluster:
        pos_heat = cg.ax_heatmap.get_position()
        pos_dend = cg.ax_col_dendrogram.get_position()
        cg.ax_col_dendrogram.set_position([
            pos_heat.x0, pos_dend.y0, pos_heat.width, pos_dend.height
        ])

    # adjust tick-label font size
    ax = cg.ax_heatmap
    xlabels = ax.get_xticklabels()
    if xlabels:
        size = label_size
        if len(xlabels) > 10:
            size = max(label_size * 0.8, label_size * (10 / len(xlabels)))
        ax.set_xticklabels(
            [t.get_text() for t in xlabels],
            rotation=col_angle, ha="right", fontsize=size
        )

    ylabels = ax.get_yticklabels()
    if ylabels:
        size = label_size
        if len(ylabels) > 15:
            size = max(label_size * 0.8, label_size * (15 / len(ylabels)))
        ax.set_yticklabels(
            [t.get_text() for t in ylabels],
            rotation=row_angle, va="center", fontsize=size
        )

    if not show_dendrogram:
        cg.ax_row_dendrogram.set_visible(False)
        cg.ax_col_dendrogram.set_visible(False)

    plt.tight_layout()
    Path(out_path).parent.mkdir(exist_ok=True)
    cg.figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(cg.figure)
    return None


# ---------------------------------------------------------------------- #
#                               PLS-DA                                   #
# ---------------------------------------------------------------------- #
def plot_plsda(
        data: pd.DataFrame,
        *,
        save_path: str | Path = "pls_da_plot.png",
        size: float = 60,
        ellipse: bool = True,
        dpi: int = 600,
        names: bool = False,
        labsize: float = 3,
        show: bool = False
):
    """
    Draw LV1–LV2 score plot and return R²X / R²Y / Q².

    Returns
    -------
    ggplot object, R²X (float), R²Y (float), Q² (float)
    """
    lv_scores, _, r2x, r2y, q2 = plsda(data)

    df = lv_scores.copy()
    df["Group"] = data.iloc[:, 1].astype(str).values
    if names:
        df["Sample"] = data.iloc[:, 0].astype(str).values

    groups = sorted(df["Group"].unique())
    cmap = dict(zip(groups, _pal(len(groups))))

    x_pad = (df.LV1.max() - df.LV1.min()) * 0.05
    y_pad = (df.LV2.max() - df.LV2.min()) * 0.05

    g = (
            ggplot(df, aes("LV1", "LV2", color="Group"))
            + geom_point(size=size / 20)
            + scale_color_manual(values=cmap)
            + scale_x_continuous(expand=(0, x_pad))
            + scale_y_continuous(expand=(0, y_pad))
            + labs(x="Component 1", y="Component 2")
            + theme_minimal(base_size=11)
            + theme(
        legend_position="right",
        plot_title=element_text(weight="bold", ha="center"),
    )
    )

    if ellipse:
        g += stat_ellipse(level=0.95, type="norm", show_legend=False)
    if names:
        g += geom_text(aes(label="Sample"), size=labsize, adjust_text={})

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True)
        fig = g.draw()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    if show:
        print(g)

    return g, r2x, r2y, q2
