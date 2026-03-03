from pathlib import Path
import os
import multiprocessing as mp
import atexit
import hashlib
import itertools as it
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
import tempfile

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
from ..stat._utils import ensure_sample_group_columns

# silence plotnine’s “saving with transparency” warnings, etc.
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------- #
#                                helpers                                 #
# ---------------------------------------------------------------------- #
def _plot_box(
        df: pd.DataFrame,
        st: pd.DataFrame,
        order: list[str],
        *,
        points: str = "all",
        point_size: float = 1.2,
        point_alpha: float = 0.5,
        point_color: str = "black",
        **_,
):
    """Internal: draw a box-and-jitter plot with significance brackets."""
    df["Group"] = pd.Categorical(df["Group"], categories=order, ordered=True)
    y_vals = pd.to_numeric(df["Value"], errors="coerce")
    if y_vals.notna().any():
        y_raw_max = float(np.nanmax(y_vals.to_numpy(dtype=float)))
        y_raw_min = float(np.nanmin(y_vals.to_numpy(dtype=float)))
    else:
        y_raw_max, y_raw_min = 0.0, 0.0

    y_span = y_raw_max - y_raw_min
    if not np.isfinite(y_span) or y_span <= 0:
        y_span = abs(y_raw_max) if np.isfinite(y_raw_max) and y_raw_max != 0 else 1.0

    n_sig = 0
    if st is not None and not st.empty and "p_value" in st.columns:
        pvals = pd.to_numeric(st["p_value"], errors="coerce")
        n_sig = int((pvals <= 0.05).sum())

    y_max_bracket = y_raw_max + y_span * (0.10 + 0.05 * max(n_sig, 1))

    def _point_df(mode: str) -> pd.DataFrame | None:
        mode = (mode or "all").lower()
        if mode == "none":
            return None
        if mode == "all":
            return df
        if mode != "non_outliers":
            raise ValueError("points must be 'all', 'non_outliers', or 'none'.")

        # Per-group 1.5*IQR outlier filtering (for points only; boxplot still uses all data)
        tmp = df.copy()
        tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce")

        def _mask_non_outliers(v: pd.Series) -> pd.Series:
            v = pd.to_numeric(v, errors="coerce")
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                return v.notna()
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            return v.between(lo, hi) & v.notna()

        keep = tmp.groupby("Group", observed=False)["Value"].transform(_mask_non_outliers)
        return tmp.loc[keep]

    p_df = _point_df(points)
    g = (
            ggplot(df, aes("Group", "Value", fill="Group"))
            + geom_boxplot(
        width=.35, colour="black", size=.7,
        outlier_shape="", alpha=.85, show_legend=False
    )
            + scale_fill_manual(values=_pal(len(order)))
            + scale_x_discrete(limits=order)
            + scale_y_continuous(
        limits=(None, y_max_bracket) if np.isfinite(y_max_bracket) else None,
        labels=lambda l: [f"{v:.1e}" for v in l]
    )
            + theme_classic(base_size=13)
            + theme(legend_position="none")
    )
    if p_df is not None and not p_df.empty:
        g = g + geom_point(
            data=p_df,
            mapping=aes("Group", "Value"),
            position=position_jitter(width=.15),
            size=point_size,
            alpha=point_alpha,
            colour=point_color,
            show_legend=False,
        )

    return _annot(g, st, order, y_top=y_raw_max, y_span=y_span)


def _plot_bar(df: pd.DataFrame, st: pd.DataFrame, order: list[str], **_):
    """Internal: draw a mean ± SE bar plot with significance brackets."""
    df["Group"] = pd.Categorical(df["Group"], categories=order, ordered=True)

    summ = (
        df.groupby("Group", observed=False)["Value"]
        .agg(["mean", "std", "count"])
        .reindex(order)
        .reset_index()
    )
    summ["se"] = np.where(
        summ["count"] > 0,
        summ["std"] / np.sqrt(summ["count"]),
        np.nan,
    )

    y_hi = summ["mean"] + summ["se"]
    y_lo = summ["mean"] - summ["se"]
    y_top_raw = float(np.nanmax(y_hi.to_numpy(dtype=float))) if y_hi.notna().any() else 0.0
    y_low_raw = float(np.nanmin(y_lo.to_numpy(dtype=float))) if y_lo.notna().any() else 0.0

    y_span = y_top_raw - y_low_raw
    if not np.isfinite(y_span) or y_span <= 0:
        y_span = abs(y_top_raw) if np.isfinite(y_top_raw) and y_top_raw != 0 else 1.0

    n_sig = 0
    if st is not None and not st.empty and "p_value" in st.columns:
        pvals = pd.to_numeric(st["p_value"], errors="coerce")
        n_sig = int((pvals <= 0.05).sum())

    y_limit = y_top_raw + y_span * (0.10 + 0.05 * max(n_sig, 1))
    y_min_limit = min(0.0, y_low_raw) if np.isfinite(y_low_raw) else 0.0
    if not np.isfinite(y_limit):
        y_limit = 1.0
    if y_min_limit >= y_limit:
        y_limit = y_min_limit + 1.0

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
        limits=(y_min_limit, y_limit),
        expand=(0, 0),
        labels=lambda l: [f"{v:.1e}" for v in l]
    )
            + labs(y="Mean ± SE")
            + theme_classic(base_size=13)
            + theme(legend_position="none")
    )

    return _annot(g, st, order, y_top=y_top_raw, y_span=y_span)


# kind → drawing function, and default folders for auto-save
_KIND = {"box": _plot_box, "bar": _plot_bar}
FOLDER = {k: f"{k}plot" for k in _KIND}


# ---------------------------------------------------------------------- #
#                              core workers                              #
# ---------------------------------------------------------------------- #
_INVALID_CHARS_RE = re.compile(r'[\\/*?:"<>|]')


def _safe_filename(raw: str, tail: str) -> str:
    """
    Sanitize a string so it can safely be used as a file name.
    Keeps at least 10 characters and appends a short hash if truncated.
    """
    name = _INVALID_CHARS_RE.sub("_", str(raw)).replace("\n", "_").strip()
    if not name:
        name = "plot"

    room = 255 - len(tail)
    if len(name) > room:
        prefix = name[: max(10, room - 8)]  # keep ≥10 characters
        suffix = hashlib.blake2b(name.encode(), digest_size=3).hexdigest()
        name = f"{prefix}_{suffix}"
    return f"{name}{tail}"


def _parse_pair(col: str, groups: list[str] | None = None):
    if groups:
        for g1, g2 in it.combinations(groups, 2):
            if col.startswith(f"({g1}, {g2})_"):
                return g1, g2
            if col.startswith(f"({g2}, {g1})_"):
                return g2, g1

    if not col.startswith("("):
        return None

    end = col.find(")_")
    if end < 0:
        return None

    inside = col[1:end]
    if ", " not in inside:
        return None

    g1, g2 = inside.rsplit(", ", 1)
    if not g1 or not g2:
        return None
    return g1, g2

# Shared-memory backed state for worker processes (spawn-safe, cross-platform)
_W_X: np.ndarray | None = None
_W_GROUP_CODES: np.ndarray | None = None
_W_P: np.ndarray | None = None
_W_ORDER: list[str] | None = None
_W_PAIRS: list[tuple[str, str]] | None = None  # (g1, g2) aligned with columns of _W_P
_W_KIND: str | None = None
_W_METAB_NAMES: list[str] | None = None
_W_OUTDIR: str | None = None
_W_SHMS: list[SharedMemory] | None = None
_W_POINTS: str = "all"
_W_POINT_SIZE: float = 1.2
_W_POINT_ALPHA: float = 0.5
_W_POINT_COLOR: str = "black"


def _close_worker_shms():
    global _W_SHMS
    if not _W_SHMS:
        return
    for shm in _W_SHMS:
        try:
            shm.close()
        except Exception:
            pass
    _W_SHMS = None


def _init_worker_shared(config: dict):
    """
    Initializer for spawned worker processes.
    Attaches to shared memory blocks and creates NumPy views.
    """
    global _W_X, _W_GROUP_CODES, _W_P, _W_ORDER, _W_PAIRS, _W_KIND, _W_METAB_NAMES, _W_OUTDIR, _W_SHMS
    global _W_POINTS, _W_POINT_SIZE, _W_POINT_ALPHA, _W_POINT_COLOR

    backend = config.get("backend", "shm")
    if backend == "shm":
        shm_x = SharedMemory(name=config["shm_x_name"])
        shm_codes = SharedMemory(name=config["shm_codes_name"])
        shm_p = SharedMemory(name=config["shm_p_name"])

        _W_SHMS = [shm_x, shm_codes, shm_p]
        atexit.register(_close_worker_shms)

        _W_X = np.ndarray(tuple(config["x_shape"]), dtype=np.float32, buffer=shm_x.buf)
        _W_GROUP_CODES = np.ndarray(tuple(config["codes_shape"]), dtype=np.int32, buffer=shm_codes.buf)
        _W_P = np.ndarray(tuple(config["p_shape"]), dtype=np.float32, buffer=shm_p.buf)
    elif backend == "memmap":
        # File-backed arrays (portable, works even when /dev/shm is unavailable)
        _W_SHMS = None
        _W_X = np.memmap(
            config["x_path"],
            dtype=np.float32,
            mode="r",
            shape=tuple(config["x_shape"]),
            order="C",
        )
        _W_GROUP_CODES = np.memmap(
            config["codes_path"],
            dtype=np.int32,
            mode="r",
            shape=tuple(config["codes_shape"]),
            order="C",
        )
        _W_P = np.memmap(
            config["p_path"],
            dtype=np.float32,
            mode="r",
            shape=tuple(config["p_shape"]),
            order="C",
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    _W_ORDER = list(config["order"])
    _W_PAIRS = list(config["pairs"])
    _W_KIND = str(config["kind"])
    _W_METAB_NAMES = list(config["metab_names"])
    _W_OUTDIR = str(config["outdir"])
    _W_POINTS = str(config.get("points", "all"))
    _W_POINT_SIZE = float(config.get("point_size", 1.2))
    _W_POINT_ALPHA = float(config.get("point_alpha", 0.5))
    _W_POINT_COLOR = str(config.get("point_color", "black"))


def _worker_plot_one(idx: int):
    """Draw and save one plot (by metabolite index) using shared-memory state."""
    if (
        _W_X is None
        or _W_GROUP_CODES is None
        or _W_P is None
        or _W_ORDER is None
        or _W_PAIRS is None
        or _W_KIND is None
        or _W_METAB_NAMES is None
        or _W_OUTDIR is None
    ):
        raise RuntimeError("Worker called without shared-memory state initialized.")

    metab = _W_METAB_NAMES[idx]
    codes = _W_GROUP_CODES
    # Map codes → group labels without building a pandas categorical
    group_labels = np.asarray([_W_ORDER[c] for c in codes], dtype=object)

    values = _W_X[:, idx].astype(float, copy=False)
    df = pd.DataFrame({"Group": group_labels, "Value": values})

    p_row = _W_P[idx, :].astype(float, copy=False)
    st = pd.DataFrame(
        {
            "group1": [g1 for (g1, _) in _W_PAIRS],
            "group2": [g2 for (_, g2) in _W_PAIRS],
            "p_value": p_row,
        }
    )

    if _W_KIND == "box":
        base = _KIND[_W_KIND](
            df,
            st,
            _W_ORDER,
            points=_W_POINTS,
            point_size=_W_POINT_SIZE,
            point_alpha=_W_POINT_ALPHA,
            point_color=_W_POINT_COLOR,
        )
    else:
        base = _KIND[_W_KIND](df, st, _W_ORDER)

    g = base + labs(title=metab, x="Group") + theme(plot_title=element_text(weight="bold"))

    Path(_W_OUTDIR).mkdir(exist_ok=True)
    fname = _safe_filename(metab, f"_{_W_KIND}.png")

    fig = g.draw()
    fig.set_size_inches(6, 6)
    fig.savefig(Path(_W_OUTDIR, fname), dpi=600, pil_kwargs={"compress_level": 3})
    plt.close(fig)

    del g, fig, df, st
    import gc
    gc.collect()


# ---------------------------------------------------------------------- #
#                             public helpers                             #
# ---------------------------------------------------------------------- #
def _plot_multi(
        data: pd.DataFrame,
        stats_res: pd.DataFrame,
        *,
        kind: str = "box",
        test_type: str = "t-test",
        significant_only: bool = False,
        alpha: float = 0.05,
        order: list[str] | None = None,
        max_workers: int | None = None,
        restart_every: int | None = None,
        points: str = "all",
        point_size: float = 1.2,
        point_alpha: float = 0.5,
        point_color: str = "black",
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
    significant_only : bool
        If True, plot only variables with at least one p-value <= `alpha`
        for the selected `test_type`.
    alpha : float
        Significance threshold used when `significant_only=True`.
    order : list[str] | None
        Order of categorical groups on the x-axis.
    max_workers : int | None
        Number of worker processes. Defaults to min(4, cpu_count). Use 1 to disable parallelism.
    restart_every : int | None
        If set to a positive integer, restart the worker pool every N plots to bound memory usage
        from third-party libraries. Default None (no restarts, fastest).
    points : {"all", "non_outliers", "none"}
        For boxplots only: whether to draw individual sample points. "non_outliers" hides points outside 1.5×IQR.
    point_size, point_alpha, point_color : float, float, str
        For boxplots only: styling for the point layer.
    """
    if kind not in _KIND:
        raise ValueError("kind must be 'box' or 'bar'.")

    if test_type.lower() not in ("t-test", "u-test", "scheffe", "dunn"):
        raise ValueError(
            "test_type must be 't-test', 'u-test', 'scheffe', or 'dunn'."
        )
    if not isinstance(significant_only, (bool, np.bool_)):
        raise TypeError("significant_only must be a boolean.")

    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise TypeError("alpha must be a float in [0, 1].") from exc
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    data = ensure_sample_group_columns(data)
    # Keep this low-copy: avoid assign-chaining on a very wide table
    group_series = data["Group"]
    mask = group_series.notna()
    if not bool(mask.all()):
        data = data.loc[mask]
        group_series = data["Group"]
    # Convert group labels to plain str without copying numeric blocks
    group_str = group_series.astype("string").astype(str)

    # match the exact column name pattern in stats_res
    test_key = test_type.lower()
    if test_key == "t-test":
        test_key = "ttest"
    elif test_key == "u-test":
        test_key = "utest"

    if order is None:
        order = list(pd.unique(group_str))
    else:
        order = [str(x) for x in order]
        missing = set(pd.unique(group_str)) - set(order)
        if missing:
            raise ValueError(
                "Invalid group names in 'order': " + ", ".join(sorted(map(str, missing)))
            )

    pcols = [c for c in stats_res.columns if test_key in c.lower()]
    if not pcols:
        raise ValueError(f"No p-value columns found for test_type={test_type!r}.")

    pairs = []
    for c in pcols:
        pair = _parse_pair(c, order)
        if pair is None:
            continue
        g1, g2 = pair
        pairs.append((c, g1, g2))

    if not pairs:
        raise ValueError(
            f"Could not parse group pairs from p-value columns for test_type={test_type!r}."
        )

    metabolites = [c for c in data.columns[2:] if c in stats_res.index]
    if not metabolites:
        return None

    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)
    max_workers = max(1, int(max_workers))

    # Encode groups as categorical codes (compact, fast)
    group_cat = pd.Categorical(group_str, categories=order, ordered=True)
    valid = group_cat.codes >= 0
    if not bool(np.all(valid)):
        data = data.loc[valid]
        group_str = group_str.loc[valid]
        group_cat = pd.Categorical(group_str, categories=order, ordered=True)

    group_codes = group_cat.codes.astype(np.int32, copy=True)

    # Prepare p-value matrix aligned to metabolites × parsed pairs columns
    pair_cols = [c for (c, _, _) in pairs]
    stats_small = (
        stats_res.loc[metabolites, pair_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
    )

    if significant_only:
        sig_mask = (stats_small <= alpha).any(axis=1)
        stats_small = stats_small.loc[sig_mask]
        metabolites = list(stats_small.index)
        if not metabolites:
            return None

    # Prepare numeric matrix for metabolites (float32 to reduce shared memory footprint)
    X = (
        data.loc[:, metabolites]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32, copy=True)
    )
    P = stats_small.to_numpy(dtype=np.float32, copy=True)

    outdir = FOLDER[kind]
    metab_names = list(metabolites)
    pair_labels = [(g1, g2) for (_, g1, g2) in pairs]

    # Always use spawn for OS portability (Windows/macOS compatible).
    ctx = mp.get_context("spawn")

    # Shared backing store: prefer SharedMemory, fall back to file-backed memmap
    backend = "shm"
    tmpdir_cm = None
    shm_x = shm_codes = shm_p = None
    worker_cfg: dict

    try:
        try:
            shm_x = SharedMemory(create=True, size=X.nbytes)
            shm_codes = SharedMemory(create=True, size=group_codes.nbytes)
            shm_p = SharedMemory(create=True, size=P.nbytes)

            np.ndarray(X.shape, dtype=np.float32, buffer=shm_x.buf)[:] = X
            np.ndarray(group_codes.shape, dtype=np.int32, buffer=shm_codes.buf)[:] = group_codes
            np.ndarray(P.shape, dtype=np.float32, buffer=shm_p.buf)[:] = P

            worker_cfg = dict(
                backend="shm",
                shm_x_name=shm_x.name,
                x_shape=(len(group_codes), len(metab_names)),
                shm_codes_name=shm_codes.name,
                codes_shape=(len(group_codes),),
                shm_p_name=shm_p.name,
                p_shape=(len(metab_names), len(pair_labels)),
                order=order,
                pairs=pair_labels,
                kind=kind,
                metab_names=metab_names,
                outdir=outdir,
                points=points,
                point_size=point_size,
                point_alpha=point_alpha,
                point_color=point_color,
            )
        except (PermissionError, OSError):
            backend = "memmap"
            tmpdir_cm = tempfile.TemporaryDirectory(prefix="lmsstat_plot_")
            tmpdir = tmpdir_cm.name

            x_path = os.path.join(tmpdir, "X.float32")
            codes_path = os.path.join(tmpdir, "codes.int32")
            p_path = os.path.join(tmpdir, "P.float32")

            mm_x = np.memmap(x_path, dtype=np.float32, mode="w+", shape=X.shape, order="C")
            mm_x[:] = X
            mm_x.flush()

            mm_codes = np.memmap(codes_path, dtype=np.int32, mode="w+", shape=group_codes.shape, order="C")
            mm_codes[:] = group_codes
            mm_codes.flush()

            mm_p = np.memmap(p_path, dtype=np.float32, mode="w+", shape=P.shape, order="C")
            mm_p[:] = P
            mm_p.flush()

            # close file handles in parent
            del mm_x, mm_codes, mm_p

            worker_cfg = dict(
                backend="memmap",
                x_path=x_path,
                x_shape=(len(group_codes), len(metab_names)),
                codes_path=codes_path,
                codes_shape=(len(group_codes),),
                p_path=p_path,
                p_shape=(len(metab_names), len(pair_labels)),
                order=order,
                pairs=pair_labels,
                kind=kind,
                metab_names=metab_names,
                outdir=outdir,
                points=points,
                point_size=point_size,
                point_alpha=point_alpha,
                point_color=point_color,
            )

        # Free parent copies early (most important for notebooks)
        del X, P, stats_small, data
        import gc
        gc.collect()

        workers = min(max_workers, len(metab_names))
        if workers <= 1:
            _init_worker_shared(worker_cfg)
            for i in range(len(metab_names)):
                _worker_plot_one(i)
            return None

        if restart_every is None or int(restart_every) <= 0:
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=ctx,
                    initializer=_init_worker_shared,
                    initargs=(worker_cfg,),
                ) as ex:
                    list(ex.map(_worker_plot_one, range(len(metab_names)), chunksize=1))
            except (PermissionError, OSError) as e:
                warnings.warn(
                    f"Multiprocessing is not available in this environment ({type(e).__name__}: {e}); "
                    f"falling back to serial plotting.",
                    RuntimeWarning,
                )
                _init_worker_shared(worker_cfg)
                for i in range(len(metab_names)):
                    _worker_plot_one(i)
        else:
            restart_every = max(1, int(restart_every))
            batch = restart_every * workers
            for start in range(0, len(metab_names), batch):
                stop = min(len(metab_names), start + batch)
                idxs = list(range(start, stop))
                try:
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=ctx,
                        initializer=_init_worker_shared,
                        initargs=(worker_cfg,),
                    ) as ex:
                        list(ex.map(_worker_plot_one, idxs, chunksize=1))
                except (PermissionError, OSError) as e:
                    warnings.warn(
                        f"Multiprocessing is not available in this environment ({type(e).__name__}: {e}); "
                        f"falling back to serial plotting.",
                        RuntimeWarning,
                    )
                    _init_worker_shared(worker_cfg)
                    for i in range(start, stop):
                        _worker_plot_one(i)
                    break
    finally:
        # Cleanup backing store
        if backend == "shm":
            for shm in (shm_x, shm_codes, shm_p):
                if shm is None:
                    continue
                try:
                    shm.close()
                except Exception:
                    pass
            for shm in (shm_x, shm_codes, shm_p):
                if shm is None:
                    continue
                try:
                    shm.unlink()
                except Exception:
                    pass
        if tmpdir_cm is not None:
            try:
                tmpdir_cm.cleanup()
            except Exception:
                pass


# high-level wrappers
def plot_bar(
        data,
        stats_res,
        *,
        test_type="t-test",
        significant_only: bool = False,
        alpha: float = 0.05,
        order=None,
        max_workers: int | None = None,
        restart_every: int | None = None,
):
    _plot_multi(
        data,
        stats_res,
        kind="bar",
        test_type=test_type,
        significant_only=significant_only,
        alpha=alpha,
        order=order,
        max_workers=max_workers,
        restart_every=restart_every,
    )
    return None


def plot_box(
        data,
        stats_res,
        *,
        test_type="t-test",
        significant_only: bool = False,
        alpha: float = 0.05,
        order=None,
        max_workers: int | None = None,
        restart_every: int | None = None,
        points: str = "all",
        point_size: float = 1.2,
        point_alpha: float = 0.5,
        point_color: str = "black",
):
    _plot_multi(
        data,
        stats_res,
        kind="box",
        test_type=test_type,
        significant_only=significant_only,
        alpha=alpha,
        order=order,
        max_workers=max_workers,
        restart_every=restart_every,
        points=points,
        point_size=point_size,
        point_alpha=point_alpha,
        point_color=point_color,
    )
    return None


# ---------------------------------------------------------------------- #
#                           ordination plots                             #
# ---------------------------------------------------------------------- #
def plot_pca(
        data: pd.DataFrame,
        *,
        n_components: int = 2,
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
    if n_components < 2:
        raise ValueError("plot_pca requires n_components >= 2 for a PC1–PC2 plot.")

    data = ensure_sample_group_columns(data)
    pc_scores, _, r2, q2 = pca(data, n_components=n_components)

    df = pc_scores.copy()
    df["Group"] = data["Group"].astype(str).values
    if names:
        df["Sample"] = data["Sample"].astype(str).values

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
    data = ensure_sample_group_columns(data)
    if scale:
        data = scaling(data, scale_method)

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
        n_components: int = 2,
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
    ggplot object, R²X (float), R²Y (float), Q² (float), VIP DataFrame
    """
    if n_components < 2:
        raise ValueError(
            "plot_plsda requires n_components >= 2 for an LV1–LV2 plot."
        )

    data = ensure_sample_group_columns(data)
    lv_scores, _, r2x, r2y, q2, vip_df = plsda(data, n_components=n_components)

    df = lv_scores.copy()
    df["Group"] = data["Group"].astype(str).values
    if names:
        df["Sample"] = data["Sample"].astype(str).values

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

    vips = vip_df.sort_values(by="VIP", ascending=False)

    return g, r2x, r2y, q2, vips
