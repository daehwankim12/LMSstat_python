"""
Effect-size / fold-change table for two-group comparisons.

`effect_size_table` owns the statistics (means, fold change, Cohen's d, and the
chosen test's p-value plus a BH-adjusted p-value). `plot.plot_volcano` consumes
the returned table; it does not compute anything itself.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from ._allstat import allstats
from ._utils import _sanitize_pvalues_array, preprocess_data

_P_KEY = {"t-test": "ttest", "u-test": "utest"}

_COLUMNS = [
    "feature", "group1", "group2", "mean1", "mean2",
    "fold_change", "log2fc", "cohens_d", "p_value", "p_adj",
]


def _find_p_column(stats_res, group1, group2, key):
    """Locate the (g1, g2)/(g2, g1) p-value column; two-sided p is directionless."""
    for a, b in ((group1, group2), (group2, group1)):
        col = f"({a}, {b})_{key}"
        if col in stats_res.columns:
            return col
    raise ValueError(
        f"No {key} p-value column for groups {group1!r}/{group2!r} in stats_res."
    )


def effect_size_table(data, stats_res=None, *, group_order=None, p_source="t-test"):
    """
    Build a per-feature effect-size table for a two-group comparison.

    Parameters
    ----------
    data : DataFrame
        Wide table. Col0 = Sample, Col1 = Group, remaining = features. Must
        contain exactly two groups.
    stats_res : DataFrame, optional
        Unadjusted ``allstats(data, p_adj=False)`` output. If None, it is
        computed internally. The raw p-value is read from the selected test's
        column; ``p_adj`` is always BH-adjusted within this table.
    group_order : sequence of two str, optional
        The two observed group labels in the desired (group1, group2) order.
        Controls direction only: fold_change = mean2 / mean1 and the sign of
        log2fc / cohens_d. Defaults to the sorted group order.
    p_source : {"t-test", "u-test"}
        Which two-group test's p-value to report. Defaults to "t-test".

    Returns
    -------
    DataFrame with columns:
        feature, group1, group2, mean1, mean2, fold_change, log2fc,
        cohens_d, p_value, p_adj

    Notes
    -----
    fold_change / log2fc are defined only when both group means are strictly
    positive (raw, non-negative intensity data); otherwise both are NaN — no
    pseudocount is invented. Cohen's d uses the pooled standard deviation.
    """
    if p_source not in _P_KEY:
        raise ValueError("p_source must be 't-test' or 'u-test'.")
    key = _P_KEY[p_source]

    _, groups_split, metabolite_names = preprocess_data(data)
    group_names = list(groups_split.groups.keys())
    if len(group_names) != 2:
        raise ValueError(
            f"effect_size_table requires exactly two groups, got {len(group_names)}."
        )

    if group_order is None:
        group1, group2 = group_names[0], group_names[1]
    else:
        group_order = [str(g) for g in group_order]
        if len(group_order) != 2 or set(group_order) != set(group_names):
            raise ValueError(
                "group_order must contain exactly the two observed groups: "
                f"{group_names}."
            )
        group1, group2 = group_order[0], group_order[1]

    mat1 = groups_split.get_group(group1)[metabolite_names].to_numpy(dtype=float)
    mat2 = groups_split.get_group(group2)[metabolite_names].to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean1 = np.nanmean(mat1, axis=0)
        mean2 = np.nanmean(mat2, axis=0)
        var1 = np.nanvar(mat1, axis=0, ddof=1)
        var2 = np.nanvar(mat2, axis=0, ddof=1)
    n1 = np.sum(~np.isnan(mat1), axis=0)
    n2 = np.sum(~np.isnan(mat2), axis=0)

    # Fold change / log2FC: only when both group means are strictly positive.
    both_pos = (mean1 > 0) & (mean2 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mean2 / mean1
        fold_change = np.where(both_pos, ratio, np.nan)
        log2fc = np.where(both_pos, np.log2(np.where(both_pos, ratio, 1.0)), np.nan)

    # Cohen's d with pooled standard deviation.
    df_pool = n1 + n2 - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / df_pool)
    cohens_d = np.full(len(metabolite_names), np.nan, dtype=float)
    usable = (df_pool > 0) & np.isfinite(s_pooled) & (s_pooled > 0)
    cohens_d[usable] = (mean2[usable] - mean1[usable]) / s_pooled[usable]
    # No spread but equal means → zero effect (e.g. a constant feature).
    flat_equal = (df_pool > 0) & ~usable & np.isclose(mean1, mean2)
    cohens_d[flat_equal] = 0.0

    if stats_res is None:
        stats_res = allstats(data, p_adj=False)
    p_col = _find_p_column(stats_res, group1, group2, key)
    p_value = _sanitize_pvalues_array(
        stats_res.reindex(metabolite_names)[p_col].to_numpy(dtype=float)
    )
    p_adj = _sanitize_pvalues_array(false_discovery_control(p_value))

    out = pd.DataFrame(
        {
            "feature": metabolite_names,
            "group1": group1,
            "group2": group2,
            "mean1": mean1,
            "mean2": mean2,
            "fold_change": fold_change,
            "log2fc": log2fc,
            "cohens_d": cohens_d,
            "p_value": p_value,
            "p_adj": p_adj,
        }
    )
    return out[_COLUMNS]
