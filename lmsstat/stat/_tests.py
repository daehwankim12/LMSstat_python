import itertools

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.oneway import anova_oneway

from ._utils import ensure_sample_group_columns, _sanitize_pvalues_array

_UTEST_EXACT_MAX_N = 10


def t_test(groups_split, metabolite_names) -> pd.DataFrame:
    """
    Perform independent t-tests for each metabolite between pairs of groups.
    Assumes equal variances between groups (equal_var=True).

    Args:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A grouped DataFrame
            object containing the groups to compare. The DataFrame should
            contain numeric columns corresponding to metabolite measurements.
        metabolite_names (List[str]): A list of column names representing the
            metabolites to perform the t-test on. These columns must exist
            in the DataFrame underlying groups_split.

    Returns:
        pd.DataFrame: A DataFrame where rows correspond to metabolite names
            and columns represent the p-value of the independent two-sample
            t-test for a specific pair of groups (e.g., '(GroupA, GroupB)_ttest').
            The index of the DataFrame is metabolite_names.
    """
    group_names = list(groups_split.groups.keys())
    group_combinations = list(itertools.combinations(group_names, 2))
    n_metabs = len(metabolite_names)

    numeric_data_groups = {
        group: groups_split.get_group(group)
        .loc[:, metabolite_names]
        .to_numpy(dtype=float)
        for group in group_names
    }

    t_test_results = {}

    for group_a, group_b in group_combinations:
        mat_a = numeric_data_groups[group_a]
        mat_b = numeric_data_groups[group_b]
        try:
            _, p_values = ss.ttest_ind(
                mat_a,
                mat_b,
                axis=0,
                equal_var=True,
                nan_policy="omit",
            )
            p_values = np.asarray(p_values, dtype=float)
            if p_values.ndim != 1 or p_values.shape[0] != n_metabs:
                raise ValueError("Unexpected p-value shape from t-test.")
        except Exception:
            # Fallback: per-metabolite computation (set failures to p=1)
            p_values = np.ones(n_metabs, dtype=float)
            for j in range(n_metabs):
                a = mat_a[:, j]
                b = mat_b[:, j]
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if a.size < 1 or b.size < 1:
                    continue
                try:
                    _, pv = ss.ttest_ind(a, b, equal_var=True)
                    pv = float(pv)
                except Exception:
                    pv = 1.0
                p_values[j] = pv if np.isfinite(pv) else 1.0
        p_values = _sanitize_pvalues_array(p_values)

        col_name = f"({group_a}, {group_b})_ttest"
        t_test_results[col_name] = p_values

    df_ttest = pd.DataFrame(t_test_results, index=metabolite_names)

    return df_ttest


def _decide_utest_method(groups_split, metabolite_names) -> str:
    """
    Decide once-and-for-all whether *all* U-tests will be ‘exact’
    or ‘asymptotic’.

    Rule (agreed with domain experts):
        • exact  ‑ if the largest group has <= _UTEST_EXACT_MAX_N samples
                   AND there is absolutely no tie in any metabolite column
        • otherwise asymptotic
    """
    # 1) largest group size
    max_n = groups_split.size().max()

    # 2) any ties?
    df_all = groups_split.obj[metabolite_names].select_dtypes("number")
    has_ties = df_all.apply(lambda col: col.duplicated().any()).any()

    method = "exact" if (max_n <= _UTEST_EXACT_MAX_N and not has_ties) else "asymptotic"

    return method


def u_test(groups_split, metabolite_names) -> pd.DataFrame:
    """
    Perform independent Mann-Whitney U tests for each metabolite between pairs of groups.

    Args:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A grouped DataFrame
            object containing the groups to compare. The DataFrame should
            contain numeric columns corresponding to metabolite measurements.
        metabolite_names (List[str]): A list of column names representing the
            metabolites to perform the U-test on. These columns must exist
            in the DataFrame underlying groups_split.

    Returns:
        pd.DataFrame: A DataFrame where rows correspond to metabolite names
            and columns represent the p-value of the independent Mann-Whitney U
            test for a specific pair of groups (e.g., '(GroupA, GroupB)_utest').
            The index of the DataFrame is metabolite_names.
    """
    group_names = list(groups_split.groups.keys())
    group_combinations = list(itertools.combinations(group_names, 2))
    n_metabs = len(metabolite_names)

    numeric_data_groups = {
        group: groups_split.get_group(group)
        .loc[:, metabolite_names]
        .to_numpy(dtype=float)
        for group in group_names
    }

    u_test_results = {}

    test_method = _decide_utest_method(groups_split, metabolite_names)

    for group_a, group_b in group_combinations:
        mat_a = numeric_data_groups[group_a]
        mat_b = numeric_data_groups[group_b]
        try:
            _, p_values = ss.mannwhitneyu(
                mat_a,
                mat_b,
                use_continuity=True,
                alternative="two-sided",
                axis=0,
                method=test_method,
                nan_policy="omit",
            )
            p_values = np.asarray(p_values, dtype=float)
            if p_values.ndim != 1 or p_values.shape[0] != n_metabs:
                raise ValueError("Unexpected p-value shape from u-test.")
        except Exception:
            p_values = np.ones(n_metabs, dtype=float)
            for j in range(n_metabs):
                a = mat_a[:, j]
                b = mat_b[:, j]
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if a.size < 1 or b.size < 1:
                    continue
                try:
                    _, pv = ss.mannwhitneyu(
                        a,
                        b,
                        use_continuity=True,
                        alternative="two-sided",
                        method=test_method,
                    )
                    pv = float(pv)
                except Exception:
                    pv = 1.0
                p_values[j] = pv if np.isfinite(pv) else 1.0
        p_values = _sanitize_pvalues_array(p_values)

        col_name = f"({group_a}, {group_b})_utest"
        u_test_results[col_name] = p_values

    df_utest = pd.DataFrame(u_test_results, index=metabolite_names)

    return df_utest


def anova_test(groups_split, metabolite_names) -> pd.DataFrame:
    """
    Perform an ANOVA test on groups of data using statsmodels.stats.oneway.anova_oneway.

    Args:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A grouped DataFrame object containing the groups to compare.
        metabolite_names (List[str]): A list of metabolite names.

    Returns:
        anova_results (pandas.core.frame.DataFrame): A DataFrame containing the p-value for each metabolite.
    """
    df = groups_split.obj
    groups = df["Group"].values

    anova_results = np.ones(len(metabolite_names), dtype=float)

    metabolite_data = df[metabolite_names].to_numpy(dtype=float)

    for i, _ in enumerate(metabolite_names):
        x = metabolite_data[:, i]
        mask = np.isfinite(x)
        if mask.sum() < 2:
            continue

        x_valid = x[mask]
        g_valid = groups[mask]

        if np.unique(g_valid).size < 2:
            continue
        if np.nanmin(x_valid) == np.nanmax(x_valid):
            continue

        try:
            anova_result = anova_oneway(x_valid, g_valid, use_var="equal")
            p = float(anova_result.pvalue)
        except Exception:
            p = 1.0

        anova_results[i] = p if np.isfinite(p) else 1.0

    anova_results = _sanitize_pvalues_array(anova_results)

    return pd.DataFrame(
        {"p-value_ANOVA": anova_results}, index=metabolite_names
    )


def kruskal_test(groups_split, metabolite_names) -> pd.DataFrame:
    """
    Perform a Kruskal-Wallis test on groups of data.

    Args:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A grouped DataFrame object containing the groups to compare.
        metabolite_names (List[str]): A list of metabolite names.

    Returns:
        kw_results (pandas.core.frame.DataFrame): A DataFrame containing the p-value for each metabolite.
    """
    df = groups_split.obj
    groups = df["Group"].values

    kw_results = np.ones(len(metabolite_names), dtype=float)

    metabolite_data = df[metabolite_names].to_numpy(dtype=float)

    for i, _ in enumerate(metabolite_names):
        x = metabolite_data[:, i]
        mask = np.isfinite(x)
        if mask.sum() < 2:
            continue

        values = x[mask]
        group_labels = groups[mask]

        if np.unique(group_labels).size < 2:
            continue
        if np.nanmin(values) == np.nanmax(values):
            continue

        unique_groups = np.unique(group_labels)
        group_values = [values[group_labels == g] for g in unique_groups]
        group_values = [gv for gv in group_values if gv.size > 0]
        if len(group_values) < 2:
            continue

        try:
            kw_result = ss.kruskal(*group_values)
            p = float(kw_result.pvalue)
        except Exception:
            p = 1.0

        kw_results[i] = p if np.isfinite(p) else 1.0

    kw_results = _sanitize_pvalues_array(kw_results)

    return pd.DataFrame({"p-value_KW": kw_results}, index=metabolite_names)


def norm_test(data, method='shapiro'):
    method = str(method).lower()
    if method not in {"shapiro", "normaltest"}:
        raise ValueError("Invalid method. Use 'shapiro' or 'normaltest'.")

    data = ensure_sample_group_columns(data)
    numeric = (
        data.drop(columns=["Sample", "Group"])
        .apply(pd.to_numeric, errors="coerce")
    )

    def _run(col: pd.Series):
        x = col.dropna().to_numpy(dtype=float)
        if method == 'shapiro':
            if x.size < 3:
                return np.nan, 1.0
        else:
            if x.size < 8:
                return np.nan, 1.0
        if method == 'shapiro':
            try:
                stat, p = ss.shapiro(x)
            except Exception:
                return np.nan, 1.0
        else:
            try:
                stat, p = ss.normaltest(x)
            except Exception:
                return np.nan, 1.0
        p = float(p)
        return float(stat), p if np.isfinite(p) else 1.0

    result = numeric.apply(_run, axis=0, result_type="expand")
    if method == 'shapiro':
        result.index = ["W-statistic", "p-value"]
    else:
        result.index = ["χ²", "p-value"]

    return result
