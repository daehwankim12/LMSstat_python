"""
Copyright https://github.com/maximtrp/scikit-posthocs
Part of this file is adapted from scikit_posthocs/_posthocs.py
"""

import itertools as it

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import false_discovery_control

from ._utils import _sanitize_pvalues_array, _sanitize_pvalues_df


def preprocess_groups(groups_split):
    """
    Preprocesses groups_split by padding the data with NaNs to make all groups the same length.
    Optimized version.

    Args:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A pandas DataFrameGroupBy object.

    Returns:
        tuple: A tuple containing:
            - preprocessed_data (dict): A dictionary containing preprocessed data for each group.
            - max_length (int): The maximum length of all groups after padding with NaNs.
    """
    # Calculate the maximum length and preprocess the data in a single pass
    preprocessed_data = {}
    max_length = 0
    for name, group in groups_split:
        group_data = {
            metabolite: group[metabolite].dropna().to_numpy() for metabolite in group
        }
        current_max = max(len(data) for data in group_data.values())
        max_length = max(max_length, current_max)

        preprocessed_data[name] = group_data

    # Pad the data with NaNs
    for name in preprocessed_data:
        preprocessed_data[name] = {
            metabolite: np.concatenate([
                data,
                np.full(max_length - len(data), np.nan)
            ])
            for metabolite, data in preprocessed_data[name].items()
        }

    return preprocessed_data, max_length


def posthoc_scheffe(x: np.ndarray) -> np.ndarray:
    """
    Calculate Scheffé post-hoc test p-values for multiple groups and features.

    Parameters
    ----------
    x : ndarray, shape (n_max, k, m)
        3D array with samples, groups, and features
        n_max: maximum number of samples (NaN-padded)
        k: number of groups
        m: number of features/metabolites

    Returns
    -------
    pvals : ndarray, shape (k, k, m)
        P-values for all pairwise group comparisons across all features
    """
    k = x.shape[1]

    means = np.nanmean(x, axis=0)
    counts = np.sum(~np.isnan(x), axis=0)
    N = counts.sum(axis=0)

    vars_ = np.nanvar(x, axis=0, ddof=1)
    vars_ = np.nan_to_num(vars_, nan=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        mse = np.sum((counts - 1) * vars_, axis=0) / (N - k)

        invn = 1.0 / counts
        denom = mse * (k - 1) * (invn[:, None, :] + invn[None, :, :])

        diff2 = (means[:, None, :] - means[None, :, :]) ** 2
        F = diff2 / denom

        # Degenerate case: denom==0 and diff==0 should yield F=0 (p=1), not inf (p=0).
        F = np.where((denom == 0) & (diff2 == 0), 0.0, F)
        F = np.where((denom == 0) & (diff2 != 0), np.inf, F)

    pvals = ss.f.sf(F, k - 1, N - k)
    pvals = _sanitize_pvalues_array(pvals)

    for i in range(k):
        pvals[i, i, :] = 1.0

    return pvals


def scheffe_test(groups_split, metabolite_names):
    """
    Perform vectorized Scheffé post-hoc tests for all group pairs and metabolites.

    Parameters
    ----------
    groups_split : pandas.core.groupby.DataFrameGroupBy
        Grouped DataFrame object containing the groups to compare
    metabolite_names : list
        List of metabolite/feature names to analyze

    Returns
    -------
    pandas.DataFrame
        DataFrame with p-values for all group pairs (columns) and metabolites (rows)
    """
    group_names = sorted(groups_split.groups)
    k = len(group_names)
    pair_indices = np.array(list(it.combinations(range(k), 2)))
    column_labels = [f"({group_names[i]}, {group_names[j]})_scheffe"
                     for i, j in pair_indices]

    group_matrices = [groups_split.get_group(g)
                      .loc[:, metabolite_names]
                      .to_numpy(float)
                      for g in group_names]

    n_max = max(mat.shape[0] for mat in group_matrices)
    m = len(metabolite_names)

    data_cube = np.full((n_max, k, m), np.nan, dtype=float)
    for gi, mat in enumerate(group_matrices):
        data_cube[:mat.shape[0], gi, :] = mat

    p_values_cube = posthoc_scheffe(data_cube)

    rows, cols = pair_indices.T
    result = p_values_cube[rows, cols, :].T
    result = _sanitize_pvalues_array(result)

    return _sanitize_pvalues_df(
        pd.DataFrame(result, index=metabolite_names, columns=column_labels)
    )


def posthoc_dunn(x: np.ndarray) -> np.ndarray:
    """
    Dunn pairwise p-values for all groups and metabolites in `x`.
    x.shape == (n_max, k, m)
    returns    (k, k, m)
    """
    n_max, k, m = x.shape
    flat = np.moveaxis(x, 2, 0).reshape(m, -1)
    ranks = ss.rankdata(flat, axis=1, nan_policy='omit')

    ranks3d = ranks.reshape(m, n_max, k).transpose(1, 2, 0)

    means = np.nanmean(ranks3d, axis=0)
    counts = np.sum(~np.isnan(ranks3d), axis=0)
    Ntot = counts.sum(axis=0)

    pvals = np.ones((k, k, m), dtype=float)

    for idx in range(m):
        if Ntot[idx] < 2:
            continue
        cnts = counts[:, idx]
        if np.any(cnts == 0) or cnts.size < 2:
            continue

        uniq, reps = np.unique(ranks[idx, ~np.isnan(ranks[idx])],
                               return_counts=True)
        c_ties = np.sum(reps ** 3 - reps) / (12 * (Ntot[idx] - 1)) if reps.size else 0.

        invn = 1. / cnts
        denom = np.sqrt((Ntot[idx] * (Ntot[idx] + 1) - c_ties) / 12.
                        * (invn[:, None] + invn))

        z = np.abs(means[:, idx, None] - means[:, idx]) / denom
        np.fill_diagonal(z, 0.)
        pvals[:, :, idx] = 2. * ss.norm.sf(z)

    return _sanitize_pvalues_array(pvals)


def dunn_test(groups_split, metabolite_names):
    group_names = sorted(groups_split.groups)
    k = len(group_names)
    pair_idx = np.array(list(it.combinations(range(k), 2)))
    col_labels = [f"({group_names[i]}, {group_names[j]})_dunn"
                  for i, j in pair_idx]

    mats = [groups_split.get_group(g)
            .loc[:, metabolite_names]
            .to_numpy(float)
            for g in group_names]

    n_max = max(mat.shape[0] for mat in mats)
    m = len(metabolite_names)

    cube = np.full((n_max, k, m), np.nan, dtype=float)
    for gi, mat in enumerate(mats):
        cube[:mat.shape[0], gi, :] = mat

    p_cube = posthoc_dunn(cube)
    r, c = pair_idx.T
    pvals = p_cube[r, c, :].T
    pvals = _sanitize_pvalues_array(pvals)
    pvals = np.apply_along_axis(false_discovery_control, 1, pvals)
    pvals = _sanitize_pvalues_array(pvals)

    return _sanitize_pvalues_df(
        pd.DataFrame(pvals, index=metabolite_names, columns=col_labels)
    )


def posthoc_gameshowell(a: np.ndarray) -> np.ndarray:
    """
    Calculate the p-values using the posthoc Games-Howell method.

    Parameters:
        a (np.ndarray): An array of shape (n, k) containing the data for analysis.

    Returns:
        p_values (np.ndarray): An array of shape (k, k) containing the p-values.
    """
    k = a.shape[1]
    if k < 2:
        return np.ones((k, k), dtype=float)

    group_means = np.nanmean(a, axis=0)
    group_vars = np.nanvar(a, axis=0, ddof=1)
    group_counts = np.sum(~np.isnan(a), axis=0).astype(float)

    mean_diffs = np.abs(group_means[:, np.newaxis] - group_means)
    var_over_n = group_vars / group_counts
    var_diffs = var_over_n[:, np.newaxis] + var_over_n

    with np.errstate(divide="ignore", invalid="ignore"):
        q_values = np.sqrt(2.0) * mean_diffs / np.sqrt(var_diffs)
        df = var_diffs ** 2 / (
            (var_over_n[:, np.newaxis] ** 2) / (group_counts[:, np.newaxis] - 1)
            + (var_over_n[np.newaxis, :] ** 2) / (group_counts[np.newaxis, :] - 1)
        )

    q_values = np.where((var_diffs == 0) & (mean_diffs == 0), 0.0, q_values)
    q_values = np.where((var_diffs == 0) & (mean_diffs != 0), np.inf, q_values)

    p_values = np.ones_like(q_values, dtype=float)
    valid_df = np.isfinite(df) & (df > 0)
    if np.any(valid_df):
        p_values[valid_df] = ss.studentized_range.sf(q_values[valid_df], k, df[valid_df])
    p_values = np.where((var_diffs == 0) & (mean_diffs != 0), 0.0, p_values)
    np.fill_diagonal(p_values, 1.0)

    return _sanitize_pvalues_array(p_values)


def games_howell_test(groups_split, metabolite_names):
    """
    Calculate the Games-Howell test p-values for each combination of groups and metabolites.

    Parameters:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A pandas GroupBy object containing the groups split by a certain variable.
        metabolite_names (list[str]): A list of metabolite names.

    Returns:
        p_values_df (pandas.core.frame.DataFrame): A pandas DataFrame containing the Games-Howell test p-values for each combination of groups and metabolites.
    """
    group_names = sorted(groups_split.groups.keys())
    group_combinations = list(it.combinations(group_names, 2))
    num_combinations = len(group_combinations)

    preprocessed_data, _ = preprocess_groups(groups_split)

    all_p_values = np.ones((len(metabolite_names), num_combinations), dtype=float)
    idx_map = {g: i for i, g in enumerate(group_names)}

    for metabolite_idx, metabolite in enumerate(metabolite_names):
        metabolite_array = np.column_stack(
            [preprocessed_data[name][metabolite] for name in group_names]
        )
        try:
            games_howell_results = posthoc_gameshowell(metabolite_array)
        except Exception:
            continue

        for comb_idx, (i, j) in enumerate(group_combinations):
            idx_i = idx_map[i]
            idx_j = idx_map[j]
            pv = float(games_howell_results[idx_i, idx_j])
            all_p_values[metabolite_idx, comb_idx] = pv if np.isfinite(pv) else 1.0

    all_p_values = _sanitize_pvalues_array(all_p_values)

    p_values_df = pd.DataFrame(
        all_p_values,
        index=metabolite_names,
        columns=[f"({i}, {j})_games_howell" for i, j in group_combinations],
    )

    return _sanitize_pvalues_df(p_values_df)
