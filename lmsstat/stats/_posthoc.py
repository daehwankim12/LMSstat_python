import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import false_discovery_control
import itertools as it


def preprocess_groups(groups_split):
    max_length = max(len(group) for name, group in groups_split)

    preprocessed_data = {}
    for name, group in groups_split:
        group_data = {
            metabolite: group[metabolite].dropna().values for metabolite in group
        }

        padding_sizes = {
            metabolite: max_length - len(data)
            for metabolite, data in group_data.items()
        }

        preprocessed_data[name] = {
            metabolite: np.pad(
                data, (0, padding_size), "constant", constant_values=np.nan
            )
            for metabolite, data, padding_size in zip(
                group_data.keys(), group_data.values(), padding_sizes.values()
            )
        }

    return preprocessed_data, max_length


def posthoc_scheffe(a: np.ndarray) -> np.ndarray:
    k = a.shape[1]

    group_means = np.nanmean(a, axis=0)
    group_counts = np.sum(~np.isnan(a), axis=0)
    n = np.sum(group_counts)
    group_vars = np.nanvar(a, axis=0, ddof=1)

    sin = np.sum((group_counts - 1) * group_vars) / (n - k)

    a = sin * (1.0 / group_counts[:, None] + 1.0 / group_counts) * (k - 1)

    dif = group_means[:, None] - group_means
    msd = dif**2

    f_val = np.divide(msd, a, out=np.zeros_like(msd), where=a != 0)

    p_values = ss.f.sf(f_val, k - 1, n - k)
    np.fill_diagonal(p_values, 1)

    return p_values


def scheffe_test(groups_split, metabolite_names):
    group_names = sorted(groups_split.groups.keys())
    group_combinations = list(it.combinations(group_names, 2))
    num_combinations = len(group_combinations)

    preprocessed_data, max_length = preprocess_groups(groups_split)

    all_p_values = np.zeros((len(metabolite_names), num_combinations))

    for metabolite_idx, metabolite in enumerate(metabolite_names):
        metabolite_array = np.column_stack(
            [preprocessed_data[name][metabolite] for name in group_names]
        )

        scheffe_results = posthoc_scheffe(metabolite_array)

        for comb_idx, (i, j) in enumerate(group_combinations):
            idx_i = group_names.index(i)
            idx_j = group_names.index(j)
            all_p_values[metabolite_idx, comb_idx] = scheffe_results[idx_i, idx_j]

    p_values_df = pd.DataFrame(
        all_p_values,
        index=metabolite_names,
        columns=[f"({i}, {j})_scheffe" for i, j in group_combinations],
    )

    return p_values_df


def posthoc_dunn(a: np.ndarray) -> np.ndarray:
    k = a.shape[1]
    ranks = ss.rankdata(np.ma.masked_invalid(a).compressed())

    ranks_array = np.full(a.shape, np.nan)
    ranks_array[~np.isnan(a)] = ranks

    group_means = np.nanmean(ranks_array, axis=0)

    _, counts = np.unique(ranks, return_counts=True)
    tie_sum = np.sum(counts[counts > 1] ** 3 - counts[counts > 1])
    c_ties = tie_sum / (12.0 * (np.sum(~np.isnan(a)) - 1)) if tie_sum else 0

    group_counts = np.sum(~np.isnan(a), axis=0)
    z_values = np.zeros((k, k))
    for i, j in it.combinations(range(k), 2):
        diff = np.abs(group_means[i] - group_means[j])
        denom = np.sqrt(
            (np.sum(~np.isnan(a)) * (np.sum(~np.isnan(a)) + 1) - c_ties)
            / 12.0
            * (1 / group_counts[i] + 1 / group_counts[j])
        )
        z_values[i, j] = diff / denom
        z_values[j, i] = z_values[i, j]

    p_values = 2 * ss.norm.sf(np.abs(z_values))
    np.fill_diagonal(p_values, 1)

    return p_values


def dunn_test(groups_split, metabolite_names):
    group_names = sorted(groups_split.groups.keys())
    group_combinations = list(it.combinations(group_names, 2))
    group_combinations = sorted(group_combinations, key=lambda x: x[1])
    num_combinations = len(group_combinations)

    preprocessed_data, max_length = preprocess_groups(groups_split)

    all_p_values = np.zeros((len(metabolite_names), num_combinations))

    for metabolite_idx, metabolite in enumerate(metabolite_names):
        metabolite_array = np.column_stack(
            [preprocessed_data[name][metabolite] for name in group_names]
        )

        dunn_results = posthoc_dunn(metabolite_array)

        for comb_idx, (i, j) in enumerate(group_combinations):
            idx_i = group_names.index(i)
            idx_j = group_names.index(j)
            all_p_values[metabolite_idx, comb_idx] = dunn_results[idx_i, idx_j]

    all_p_values = np.apply_along_axis(false_discovery_control, 1, all_p_values)

    p_values_df = pd.DataFrame(
        all_p_values,
        index=metabolite_names,
        columns=[f"({i}, {j})_dunn" for i, j in group_combinations],
    )

    return p_values_df
