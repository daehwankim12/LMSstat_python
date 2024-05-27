"""
Copyright https://github.com/maximtrp/scikit-posthocs
Part of this file is adapted from scikit_posthocs/_posthocs.py
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import false_discovery_control
import itertools as it


# def preprocess_groups(groups_split):
#     """
#     Preprocesses groups_split by padding the data with NaNs to make all groups the same length.
#
#     Args:
#         groups_split (pandas.core.groupby.DataFrameGroupBy): A pandas DataFrameGroupBy object.
#
#     Returns:
#         tuple: A tuple containing:
#             - preprocessed_data (dict): A dictionary containing preprocessed data for each group.
#             - max_length (int): The maximum length of all groups after padding with NaNs.
#     """
#     max_length = max(len(group) for name, group in groups_split)
#
#     preprocessed_data = {}
#     for name, group in groups_split:
#         group_data = {
#             metabolite: group[metabolite].dropna().values for metabolite in group
#         }
#
#         padding_sizes = {
#             metabolite: max_length - len(data)
#             for metabolite, data in group_data.items()
#         }
#
#         preprocessed_data[name] = {
#             metabolite: np.pad(
#                 data, (0, padding_size), "constant", constant_values=np.nan
#             )
#             for metabolite, data, padding_size in zip(
#                 group_data.keys(), group_data.values(), padding_sizes.values()
#             )
#         }
#
#     return preprocessed_data, max_length


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
    # 최대 길이 계산과 데이터 전처리를 한 번의 순회로 처리합니다.
    preprocessed_data = {}
    max_length = 0
    for name, group in groups_split:
        group_data = {metabolite: group[metabolite].dropna().to_numpy() for metabolite in group}
        current_max = max(len(data) for data in group_data.values())
        max_length = max(max_length, current_max)

        preprocessed_data[name] = group_data

    # 데이터 패딩
    for name in preprocessed_data:
        preprocessed_data[name] = {
            metabolite: np.pad(data, (0, max_length - len(data)), mode='constant', constant_values=np.nan)
            for metabolite, data in preprocessed_data[name].items()
        }

    return preprocessed_data, max_length





def posthoc_scheffe(a: np.ndarray) -> np.ndarray:
    """
    Calculate the p-values using the posthoc Scheffe method.

    Parameters:
        a (np.ndarray): An array of shape (n, k) containing the data for analysis.

    Returns:
        p_values (np.ndarray): An array of shape (k, k) containing the p-values.
    """
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
    """
    Calculate the Scheffe's test p-values for each combination of groups and metabolites.

    Parameters:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A pandas GroupBy object containing the groups split by a certain variable.
        metabolite_names (list[str]): A list of metabolite names.

    Returns:
        p_values_df (pandas.core.frame.DataFrame): A pandas DataFrame containing the Scheffe's test p-values for each combination of groups and metabolites.
    """
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

# def scheffe_test(groups_split, metabolite_names):
#     """
#     Calculate the Scheffe's test p-values for each combination of groups and metabolites.
#
#     Parameters:
#     groups_split (pandas.core.groupby.DataFrameGroupBy): A pandas GroupBy object containing the groups split by a certain variable.
#     metabolite_names (list[str]): A list of metabolite names.
#
#     Returns:
#     p_values_df (pandas.core.frame.DataFrame): A pandas DataFrame containing the Scheffe's test p-values for each combination of groups and metabolites.
#     """
#     group_names = sorted(groups_split.groups.keys())
#     group_combinations = list(it.combinations(group_names, 2))
#     num_combinations = len(group_combinations)
#
#     metabolite_data = []
#     for name, group_data in groups_split:
#         metabolite_data.append(group_data[metabolite_names].values)
#
#     metabolite_array = np.array(metabolite_data)
#     metabolite_array = np.transpose(metabolite_array, (1, 0, 2))
#
#     scheffe_results = np.apply_along_axis(posthoc_scheffe, 1, metabolite_array)
#
#     p_values = np.zeros((len(metabolite_names), num_combinations))
#     for comb_idx, (i, j) in enumerate(group_combinations):
#         idx_i = group_names.index(i)
#         idx_j = group_names.index(j)
#         p_values[:, comb_idx] = scheffe_results[:, idx_i, idx_j]
#
#     p_values_df = pd.DataFrame(
#         p_values,
#         index=metabolite_names,
#         columns=[f"({i}, {j})_scheffe" for i, j in group_combinations],
#     )
#
#     return p_values_df


# def scheffe_test(groups_split, metabolite_names):
#     """
#     Calculate the Scheffe's test p-values for each combination of groups and metabolites.
#
#     Parameters:
#     groups_split (pandas.core.groupby.DataFrameGroupBy): A pandas GroupBy object containing the groups split by a certain variable.
#     metabolite_names (list[str]): A list of metabolite names.
#
#     Returns:
#     p_values_df (pandas.core.frame.DataFrame): A pandas DataFrame containing the Scheffe's test p-values for each combination of groups and metabolites.
#     """
#     group_names = sorted(groups_split.groups.keys())
#     group_combinations = list(it.combinations(group_names, 2))
#     num_combinations = len(group_combinations)
#
#     metabolite_data = []
#     for name, group_data in groups_split:
#         metabolite_data.append(group_data[metabolite_names].values)
#
#     metabolite_array = np.array(metabolite_data)
#     metabolite_array = np.transpose(metabolite_array, (1, 0, 2))
#
#     scheffe_results = np.apply_along_axis(posthoc_scheffe, 1, metabolite_array)
#
#     idx_combinations = np.array([(group_names.index(i), group_names.index(j)) for i, j in group_combinations])
#     p_values = scheffe_results[:, idx_combinations[:, 0], idx_combinations[:, 1]].T
#
#     p_values_df = pd.DataFrame(
#         p_values,
#         index=metabolite_names,
#         columns=[f"({i}, {j})_scheffe" for i, j in group_combinations],
#     )
#
#     return p_values_df


def posthoc_dunn(a: np.ndarray) -> np.ndarray:
    """
    Calculate the p-values using the posthoc Dunn's method.

    Parameters:
        a (np.ndarray): An array of shape (n, k) containing the data for analysis.

    Returns:
        p_values (np.ndarray): An array of shape (k, k) containing the p-values.
    """
    k = a.shape[1]
    ranks = ss.rankdata(np.ma.masked_invalid(a).compressed())

    ranks_array = np.full(a.shape, np.nan)
    ranks_array[~np.isnan(a)] = ranks

    group_means = np.nanmean(ranks_array, axis=0)

    _, counts = np.unique(ranks, return_counts=True)
    tie_sum = np.sum(counts[counts > 1] ** 3 - counts[counts > 1])
    c_ties = tie_sum / (12.0 * (np.sum(~np.isnan(a)) - 1)) if tie_sum else 0

    group_counts = np.sum(~np.isnan(a), axis=0)
    denom = np.sqrt((np.sum(~np.isnan(a)) * (np.sum(~np.isnan(a)) + 1) - c_ties) / 12.0 * (1 / group_counts[:, None] + 1 / group_counts))
    z_values = np.abs(group_means[:, None] - group_means) / denom
    np.fill_diagonal(z_values, 0)

    p_values = 2 * ss.norm.sf(z_values)
    np.fill_diagonal(p_values, 1)

    return p_values


def dunn_test(groups_split, metabolite_names):
    """
    Calculate the Dunn's test p-values for each combination of groups and metabolites.

    Parameters:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A pandas GroupBy object containing the groups split by a certain variable.
        metabolite_names (list[str]): A list of metabolite names.

    Returns:
        p_values_df (pandas.core.frame.DataFrame): A pandas DataFrame containing the Dunn's test p-values for each combination of groups and metabolites.
    """
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


def posthoc_gameshowell(a: np.ndarray) -> np.ndarray:
    """
    Calculate the p-values using the posthoc Games-Howell method.

    Parameters:
        a (np.ndarray): An array of shape (n, k) containing the data for analysis.

    Returns:
        p_values (np.ndarray): An array of shape (k, k) containing the p-values.
    """
    k = a.shape[1]

    group_means = np.nanmean(a, axis=0)
    group_vars = np.nanvar(a, axis=0, ddof=1)
    group_counts = np.sum(~np.isnan(a), axis=0)

    mean_diffs = group_means[:, np.newaxis] - group_means
    var_diffs = group_vars[:, np.newaxis] / group_counts[:, np.newaxis] + group_vars / group_counts
    denom = np.sqrt(var_diffs)

    q_values = mean_diffs / denom
    np.fill_diagonal(q_values, 0)

    # Calculate Welch's degrees of freedom
    df = var_diffs ** 2 / ((group_vars[:, np.newaxis] / group_counts[:, np.newaxis]) ** 2 / (group_counts[:, np.newaxis] - 1) +
                           (group_vars / group_counts) ** 2 / (group_counts - 1))

    p_values = 2 * ss.t.sf(np.abs(q_values), df)
    np.fill_diagonal(p_values, 1)

    return p_values


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

    preprocessed_data, max_length = preprocess_groups(groups_split)

    all_p_values = np.zeros((len(metabolite_names), num_combinations))

    for metabolite_idx, metabolite in enumerate(metabolite_names):
        metabolite_array = np.column_stack(
            [preprocessed_data[name][metabolite] for name in group_names]
        )

        games_howell_results = posthoc_gameshowell(metabolite_array)

        for comb_idx, (i, j) in enumerate(group_combinations):
            idx_i = group_names.index(i)
            idx_j = group_names.index(j)
            all_p_values[metabolite_idx, comb_idx] = games_howell_results[idx_i, idx_j]

    all_p_values = np.apply_along_axis(false_discovery_control, 1, all_p_values)

    p_values_df = pd.DataFrame(
        all_p_values,
        index=metabolite_names,
        columns=[f"({i}, {j})_games_howell" for i, j in group_combinations],
    )

    return p_values_df


# def games_howell_test(groups_split, metabolite_names):
#     """
#     Calculate the Games-Howell test p-values for each combination of groups and metabolites.
#
#     Parameters:
#         groups_split (pandas.core.groupby.DataFrameGroupBy): A pandas GroupBy object containing the groups split by a certain variable.
#         metabolite_names (list[str]): A list of metabolite names.
#
#     Returns:
#         p_values_df (pandas.core.frame.DataFrame): A pandas DataFrame containing the Games-Howell test p-values for each combination of groups and metabolites.
#     """
#     group_names = sorted(groups_split.groups.keys())
#     group_combinations = list(it.combinations(group_names, 2))
#     num_combinations = len(group_combinations)
#
#     preprocessed_data, max_length = preprocess_groups(groups_split)
#
#     all_p_values = np.zeros((len(metabolite_names), num_combinations))
#
#     for metabolite_idx, metabolite in enumerate(metabolite_names):
#         metabolite_array = np.column_stack(
#             [preprocessed_data[name][metabolite] for name in group_names]
#         )
#
#         games_howell_results = posthoc_gameshowell(metabolite_array)
#
#         # 1차원 인덱스 배열 생성
#         comb_indices = np.array([group_names.index(i) for i, _ in group_combinations])
#         comb_indices_j = np.array([group_names.index(j) for _, j in group_combinations])
#
#         # Broadcasting 활용
#         all_p_values[metabolite_idx, :] = games_howell_results[comb_indices, comb_indices_j]
#
#
#     all_p_values = np.apply_along_axis(false_discovery_control, 1, all_p_values)
#
#     p_values_df = pd.DataFrame(
#         all_p_values,
#         index=metabolite_names,
#         columns=[f"({i}, {j})_games_howell" for i, j in group_combinations],
#     )
#
#     return p_values_df
