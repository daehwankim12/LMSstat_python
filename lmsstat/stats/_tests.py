import itertools
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.oneway import anova_oneway


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

    numeric_data_groups = {
        group: groups_split.get_group(group)
               .loc[:, metabolite_names]
        .select_dtypes("number")
        .to_numpy(dtype=float)
        for group in group_names
    }

    t_test_results = {}

    for group_a, group_b in group_combinations:
        mat_a = numeric_data_groups[group_a]
        mat_b = numeric_data_groups[group_b]
        _, p_values = ss.ttest_ind(mat_a, mat_b,
                                   axis=0,
                                   equal_var=True,
                                   nan_policy="omit")

        col_name = f"({group_a}, {group_b})_ttest"
        t_test_results[col_name] = p_values

    df_ttest = pd.DataFrame(t_test_results, index=metabolite_names)

    return df_ttest


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

    numeric_data_groups = {
        group: groups_split.get_group(group)
               .loc[:, metabolite_names]
        .select_dtypes("number")
        .to_numpy(dtype=float)
        for group in group_names
    }

    u_test_results = {}

    for group_a, group_b in group_combinations:
        mat_a = numeric_data_groups[group_a]
        mat_b = numeric_data_groups[group_b]
        _, p_values = ss.mannwhitneyu(mat_a, mat_b,
                                      use_continuity=True,
                                      alternative='two-sided',
                                      axis=0,
                                      nan_policy='omit')

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

    anova_results = np.zeros(len(metabolite_names))

    metabolite_data = df[metabolite_names].values

    for i, _ in enumerate(metabolite_names):
        mask = ~np.isnan(metabolite_data[:, i])

        anova_result = anova_oneway(
            metabolite_data[mask, i],
            groups[mask],
            use_var="equal"
        )
        anova_results[i] = anova_result.pvalue

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

    kw_results = np.zeros(len(metabolite_names))

    metabolite_data = df[metabolite_names].values

    for i, _ in enumerate(metabolite_names):
        mask = ~np.isnan(metabolite_data[:, i])

        values = metabolite_data[mask, i]
        group_labels = groups[mask]

        unique_groups = np.unique(group_labels)
        group_values = [values[group_labels == g] for g in unique_groups]

        kw_result = ss.kruskal(*group_values)
        kw_results[i] = kw_result.pvalue

    return pd.DataFrame({"p-value_KW": kw_results}, index=metabolite_names)


def norm_test(data, method='shapiro'):
    data = data.rename(columns={data.columns[0]: "Sample", data.columns[1]: "Group"})
    data.drop(columns=["Sample", "Group"], inplace=True)

    if method == 'shapiro':
        result = data.apply(func=ss.shapiro, axis=0)
        result.index = ["W-statistic", "p-value"]
    else:
        result = data.apply(func=ss.normaltest, axis=0)
        result.index = ["χ²", "p-value"]

    return result
