import itertools

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.oneway import anova_oneway


def t_test(groups_split, metabolite_names) -> pd.DataFrame:
    """
    Perform t-test on groups to compare their means.

    Args:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A grouped DataFrame object containing the groups to compare.
        metabolite_names (List[str]): A list of metabolite names.

    Returns:
        df_ttest (pandas.core.frame.DataFrame): A DataFrame containing the t-test results for each combination of group pairs.
    """
    group_names = list(groups_split.groups.keys())
    group_combinations = list(itertools.combinations(group_names, 2))

    numeric_data_groups = {
        group: groups_split.get_group(group)
        .select_dtypes(include=[float, "float64", "int", "int64"])
        .to_numpy()
        for group in group_names
    }

    column_names = [f"({combo[0]}, {combo[1]})_ttest" for combo in group_combinations]
    df_ttest = pd.DataFrame(index=metabolite_names, columns=column_names)

    # Instead of using ProcessPoolExecutor and futures,
    # perform the t-test directly inside the loop
    for combo in group_combinations:
        col_name = f"({combo[0]}, {combo[1]})_ttest"
        # Call the t-test directly without using a separate worker function
        _, p_value = ss.ttest_ind(
            numeric_data_groups[combo[0]],
            numeric_data_groups[combo[1]],
            equal_var=False,
        )
        # Assign the p_value to all rows in the column, assuming that's the intent
        df_ttest[col_name] = p_value

    return df_ttest


def u_test(groups_split, metabolite_names) -> pd.DataFrame:
    """
    Perform u-test on groups to compare their means.

    Args:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A grouped DataFrame object containing the groups to compare.
        metabolite_names (List[str]): A list of metabolite names.

    Returns:
        df_utest (pandas.core.frame.DataFrame): A DataFrame containing the u-test results for each combination of group pairs.
    """
    group_names = list(groups_split.groups.keys())
    group_combinations = list(itertools.combinations(group_names, 2))

    numeric_data_groups = {
        group: groups_split.get_group(group)
        .select_dtypes(include=[float, "float64", "int", "int64"])
        .to_numpy()
        for group in group_names
    }

    column_names = [f"({combo[0]}, {combo[1]})_utest" for combo in group_combinations]
    df_utest = pd.DataFrame(index=metabolite_names, columns=column_names)

    # Perform the U-test directly inside the loop
    for combo in group_combinations:
        col_name = f"({combo[0]}, {combo[1]})_utest"
        # Call the Mann-Whitney U test directly without using a separate worker function
        _, p_value = ss.mannwhitneyu(
            numeric_data_groups[combo[0]],
            numeric_data_groups[combo[1]],
            alternative="two-sided",
            use_continuity=True,
        )
        # Assign the p_value to all rows in the column, assuming that's the intent
        df_utest[col_name] = p_value

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
    group_data = {
        name: group.dropna(subset=metabolite_names).drop(columns=["Sample"])
        for name, group in groups_split
    }
    merged_data = pd.concat(group_data.values(), ignore_index=True)
    group = merged_data.pop("Group").values

    metabolite_data = merged_data[metabolite_names].values
    anova_results = np.zeros(len(metabolite_names))

    for i, _metabolite in enumerate(metabolite_names):
        anova_result = anova_oneway(metabolite_data[:, i], group, welch_correction=True)
        anova_results[i] = anova_result.pvalue

    anova_results = pd.DataFrame(
        {"p-value_ANOVA": anova_results}, index=metabolite_names
    )
    return anova_results


def kruskal_test(groups_split, metabolite_names) -> pd.DataFrame:
    """
    Perform a Kruskal-Wallis test on groups of data.

    Args:
        groups_split (pandas.core.groupby.DataFrameGroupBy): A grouped DataFrame object containing the groups to compare.
        metabolite_names (List[str]): A list of metabolite names.

    Returns:
        kw_results (pandas.core.frame.DataFrame): A DataFrame containing the p-value for each metabolite.
    """
    # Prepare a dictionary to hold pre-fetched group data
    group_data = {
        name: group.dropna(subset=metabolite_names) for name, group in groups_split
    }

    # Prepare the DataFrame structure outside the loop
    kw_results = np.zeros(len(metabolite_names))
    # Iterate over metabolites only once
    for i, metabolite in enumerate(metabolite_names):
        metabolite_data = [group[metabolite] for group in group_data.values()]

        # Perform the ANOVA
        kw_result = ss.kruskal(*metabolite_data)
        kw_results[i] = kw_result.pvalue

        # Store the results
    kw_results = pd.DataFrame({"p-value_KW": kw_results}, index=metabolite_names)

    return kw_results


def norm_test(data):
    data = data.rename(columns={data.columns[0]: "Sample", data.columns[1]: "Group"})
    data.drop(columns=["Sample", "Group"], inplace=True)

    result = data.apply(func=ss.shapiro, axis=0)
    result.index = ["W-statistic", "p-value"]

    return result
