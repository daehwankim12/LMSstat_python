import itertools

import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind


def t_test(groups_split, metabolite_names) -> pd.DataFrame:
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
        try:
            # Call the t-test directly without using a separate worker function
            _, p_value = ttest_ind(
                numeric_data_groups[combo[0]],
                numeric_data_groups[combo[1]],
                equal_var=False,
            )
            # Assign the p_value to all rows in the column, assuming that's the intent
            df_ttest[col_name] = p_value
        except Exception as e:
            print(f"Failed to complete t-test for combination {combo}: {e}")

    return df_ttest


def u_test(groups_split, metabolite_names) -> pd.DataFrame:
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
        try:
            # Call the Mann-Whitney U test directly without using a separate worker function
            _, p_value = mannwhitneyu(
                numeric_data_groups[combo[0]],
                numeric_data_groups[combo[1]],
                alternative="two-sided",
                use_continuity=True,
            )
            # Assign the p_value to all rows in the column, assuming that's the intent
            df_utest[col_name] = p_value
        except Exception as e:
            print(f"Failed to complete u-test for combination {combo}: {e}")

    return df_utest


def anova_test(groups_split, metabolite_names) -> pd.DataFrame:
    # Prepare a dictionary to hold pre-fetched group data
    group_data = {
        name: group.dropna(subset=metabolite_names) for name, group in groups_split
    }

    # Prepare the DataFrame structure outside the loop
    anova_results = pd.DataFrame(index=metabolite_names, columns=["p-value_ANOVA"])

    # Iterate over metabolites only once
    for metabolite in metabolite_names:
        metabolite_data = [group[metabolite] for group in group_data.values()]

        # Perform the ANOVA
        _, p_value = f_oneway(*metabolite_data)

        # Store the results
        anova_results.at[metabolite, "p-value_ANOVA"] = p_value

    return anova_results.astype(float)


def kruskal_test(groups_split, metabolite_names) -> pd.DataFrame:
    # Prepare a dictionary to hold pre-fetched group data
    group_data = {
        name: group.dropna(subset=metabolite_names) for name, group in groups_split
    }

    # Prepare the DataFrame structure outside the loop
    kw_results = pd.DataFrame(index=metabolite_names, columns=["p-value_KW"])

    # Iterate over metabolites only once
    for metabolite in metabolite_names:
        metabolite_data = [group[metabolite] for group in group_data.values()]

        # Perform the ANOVA
        _, p_value = kruskal(*metabolite_data)

        # Store the results
        kw_results.at[metabolite, "p-value_KW"] = p_value

    return kw_results.astype(float)
