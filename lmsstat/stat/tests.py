import itertools

import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind


def t_test(groups_split, metabolite_names) -> pd.DataFrame:
    group_names = groups_split.groups.keys()
    group_combinations = list(itertools.combinations(group_names, 2))

    column_names = [f"({combo[0]}, {combo[1]})_ttest" for combo in group_combinations]
    df_ttest = pd.DataFrame(index=metabolite_names, columns=column_names)

    for combo in group_combinations:
        col_name = f"({combo[0]}, {combo[1]})_ttest"
        group1_data = (
            groups_split.get_group(combo[0])
            .select_dtypes(include=[float, "float64", "int", "int64"])
            .to_numpy()
        )
        group2_data = (
            groups_split.get_group(combo[1])
            .select_dtypes(include=[float, "float64", "int", "int64"])
            .to_numpy()
        )
        _, p_value = ttest_ind(group1_data, group2_data, equal_var=False)
        df_ttest[col_name] = p_value

    return df_ttest


def u_test(groups_split, metabolite_names) -> pd.DataFrame:
    group_names = groups_split.groups.keys()
    group_combinations = list(itertools.combinations(group_names, 2))

    column_names = [f"({combo[0]}, {combo[1]})_utest" for combo in group_combinations]
    df_utest = pd.DataFrame(index=metabolite_names, columns=column_names)

    for combo in group_combinations:
        col_name = f"({combo[0]}, {combo[1]})_utest"
        group1_data = (
            groups_split.get_group(combo[0])
            .select_dtypes(include=[float, "float64", "int", "int64"])
            .to_numpy()
        )
        group2_data = (
            groups_split.get_group(combo[1])
            .select_dtypes(include=[float, "float64", "int", "int64"])
            .to_numpy()
        )
        _, p_value = mannwhitneyu(group1_data, group2_data, alternative="two-sided")
        df_utest[col_name] = p_value

    return df_utest


def anova_test(groups_split, metabolite_names) -> pd.DataFrame:
    anova_results = pd.DataFrame(index=metabolite_names, columns=["p-value_ANOVA"])

    for metabolite in metabolite_names:
        metabolite_data = [group[metabolite].dropna() for name, group in groups_split]

        _, p_value = f_oneway(*metabolite_data)

        anova_results.at[metabolite, "p-value_ANOVA"] = p_value

    return anova_results


def kruskal_test(groups_split, metabolite_names) -> pd.DataFrame:
    kw_results = pd.DataFrame(index=metabolite_names, columns=["p-value_KW"])

    for metabolite in metabolite_names:
        metabolite_data = [group[metabolite].dropna() for name, group in groups_split]

        _, p_value = kruskal(*metabolite_data)

        kw_results.at[metabolite, "p-value_KW"] = p_value

    return kw_results
