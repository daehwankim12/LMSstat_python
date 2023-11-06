import itertools as it
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as ss
from pandas import DataFrame
from statsmodels.sandbox.stats.multicomp import multipletests


def __convert_to_df(
    a: Union[list, np.ndarray, DataFrame],
    val_col: str = "vals",
    group_col: str = "groups",
    val_id: int = None,
    group_id: int = None,
) -> Tuple[DataFrame, str, str]:
    if not group_col:
        group_col = "groups"
    if not val_col:
        val_col = "vals"

    if isinstance(a, DataFrame):
        x = a.copy()
        if not {group_col, val_col}.issubset(a.columns):
            raise ValueError(
                "Specify correct column names using `group_col` and `val_col` args"
            )
        return x, val_col, group_col

    elif isinstance(a, list) or (isinstance(a, np.ndarray) and not a.shape.count(2)):
        grps_len = map(len, a)
        grps = list(it.chain(*[[i + 1] * l for i, l in enumerate(grps_len)]))
        vals = list(it.chain(*a))

        return DataFrame({val_col: vals, group_col: grps}), val_col, group_col

    elif isinstance(a, np.ndarray):
        # cols ids not defined
        # trying to infer
        if not all([val_id, group_id]):
            if np.argmax(a.shape):
                a = a.T

            ax = [np.unique(a[:, 0]).size, np.unique(a[:, 1]).size]

            if not np.diff(ax).item():
                raise ValueError(
                    "Cannot infer input format.\nPlease specify `val_id` and `group_id` args"
                )

            __val_col = np.argmax(ax)
            __group_col = np.argmin(ax)
            cols = {__val_col: val_col, __group_col: group_col}
        else:
            cols = {val_id: val_col, group_id: group_col}

        cols_vals = dict(sorted(cols.items())).values()
        return DataFrame(a, columns=cols_vals), val_col, group_col


def posthoc_scheffe(
    a: Union[list, np.ndarray, DataFrame],
    val_col: str = None,
    group_col: str = None,
    sort: bool = False,
) -> DataFrame:
    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col], ascending=True) if sort else x

    groups = x[_group_col].unique()
    x_grouped = x.groupby(_group_col)[_val_col]
    ni = x_grouped.count()
    xi = x_grouped.mean()
    si = x_grouped.var()
    n = ni.sum()
    sin = 1.0 / (n - groups.size) * np.sum(si * (ni - 1.0))

    def compare(i, j):
        dif = xi.loc[i] - xi.loc[j]
        A = sin * (1.0 / ni.loc[i] + 1.0 / ni.loc[j]) * (groups.size - 1.0)
        f_val = dif**2.0 / A
        return f_val

    vs = np.zeros((groups.size, groups.size), dtype=float)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    combs = it.combinations(range(groups.size), 2)

    for i, j in combs:
        vs[i, j] = compare(groups[i], groups[j])

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    p_values = ss.f.sf(vs, groups.size - 1.0, n - groups.size)

    np.fill_diagonal(p_values, 1)
    return DataFrame(p_values, index=groups, columns=groups)


def posthoc_dunn(
    a: Union[list, np.ndarray, DataFrame],
    val_col: str = None,
    group_col: str = None,
    p_adjust: str = None,
    sort: bool = True,
) -> DataFrame:
    def compare_dunn(i, j):
        diff = np.abs(x_ranks_avg.loc[i] - x_ranks_avg.loc[j])
        A = n * (n + 1.0) / 12.0
        B = 1.0 / x_lens.loc[i] + 1.0 / x_lens.loc[j]
        z_value = diff / np.sqrt((A - x_ties) * B)
        p_value = 2.0 * ss.norm.sf(np.abs(z_value))
        return p_value

    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    n = len(x.index)
    x_groups_unique = x[_group_col].unique()
    x_len = x_groups_unique.size
    x_lens = x.groupby(_group_col)[_val_col].count()

    x["ranks"] = x[_val_col].rank()
    x_ranks_avg = x.groupby(_group_col)["ranks"].mean()

    # ties
    vals = x.groupby("ranks").count()[_val_col].values
    tie_sum = np.sum(vals[vals != 1] ** 3 - vals[vals != 1])
    tie_sum = tie_sum or 0
    x_ties = tie_sum / (12.0 * (n - 1))

    vs = np.zeros((x_len, x_len))
    combs = it.combinations(range(x_len), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    for i, j in combs:
        vs[i, j] = compare_dunn(x_groups_unique[i], x_groups_unique[j])

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    return DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)


def scheffe_test(groups_split, metabolite_names):
    group_names = sorted(groups_split.groups.keys())
    group_combinations = list(it.combinations(group_names, 2))

    # Prepare DataFrame to hold the p-values
    p_values_df = pd.DataFrame(
        index=metabolite_names,
        columns=[f"({i}, {j})_scheffe" for i, j in group_combinations],
    )

    for metabolite in metabolite_names:
        metabolite_data = [
            group[metabolite].dropna().values
            for name, group in groups_split
            if name in group_names
        ]

        scheffe_results = posthoc_scheffe(metabolite_data)

        # Remap the indices of the results to match the group names
        scheffe_results.index = group_names
        scheffe_results.columns = group_names

        for i, j in group_combinations:
            p_values_df.loc[metabolite, f"({i}, {j})_scheffe"] = scheffe_results.loc[
                i, j
            ]

    return p_values_df


def dunn_test(groups_split, metabolite_names):
    group_names = sorted(groups_split.groups.keys())
    group_combinations = list(it.combinations(group_names, 2))

    # Prepare DataFrame to hold the p-values
    p_values_df = pd.DataFrame(
        index=metabolite_names,
        columns=[f"({i}, {j})_dunn" for i, j in group_combinations],
    )

    for metabolite in metabolite_names:
        metabolite_data = [
            group[metabolite].dropna().values
            for name, group in groups_split
            if name in group_names
        ]

        dunn_results = posthoc_dunn(metabolite_data)

        # Remap the indices of the results to match the group names
        dunn_results.index = group_names
        dunn_results.columns = group_names

        for i, j in group_combinations:
            p_values_df.loc[metabolite, f"({i}, {j})_dunn"] = dunn_results.loc[i, j]

    return p_values_df
