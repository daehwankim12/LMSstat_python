from ._tests import t_test, u_test, anova_test, kruskal_test
from ._posthoc import scheffe_test, dunn_test
from ._utils import preprocess_data, p_adjust

import os
import pandas as pd


def allstats(filedir, p_adj=True):
    """
    Generates a statistical analysis of the given file.

    Parameters:
        filedir (str): The directory of the file to be analyzed.

        p_adj (bool, optional): Whether to perform p-value adjustment. Defaults to True.

    Returns:
        pandas.DataFrame: The statistical analysis results.
    """
    assert isinstance(filedir, str)
    assert os.path.isfile(filedir)

    data = pd.read_csv(filedir)

    _, groups_split, metabolite_names = preprocess_data(data)

    num_groups = len(groups_split)

    assert num_groups > 1, "Number of groups must be greater than 1"

    result_t = t_test(groups_split, metabolite_names)
    result_u = u_test(groups_split, metabolite_names)
    if num_groups > 2:
        result_anova = anova_test(groups_split, metabolite_names)
        result_kruskal = kruskal_test(groups_split, metabolite_names)
        result_scheffe = scheffe_test(groups_split, metabolite_names)
        result_dunn = dunn_test(groups_split, metabolite_names)

    if p_adj:
        result_t = p_adjust(result_t)
        result_u = p_adjust(result_u)
        if num_groups > 2:
            result_anova = p_adjust(result_anova)
            result_kruskal = p_adjust(result_kruskal)
    if num_groups == 2:
        result = pd.concat([result_t, result_u], axis=1)
    else:
        result = pd.concat(
            [
                result_t,
                result_u,
                result_anova,
                result_scheffe,
                result_kruskal,
                result_dunn,
            ],
            axis=1,
        )

    return result
