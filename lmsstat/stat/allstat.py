import tests
from posthoc import scheffe_test, dunn_test
from preprocessing import preprocess_data

import os
import pandas as pd


def allstats(filedir):
    assert isinstance(filedir, str)
    assert os.path.isfile(filedir)

    data = pd.read_csv(filedir)

    data_final_raw, groups_split, metabolite_names = preprocess_data(data)

    result_t = tests.t_test(groups_split, metabolite_names)
    result_u = tests.u_test(groups_split, metabolite_names)
    result_anova = tests.anova_test(groups_split, metabolite_names)
    result_kruskal = tests.kruskal_test(groups_split, metabolite_names)
    result_scheffe = scheffe_test(groups_split, metabolite_names)
    result_dunn = dunn_test(groups_split, metabolite_names)

    result = pd.concat(
        [result_t, result_u, result_anova, result_scheffe, result_kruskal, result_dunn],
        axis=1,
    )

    return result
