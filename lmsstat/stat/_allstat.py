import pandas as pd

from ._posthoc import dunn_test, games_howell_test, scheffe_test
from ._tests import anova_test, kruskal_test, t_test, u_test
from ._utils import _sanitize_pvalues_df, p_adjust, preprocess_data


def allstats(data, p_adj=True, anova_use_var="equal", posthoc="scheffe"):
    """
    Generates a statistical analysis of the given data.

    Parameters:
        data (pd.DataFrame): DataFrame to be analyzed.
            Column 0 = Sample, Column 1 = Group, remaining columns = metabolites/features.

        p_adj (bool, optional): Whether to perform p-value adjustment. Defaults to True.

        anova_use_var ({"equal", "unequal"}, optional): Variance assumption for the
            ANOVA (3+ groups). "equal" is the classic one-way ANOVA; "unequal" is
            Welch's ANOVA. Defaults to "equal".

        posthoc ({"scheffe", "games_howell", "both"}, optional): Which parametric
            post-hoc test(s) to run for 3+ groups. The non-parametric Dunn post-hoc
            is always included. Defaults to "scheffe". Selection is explicit (no
            per-feature auto-selection) so result columns stay reproducible.

    Returns:
        pandas.DataFrame: The statistical analysis results.
    """
    if posthoc not in ("scheffe", "games_howell", "both"):
        raise ValueError("posthoc must be 'scheffe', 'games_howell', or 'both'.")
    if anova_use_var not in ("equal", "unequal"):
        raise ValueError("anova_use_var must be 'equal' or 'unequal'.")

    _, groups_split, metabolite_names = preprocess_data(data)

    num_groups = len(groups_split)

    if num_groups <= 1:
        raise ValueError("Number of groups must be greater than 1")

    result_t = t_test(groups_split, metabolite_names)
    result_u = u_test(groups_split, metabolite_names)
    if num_groups > 2:
        result_anova = anova_test(
            groups_split, metabolite_names, use_var=anova_use_var
        )
        result_kruskal = kruskal_test(groups_split, metabolite_names)
        parametric_posthoc = []
        if posthoc in ("scheffe", "both"):
            parametric_posthoc.append(scheffe_test(groups_split, metabolite_names))
        if posthoc in ("games_howell", "both"):
            parametric_posthoc.append(
                games_howell_test(groups_split, metabolite_names)
            )
        result_dunn = dunn_test(groups_split, metabolite_names)

    if p_adj:
        result_t = p_adjust(result_t)
        result_u = p_adjust(result_u)
        if num_groups > 2:
            result_anova = p_adjust(result_anova)
            result_kruskal = p_adjust(result_kruskal)
    if num_groups == 2:
        out = pd.concat([result_t, result_u], axis=1)
    else:
        out = pd.concat(
            [
                result_t,
                result_u,
                result_anova,
                *parametric_posthoc,
                result_kruskal,
                result_dunn,
            ],
            axis=1,
        )
    return _sanitize_pvalues_df(out)
