import os

import altair as alt
import pandas as pd

from ..stats import scaling


def plot_pca(pc_scores, group):
    pc_scores["Group"] = group
    points = (
        alt.Chart(pc_scores)
        .mark_circle(size=60)
        .encode(
            x="PC1:Q", y="PC2:Q", color="Group:N", tooltip=["Group:N", "PC1:Q", "PC2:Q"]
        )
    )

    return points


def plot_one_box(metabolite):
    box = (
        alt.Chart(metabolite)
        .mark_boxplot()
        .encode(x="Group:N", y="Value:Q", color="Group:N")
    )

    return box


def plot_box(data, scale=False, scale_method="auto"):
    data = data.rename(columns={data.columns[0]: "Sample", data.columns[1]: "Group"})
    if scale:
        data = scaling(data, method=scale_method)

    group = data["Group"]
    data = data.drop(columns=["Sample", "Group"])

    if not os.path.exists("boxplots"):
        os.makedirs("boxplots", exist_ok=True)

    for metabolite in data.columns:
        one_data = pd.DataFrame({"Group": group, "Value": data[metabolite]})
        box = plot_one_box(one_data)
        box.save(f"boxplots/box_{metabolite}.png")

    return None


def plot_one_dot(metabolite):
    dot = (
        alt.Chart(metabolite)
        .mark_circle(size=60)
        .encode(x="Group:N", y="Value:Q", color="Group:N", xOffset="jitter:Q")
        .transform_calculate(
            # Generate Gaussian jitter with a Box-Muller transform
            jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
        )
    )

    return dot


def plot_dot(data, scale=False, scale_method="auto"):
    data = data.rename(columns={data.columns[0]: "Sample", data.columns[1]: "Group"})
    if scale:
        data = scaling(data, method=scale_method)

    group = data["Group"]
    data = data.drop(columns=["Sample", "Group"])

    if not os.path.exists("dotplots"):
        os.makedirs("dotplots", exist_ok=True)

    for metabolite in data.columns:
        one_data = pd.DataFrame({"Group": group, "Value": data[metabolite]})
        dot = plot_one_dot(one_data)
        dot.save(f"dotplots/dot_{metabolite}.png")

    return None


# TODO: Implement plot_violin
# def plot_one_violin(metabolite, metabolite_name):
#     violin = alt.Chart(metabolite).transform_density(
#         density='Value',
#         as_=['Value', 'density'],
#         extent=[metabolite['Value'].min(), metabolite['Value'].max()],
#         groupby=['Group']
#     ).mark_area(orient='horizontal').encode(
#         y=alt.Y('Value:Q', title=metabolite_name),
#         color='Group:N',
#         x=alt.X(
#             'density:Q',
#             stack='center',
#             impute=None,
#             title=None,
#             axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True)
#         ),
#         column=alt.Column(
#             'Group:N',
#             header=alt.Header(
#                 titleOrient='bottom',
#                 labelOrient='bottom',
#                 labelPadding=0
#             )
#         )
#     ).properties(
#         width=300,
#         height=200
#     )
#
#     return violin
#
#
# def plot_violin(data):
#     data = data.rename(columns={data.columns[0]: 'Sample', data.columns[1]: 'Group'})
#     group = data['Group']
#     data = data.drop(columns=['Sample', 'Group'])
#
#     if not os.path.exists('violinplots'):
#         os.makedirs('violinplots', exist_ok=True)
#
#     for metabolite in data.columns:
#         one_data = pd.DataFrame({'Group': group, 'Value': data[metabolite]})
#         violin = plot_one_violin(one_data, metabolite)
#         violin.save(f'violinplots/violin_{metabolite}.png', format='png')
#
#     return None


# TODO: remove the following function
def melting(data):
    data = data.set_index("Sample")
    data = data.melt(var_name="variable", value_name="value")

    return data


# TODO: Using seaborn to plot heatmap
def plot_heatmap(data, scale=False, scale_method="auto"):
    if scale:
        data = scaling(data, method=scale_method)
    data = data.rename(columns={data.columns[0]: "Sample", data.columns[1]: "Group"})
    data = melting(data)

    heatmap = (
        alt.Chart(data)
        .mark_rect()
        .encode(x="variable:N", y="Sample:N", color="value:Q")
    )
    heatmap.save("heatmap.png")

    return None
