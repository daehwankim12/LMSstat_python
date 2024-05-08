import pandas as pd
from scipy.stats import false_discovery_control


def preprocess_data(data):
    # Rename columns
    data = data.rename(columns={data.columns[0]: "Sample", data.columns[1]: "Group"})

    # Convert the "Group" column to character type (object in pandas)
    data["Group"] = data["Group"].astype(str)

    # Sort the data by "Group"
    data = data.sort_values(by="Group")

    # Convert relevant columns to numeric using apply() and to_numeric()
    cols_to_convert = data.columns[2:]
    data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors="coerce")

    # Convert the DataFrame to a data table (not necessary in Python)
    data_final_raw = data.drop(columns=["Sample", "Group"])

    # Split the data table by group
    groups_split = data.groupby("Group")

    metabolite_names = list(data_final_raw.columns)

    return data_final_raw, groups_split, metabolite_names


def p_adjust(mat):
    return mat.apply(
        false_discovery_control,
        axis=0,
        raw=True,
    )


def correlation(data, axis="sample"):
    if axis == "sample":
        return data.corr()
    elif axis == "metabolite":
        return data.transpose().corr()
