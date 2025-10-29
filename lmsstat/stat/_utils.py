import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from sklearn.preprocessing import StandardScaler


# from sklearn.model_selection import KFold


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


def correlation(data, axis="sample", method="pearson"):
    data = data.iloc[:, 2:]
    axis = axis.lower()
    if axis == "sample":
        return data.transpose().corr(method=method)
    elif axis == "metabolite":
        return data.corr(method=method)
    else:
        raise ValueError("Invalid axis. Use 'sample' or 'metabolite'.")


def scaling(data, method="auto"):
    data = data.rename(columns={data.columns[0]: "Sample", data.columns[1]: "Group"})
    data_raw = data.iloc[:, 2:]
    scaled_data = data_raw.copy()
    scaler = StandardScaler()
    scaler.fit(scaled_data)

    if method == "auto":
        scaler.scale_ = np.std(scaled_data, axis=0, ddof=1).to_list()
        scaled_data = pd.DataFrame(
            scaler.transform(scaled_data), columns=scaled_data.columns
        )
    elif method == "pareto":
        scaler.scale_ = np.sqrt(np.std(scaled_data, axis=0, ddof=1)).to_list()
        scaled_data = pd.DataFrame(
            scaler.transform(scaled_data), columns=scaled_data.columns
        )
    else:
        raise ValueError("Invalid scaling method.")

    return pd.concat([data[["Sample", "Group"]].reset_index(drop=True),
                  scaled_data.reset_index(drop=True)], axis=1)
