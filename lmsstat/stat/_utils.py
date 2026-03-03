import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

# from sklearn.model_selection import KFold


def _sanitize_pvalues_array(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(p, 0.0, 1.0)


def _sanitize_pvalues_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return out.clip(lower=0.0, upper=1.0)


def ensure_sample_group_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the first two columns to be named 'Sample' and 'Group'.

    This project treats column 0 as Sample and column 1 as Group (strict-by-position).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.shape[1] < 2:
        raise ValueError("data must have at least two columns: Sample and Group.")

    cols = list(data.columns)
    cols[0] = "Sample"
    cols[1] = "Group"
    if len(set(cols)) != len(cols):
        raise ValueError(
            "Column names become duplicated after standardizing the first two "
            "columns to 'Sample' and 'Group'. Rename your input columns so all "
            "column names are unique."
        )

    out = data.copy(deep=False)
    out.columns = cols
    return out


def preprocess_data(data):
    # Rename columns
    data = ensure_sample_group_columns(data)

    # Convert the "Group" column to character type (object in pandas)
    data = (
        data.assign(Group=data["Group"].astype("string"))
        .dropna(subset=["Group"])
        .assign(Group=lambda df: df["Group"].astype(str))
    )

    # Sort the data by "Group"
    data = data.sort_values(by="Group")

    # Convert relevant columns to numeric using apply() and to_numeric()
    cols_to_convert = data.columns[2:]
    numeric = data[cols_to_convert].apply(pd.to_numeric, errors="coerce")
    data = pd.concat([data[["Sample", "Group"]].reset_index(drop=True), numeric.reset_index(drop=True)], axis=1)

    # Convert the DataFrame to a data table (not necessary in Python)
    data_final_raw = numeric.reset_index(drop=True)

    # Split the data table by group
    groups_split = data.groupby("Group")

    metabolite_names = list(data_final_raw.columns)

    return data_final_raw, groups_split, metabolite_names


def p_adjust(mat):
    mat = _sanitize_pvalues_df(mat)
    adj = mat.apply(false_discovery_control, axis=0, raw=True)
    return _sanitize_pvalues_df(adj)


def correlation(data, axis="sample", method="pearson"):
    data = ensure_sample_group_columns(data)
    data = data.drop(columns=["Sample", "Group"])
    axis = axis.lower()
    if axis == "sample":
        return data.transpose().corr(method=method)
    elif axis == "metabolite":
        return data.corr(method=method)
    else:
        raise ValueError("Invalid axis. Use 'sample' or 'metabolite'.")


def scaling(data, method="auto"):
    data = ensure_sample_group_columns(data)
    numeric = (
        data.drop(columns=["Sample", "Group"])
        .apply(pd.to_numeric, errors="coerce")
    )

    X = numeric.to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0)
    Xc = X - mu

    std = np.nanstd(X, axis=0, ddof=1)
    if method == "auto":
        denom = std
    elif method == "pareto":
        denom = np.sqrt(std)
    else:
        raise ValueError("Invalid scaling method.")

    denom = np.where(~np.isfinite(denom) | (denom == 0), 1.0, denom)
    Xs = Xc / denom
    scaled = pd.DataFrame(Xs, columns=numeric.columns)

    return pd.concat(
        [
            data[["Sample", "Group"]].reset_index(drop=True),
            scaled.reset_index(drop=True),
        ],
        axis=1,
    )
