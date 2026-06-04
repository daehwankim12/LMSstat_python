"""
Preprocessing utilities for metabolomics tables.

Every function in this module follows the same contract:
    * the input DataFrame is never mutated;
    * a new DataFrame is returned with the first two columns standardized to
      ``Sample`` and ``Group`` (unchanged values) followed by the feature
      columns in their original order;
    * feature columns are coerced to numeric (non-numeric → NaN).
"""

import warnings

import numpy as np
import pandas as pd

from ._utils import ensure_sample_group_columns


def _split(data: pd.DataFrame):
    """Return (Sample/Group frame, numeric feature frame) without mutating input."""
    data = ensure_sample_group_columns(data)
    numeric = (
        data.drop(columns=["Sample", "Group"])
        .apply(pd.to_numeric, errors="coerce")
    )
    return data, numeric


def _reassemble(data: pd.DataFrame, numeric: pd.DataFrame) -> pd.DataFrame:
    """Glue Sample/Group back onto a transformed numeric block, preserving order."""
    return pd.concat(
        [
            data[["Sample", "Group"]].reset_index(drop=True),
            numeric.reset_index(drop=True),
        ],
        axis=1,
    )


def impute_missing(data: pd.DataFrame, method: str = "half_min") -> pd.DataFrame:
    """
    Impute missing feature values.

    Parameters
    ----------
    data : DataFrame
        Wide table. Col0 = Sample, Col1 = Group, remaining = features.
    method : {"half_min", "min", "knn"}
        * ``"min"``      – replace NaN in each feature with that feature's
          minimum observed value.
        * ``"half_min"`` – replace NaN with half the feature's minimum observed
          value (a common below-limit-of-detection imputation for LC-MS data).
        * ``"knn"``      – k-nearest-neighbours imputation across samples.

    Notes
    -----
    A feature with no observed values cannot be imputed and is left as NaN.
    """
    if method not in ("half_min", "min", "knn"):
        raise ValueError("method must be 'half_min', 'min', or 'knn'.")

    data, numeric = _split(data)

    if method in ("min", "half_min"):
        mins = numeric.min(axis=0, skipna=True)
        fill = mins if method == "min" else mins / 2.0
        numeric = numeric.fillna(fill)
    else:  # knn
        from sklearn.impute import KNNImputer

        n_samples = numeric.shape[0]
        n_neighbors = max(1, min(5, n_samples - 1))
        X = numeric.to_numpy(dtype=float)
        imputed = KNNImputer(n_neighbors=n_neighbors).fit_transform(X)
        numeric = pd.DataFrame(
            imputed, columns=numeric.columns, index=numeric.index
        )

    return _reassemble(data, numeric)


def log_transform(data: pd.DataFrame, base: float = 2, offset: float = 1.0) -> pd.DataFrame:
    """
    Apply a logarithmic transform ``log_base(x + offset)`` to feature columns.

    Parameters
    ----------
    data : DataFrame
        Wide table. Col0 = Sample, Col1 = Group, remaining = features.
    base : float
        Logarithm base (must be > 0 and != 1). Common values: 2, 10, ``np.e``.
    offset : float
        Added before the log to keep zeros finite. Defaults to 1.0.

    Raises
    ------
    ValueError
        If ``base`` is invalid, or any observed value would be non-positive
        after adding ``offset`` (suggest a larger offset). NaNs are preserved.
    """
    if not np.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("base must be a positive number other than 1.")

    data, numeric = _split(data)

    shifted = numeric.to_numpy(dtype=float) + offset
    finite = np.isfinite(shifted)
    if np.any(finite & (shifted <= 0)):
        raise ValueError(
            "Some values are <= 0 after adding offset; increase offset before "
            "log-transforming."
        )

    logged = np.log(shifted) / np.log(base)
    numeric = pd.DataFrame(logged, columns=numeric.columns, index=numeric.index)
    return _reassemble(data, numeric)


def _safe_denominator(denom: np.ndarray) -> np.ndarray:
    """Replace non-finite or zero denominators with 1.0 (leaves the row unscaled)."""
    return np.where(~np.isfinite(denom) | (denom == 0), 1.0, denom)


def normalize(data: pd.DataFrame, method: str = "median") -> pd.DataFrame:
    """
    Sample-wise (row) normalization to correct for dilution / loading differences.

    Parameters
    ----------
    data : DataFrame
        Wide table. Col0 = Sample, Col1 = Group, remaining = features.
    method : {"median", "total_area", "pqn"}
        * ``"median"``     – divide each sample by its median feature value.
        * ``"total_area"`` – divide each sample by the sum of its features
          (constant-sum / integral normalization).
        * ``"pqn"``        – Probabilistic Quotient Normalization (Dieterle 2006):
          integral-normalize, build a median reference spectrum, then divide each
          sample by the median quotient against the reference.

    Notes
    -----
    NaNs are ignored when computing per-sample factors and preserved in the
    output. Degenerate factors (zero / non-finite) leave the sample unscaled.
    """
    if method not in ("median", "total_area", "pqn"):
        raise ValueError("method must be 'median', 'total_area', or 'pqn'.")

    data, numeric = _split(data)
    X = numeric.to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "total_area":
            denom = _safe_denominator(np.nansum(X, axis=1, keepdims=True))
            Xn = X / denom
        elif method == "median":
            denom = _safe_denominator(np.nanmedian(X, axis=1, keepdims=True))
            Xn = X / denom
        else:  # pqn
            row_sum = _safe_denominator(np.nansum(X, axis=1, keepdims=True))
            X1 = X / row_sum
            ref = np.nanmedian(X1, axis=0)
            ref_valid = np.isfinite(ref) & (ref > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                quot = np.where(ref_valid[None, :], X1 / ref, np.nan)
            dil = _safe_denominator(np.nanmedian(quot, axis=1, keepdims=True))
            Xn = X1 / dil

    numeric = pd.DataFrame(Xn, columns=numeric.columns, index=numeric.index)
    return _reassemble(data, numeric)
