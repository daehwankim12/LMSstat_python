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
    """Glue Sample/Group back onto a transformed numeric block.

    Preserves both the original column order and the input row index. ``data``
    and ``numeric`` share the index produced by ``_split``, so they align.
    """
    return pd.concat([data[["Sample", "Group"]], numeric], axis=1)


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
          Intended for raw, non-negative intensity data; with negative values
          half of a negative minimum is more negative, which is rarely meaningful.
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

        # KNNImputer drops all-missing columns, so impute only the columns with
        # at least one observed value and leave all-NaN features as NaN.
        observed = numeric.columns[numeric.notna().any(axis=0)]
        if len(observed) > 0:
            n_samples = numeric.shape[0]
            n_neighbors = max(1, min(5, n_samples - 1))
            imputed = KNNImputer(n_neighbors=n_neighbors).fit_transform(
                numeric[observed].to_numpy(dtype=float)
            )
            numeric = numeric.copy()
            numeric[observed] = imputed

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
    if not np.isfinite(offset):
        raise ValueError("offset must be a finite number.")

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


def rsd_filter(
        data: pd.DataFrame,
        *,
        qc_label: str = "QC",
        max_rsd: float = 30.0,
        return_rsd: bool = False,
):
    """
    Drop features whose %RSD across the QC samples exceeds ``max_rsd``.

    %RSD (a.k.a. CV) is the standard LC-MS quality metric for feature
    reliability: ``100 * std(ddof=1) / mean`` computed over the pooled QC
    samples. Only features (columns) are filtered; samples are untouched.

    Parameters
    ----------
    data : DataFrame
        Wide table. Col0 = Sample, Col1 = Group, remaining = features. Expects
        raw, non-negative intensities.
    qc_label : str
        Group label that marks the QC (pooled) samples. Defaults to "QC".
    max_rsd : float
        %RSD cutoff (finite, non-negative). Features with QC %RSD <= max_rsd
        are kept.
    return_rsd : bool
        If True, also return a Series of the QC %RSD for *all* original
        features (so the caller can see what was dropped).

    Returns
    -------
    DataFrame, or (DataFrame, Series) when ``return_rsd=True``.

    Notes
    -----
    A feature is dropped (treated as infinite %RSD) when it cannot be assessed:
    fewer than two finite QC values, or a non-positive QC mean. A constant QC
    feature (zero spread, positive mean) has 0% RSD and is kept. Requires at
    least two QC samples.
    """
    if not (np.isfinite(max_rsd) and max_rsd >= 0):
        raise ValueError("max_rsd must be a finite, non-negative number.")

    data, numeric = _split(data)
    qc_mask = (data["Group"].astype(str) == str(qc_label)).to_numpy()
    n_qc = int(qc_mask.sum())
    if n_qc < 2:
        raise ValueError(
            f"rsd_filter needs at least two QC samples labelled {qc_label!r}; "
            f"found {n_qc}."
        )

    qc_arr = numeric.to_numpy(dtype=float)[qc_mask]
    n_finite = np.sum(np.isfinite(qc_arr), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(qc_arr, axis=0)
        std = np.nanstd(qc_arr, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rsd = 100.0 * std / mean
    # Unassessable features → infinite %RSD (always dropped).
    degenerate = (n_finite < 2) | ~np.isfinite(mean) | (mean <= 0) | np.isnan(rsd)
    rsd = np.where(degenerate, np.inf, rsd)

    rsd_series = pd.Series(rsd, index=numeric.columns, name="rsd")
    keep = numeric.columns[rsd <= max_rsd]
    filtered = _reassemble(data, numeric[keep])

    if return_rsd:
        return filtered, rsd_series
    return filtered
