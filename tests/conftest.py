import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from lmsstat.stat._utils import preprocess_data


@pytest.fixture(autouse=True, scope="session")
def matplotlib_agg_backend():
    matplotlib.use("Agg")
    yield


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    yield
    plt.close("all")


def _make_data(rng, n_per_group, group_labels, n_metab, means):
    """Build a wide DataFrame: Sample, Group, Met_0 … Met_{n-1}."""
    rows = []
    idx = 0
    for g, mu in zip(group_labels, means):
        for _ in range(n_per_group):
            row = {"Sample": f"S{idx:03d}", "Group": g}
            for j in range(n_metab):
                row[f"Met_{j}"] = rng.normal(mu + j * 0.1, 1.0)
            rows.append(row)
            idx += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def two_group_data():
    rng = np.random.default_rng(42)
    return _make_data(rng, n_per_group=10, group_labels=["A", "B"],
                      n_metab=5, means=[0.0, 3.0])


@pytest.fixture()
def three_group_data():
    rng = np.random.default_rng(123)
    return _make_data(rng, n_per_group=10, group_labels=["X", "Y", "Z"],
                      n_metab=5, means=[0.0, 2.0, 5.0])


@pytest.fixture()
def heteroscedastic_data():
    """Three groups with very different variances and unequal sizes.

    Welch's ANOVA and the classic equal-variance ANOVA diverge meaningfully
    here (not just by sampling noise), so it is a fair test of use_var.
    """
    rng = np.random.default_rng(7)
    specs = [("X", 12, 0.0, 0.5), ("Y", 8, 0.5, 3.0), ("Z", 20, 1.0, 8.0)]
    rows = []
    idx = 0
    for g, n, mu, sd in specs:
        for _ in range(n):
            row = {"Sample": f"H{idx:03d}", "Group": g}
            for j in range(5):
                row[f"Met_{j}"] = rng.normal(mu, sd)
            rows.append(row)
            idx += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def preprocessed_two_groups(two_group_data):
    raw, gs, names = preprocess_data(two_group_data)
    return two_group_data, raw, gs, names


@pytest.fixture()
def preprocessed_three_groups(three_group_data):
    raw, gs, names = preprocess_data(three_group_data)
    return three_group_data, raw, gs, names


@pytest.fixture()
def small_exact_data():
    """12 samples (6 per group), all unique floats — forces exact U-test path."""
    rng = np.random.default_rng(99)
    rows = []
    for i in range(6):
        row = {"Sample": f"E{i}", "Group": "G1"}
        for j in range(5):
            row[f"Met_{j}"] = rng.uniform(0, 100)
        rows.append(row)
    for i in range(6):
        row = {"Sample": f"E{i+6}", "Group": "G2"}
        for j in range(5):
            row[f"Met_{j}"] = rng.uniform(0, 100)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def data_with_nans(two_group_data):
    df = two_group_data.copy()
    df.iloc[0, 2] = np.nan
    df.iloc[5, 3] = np.nan
    df.iloc[10, 4] = np.nan
    return df


@pytest.fixture()
def data_with_constant_col(two_group_data):
    df = two_group_data.copy()
    df["Met_4"] = 5.0
    return df
