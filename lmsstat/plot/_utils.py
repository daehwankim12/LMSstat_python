import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.pyplot import colormaps
from plotnine import geom_segment, aes, annotate
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def scaling(data: pd.DataFrame, *, method: str = "auto") -> pd.DataFrame:
    """
    Scale the numeric columns in a tidy-format DataFrame.

    Parameters
    ----------
    data : DataFrame
        First column = Sample, second column = Group, remaining = numeric variables.
    method : {"auto", "pareto"}
        auto   -> unit variance (z-score) scaling
        pareto -> sqrt-SD scaling

    Returns
    -------
    DataFrame
        Same structure but scaled numeric values.
    """
    if method not in ("auto", "pareto"):
        raise ValueError("Invalid scaling method: choose 'auto' or 'pareto'.")

    data = data.rename(columns={data.columns[0]: "Sample",
                                data.columns[1]: "Group"})
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

    return pd.concat([data[["Sample", "Group"]], scaled_data], axis=1)


def pca(data: pd.DataFrame,
        n_components: int = 2,
        scale: bool = True,
        cv_splits: int = 7,
        random_state: int = 42):
    """
    Perform PCA and calculate cumulative Q² via cross-validation.

    Parameters
    ----------
    data : DataFrame
        First column = Sample, second = Group, remaining = numeric variables.
    n_components : int
        Number of principal components to keep.
    scale : bool
        If True, apply auto-scaling before PCA.
    cv_splits : int
        Number of folds for K-fold CV (between 2 and n_samples).
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    pc_scores : DataFrame
        Sample scores for each component.
    pc_loadings : DataFrame
        Variable loadings for each component.
    pc_r2 : float
        Cumulative variance explained (R²X).
    pc_q2 : float
        Cumulative predictive ability (Q²) from CV.
    """
    # --- Pre-processing --------------------------------------------------
    if scale:
        data = scaling(data, method="auto")

    data = data.rename(columns={data.columns[0]: "Sample",
                                data.columns[1]: "Group"})
    X_df = data.drop(columns=["Sample", "Group"])
    X = X_df.to_numpy(float)

    # --- Fit PCA on the full data ---------------------------------------
    pc = PCA(n_components=n_components).fit(X)

    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    pc_scores = pd.DataFrame(pc.transform(X), columns=pc_cols)
    pc_loadings = pd.DataFrame(pc.components_.T,
                               index=X_df.columns,
                               columns=pc_cols)
    pc_r2 = pc.explained_variance_ratio_.sum()

    # --- Cross-validation for Q² ----------------------------------------
    n_samples = X.shape[0]
    cv_splits = min(max(2, cv_splits), n_samples)
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    press_contrib = np.zeros(n_components)
    ss_contrib = np.zeros(n_components)

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]

        # Center data within the fold
        mu_fold = X_tr.mean(axis=0)
        R_tr = X_tr - mu_fold
        R_te = X_te - mu_fold

        for a in range(n_components):
            # Total sum of squares before extracting component a
            ss_now = np.sum(R_te ** 2)
            ss_contrib[a] += ss_now
            if np.isclose(ss_now, 0.0):
                R_tr.fill(0.0)
                R_te.fill(0.0)
                break

            # One-component PCA on training residuals
            if np.isclose(np.sum(R_tr ** 2), 0.0):
                P_a = np.zeros((1, R_tr.shape[1]))
            else:
                pca_one = PCA(n_components=1, random_state=random_state).fit(R_tr)
                P_a = pca_one.components_

            # Project test residuals and compute PRESS
            T_te = R_te @ P_a.T
            R_te_hat = T_te @ P_a
            press_contrib[a] += np.sum((R_te - R_te_hat) ** 2)

            # Deflate residuals for next component
            T_tr = R_tr @ P_a.T
            R_tr -= T_tr @ P_a
            R_te -= R_te_hat

    # --- Aggregate Q² ----------------------------------------------------
    ratios = []
    for a in range(n_components):
        ss_a = ss_contrib[a]
        pr_a = press_contrib[a]
        ratio = 1.0 if np.isclose(ss_a, 0.0) else pr_a / ss_a
        ratio = max(ratio, -0.1)
        ratios.append(ratio)

    pc_q2 = 1.0 - np.prod(ratios)
    return pc_scores, pc_loadings, pc_r2, pc_q2


def _pal(n: int):
    """
    Return n hex colors from Matplotlib's tab20 colormap.
    """
    cmap = colormaps.get_cmap("tab20")
    return [to_hex(cmap(i % cmap.N)) for i in range(n)]


# ───────── Bracket + Star annotation helper ─────────
def _annot(gg,
           stat_tbl: pd.DataFrame,
           order: list[str],
           y_top: float,
           *,
           offset: float = 0.05,
           step: float = 0.05,
           tip: float = 0.01,
           star: int = 10,
           line: float = 0.15):
    """
    Add bracket-style significance annotations to a plotnine object.

    Parameters
    ----------
    gg : plotnine object
        The base plot.
    stat_tbl : DataFrame
        Must contain columns {'group1','group2','p_value'}.
    order : list[str]
        Categorical ordering of groups on the x-axis.
    y_top : float
        Highest y value among the plotted objects.
    offset, step, tip : float
        Control vertical positioning of brackets.
    star : int
        Text size of significance stars.
    line : float
        Line thickness.
    """
    # Filter by significance level p ≤ .05
    stat_tbl = stat_tbl.loc[stat_tbl.p_value <= 0.05]
    if stat_tbl.empty or not {'group1', 'group2', 'p_value'}.issubset(stat_tbl.columns):
        return gg

    level = 0
    for _, r in stat_tbl.iterrows():
        if r.group1 not in order or r.group2 not in order:
            continue
        x1, x2 = order.index(r.group1) + 1, order.index(r.group2) + 1
        y = y_top * (1 + offset + step * level)
        y2 = y - tip * y_top
        s = '**' if r.p_value <= 0.01 else '*'
        level += 1

        gg += geom_segment(aes(x=x1, xend=x2, y=y, yend=y), size=line)
        gg += geom_segment(aes(x=x1, xend=x1, y=y, yend=y2), size=line)
        gg += geom_segment(aes(x=x2, xend=x2, y=y, yend=y2), size=line)
        gg += annotate('text',
                       x=(x1 + x2) / 2,
                       y=y,
                       label=s,
                       size=star,
                       ha='center',
                       va='bottom')
    return gg


def melting(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide-format DataFrame (Sample as index) to tidy long format.
    """
    data = data.set_index("Sample")
    return data.melt(var_name="variable", value_name="value")


def plsda(data: pd.DataFrame,
          n_components: int = 2,
          scale: bool = True,
          cv_splits: int = 7,
          random_state: int = 42):
    """
    Perform PLS-DA and compute cumulative R²X, R²Y, Q².

    Parameters
    ----------
    data : DataFrame
        First column = Sample, second = Group, remaining = numeric variables.
    n_components : int
        Number of latent variables.
    scale : bool
        If True, apply auto scaling before PLS-DA.
    cv_splits : int
        Number of CV folds (2 ≤ cv_splits ≤ n_samples).
    random_state : int
        RNG seed.

    Returns
    -------
    lv_scores : DataFrame
        Sample scores for latent variables.
    lv_loadings : DataFrame
        Variable loadings.
    r2x_cum : float
        Cumulative R²X.
    r2y_cum : float
        Cumulative R²Y.
    q2_cum : float
        Cumulative Q² from CV.
    """
    # --- Pre-processing --------------------------------------------------
    if scale:
        data = scaling(data, "auto")

    data = data.rename(columns={data.columns[0]: "Sample",
                                data.columns[1]: "Group"})
    X_df = data.drop(columns=["Sample", "Group"])
    X = X_df.to_numpy(float)

    y_labels = data["Group"].astype(str)
    Y_df = pd.get_dummies(y_labels)  # one-hot encoding
    Y = Y_df.to_numpy(float)

    # --- Fit PLS-DA on full data ----------------------------------------
    pls = PLSRegression(n_components=n_components,
                        scale=False,
                        max_iter=200).fit(X, Y)

    lv_cols = [f"LV{i + 1}" for i in range(n_components)]
    lv_scores = pd.DataFrame(pls.x_scores_, columns=lv_cols)
    lv_loadings = pd.DataFrame(pls.x_loadings_,
                               index=X_df.columns,
                               columns=lv_cols)

    r2y_cum = pls.score(X, Y)

    # R²X based on reconstruction
    X_mean = pls._x_mean
    Xc = X - X_mean
    X_hat = pls.x_scores_ @ pls.x_loadings_.T
    sse_x = np.sum((Xc - X_hat) ** 2)
    tss_x = np.sum(Xc ** 2)
    r2x_cum = 0.0 if np.isclose(tss_x, 0.0) else 1.0 - sse_x / tss_x

    # --- Cross-validated Q² --------------------------------------------
    n_samples = X.shape[0]
    cv_splits = min(max(2, cv_splits), n_samples)
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    press = np.zeros(n_components)
    ss = np.zeros(n_components)

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]

        pls_fold = PLSRegression(n_components=n_components,
                                 scale=False,
                                 max_iter=200).fit(X_tr, Y_tr)

        W = pls_fold.x_weights_
        P = pls_fold.x_loadings_
        Q = pls_fold.y_loadings_
        X0 = pls_fold._x_mean
        Y0 = pls_fold._y_mean
        W_star = W @ np.linalg.inv(P.T @ W)

        Xc_te = X_te - X0
        Y_pred_prev = np.tile(Y0, (len(te_idx), 1))

        for a in range(n_components):
            ss_now = np.sum((Y_te - Y_pred_prev) ** 2)
            ss[a] += ss_now
            if np.isclose(ss_now, 0.0):
                break

            B_a = W_star[:, :a + 1] @ Q[:, :a + 1].T
            Y_hat = Xc_te @ B_a + Y0
            press[a] += np.sum((Y_te - Y_hat) ** 2)
            Y_pred_prev = Y_hat

    ratios = []
    for a in range(n_components):
        r = 1.0 if np.isclose(ss[a], 0.0) else press[a] / ss[a]
        r = max(r, -0.1)
        ratios.append(r)

    q2_cum = 1.0 - np.prod(ratios)
    return lv_scores, lv_loadings, r2x_cum, r2y_cum, q2_cum
