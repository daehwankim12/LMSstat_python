import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.pyplot import colormaps
from plotnine import geom_segment, aes, annotate
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def scaling(data, method="auto"):
    if method not in ("auto", "pareto"):
        raise ValueError("Invalid scaling method.")
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

    return pd.concat([data[["Sample", "Group"]], scaled_data], axis=1)


def pca(data, n_components=2, scale=True, cv_splits=7, random_state=42):
    """
    PCA 수행 및 교차검증 Q² 계산 함수

    Parameters
    ----------
    data : DataFrame - 첫 열 Sample, 두 번째 Group, 나머지 수치형 변수
    n_components : int - 최대 주성분 수
    scale : bool - True면 데이터 스케일링 수행
    cv_splits : int - 교차검증 분할 수 (2 이상, 샘플 수 이하)
    random_state : int - 랜덤 시드

    Returns
    -------
    pc_scores : DataFrame - 주성분 점수
    pc_loadings : DataFrame - 주성분 적재값
    pc_r2 : float - 설명된 분산 비율(R²)
    pc_q2 : float - 교차검증 예측력(Q²)
    """
    # 데이터 전처리
    if scale:
        data = scaling(data, method="auto")

    data = data.rename(columns={data.columns[0]: "Sample", data.columns[1]: "Group"})
    X_df = data.drop(columns=["Sample", "Group"])
    X = X_df.to_numpy(float)

    # 전체 데이터에 대한 PCA 수행
    pc = PCA(n_components=n_components).fit(X)

    # 결과 데이터프레임으로 변환
    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    pc_scores = pd.DataFrame(pc.transform(X), columns=pc_cols)
    pc_loadings = pd.DataFrame(pc.components_.T, index=X_df.columns, columns=pc_cols)
    pc_r2 = pc.explained_variance_ratio_.sum()

    # 교차검증 Q² 계산
    n_samples = X.shape[0]
    cv_splits = min(max(2, cv_splits), n_samples)  # 분할 수 범위 제한
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # 각 성분별 PRESS와 SS 저장 배열
    press_contrib = np.zeros(n_components)
    ss_contrib = np.zeros(n_components)

    # 교차검증 수행
    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]

        # 훈련/테스트 데이터 중심화
        mu_fold = X_tr.mean(axis=0)
        R_tr = X_tr - mu_fold
        R_te = X_te - mu_fold

        # 각 성분에 대해 반복
        for a in range(n_components):
            # 현재 성분 추출 전 테스트 잔차 제곱합
            ss_now = np.sum(R_te ** 2)
            ss_contrib[a] += ss_now

            # 분산이 없으면 다음 폴드로 진행
            if np.isclose(ss_now, 0.0):
                R_tr.fill(0.0)
                R_te.fill(0.0)
                break

            # 현재 훈련 잔차에 대해 1-성분 PCA 수행
            if np.isclose(np.sum(R_tr ** 2), 0.0):
                P_a = np.zeros((1, R_tr.shape[1]))
            else:
                pca_one = PCA(n_components=1, random_state=random_state).fit(R_tr)
                P_a = pca_one.components_

            # 테스트 잔차 예측 및 PRESS 계산
            T_te = R_te @ P_a.T
            R_te_hat = T_te @ P_a
            press_contrib[a] += np.sum((R_te - R_te_hat) ** 2)

            # 다음 성분을 위한 잔차 업데이트
            T_tr = R_tr @ P_a.T
            R_tr -= T_tr @ P_a
            R_te -= R_te_hat

    # PRESS/SS 비율 계산 및 Q² 도출
    ratios = []
    for a in range(n_components):
        ss_a = ss_contrib[a]
        pr_a = press_contrib[a]
        ratio = 1.0 if np.isclose(ss_a, 0.0) else pr_a / ss_a
        ratio = max(ratio, -0.1)  # 하한값 설정
        ratios.append(ratio)

    pc_q2 = 1.0 - np.prod(ratios)

    return pc_scores, pc_loadings, pc_r2, pc_q2


def _pal(n: int):
    cmap = colormaps.get_cmap("tab20")
    return [to_hex(cmap(i % cmap.N)) for i in range(n)]


# ───────── 브래킷 + 별 ─────────
# ───────────────────────── 1) _annot (offset=0.05)
def _annot(g, st, order, y_top,
           offset=.05, step=.05, tip=.01,
           star=10, line=.15):
    # p≤.05 로 필터
    st = st.loc[st.p_value <= .05]
    if st.empty or not {'group1', 'group2', 'p_value'}.issubset(st.columns):
        return g

    level = 0
    for _, r in st.iterrows():
        if r.group1 not in order or r.group2 not in order:
            continue
        x1, x2 = order.index(r.group1) + 1, order.index(r.group2) + 1
        y = y_top * (1 + offset + step * level)  # ← 5 % 마진
        y2 = y - tip * y_top
        s = '**' if r.p_value <= .01 else '*'
        level += 1

        g += geom_segment(aes(x=x1, xend=x2, y=y, yend=y), size=line)
        g += geom_segment(aes(x=x1, xend=x1, y=y, yend=y2), size=line)
        g += geom_segment(aes(x=x2, xend=x2, y=y, yend=y2), size=line)
        g += annotate('text', x=(x1 + x2) / 2, y=y, label=s,
                      size=star, ha='center', va='bottom')
    return g


def melting(data):
    data = data.set_index("Sample")
    data = data.melt(var_name="variable", value_name="value")

    return data


def plsda(data: pd.DataFrame,
          n_components: int = 2,
          scale: bool = True,
          cv_splits: int = 7,
          random_state: int = 42):
    """
    PLS-DA + R2X, R2Y, Q2 계산
    ---------------------------------
    첫 열 : Sample  │  두 번째 : Group  │  나머지 : 수치 변수
    """
    # 0) 전처리 -------------------------------------------------------
    if scale:
        data = scaling(data, "auto")
    data = data.rename(columns={data.columns[0]: "Sample",
                                data.columns[1]: "Group"})
    X_df = data.drop(columns=["Sample", "Group"])
    X = X_df.to_numpy(float)
    ylab = data["Group"].astype(str)
    Y_df = pd.get_dummies(ylab)  # one-hot
    Y = Y_df.to_numpy(float)

    # 1) 전체 데이터 PLS-DA ------------------------------------------
    pls = PLSRegression(n_components=n_components,
                        scale=False, max_iter=200).fit(X, Y)

    lv_cols = [f"LV{i + 1}" for i in range(n_components)]
    lv_scores = pd.DataFrame(pls.x_scores_, columns=lv_cols)
    lv_loadings = pd.DataFrame(pls.x_loadings_,
                               index=X_df.columns, columns=lv_cols)

    # -- R2Y(cum) : scikit-learn score ------------------------------
    r2y_cum = pls.score(X, Y)

    # -- R2X(cum) : 재구성 기반 -------------------------------------
    X_mean = pls._x_mean
    Xc = X - X_mean
    X_hat = pls.x_scores_ @ pls.x_loadings_.T  # (n, p)
    sse_x = np.sum((Xc - X_hat) ** 2)
    tss_x = np.sum(Xc ** 2)
    r2x_cum = 0.0 if np.isclose(tss_x, 0) else 1 - sse_x / tss_x

    # 2) Q²(cum) -----------------------------------------------------
    n_samples = X.shape[0]
    cv_splits = min(max(2, cv_splits), n_samples)
    kf = KFold(n_splits=cv_splits, shuffle=True,
               random_state=random_state)

    press = np.zeros(n_components)
    ss = np.zeros(n_components)

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]

        pls_fold = PLSRegression(n_components=n_components,
                                 scale=False, max_iter=200).fit(X_tr, Y_tr)

        # 재사용 행렬
        W = pls_fold.x_weights_
        P = pls_fold.x_loadings_
        Q = pls_fold.y_loadings_
        X0 = pls_fold._x_mean
        Y0 = pls_fold._y_mean
        Wstar = W @ np.linalg.inv(P.T @ W)

        Xc_te = X_te - X0
        Y_pred_prev = np.tile(Y0, (len(te_idx), 1))

        for a in range(n_components):
            # SS_{a-1}
            ss_now = np.sum((Y_te - Y_pred_prev) ** 2)
            ss[a] += ss_now
            if np.isclose(ss_now, 0):
                break

            B_a = Wstar[:, :a + 1] @ Q[:, :a + 1].T
            Y_hat = Xc_te @ B_a + Y0
            press[a] += np.sum((Y_te - Y_hat) ** 2)
            Y_pred_prev = Y_hat

    ratios = []
    for a in range(n_components):
        r = 1.0 if np.isclose(ss[a], 0) else press[a] / ss[a]
        r = max(r, -0.1)  # truncate
        ratios.append(r)
    q2_cum = 1.0 - np.prod(ratios)

    return lv_scores, lv_loadings, r2x_cum, r2y_cum, q2_cum
