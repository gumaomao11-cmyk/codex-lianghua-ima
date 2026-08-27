# -*- coding: utf-8 -*-
"""因子正交化：剔除传统量价 Beta，保留纯文本 Alpha。"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def orthogonalize_gs(target, controls):
    """
    Gram-Schmidt 正交化：用线性回归剔除 controls 对 target 的解释部分，返回残差。
    target: Series
    controls: DataFrame（与 target 同索引）
    返回：残差 Series（均值为 0）
    """
    y = target.copy()
    X = controls.copy()
    # 去掉全空行
    valid = y.notna() & X.notna().all(axis=1)
    y_valid = y[valid]
    X_valid = X[valid]
    if len(y_valid) < 10 or X_valid.shape[1] == 0:
        return pd.Series(np.nan, index=target.index)
    model = LinearRegression().fit(X_valid, y_valid)
    pred = pd.Series(np.nan, index=target.index)
    pred[valid] = model.predict(X_valid)
    resid = y - pred
    return resid


def orthogonalize_pca(factors, n_components=5):
    """
    PCA 正交化：返回前 n 个主成分得分。
    factors: DataFrame，列为原始因子
    """
    factors = factors.dropna()
    if len(factors) < 20:
        return pd.DataFrame(index=factors.index)
    pca = PCA(n_components=min(n_components, factors.shape[1]))
    scores = pca.fit_transform(factors)
    cols = [f"pca_{i+1}" for i in range(scores.shape[1])]
    return pd.DataFrame(scores, index=factors.index, columns=cols)


def orthogonalize_factors(df, text_cols, control_cols, method="gs"):
    """
    对 DataFrame 中的多个文本因子做正交化。
    df: 包含 date, ticker 和因子列的 DataFrame
    text_cols: 需要正交化的文本因子列名
    control_cols: 控制变量列名（传统量价因子）
    method: 'gs' 或 'pca'
    返回：添加了 *_ortho 列的 DataFrame
    """
    df = df.copy()
    if method == "pca":
        pca_df = orthogonalize_pca(df[text_cols + control_cols], n_components=min(5, len(text_cols)))
        for c in pca_df.columns:
            df[c] = pca_df[c].reindex(df.index)
        return df

    # Gram-Schmidt
    out_cols = []
    for tcol in text_cols:
        ocol = f"{tcol}_ortho"
        out_cols.append(ocol)
        resids = []
        for tk, g in df.groupby("ticker"):
            target = g[tcol]
            controls = g[control_cols] if all(c in g.columns for c in control_cols) else pd.DataFrame(index=g.index)
            resid = orthogonalize_gs(target, controls)
            resid.name = ocol
            resids.append(resid)
        df[ocol] = pd.concat(resids).reindex(df.index)
    return df
