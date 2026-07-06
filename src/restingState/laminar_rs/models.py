# laminar_rs/models.py
from __future__ import annotations
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from typing import Dict, List, Optional, Tuple

from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


def plot_horizontal_correlation_bar_partial(
        layers: List[np.ndarray],
        gradient: np.ndarray,
        out_dir,
        fname: str,
        layer_names: Optional[List[str]] = None,
        title: str = "Effective connectivity vs. laminar indices",
        xlabel: str = "Association with send/receive gradient",
        xlim: Tuple[float, float] = (-1, 1),
        alpha: float = 0.05,
        robust_se: str = "HC3",
        do_fdr: bool = True,
        ridge_alpha: float = None,
):
    """
    Functional version of plot_horizontal_correlation_bar.
    """
    import os
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    X_list = [np.asarray(l).ravel() for l in layers]
    y = np.asarray(gradient).ravel()
    if any(len(l) != len(y) for l in X_list):
        raise ValueError("All layer vectors must have same length as gradient.")
    p = len(X_list)

    if layer_names is None:
            if p == 2:
                layer_names = ["Superficial", "Deep"]
            elif p == 3:
                layer_names = ["Superficial", "Middle", "Deep"]
            else:
                layer_names = [f"Layer {i+1}" for i in range(p)]
    elif len(layer_names) != p:
        raise ValueError(
            f"layer_names has {len(layer_names)} entries but {p} layers were given."
        )

    if layer_names is None:
        layer_names = [f"Layer {i+1}" for i in range(p)]

    X = np.column_stack(X_list)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[valid, :], y[valid]

    marg_r, marg_p = zip(*[pearsonr(y, X[:, i]) for i in range(p)])
    marg_r, marg_p = np.asarray(marg_r), np.asarray(marg_p)

    y_z = zscore(y, ddof=1)
    X_z = zscore(X, axis=0, ddof=1)
    vifs = [variance_inflation_factor(X_z, i) for i in range(p)]

    X_fit = sm.add_constant(X_z)
    ols = sm.OLS(y_z, X_fit)
    res = ols.fit(cov_type=robust_se) if robust_se else ols.fit()
    betas = res.params[1:]
    tvals = res.tvalues[1:]
    pvals = res.pvalues[1:]
    df_resid = int(res.df_resid)
    R2_full = float(res.rsquared)

    partial_r = np.sign(tvals) * np.sqrt((tvals**2) / (tvals**2 + df_resid))

    unique_r2 = []
    for k in range(p):
        cols = [i for i in range(p) if i != k]
        res_red = sm.OLS(y_z, sm.add_constant(X_z[:, cols])).fit(cov_type=robust_se) if robust_se \
            else sm.OLS(y_z, sm.add_constant(X_z[:, cols])).fit()
        unique_r2.append(R2_full - float(res_red.rsquared))
    unique_r2 = np.asarray(unique_r2)

    ridge_info = {}
    if ridge_alpha is not None:
        lam = float(ridge_alpha)
        XtX = X_z.T @ X_z
        beta_ridge = np.linalg.solve(XtX + lam * np.eye(p), X_z.T @ y_z)
        ridge_info = {"ridge_alpha": lam, "beta_ridge_std": beta_ridge}

    if do_fdr:
        rej, p_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    else:
        rej, p_fdr = np.array([p < alpha for p in pvals]), pvals

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y_pos = np.arange(p)
    ax.barh(y_pos, partial_r)
    ax.plot(marg_r, y_pos, "o", markersize=5)
    ax.axvline(0, lw=1)
    ax.set_yticks(y_pos, labels=layer_names)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for i, (pf, ur2, sig) in enumerate(zip(p_fdr, unique_r2, rej)):
        mark = "★" if sig else ""
        ax.text(xlim[1], i,
                f" {mark} p(FDR)={pf:.3g} · ΔR²={ur2:.3f}",
                va="center", ha="left", fontsize=8)

    fig.tight_layout()
    path = f"{out_dir}/{fname}"
    fig.savefig(path, dpi=500, bbox_inches="tight")

    df = pd.DataFrame({
        "predictor": layer_names,
        "marginal_r": marg_r,
        "marginal_p": marg_p,
        "beta_std": betas,
        "partial_r": partial_r,
        "p": pvals,
        "p_fdr": p_fdr,
        "sig_fdr": rej,
        "unique_R2": unique_r2,
        "VIF": vifs,
        "R2_full_model": R2_full,
        "df_resid": df_resid,
    })
    if ridge_info:
        for i, name in enumerate(layer_names):
            df.loc[i, "beta_ridge_std"] = ridge_info["beta_ridge_std"][i]
        df.attrs.update(ridge_info)

    csv_path = f"{out_dir}/{fname.rsplit('.', 1)[0]}.csv"
    df.to_csv(csv_path, index=False)

    print(df)
    if any(v > 5 for v in vifs):
        print(f"[warn] High collinearity (VIF>5): {vifs}. Consider ridge_alpha=1.0.")

    return df, res


def plot_horizontal_correlation_bar(
        layers: List[np.ndarray],
        gradient: np.ndarray,
        out_dir,
        fname: str,
        layer_names: Optional[List[str]] = ["Superficial", "Middle", "Deep"],
        title: str = "Laminar indices vs. gradient",
        xlabel: str = "Pearson r",
        xlim: Tuple[float, float] = (-1, 1),
        alpha: float = 0.05,
        do_fdr: bool = True,
):
    """
    Plot per-layer marginal Pearson correlation between each laminar index
    and a target gradient.
    """
    import os
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    X_list = [np.asarray(l).ravel() for l in layers]
    y = np.asarray(gradient).ravel()
    if any(len(l) != len(y) for l in X_list):
        raise ValueError("All layer vectors must have same length as gradient.")
    p = len(X_list)
    if layer_names is None:
        layer_names = [f"Layer {i+1}" for i in range(p)]

    X = np.column_stack(X_list)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[valid, :], y[valid]

    rs, ps = zip(*[pearsonr(y, X[:, i]) for i in range(p)])
    rs, ps = np.asarray(rs), np.asarray(ps)

    if do_fdr:
        rej, p_adj, _, _ = multipletests(ps, alpha=alpha, method="fdr_bh")
    else:
        rej, p_adj = ps < alpha, ps

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y_pos = np.arange(p)
    ax.barh(y_pos, rs)
    ax.axvline(0, lw=1)
    ax.set_yticks(y_pos, labels=layer_names)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    p_label = "p(FDR)" if do_fdr else "p"
    for i, (pa, sig) in enumerate(zip(p_adj, rej)):
        mark = "★" if sig else ""
        ax.text(xlim[1], i, f" {mark} {p_label}={pa:.3g}",
                va="center", ha="left", fontsize=8)

    fig.tight_layout()
    path = f"{out_dir}/{fname}"
    fig.savefig(path, dpi=500, bbox_inches="tight")

    df = pd.DataFrame({
        "predictor": layer_names,
        "r": rs,
        "p": ps,
        "p_adj": p_adj,
        "sig": rej,
        "n": int(valid.sum()),
    })

    csv_path = f"{out_dir}/{fname.rsplit('.', 1)[0]}.csv"
    df.to_csv(csv_path, index=False)

    print(df)
    return df