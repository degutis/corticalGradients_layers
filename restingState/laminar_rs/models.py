# laminar_rs/models.py
from __future__ import annotations
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_ff_fb_models(
        layers: List[np.ndarray],
        y_send: np.ndarray,
        y_recv: np.ndarray,
        out_dir,
        fname: str,
        robust_se: str = "HC3",
        fdr_alpha: float = 0.05,
        orthogonalize: bool = True,
        xlim: Tuple[float, float] = (-0.6, 0.6),
        dpi: int = 500,
):
    """
    Functional version of run_ff_fb_models.
    """
    import os

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    Sup, Mid, Deep = [np.asarray(v).ravel() for v in layers]
    y_send = np.asarray(y_send).ravel()
    y_recv = np.asarray(y_recv).ravel()
    M = np.column_stack([Sup, Mid, Deep, y_send, y_recv])
    valid = np.all(np.isfinite(M), axis=1)
    Sup, Mid, Deep, y_send, y_recv = [v[valid] for v in (Sup, Mid, Deep, y_send, y_recv)]

    Sup_z, Mid_z, Deep_z = [zscore(v, ddof=1) for v in (Sup, Mid, Deep)]

    C_FB = -0.5*Sup_z + 1.0*Mid_z - 0.5*Deep_z
    C_SD = 0.5*Sup_z + 0.0*Mid_z - 0.5*Deep_z

    if orthogonalize:
        proj = np.dot(C_SD, C_FB) / np.dot(C_FB, C_FB)
        C_SD = C_SD - proj * C_FB

    C_FB = zscore(C_FB, ddof=1)
    C_SD = zscore(C_SD, ddof=1)

    def fit_one(y):
        y_z = zscore(y, ddof=1)
        X = np.column_stack([C_FB, C_SD])
        X = sm.add_constant(X)
        res = sm.OLS(y_z, X).fit(cov_type=robust_se) if robust_se else sm.OLS(y_z, X).fit()
        betas = res.params[1:]
        tvals = res.tvalues[1:]
        pvals = res.pvalues[1:]
        df_res = res.df_resid
        pr = np.sign(tvals) * np.sqrt((tvals**2) / (tvals**2 + df_res))
        R2_full = res.rsquared
        dR2 = []
        for k in (0, 1):
            cols = [i for i in (0, 1) if i != k]
            X_red = np.column_stack([C_FB, C_SD])[:, cols]
            X_red = sm.add_constant(X_red)
            r2_red = sm.OLS(y_z, X_red).fit(cov_type=robust_se).rsquared if robust_se \
                else sm.OLS(y_z, X_red).fit().rsquared
            dR2.append(R2_full - r2_red)
        return betas, pr, np.asarray(pvals), np.asarray(dR2), R2_full

    b_send, pr_send, p_send, dR2_send, R2_send = fit_one(y_send)
    b_recv, pr_recv, p_recv, dR2_recv, R2_recv = fit_one(y_recv)
    y_diff = zscore(y_send, ddof=1) - zscore(y_recv, ddof=1)
    b_diff, pr_diff, p_diff, dR2_diff, R2_diff = fit_one(y_diff)

    rows = []
    for outcome, bs, prs, ps, dR2, R2 in [
        ("send", b_send, pr_send, p_send, dR2_send, R2_send),
        ("recv", b_recv, pr_recv, p_recv, dR2_recv, R2_recv),
        ("diff", b_diff, pr_diff, p_diff, dR2_diff, R2_diff),
    ]:
        rows += [
            {"outcome": outcome, "predictor": "FB", "beta_std": bs[0], "partial_r": prs[0], "p": ps[0],
             "unique_R2": dR2[0], "R2_full": R2},
            {"outcome": outcome, "predictor": "SD", "beta_std": bs[1], "partial_r": prs[1], "p": ps[1],
             "unique_R2": dR2[1], "R2_full": R2},
        ]
    df = pd.DataFrame(rows)

    df["p_FDR"] = multipletests(df["p"].values, alpha=fdr_alpha, method="fdr_bh")[1]
    df["sig_FDR"] = df["p_FDR"] < fdr_alpha

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True, sharey=True)
    outcomes = ["send", "recv", "diff"]
    preds = ["FB", "SD"]

    for j, outcome in enumerate(outcomes):
        ax = axes[j]
        sub = df[df["outcome"] == outcome].set_index("predictor").loc[preds].reset_index()
        y_pos = np.arange(len(preds))
        ax.barh(y_pos, sub["partial_r"].values)
        ax.axvline(0, lw=1)
        ax.set_xlim(*xlim)
        if j == 0:
            ax.set_yticks(y_pos, labels=preds)
        else:
            ax.set_yticks(y_pos, labels=["", ""])
        ax.set_title(outcome)
        ax.invert_yaxis()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        for i, row in sub.iterrows():
            mark = "★" if row["sig_FDR"] else ""
            ax.text(xlim[1], i,
                    f" {mark} p(FDR)={row['p_FDR']:.3g} · ΔR²={row['unique_R2']:.3f}",
                    va="center", ha="left", fontsize=8)

    fig.suptitle("Partial correlations of FF/FB contrasts with rDCM outcomes")
    fig.supxlabel("Partial correlation (unique effect)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig_path = f"{out_dir}/{fname}"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    base = os.path.splitext(fname)[0]
    csv_path = f"{out_dir}/{base}.csv"
    df.to_csv(csv_path, index=False)
    return df, fig_path, csv_path


def plot_horizontal_correlation_bar(
        layers: List[np.ndarray],
        gradient: np.ndarray,
        out_dir,
        fname: str,
        layer_names: Optional[List[str]] = None,
        title: str = "Effective connectivity vs. laminar indices",
        xlabel: str = "Association with send/receive gradient",
        xlim: Tuple[float, float] = (-0.8, 0.8),
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