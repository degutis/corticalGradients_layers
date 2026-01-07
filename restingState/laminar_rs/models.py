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
import laminar_rs.schaefer_stats as schaefer_stats


def run_ff_fb_models(
        layers: List[np.ndarray],
        y_send: np.ndarray,
        y_recv: np.ndarray,
        out_dir,
        fname: str,
        robust_se: str = "HC3",
        orthogonalize: bool = True,
        xlim: Tuple[float, float] = (-0.6, 0.6),
        dpi: int = 500,
        spin_n_perm: int = 0,
        spin_random_state: int = 0,
):
    """
    Fit FF/FB models and (optionally) run ENIGMA-style spin permutation tests.

    Parameters
    ----------
    layers : list of np.ndarray
        [Sup, Mid, Deep] laminar intra-regional indices (Schaefer-400 order).
    y_send, y_recv : np.ndarray
        rDCM efferent (send) and afferent (recv) strengths (Schaefer-400).
    out_dir : path-like
        Output directory for figure + CSV.
    fname : str
        Output figure filename (CSV will use same base name).
    robust_se : str or None
        Covariance type for statsmodels OLS (e.g., 'HC3'); if None, use classical.
    orthogonalize : bool
        Whether to orthogonalize C_SD w.r.t. C_FB.
    xlim : (float, float)
        X-axis limits for bar plots.
    dpi : int
        Figure DPI.
    spin_n_perm : int
        Number of ENIGMA-style spin permutations for ΔR² null (0 = no spins).
    spin_random_state : int
        RNG seed passed to ENIGMA permutation generator.

    Returns
    -------
    df : pandas.DataFrame
        Summary stats for each outcome × predictor, including parametric p,
        unique_R2, and (if spin_n_perm>0) spin-based p_spin.
    fig_path : str
        Path to the saved figure.
    csv_path : str
        Path to the saved CSV.
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    #  Prepare data and contrasts
    # -----------------------------
    Sup, Mid, Deep = [np.asarray(v).ravel() for v in layers]
    y_send = np.asarray(y_send).ravel()
    y_recv = np.asarray(y_recv).ravel()

    M = np.column_stack([Sup, Mid, Deep, y_send, y_recv])
    valid = np.all(np.isfinite(M), axis=1)
    Sup, Mid, Deep, y_send, y_recv = [v[valid] for v in (Sup, Mid, Deep, y_send, y_recv)]

    Sup_z, Mid_z, Deep_z = [zscore(v, ddof=1) for v in (Sup, Mid, Deep)]

    # FF/FB contrasts (Froudist-Walsh style)
    C_FB = -0.5 * Sup_z + 1.0 * Mid_z - 0.5 * Deep_z
    C_SD = 0.5 * Sup_z + 0.0 * Mid_z - 0.5 * Deep_z

    if orthogonalize:
        proj = np.dot(C_SD, C_FB) / np.dot(C_FB, C_FB)
        C_SD = C_SD - proj * C_FB

    C_FB = zscore(C_FB, ddof=1)
    C_SD = zscore(C_SD, ddof=1)

    def fit_one(y):
        y_z = zscore(y, ddof=1)
        X = np.column_stack([C_FB, C_SD])
        X = sm.add_constant(X)
        if robust_se:
            res = sm.OLS(y_z, X).fit(cov_type=robust_se)
        else:
            res = sm.OLS(y_z, X).fit()
        betas = res.params[1:]
        tvals = res.tvalues[1:]
        pvals = res.pvalues[1:]
        df_res = res.df_resid
        # Partial r from t and df
        pr = np.sign(tvals) * np.sqrt((tvals**2) / (tvals**2 + df_res))
        R2_full = res.rsquared
        dR2 = []
        for k in (0, 1):
            cols = [i for i in (0, 1) if i != k]
            X_red = np.column_stack([C_FB, C_SD])[:, cols]
            X_red = sm.add_constant(X_red)
            if robust_se:
                r2_red = sm.OLS(y_z, X_red).fit(cov_type=robust_se).rsquared
            else:
                r2_red = sm.OLS(y_z, X_red).fit().rsquared
            dR2.append(R2_full - r2_red)
        return betas, pr, np.asarray(pvals), np.asarray(dR2), R2_full

    # -----------------------------
    #  Fit models: send / recv / diff
    # -----------------------------
    b_send, pr_send, p_send, dR2_send, R2_send = fit_one(y_send)
    b_recv, pr_recv, p_recv, dR2_recv, R2_recv = fit_one(y_recv)
    y_diff = zscore(y_send, ddof=1) - zscore(y_recv, ddof=1)
    b_diff, pr_diff, p_diff, dR2_diff, R2_diff = fit_one(y_diff)

    # -----------------------------
    #  Assemble parametric results
    # -----------------------------
    rows = []
    for outcome, bs, prs, ps, dR2, R2 in [
        ("send", b_send, pr_send, p_send, dR2_send, R2_send),
        ("recv", b_recv, pr_recv, p_recv, dR2_recv, R2_recv),
        ("diff", b_diff, pr_diff, p_diff, dR2_diff, R2_diff),
    ]:
        rows += [
            {
                "outcome": outcome,
                "predictor": "FB",
                "beta_std": bs[0],
                "partial_r": prs[0],
                "p": ps[0],
                "unique_R2": dR2[0],
                "R2_full": R2,
            },
            {
                "outcome": outcome,
                "predictor": "SD",
                "beta_std": bs[1],
                "partial_r": prs[1],
                "p": ps[1],
                "unique_R2": dR2[1],
                "R2_full": R2,
            },
        ]
    df = pd.DataFrame(rows)

    # -----------------------------
    #  Optional ENIGMA spin test on ΔR²
    # -----------------------------
    if spin_n_perm > 0:
        # We assume Schaefer-400 and that all valid parcels are used.
        n_parc = Sup_z.size
        if n_parc != 400:
            raise ValueError(
                f"Spin FF/FB models currently assumes 400 parcels (Schaefer-400), got {n_parc}."
            )

        # Observed unique_R2 as 3×2 matrix: outcomes x predictors
        # outcomes: 0=send,1=recv,2=diff ; predictors: 0=FB,1=SD
        dR2_emp = np.vstack([dR2_send, dR2_recv, dR2_diff])  # shape (3,2)

        # Z-scored outcomes used in spins
        y_send_z = zscore(y_send, ddof=1)
        y_recv_z = zscore(y_recv, ddof=1)
        y_diff_z = zscore(y_diff, ddof=1)
        ys = [y_send_z, y_recv_z, y_diff_z]

        # ENIGMA-style permutations (parcel-level spins)
        perm_id = schaefer_stats._get_schaefer400_perm_id(
            n_perm=spin_n_perm,
            random_state=spin_random_state,
        )
        # perm_id: shape (400, spin_n_perm)

        # Counts of null exceedances for each outcome × predictor
        exceed_counts = np.zeros((3, 2), dtype=int)

        for j in range(spin_n_perm):
            idx = perm_id[:, j].astype(int)

            # Permute FF/FB contrasts
            C_FB_perm = C_FB[idx]
            C_SD_perm = C_SD[idx]

            # Full design with permuted predictors
            X_full_perm = np.column_stack([C_FB_perm, C_SD_perm])
            X_full_perm = sm.add_constant(X_full_perm, has_constant="add")

            # Reduced designs for unique_R2
            X_red_FBonly = sm.add_constant(C_FB_perm.reshape(-1, 1), has_constant="add")
            X_red_SDonly = sm.add_constant(C_SD_perm.reshape(-1, 1), has_constant="add")

            for out_idx, y_z in enumerate(ys):
                # Full model
                res_full = sm.OLS(y_z, X_full_perm).fit()
                R2_full_perm = res_full.rsquared

                # Unique R² for FB (drop SD)
                res_red_SD = sm.OLS(y_z, X_red_FBonly).fit()
                dR2_FB_perm = R2_full_perm - res_red_SD.rsquared
                if dR2_FB_perm >= dR2_emp[out_idx, 0]:
                    exceed_counts[out_idx, 0] += 1

                # Unique R² for SD (drop FB)
                res_red_FB = sm.OLS(y_z, X_red_SDonly).fit()
                dR2_SD_perm = R2_full_perm - res_red_FB.rsquared
                if dR2_SD_perm >= dR2_emp[out_idx, 1]:
                    exceed_counts[out_idx, 1] += 1

        p_spin_mat = (exceed_counts + 1) / (spin_n_perm + 1)

        # Map back into df rows
        p_spin_list = []
        for _, row in df.iterrows():
            out = row["outcome"]
            pred = row["predictor"]
            out_idx = {"send": 0, "recv": 1, "diff": 2}[out]
            pred_idx = {"FB": 0, "SD": 1}[pred]
            p_spin_list.append(p_spin_mat[out_idx, pred_idx])

        df["p_spin"] = np.asarray(p_spin_list, float)

    # -----------------------------
    #  Plot
    # -----------------------------
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
            txt = f" p={row['p']:.3g} · ΔR²={row['unique_R2']:.3f}"
            if "p_spin" in row and not np.isnan(row["p_spin"]):
                txt = f" p={row['p']:.3g} · p_spin={row['p_spin']:.3g} · ΔR²={row['unique_R2']:.3f}"
            ax.text(
                xlim[1],
                i,
                txt,
                va="center",
                ha="left",
                fontsize=8,
            )

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
        layer_names: Optional[List[str]] = ["Superficial", "Middle", "Deep"],
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