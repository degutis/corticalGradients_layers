import numpy as np
import nibabel as nib
from scipy.stats import f_oneway
from brainspace.datasets import load_conte69
from brainspace.null_models import SpinPermutations

# ---------------- helpers: read label.gii & map names -> Schaefer-7 index ----------------
def _load_label_gii(path):
    g = nib.load(path)
    labs = np.asarray(g.agg_data(), dtype=int).squeeze()
    lt = g.labeltable
    key_to_name = {lab.key: lab.label for lab in lt.labels}
    return labs, key_to_name

def _schaefer7_from_name(name: str) -> int:
    n = name.lower()
    if 'vis' in n: return 0
    if 'som' in n or 'sommot' in n: return 1
    if 'dorsattn' in n or ('dors' in n and 'attn' in n): return 2
    if 'ventattn' in n or 'salventattn' in n or ('vent' in n and 'attn' in n) or 'sal' in n: return 3
    if 'limbic' in n: return 4
    if 'cont' in n or 'control' in n or 'frontoparietal' in n or 'fp' in n: return 5
    if 'default' in n: return 6
    raise ValueError(f"Unrecognized Schaefer-7 network in label name: {name}")

RSN7_NAMES = ['Visual','Somatomotor','Dorsal Attn','Ventral/Salience','Limbic','Control','Default']

# ------------- build atlas bookkeeping: parcel order (uL,uR), per-vertex parcel ids, RSN per parcel -------------
def build_schaefer400_bookkeeping(schaefer_label_L, schaefer_label_R):
    L_lab, L_map = _load_label_gii(schaefer_label_L)
    R_lab, R_map = _load_label_gii(schaefer_label_R)

    # unique parcel keys > 0 (0 is medial wall)
    uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
    uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))

    # network index per parcel in uL/uR order
    nets0 = []
    for k in uL: nets0.append(_schaefer7_from_name(L_map[k]))
    for k in uR: nets0.append(_schaefer7_from_name(R_map[k]))
    networks0 = np.asarray(nets0, dtype=int)   # length 400

    # for fast vertex→parcel lookup
    # build list of vertex indices for each parcel (left, right)
    parcel_verts_L = [np.where(L_lab == k)[0] for k in uL]
    parcel_verts_R = [np.where(R_lab == k)[0] for k in uR]

    # medial wall masks (NaN during spin, as per BrainSpace tutorial)
    mw_L = (L_lab == 0)
    mw_R = (R_lab == 0)

    return (uL, uR, networks0, parcel_verts_L, parcel_verts_R, mw_L, mw_R)

# ------------- upsample parcel values (400,) to per-vertex arrays (32k per hemi) -------------
def parcel_to_vertices(D400, uL, uR, parcel_verts_L, parcel_verts_R, nL, nR, mw_L=None, mw_R=None):
    vL = np.full(nL, np.nan, float)
    vR = np.full(nR, np.nan, float)

    # assume D is ordered [uL, uR] (same as your plotting code). If not, reorder upstream.
    assert D400.shape[0] == (len(uL) + len(uR)), "D has wrong length for Schaefer-400."

    # fill vertices with their parcel's value
    for i, idxs in enumerate(parcel_verts_L):
        vL[idxs] = float(D400[i])
    for j, idxs in enumerate(parcel_verts_R):
        vR[idxs] = float(D400[len(uL) + j])

    # force medial wall to NaN
    if mw_L is not None: vL[mw_L] = np.nan
    if mw_R is not None: vR[mw_R] = np.nan
    return vL, vR

# ------------- fold a vertex map back to 400 parcels by mean within each parcel -------------
def vertices_to_parcels(v_full, nL, parcel_verts_L, parcel_verts_R, n_parc_L):
    vL = v_full[:nL]
    vR = v_full[nL:]
    out = np.empty(n_parc_L + len(parcel_verts_R), float)
    for i, idxs in enumerate(parcel_verts_L):
        out[i] = np.nanmean(vL[idxs])
    for j, idxs in enumerate(parcel_verts_R):
        out[n_parc_L + j] = np.nanmean(vR[idxs])
    return out

# ------------- compute ANOVA F across the 7 RSNs -------------
def anova_F_by_network(D400, networks0):
    groups = [D400[networks0 == k] for k in range(7)]
    F, _ = f_oneway(*groups)  # parametric p ignored; we’ll do spin-based p
    return float(F)

# ------------- main: spin-ANOVA for one layer -------------
def spin_anova_single_layer(D400, uL, uR, networks0, parcel_verts_L, parcel_verts_R, mw_L, mw_R,
                            n_perm=10000, random_state=0):
    # spheres (fsLR/Conte69 32k)
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)  # returns BrainSpace mesh objects
    nL = sphere_lh.n_points
    nR = sphere_rh.n_points

    # observed F on parcels
    F_obs = anova_F_by_network(D400, networks0)

    # upsample to vertices (so spins are spatially sensible)
    vL, vR = parcel_to_vertices(D400, uL, uR, parcel_verts_L, parcel_verts_R, nL, nR, mw_L, mw_R)

    # spin permutations
    sp = SpinPermutations(n_rep=n_perm, random_state=random_state)
    sp.fit(sphere_lh, points_rh=sphere_rh)
    # returns n_perm rotations for each hemi; stack to full-brain vertex maps (rows = perms)
    Vrot = np.hstack(sp.randomize(vL, vR))  # shape: (n_perm, nL+nR)

    # fold each rotation back to parcels, compute F
    F_perm = np.empty(n_perm, float)
    for i in range(n_perm):
        Dperm = vertices_to_parcels(Vrot[i], nL, parcel_verts_L, parcel_verts_R, len(uL))
        F_perm[i] = anova_F_by_network(Dperm, networks0)

    # spin-based p-value (upper tail because larger F = more between-group variance)
    p_spin = (np.sum(F_perm >= F_obs) + 1) / (n_perm + 1)

    # descriptive per-network means (observed)
    net_means = np.array([D400[networks0 == k].mean() for k in range(7)])
    net_ns    = np.array([np.sum(networks0 == k) for k in range(7)])

    return dict(F_obs=F_obs, p_spin=p_spin, net_means=net_means, net_ns=net_ns)

# ------------- convenience wrapper for your three layers -------------
def spin_anova_layers(D_deep, D_mid, D_sup,
                      schaefer_label_L="/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
                      schaefer_label_R="/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
                      n_perm=10000, random_state=0):
    uL, uR, networks0, parcel_verts_L, parcel_verts_R, mw_L, mw_R = \
        build_schaefer400_bookkeeping(schaefer_label_L, schaefer_label_R)

    results = {}
    for layer_name, D in [('Deep', D_deep), ('Middle', D_mid), ('Superficial', D_sup)]:
        D = np.asarray(D, float).squeeze()
        if D.shape[0] != (len(uL) + len(uR)):
            raise ValueError(f"{layer_name}: expected length {len(uL)+len(uR)} but got {D.shape[0]}. "
                             "Make sure D is ordered [uL, uR] (left parcels then right) as in your plotting code.")
        res = spin_anova_single_layer(D, uL, uR, networks0, parcel_verts_L, parcel_verts_R, mw_L, mw_R,
                                      n_perm=n_perm, random_state=random_state)
        results[layer_name] = res

    return results, RSN7_NAMES

# ---------------- example call ----------------
# results, rsn_names = spin_anova_layers(D_deep, D_mid, D_sup,
#                                        schaefer_label_L="/path/to/Schaefer400.L.label.gii",
#                                        schaefer_label_R="/path/to/Schaefer400.R.label.gii",
#                                        n_perm=10000, random_state=0)
# for lyr, r in results.items():
#     print(f"\n{lyr}: F={r['F_obs']:.3f}, p_spin={r['p_spin']:.4g}")
#     for k, (m, n) in enumerate(zip(r['net_means'], r['net_ns'])):
#         print(f"  {RSN7_NAMES[k]:>16s}: mean={m: .4f} (n={n})")


def save_spin_results_csv(results, rsn_names, out_csv):
    """
    Save layer-wise spin-ANOVA results to a CSV.

    Each row = one (layer, network).
    Columns: layer, network, net_mean, net_n, F_obs_layer, p_spin_layer
    (F_obs_layer and p_spin_layer repeat across networks for that layer.)
    """
    try:
        import pandas as pd
        rows = []
        for layer, r in results.items():
            F_obs = float(r["F_obs"])
            pval  = float(r["p_spin"])
            means = np.asarray(r["net_means"]).ravel()
            ns    = np.asarray(r["net_ns"]).ravel()
            for k, name in enumerate(rsn_names):
                rows.append({
                    "layer": layer,
                    "network": name,
                    "net_mean": float(means[k]),
                    "net_n": int(ns[k]),
                    "F_obs_layer": F_obs,
                    "p_spin_layer": pval
                })
        df = pd.DataFrame(rows, columns=["layer","network","net_mean","net_n","F_obs_layer","p_spin_layer"])
        df.to_csv(out_csv, index=False)
        print(f"Saved CSV to: {out_csv}")
    except ImportError:
        # Fallback without pandas
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer","network","net_mean","net_n","F_obs_layer","p_spin_layer"])
            for layer, r in results.items():
                F_obs = float(r["F_obs"])
                pval  = float(r["p_spin"])
                means = np.asarray(r["net_means"]).ravel()
                ns    = np.asarray(r["net_ns"]).ravel()
                for k, name in enumerate(rsn_names):
                    writer.writerow([layer, name, float(means[k]), int(ns[k]), F_obs, pval])
        print(f"Saved CSV to: {out_csv}")


def spin_network_layer_interaction(
    D_deep, D_mid, D_sup,
    schaefer_label_L="/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
    schaefer_label_R="/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
    n_perm=10000, random_state=0, batch_size=50
):
    """
    Spin-test for the Network × Layer interaction (3 layers × 7 networks).

    Returns:
      dict with:
        - F_int_obs: observed F for the 12-df interaction block
        - p_int_spin: spin-based p-value (upper-tail)
        - df1: 12 (=(3-1)*(7-1))
        - df2: residual df under full model
        - cell_means: (7,3) array of mean values per (network, layer)
        - net_ns: length-7 array (# parcels per network)
        - layer_names: ['Deep','Middle','Superficial']
    """
    # --- atlas bookkeeping (from your helpers) ---
    uL, uR, networks0, parcel_verts_L, parcel_verts_R, mw_L, mw_R = \
        build_schaefer400_bookkeeping(schaefer_label_L, schaefer_label_R)
    n_parc = networks0.size

    # spheres (fsLR/Conte69 32k)
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    nL, nR = sphere_lh.n_points, sphere_rh.n_points

    # stack layers
    Y_layers = [np.asarray(D, float).squeeze() for D in (D_deep, D_mid, D_sup)]
    for idx, D in enumerate(Y_layers):
        if D.shape[0] != n_parc:
            raise ValueError(f"Layer {idx} length {D.shape[0]} != {n_parc} parcels. "
                             "Make sure vectors are ordered [uL, uR].")

    # --- design & F for interaction (compare full vs no-interaction) ---
    def _F_interaction(y_layers):
        y = np.concatenate(y_layers, axis=0)     # length = 3N
        net = np.tile(networks0, 3)              # 0..6
        layer = np.repeat([0,1,2], n_parc)       # 0/1/2

        # Base model: intercept + Layer (2 df) + Network (6 df)
        X_intercept = np.ones((y.size, 1))
        X_layer = np.column_stack([(layer == 1).astype(float),
                                   (layer == 2).astype(float)])            # 2 dummies
        X_net = np.column_stack([(net == k).astype(float) for k in range(1,7)])  # 6 dummies
        X_red = np.column_stack([X_intercept, X_layer, X_net])

        # Full model adds Layer×Network interactions (2*6 = 12 df)
        X_int = []
        for l in (1, 2):
            for k in range(1, 7):
                X_int.append(((layer == l) & (net == k)).astype(float))
        X_int = np.column_stack(X_int)
        X_full = np.column_stack([X_red, X_int])

        # Least-squares fits and F
        beta_f, *_ = np.linalg.lstsq(X_full, y, rcond=None)
        rss_f = float(np.sum((y - X_full @ beta_f) ** 2))

        beta_r, *_ = np.linalg.lstsq(X_red, y, rcond=None)
        rss_r = float(np.sum((y - X_red @ beta_r) ** 2))

        df1 = X_full.shape[1] - X_red.shape[1]   # 12
        df2 = y.size - X_full.shape[1]
        F = ((rss_r - rss_f) / df1) / (rss_f / df2)
        return float(F), df1, int(df2)

    # observed interaction F
    F_int_obs, df1, df2 = _F_interaction(Y_layers)

    # precompute vertex maps for each layer (so we rotate values, not labels)
    vLs, vRs = [], []
    for D in Y_layers:
        vL, vR = parcel_to_vertices(D, uL, uR, parcel_verts_L, parcel_verts_R, nL, nR, mw_L, mw_R)
        vLs.append(vL); vRs.append(vR)

    # spin permutations (same rotations across layers per permutation), in batches
    rng = np.random.RandomState(random_state)
    F_perm = np.empty(n_perm, float)
    filled = 0
    while filled < n_perm:
        m = min(batch_size, n_perm - filled)
        sp = SpinPermutations(n_rep=m, random_state=int(rng.randint(1e9)))
        sp.fit(sphere_lh, points_rh=sphere_rh)

        # rotate each layer with the *same* set of spins
        rotL_list, rotR_list = [], []
        for vL, vR in zip(vLs, vRs):
            rL, rR = sp.randomize(vL, vR)  # shapes: (m, nL), (m, nR)
            rotL_list.append(rL); rotR_list.append(rR)

        # fold each perm back to parcels for all layers, compute F_int
        for i in range(m):
            Dperm_layers = []
            for rL, rR in zip(rotL_list, rotR_list):
                v_full = np.concatenate([rL[i], rR[i]])
                Dp = vertices_to_parcels(v_full, nL, parcel_verts_L, parcel_verts_R, len(uL))
                Dperm_layers.append(Dp)
            Fp, _, _ = _F_interaction(Dperm_layers)
            F_perm[filled + i] = Fp
        filled += m

    p_int_spin = (np.sum(F_perm >= F_int_obs) + 1) / (n_perm + 1)

    # cell means (7 networks × 3 layers) and counts
    means_by_net_by_layer = np.zeros((7, 3), float)
    for l, D in enumerate(Y_layers):
        for k in range(7):
            means_by_net_by_layer[k, l] = float(np.nanmean(D[networks0 == k]))
    net_ns = np.array([int(np.sum(networks0 == k)) for k in range(7)])

    return dict(
        F_int_obs=float(F_int_obs),
        p_int_spin=float(p_int_spin),
        df1=int(df1),
        df2=int(df2),
        cell_means=means_by_net_by_layer,  # shape (7, 3) in order RSN x [Deep, Middle, Superficial]
        net_ns=net_ns,
        layer_names=['Deep','Middle','Superficial']
    )