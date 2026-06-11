# laminar_rs/graph_metrics.py
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

import hcp_utils as hcp


# ---------- rich club, most common members ----------

def rich_club_sweep(connectivity_matrix: np.ndarray,
                    deg_cutoff_percentile: float = 95,
                    normalized: bool = True,
                    seed: int = 33) -> Tuple[float, List[int]]:
    G = nx.from_numpy_array(connectivity_matrix)
    phi_raw = nx.rich_club_coefficient(G, normalized=False)

    if normalized:
        phi_rand = nx.rich_club_coefficient(G, normalized=False, seed=seed)
        phi = {}
        for k, v in phi_raw.items():
            denom = phi_rand.get(k, 0.0)
            phi[k] = (v / denom) if denom > 0 else np.nan
    else:
        phi = phi_raw

    ks = np.array(sorted(phi))
    phis = np.array([phi[k] for k in ks])
    valid = ~np.isnan(phis)
    phi_auc = np.trapz(phis[valid], x=ks[valid])

    degrees = np.array([d for _, d in G.degree()])
    deg_cut = np.percentile(degrees, deg_cutoff_percentile)
    rich_club_nodes = [n for n, d in G.degree() if d >= deg_cut]

    return phi_auc, rich_club_nodes


def most_common_members(members_list: List[np.ndarray],
                        N: int,
                        min_frac: float = 0.8):
    """
    Functional version of most_common_members.
    """
    T = len(members_list)
    membership = np.zeros((N, T), dtype=int)
    for t, mids in enumerate(members_list):
        membership[mids, t] = 1
    freq = membership.sum(axis=1) / T
    stable = np.where(freq >= min_frac)[0]
    return stable, freq


# ---------- connectograms ----------

def get_top_percent_edges(mat: np.ndarray, percent: float = 20) -> List[Tuple[int, int]]:
    """
    Functional version of get_top_percent_edges.
    """
    assert 0 < percent <= 100, "Percent must be between 0 and 100."
    triu_indices = np.triu_indices_from(mat, k=1)
    edge_weights = mat[triu_indices]
    num_edges = len(edge_weights)
    k = int(np.ceil(num_edges * percent / 100.0))
    if k == 0:
        return []

    top_k_indices = np.argpartition(edge_weights, -k)[-k:]
    edges = [(triu_indices[0][i], triu_indices[1][i]) for i in top_k_indices]
    return edges


def _plot_edges(fig, edges, pos, color, name):
    edge_x, edge_y = [], []
    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.1, color=color),
        mode="lines",
        name=name,
        hoverinfo="none"
    ))


def plot_connectogram(connectivity_matrix: np.ndarray,
                      out_dir,
                      name: str,
                      layer: str,
                      color: str = "red",
                      n: int = 360,
                      percent: float = 20):
    """
    Functional version of plotConnectogram.
    """
    import plotly.graph_objects as go
    import os

    out_dir = str(out_dir)
    os.makedirs(f"{out_dir}/{name}/Connectogram", exist_ok=True)

    labels = hcp.mmp.labels
    G = nx.Graph()
    G.add_nodes_from(range(n))
    pos = nx.circular_layout(G)

    edges = get_top_percent_edges(connectivity_matrix, percent=percent)
    fig = go.Figure()
    _plot_edges(fig, edges, pos, color=color, name=layer)

    for node, (x, y) in pos.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            text=labels[node],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=1, color="gray"),
            showlegend=False
        ))

    fig.update_layout(
        width=1200,
        height=1200,
        showlegend=True,
        title="Multi-layer Connectogram",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.write_image(f"{out_dir}/{name}/Connectogram/Crossings_{layer}.png")


def plot_connectogram_allInOne(layer1: np.ndarray,
                               layer2: np.ndarray,
                               layer3: np.ndarray,
                               out_dir,
                               name: str,
                               percent: float = 20,
                               n: int = 360):
    """
    Functional version of plotConnectogram_allInOne.
    """
    import plotly.graph_objects as go
    import os

    out_dir = str(out_dir)
    os.makedirs(f"{out_dir}/{name}/Connectogram", exist_ok=True)

    labels = hcp.mmp.labels
    G = nx.Graph()
    G.add_nodes_from(range(n))
    pos = nx.circular_layout(G)

    edges1 = get_top_percent_edges(layer1, percent=percent)
    edges2 = get_top_percent_edges(layer2, percent=percent)
    edges3 = get_top_percent_edges(layer3, percent=percent)

    fig = go.Figure()
    _plot_edges(fig, edges1, pos, "red", "Superficial")
    _plot_edges(fig, edges2, pos, "green", "Middle")
    _plot_edges(fig, edges3, pos, "blue", "Deep")

    for node, (x, y) in pos.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            text=labels[node],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=1, color="gray"),
            showlegend=False
        ))

    fig.update_layout(
        width=1200,
        height=1200,
        showlegend=True,
        title="Multi-layer Connectogram",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.write_image(f"{out_dir}/{name}/Connectogram/Crossings_All.png")


# ---------- degree distribution & modularity ----------

def run_degree_distribution(M: np.ndarray, out_dir, name: str, layer_name: str):
    """
    Functional version of runDegreeDistribution.
    """
    import os
    out_dir = str(out_dir)
    os.makedirs(f"{out_dir}/{name}", exist_ok=True)

    G = nx.from_numpy_array(M)
    hist = nx.degree_histogram(G)
    counts = np.array(hist)

    N = G.number_of_nodes()
    p_k = counts / N
    F = np.cumsum(p_k)
    ccdf = 1 - F
    k = np.arange(len(ccdf)-1)

    comps = nx.number_connected_components(G)
    print(f"{comps} connected component(s)")

    plt.figure(figsize=(6, 4))
    plt.step(k, ccdf[:-1], where="post", marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree $k$ (log scale)")
    plt.ylabel(r"$1 - F(k) = P(K > k)$ (log scale)")
    plt.title("Node‐Degree CCDF (NetworkX)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}/DegreePlot_{layer_name}.png", bbox_inches="tight")
    plt.close()


def modularity(A: np.ndarray, network_assignments_path: str = "cortex_parcel_network_assignments.txt") -> float:
    """Functional version of modularity()."""
    labels = np.loadtxt(network_assignments_path, dtype=int)
    k = A.sum(axis=1)
    m2 = k.sum()
    Q = 0.0

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        lc = A[np.ix_(idx, idx)].sum()
        kc = k[idx].sum()
        Q += (lc / m2) - (kc / m2) ** 2

    return Q


# ---------- eigenvector centrality ----------

def eigenvector_centrality_calc(adj_matrix: np.ndarray,
                                weight=None):
    """
    Functional version of eigenvector_centrality_calc.
    Returns:
        centrality_arr : shape (N_nodes,)
        one_hot_centrality : shape (360,3) with 1 at layer with max centrality.
    """
    G = nx.from_numpy_array(adj_matrix)
    centrality = nx.eigenvector_centrality(G, max_iter=1000, weight=weight)
    centrality_arr = np.array([centrality[i] for i in range(G.number_of_nodes())])

    mat = centrality_arr.reshape(360, 3)
    one_hot_centrality = np.zeros_like(mat, dtype=int)
    idx_max = np.argmax(mat, axis=1)
    one_hot_centrality[np.arange(360), idx_max] = 1
    return centrality_arr, one_hot_centrality


def eigenvector_centrality_plot(
        centrality: np.ndarray,
        one_hot_centrality: np.ndarray,
        out_dir,
        name: str,
        additionalName: str = "",
):
    """
    Functional version of eigenvector_centrality_plot.
    """
    import os
    out_dir = str(out_dir)
    os.makedirs(f"{out_dir}/{name}", exist_ok=True)

    cats = np.loadtxt("cortex_parcel_network_assignments.txt", dtype=int)
    subs = centrality.shape[-1]

    counts = np.zeros((12, 3, subs), dtype=int)
    averages = np.zeros((12, 3, subs), dtype=float)

    for s in range(subs):
        mat = centrality[:, s].reshape(360, 3)
        curr_one_hot = one_hot_centrality[:, :, s]
        for k in range(1, 13):
            mask = (cats == k)
            counts[k-1, :, s] = curr_one_hot[mask, :].sum(axis=0)
            data = mat[mask, :]
            averages[k-1, :, s] = data.mean(axis=0)

    tick_labels = [
        "Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular",
        "Dorsal-Attentional", "Language", "Frontoparietal", "Auditory",
        "Default", "Posterior-Multimodal", "Ventral-Multimodal", "Orbito-Affective"
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharey=True)
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        x = np.arange(1, 4)
        heights = np.mean(counts[idx, :, :], axis=-1)
        errors = np.std(counts[idx, :, :], axis=-1)/np.sqrt(subs)

        stat, pval = f_oneway(
            counts[idx, 0, :],
            counts[idx, 1, :],
            counts[idx, 2, :]
        )

        bars = ax.bar(
            x, heights,
            yerr=errors,
            capsize=5,
            edgecolor="black"
        )

        sig = "*" if pval < 0.05 else ""
        ax.text(
            0.5, 0.95,
            f"p = {pval:.3f}{sig}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
        )

        ax.set_title(tick_labels[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(["Deep", "Middle", "Superficial"])
        ax.set_ylabel("Parcel count - eigenvector centrality \n(± SEM)")

        for i, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + errors[i] + 0.01*h,
                f"{h:.4f}",
                ha="center",
                va="bottom"
            )

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}/EigvecsBelongingToEachRSN_EigCentrality_AcrossSubs{additionalName}.png",
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharey=True)
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        x = np.arange(1, 4)
        heights = np.mean(averages[idx, :, :], axis=-1)
        errors = np.std(averages[idx, :, :], axis=-1)/np.sqrt(subs)

        stat, pval = f_oneway(
            averages[idx, 0, :],
            averages[idx, 1, :],
            averages[idx, 2, :]
        )

        bars = ax.bar(
            x, heights,
            yerr=errors,
            capsize=5,
            edgecolor="black"
        )

        sig = "*" if pval < 0.05 else ""
        ax.text(
            0.5, 0.95,
            f"p = {pval:.3f}{sig}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
        )

        ax.set_title(tick_labels[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(["Deep", "Middle", "Superficial"])
        ax.set_ylabel("Mean Centrality\n(± SEM)")

        for i, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + errors[i] + 0.01*h,
                f"{h:.4f}",
                ha="center",
                va="bottom"
            )

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}/EigvecsBelongingToEachRSN_EigCentrality_MeanSEM_AcrossSubs{additionalName}.png",
                bbox_inches="tight")
    plt.close(fig)


def eigenvector_centrality_plot_avg(
        centrality: np.ndarray,
        one_hot_centrality: np.ndarray,
        out_dir,
        name: str,
        additionalName: str = "",
):
    """
    Functional version of eigenvector_centrality_plot_avg.
    """
    import os
    out_dir = str(out_dir)
    os.makedirs(f"{out_dir}/{name}", exist_ok=True)

    cats = np.loadtxt("cortex_parcel_network_assignments.txt", dtype=int)

    counts = np.zeros((12, 3), dtype=int)
    averages = np.zeros((12, 3), dtype=float)
    sem = np.zeros((12, 3), dtype=float)

    mat = centrality.reshape(360, 3)
    for k in range(1, 13):
        mask = (cats == k)
        counts[k-1, :] = one_hot_centrality[mask, :].sum(axis=0)
        data = mat[mask, :]
        averages[k-1, :] = data.mean(axis=0)
        sem[k-1, :] = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])

    tick_labels = [
        "Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular",
        "Dorsal-Attentional", "Language", "Frontoparietal", "Auditory",
        "Default", "Posterior-Multimodal", "Ventral-Multimodal", "Orbito-Affective"
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharey=True)
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        x = np.arange(1, 4)
        heights = counts[idx, :]
        bars = ax.bar(
            x, heights,
            capsize=5,
            edgecolor="black"
        )

        ax.set_title(tick_labels[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(["Deep", "Middle", "Superficial"])
        ax.set_ylabel("Parcel count - eigenvector centrality\n(± SEM)")

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01*h,
                f"{h:.4f}",
                ha="center",
                va="bottom"
            )

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}/EigvecsBelongingToEachRSN_EigCentrality_AverageAdj{additionalName}.png",
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharey=True)
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        x = np.arange(1, 4)
        heights = averages[idx, :]
        errors = sem[idx, :]

        bars = ax.bar(
            x, heights,
            yerr=errors,
            capsize=5,
            edgecolor="black"
        )

        ax.set_title(tick_labels[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(["Deep", "Middle", "Superficial"])
        ax.set_ylabel("Mean Centrality\n(± SEM)")

        for i, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + errors[i] + 0.01*h,
                f"{h:.4f}",
                ha="center",
                va="bottom"
            )

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}/EigvecsBelongingToEachRSN_EigCentrality_Count_AverageAdj{additionalName}.png",
                bbox_inches="tight")
    plt.close(fig)