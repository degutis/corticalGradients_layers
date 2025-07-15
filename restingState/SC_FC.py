import os
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
import matplotlib.pyplot as plt

def load_data(dir_func = '../../highRes_resting/derivatives/correlations/sub-50/smallGap/WithinLayer/FC_matrix.npy', dir_SC = '../../highRes_resting/derivatives/correlations/structuralMatrix/adjacency_matrix.npy'):

    sc = np.load(dir_SC)
    rs = np.load(dir_func)
    
    return sc, rs


def compute_normalized_laplacian(W: np.ndarray):
    """
    Compute symmetric normalized Laplacian L = I - D^{-1/2} W D^{-1/2}.
    Returns L and the symmetrically normalized weight matrix.
    """
    d = np.sum(W, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    W_symm = D_inv_sqrt @ W @ D_inv_sqrt
    L = np.eye(W.shape[0]) - W_symm
    return L, W_symm


def laplacian_eigendecomp(L: np.ndarray):
    """
    Perform eigendecomposition, sort ascending, return eigenvalues and eigenvectors.
    """
    lambdas, U = np.linalg.eigh(L)
    idx = np.argsort(lambdas)
    return lambdas[idx], U[:, idx]


def weighted_zero_crossings(U: np.ndarray, W: np.ndarray, threshold: float = 1.0):
    """
    Count weighted zero crossings per eigenvector: for each pair (i,j),
    if signs differ and W[i,j]>threshold, increment.
    Returns array of length n_harmonics.
    """
    n, k = U.shape
    wZC = np.zeros(k, dtype=int)
    for u in range(k):
        uu = U[:, u]
        count = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if uu[i] * uu[j] < 0 and W[i, j] > threshold:
                    count += 1
        wZC[u] = count
    return wZC


def project_and_compute_psd(U: np.ndarray, X_RS: np.ndarray):
    """
    Project z-scored fMRI signals onto structural harmonics, compute PSD:
      X_hat_L[:, :, s] = U^T @ zX_RS[:, :, s]
      PSD = mean(abs(X_hat_L)^2, axis=time)
    Returns PSD (n_harmonics x n_subjects).
    """
    # z-score each ROI timecourse along time axis
    zX = zscore(X_RS, axis=1, ddof=0)
    n_roi, T, n_subj = X_RS.shape
    X_hat = np.zeros((n_roi, T, n_subj))
    for s in range(n_subj):
        X_hat[:, :, s] = U.T @ zX[:, :, s]
    power = np.abs(X_hat) ** 2
    PSD = np.mean(power, axis=1)  # average across time
    return PSD


def plot_zero_crossings(wZC: np.ndarray):
    plt.figure()
    plt.plot(wZC)
    plt.title('Weighted Zero Crossings')
    plt.xlabel('Harmonic index')
    plt.ylabel('Weighted zero crossings')
    plt.show()


def plot_psd(lambdas: np.ndarray, PSD: np.ndarray):
    avg = np.mean(PSD, axis=1)
    std_dev = np.std(PSD, axis=1)
    lower = avg - std_dev
    upper = avg + std_dev
    valid = (np.max(PSD, axis=1) > 0) & (np.min(PSD, axis=1) > 0) & (avg > 0)
    plt.figure()
    plt.fill_between(lambdas[valid], lower[valid], upper[valid], alpha=0.3)
    plt.plot(lambdas, avg)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Harmonic frequency')
    plt.ylabel('Power')
    plt.title('Energy Spectral Density')
    plt.show()


def compute_cutoff_frequency(lambdas: np.ndarray, PSD: np.ndarray, percentile: float = 0.5) -> float:
    """
    Determine cutoff lambda where cumulative PSD reaches given percentile of total power.
    """
    mPSD = np.mean(PSD, axis=1)
    total_area = np.trapz(mPSD)
    cum = np.cumsum(mPSD)
    idx = np.searchsorted(cum, percentile * total_area)
    return lambdas[min(idx, len(lambdas) - 1)]


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    W, X_RS = load_data(base)
    L, W_symm = compute_normalized_laplacian(W)
    lambdas, U = laplacian_eigendecomp(L)
    wZC = weighted_zero_crossings(U, W)
    plot_zero_crossings(wZC)
    PSD = project_and_compute_psd(U, X_RS)
    plot_psd(lambdas, PSD)
    cutoff = compute_cutoff_frequency(lambdas, PSD)
    print(f'Cutoff frequency (lambda): {cutoff}')