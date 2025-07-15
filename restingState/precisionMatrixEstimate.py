import numpy as np
from numpy.linalg import inv

def regularized_precision(corr, gamma):
    """
    Compute the Tikhonov-regularized precision matrix:
        (corr + γ I)^{-1}

    Parameters
    ----------
    corr : array-like, shape (p, p)
        A symmetric positive-semidefinite correlation matrix.
    gamma : float
        Regularization parameter γ ≥ 0.

    Returns
    -------
    precision : ndarray, shape (p, p)
        The regularized precision matrix.
    """
    p = corr.shape[0]
    return inv(corr + gamma * np.eye(p))


# --- Convenience wrapper if you want a named function ---
def precision_from_corr_matrix(corr_matrix, gamma):
    """
    Given a correlation matrix and γ, return its regularized precision.
    """
    return regularized_precision(corr_matrix, gamma)










import numpy as np
from numpy.linalg import inv, eigvals
from scipy.linalg import eigh

def compute_correlation(X):
    """
    Compute the sample Pearson correlation matrix of shape (p, p)
    from data X of shape (n_samples, p_features).
    """
    X = X - X.mean(axis=0)
    cov = np.dot(X.T, X) / (X.shape[0] - 1)
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)

def compute_group_precision(corrs, eps=1e-6):
    """
    Given a list of subject correlation matrices, compute
    the average correlation and its inverse (precision).
    """
    mean_corr = np.mean(corrs, axis=0)
    # ensure positive definiteness
    min_eig = np.min(eigvals(mean_corr))
    if min_eig < eps:
        mean_corr += (eps - min_eig) * np.eye(mean_corr.shape[0])
    return inv(mean_corr)

def regularized_precision(corr, gamma):
    """
    Compute (corr + gamma * I)^{-1}.
    """
    p = corr.shape[0]
    return inv(corr + gamma * np.eye(p))

def frobenius_distance(A, B):
    return np.linalg.norm(A - B, 'fro')


def optimize_gamma(subject_datas, gammas, metric=frobenius_distance):
    """
    For each candidate gamma, compute regularized precisions for all subjects,
    sum their distances to the unregularized group precision, and pick the gamma
    that minimizes this sum.
    
    Parameters
    ----------
    subject_datas : list of arrays, each of shape (n_samples_i, p)
    gammas         : 1d array of candidate gamma values
    """
    # 1) compute all subject correlation matrices
    corrs = [compute_correlation(X) for X in subject_datas]
    
    # 2) compute group precision from unregularized mean corr
    group_prec = compute_group_precision(corrs)
    
    # 3) for each gamma, score it
    scores = []
    for g in gammas:
        regs = [regularized_precision(C, g) for C in corrs]
        dists = [metric(R, group_prec) for R in regs]
        scores.append(np.sum(dists))
    
    best_idx = np.argmin(scores)
    return gammas[best_idx], scores

def estimate_subject_precisions(subject_datas, gamma=None, gamma_grid=None):
    """
    If gamma is provided, use it; else optimize over gamma_grid.
    Returns:
      gamma_opt, list_of_precisions, gamma_scores (if optimized)
    """
    # compute correlations once
    corrs = [compute_correlation(X) for X in subject_datas]
    # compute group precision
    group_prec = compute_group_precision(corrs)

    if gamma is None:
        if gamma_grid is None:
            # default search grid
            gamma_grid = np.logspace(-4, 1, 50)
        gamma_opt, scores = optimize_gamma(subject_datas, gamma_grid)
    else:
        gamma_opt = gamma
        scores = None

    # compute final regularized precisions
    precisions = [regularized_precision(C, gamma_opt) for C in corrs]
    return gamma_opt, precisions, scores

# ===== Example usage =====
if __name__ == "__main__":
    # simulate some data: 5 subjects, each 30 time‐points, 10 variables
    rng = np.random.RandomState(0)
    subject_data = [rng.randn(30, 10) for _ in range(5)]
    
    # estimate
    gamma_opt, precisions, gamma_scores = estimate_subject_precisions(
        subject_data,
        gamma_grid=np.logspace(-3, 0, 40)
    )
    print("Optimal γ:", gamma_opt)
    for i, P in enumerate(precisions):
        print(f"Subject {i} precision matrix shape: {P.shape}")