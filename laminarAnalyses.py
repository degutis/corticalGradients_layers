import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

def runClusterAnalysis(eigvecs_list, threshold=0.3):

    eigvecs_array = np.array([v / np.linalg.norm(v) for v in eigvecs_list])

    D = squareform(pdist(eigvecs_array, metric=sign_invariant_distance))

    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(D)
    
    cluster_groups = defaultdict(list)
    for i, cluster_id in enumerate(labels):
        cluster_groups[cluster_id].append(i)

    return cluster_groups, labels

# Pairwise distance matrix (sign-invariant)
def sign_invariant_distance(u, v):
    return 1 - np.abs(np.dot(u, v))

def convert_eigvals_to_list(eigvecs, eigvals, N, num_layers):
    
    eigvecs_list = []
    eigvalue_list = []
    source_info = []

    for layer_idx in range(num_layers):
        row_start = layer_idx * N
        row_end = row_start + N

        # Get the 360x1080 block for this layer (rows slice, all columns)
        layer_eigvecs = eigvecs[row_start:row_end, :]

        row_idx = layer_idx + 1
        col_idx = layer_idx + 1

        for i in range(layer_eigvecs.shape[1]):
            eigvec = layer_eigvecs[:, i]
            eigvecs_list.append(eigvec / np.linalg.norm(eigvec))
            #source_info.append((layer_idx + 1, i))  # layer 1-based, eigenvector index
            source_info.append((row_idx, col_idx, i))
            eigvalue_list.append(eigvals[i])

    return eigvecs_list, eigvalue_list, source_info


def plotEigvectors_similar_distinct(eigvecs_list, eigvalue_list, source_info, cluster_groups, restStateSub, eigenvalue_threshold, cluster_threshold, name):
    
    for cluster_id, indices in cluster_groups.items():
        
        if len(indices) == 1:
            continue
        eigvecs_to_plot = [eigvecs_list[i] for i in indices]
        meta = [source_info[i] for i in indices]
        titles = [f"(r{r},c{c}) eig{e}" for (r, c, e) in meta]
        Xp = np.stack(eigvecs_to_plot, axis=1)

        # Build filename from source info
        name_str = "-".join([f"r{r}_c{c}_e{e}" for (r, c, e) in meta])
        eig_label = f"{name_str}"

        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(
            Xp, eig_label, name=name, titles=titles, folder_name="SimilarDissimilar"
            )


    for cluster_id, indices in cluster_groups.items():
        if len(indices) > 1:
            continue  # Only consider singleton clusters

        i = indices[0]
        if not(0 < eigvalue_list[i] < eigenvalue_threshold):
            continue

        r, c, _ = source_info[i]
        v_i = eigvecs_list[i] / np.linalg.norm(eigvecs_list[i])

        # Compare against others from the same region-pair
        similar_found = False
        for j, (rj, cj, _) in enumerate(source_info):
            if (rj, cj) == (r, c) and j != i:
                eigval_j = eigvalue_list[j]
                if not (0 < eigval_j < eigenvalue_threshold):
                    continue
                
                v_j = eigvecs_list[j] / np.linalg.norm(eigvecs_list[j])
                similarity = np.abs(np.dot(v_i, v_j))
                if similarity >= (1 - cluster_threshold):
                    similar_found = True
                    break

        if similar_found:
            continue

        # Passed distinctness check → plot
        eigvecs_to_plot = [eigvecs_list[i]]
        meta = [source_info[i]]
        titles = [f"Distinct_(r{r},c{c}) eig{e}" for (r, c, e) in meta]
        Xp = np.stack(eigvecs_to_plot, axis=1)

        name_str = f"r{r}_c{c}_e{meta[0][2]}"
        eig_label = name_str

        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(
            Xp, eig_label, name=name, titles=titles, folder_name="SimilarDissimilar"
        )


