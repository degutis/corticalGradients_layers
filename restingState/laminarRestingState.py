import numpy as np
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import hcp_utils as hcp
import warnings
from collections import defaultdict
import plotly.graph_objects as go
import networkx as nx
from scipy.signal import resample
from tqdm import tqdm
from collections import Counter
from scipy.stats import f_oneway
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score





class LaminarRestingState:
    def __init__(self, data_dir, N, setThresh, num_layers = 3, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-01/HCP-MM1_in-func.nii"):
        
        self.data_dir = data_dir
        self.N = N
        self.setThresh = setThresh
        self.atlas_dir = atlas_dir
        self.num_layers = num_layers
        self.npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
      
    def plotReliability(self, TR=3.2, min_minutes=1, n_iterations=500):
        """
        Computes within‐subject FC reliability per layer AND across ALL parcels from ALL layers,
        using matched‐window design: for each window length tmin, sample two independent
        segments of tmin minutes each.
        """
        # how many volumes in one minute
        volumes_per_minute = int(round(60.0 / TR))

        # --- 1) collect layer‐grouped file lists ---
        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        layer_groups = defaultdict(list)
        for fname in npy_files:
            try:
                layer_num = int(fname.split('_')[-1].replace('.npy',''))
            except Exception as e:
                raise ValueError(f"Could not extract layer number from filename: {fname}") from e
            layer_groups[layer_num].append(os.path.join(self.data_dir, fname))

        reliability_results = {}

        # --- 2) per‐layer curves ---
        for layer_num in sorted(layer_groups):
            files = layer_groups[layer_num]
            print(f"\nLayer {layer_num}: {len(files)} run(s)")

            # load + concat runs → data: [n_parcels, total_timepoints]
            runs = [np.load(fp) for fp in files]
            data = np.concatenate(runs, axis=1)
            n_parcels, total_vols = data.shape
            total_mins = total_vols // volumes_per_minute
            print(f"  total_vols={total_vols}, total_mins={total_mins}")

            if total_mins < 2 * min_minutes:
                print(f"  skip (need ≥{2*min_minutes} min total)")
                continue

            iu = np.triu_indices(n_parcels, k=1)
            layer_curve = []

            # tmin is the length (minutes) of each of the two windows
            for tmin in tqdm(range(min_minutes, total_mins // 2 + 1),
                            desc=f"Layer {layer_num}"):
                win_vols = tmin * volumes_per_minute
                corrs = []

                for _ in range(n_iterations):
                    # sample two independent segments of length win_vols
                    start1 = np.random.randint(0, total_vols - win_vols + 1)
                    seg1 = data[:, start1:start1+win_vols]

                    start2 = np.random.randint(0, total_vols - win_vols + 1)
                    seg2 = data[:, start2:start2+win_vols]

                    # if by chance lengths differ (edge cases), resample
                    if seg1.shape[1] != seg2.shape[1]:
                        seg2 = resample(seg2, seg1.shape[1], axis=1)

                    # compute FCs and extract upper‐triangle
                    fc1 = np.corrcoef(seg1)
                    fc2 = np.corrcoef(seg2)
                    v1, v2 = fc1[iu], fc2[iu]

                    valid = ~np.isnan(v1) & ~np.isnan(v2)
                    if valid.sum() < 2:
                        continue

                    r = np.corrcoef(v1[valid], v2[valid])[0,1]
                    corrs.append(r)

                layer_curve.append((tmin, np.nan if not corrs else np.mean(corrs)))

            reliability_results[layer_num] = layer_curve

        # --- 3) ALL‐layers curve ---
        # group by run so that timepoints align across layers
        run_groups = defaultdict(list)
        for fps in layer_groups.values():
            for fp in fps:
                run_id = os.path.basename(fp).split('_')[1]  # e.g. 'run2'
                run_groups[run_id].append(fp)

        data_runs = []
        for run_id, fps in run_groups.items():
            fps_sorted = sorted(
                fps,
                key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.npy',''))
            )
            arrs = [np.load(fp) for fp in fps_sorted]
            T0 = arrs[0].shape[1]
            if any(a.shape[1] != T0 for a in arrs):
                raise ValueError(f"Run {run_id} has mismatched timepoints across layers")
            # stack parcels across layers
            data_runs.append(np.vstack(arrs))

        data_all = np.concatenate(data_runs, axis=1)
        n_all, total_vols_all = data_all.shape
        total_mins_all = total_vols_all // volumes_per_minute
        print(f"\nALL‐LAYERS: parcels={n_all}, total_vols={total_vols_all}, total_mins={total_mins_all}")

        if total_mins_all >= 2 * min_minutes:
            iu_all = np.triu_indices(n_all, k=1)
            all_curve = []

            for tmin in tqdm(range(min_minutes, total_mins_all // 2 + 1),
                            desc="ALL‐LAYERS"):
                win_vols = tmin * volumes_per_minute
                corrs = []

                for _ in range(n_iterations):
                    s1 = np.random.randint(0, total_vols_all - win_vols + 1)
                    seg1 = data_all[:, s1:s1+win_vols]
                    s2 = np.random.randint(0, total_vols_all - win_vols + 1)
                    seg2 = data_all[:, s2:s2+win_vols]

                    if seg1.shape[1] != seg2.shape[1]:
                        seg2 = resample(seg2, seg1.shape[1], axis=1)

                    fc1 = np.corrcoef(seg1)
                    fc2 = np.corrcoef(seg2)
                    v1, v2 = fc1[iu_all], fc2[iu_all]

                    valid = ~np.isnan(v1) & ~np.isnan(v2)
                    if valid.sum() < 2:
                        continue

                    r = np.corrcoef(v1[valid], v2[valid])[0,1]
                    corrs.append(r)

                all_curve.append((tmin, np.nan if not corrs else np.mean(corrs)))

            reliability_results['all'] = all_curve
        else:
            print("Not enough data to compute ALL‐LAYERS curve.")

        # --- 4) Plot everything ---
        plt.figure(figsize=(10,6))
        for key, curve in sorted(reliability_results.items(), key=lambda x: str(x[0])):
            mins, rs = zip(*curve)
            label = 'All layers' if key=='all' else f'Layer {key}'
            plt.plot(mins, rs, marker='o', label=label)

        plt.ylim(0, 1)
        plt.xlabel('Window length (minutes)')
        plt.ylabel('Mean FC‐matrix corr')
        plt.title('Within‐subject FC reliability vs. data amount (matched windows)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        outpath = os.path.join(self.data_dir, 'Reliability_FC_matched.png')
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
        print(f"Saved combined plot to {outpath}")


    def get_adj_matrix_withinLayers(self):
        
        adj_matrix_within = np.empty((self.N,self.N,self.num_layers))

        for i, file in enumerate(self.npy_files):

            # print("Working on file: ", file)

            file_path = os.path.join(self.data_dir, file)
            time_series = np.load(file_path)

            corr_matrix = np.corrcoef(time_series)

            corr_matrix = np.nan_to_num(corr_matrix, nan=0)
            np.fill_diagonal(corr_matrix, 1)

            threshold = np.percentile(np.abs(corr_matrix), self.setThresh)
            adj_matrix = np.where(np.abs(corr_matrix) >= threshold, corr_matrix, 0)
            adj_matrix_within[:,:,i] = np.abs(adj_matrix)

        # Block matrix for within-parcel / across layer connections
        I_N = np.eye(self.N)
        adj_matrix_full = np.block([
            [adj_matrix_within[:,:,0], I_N, I_N],
            [I_N, adj_matrix_within[:,:,1], I_N],
            [I_N, I_N, adj_matrix_within[:,:,2]]
        ])

        return adj_matrix_full
    
    def get_adj_matrix_withinLayers_multRuns(self, subtractAverage=False):
        
        layer_groups = defaultdict(list)
        
        for file in self.npy_files:
            try:
                layer_str = file.split('_')[-1].replace('.npy', '')
                layer_num = int(layer_str)
                layer_groups[layer_num].append(file)
            except Exception as e:
                # raise ValueError(f"Could not extract layer number from filename: {file}") from e
                # print(f"Could not extract layer number from filename: {file}")
                continue

        sorted_layers = sorted(layer_groups.items())
        adj_matrix_within = np.empty((self.N, self.N, self.num_layers))
        adj_matrix_within_noThresh = np.empty((self.N,self.N,self.num_layers))

        for i, (layer_num, files) in enumerate(sorted_layers):
            # print(f"Processing Layer {layer_num} with {len(files)} run(s)")
            all_time_series = []

            for file in files:
                file_path = os.path.join(self.data_dir, file)
                time_series = np.load(file_path)
                all_time_series.append(time_series)

            concatenated = np.concatenate(all_time_series, axis=1)
            # print(f"Concatenated shape: {concatenated.shape}")

            # Compute correlation
            corr_matrix = np.corrcoef(concatenated)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0)
            np.fill_diagonal(corr_matrix, 0)

            # Threshold
            threshold = np.percentile(np.abs(corr_matrix), self.setThresh)
            adj_matrix = np.where(np.abs(corr_matrix) >= threshold, corr_matrix, 0)
            adj_matrix_within[:, :, i] = np.abs(adj_matrix)
            adj_matrix_within_noThresh[:,:,i] = self.fisher_z(corr_matrix)

        # 
        if subtractAverage:
            avg_matrix = np.mean(adj_matrix_within_noThresh, axis=2)
            for i in range(self.num_layers):
                adj_matrix_within_noThresh[:, :, i] -= avg_matrix


        # Build inter-layer identity matrices
        I_N = np.eye(self.N)
        blocks = []

        for i in range(self.num_layers):
            row_blocks = []
            for j in range(self.num_layers):
                if i == j:
                    row_blocks.append(adj_matrix_within[:, :, i])
                else:
                    row_blocks.append(I_N)
            blocks.append(row_blocks)

        adj_matrix_full = np.block(blocks)
        
        return adj_matrix_full, adj_matrix_within_noThresh


    def get_adj_matrix_full(self):
        
        all_series = []

        for file in self.npy_files:

            print("Working on file: ", file)

            file_path = os.path.join(self.data_dir, file)
            time_series = np.load(file_path)
            all_series.append(time_series)

        all_series_array = np.concatenate(all_series, axis=0)
        full_corr = np.corrcoef(all_series_array)
        full_corr = np.nan_to_num(full_corr, nan=0)
        np.fill_diagonal(full_corr, 1)
        threshold = np.percentile(np.abs(full_corr), self.setThresh)
        adj_full = np.where(np.abs(full_corr) >= threshold, full_corr, 0)
        
        return np.abs(adj_full), all_series_array


    def get_adj_matrix_full_multRuns(self):
        
        layer_groups = defaultdict(list)
        
        for file in self.npy_files:
            try:
                layer_str = file.split('_')[-1].replace('.npy', '')
                layer_num = int(layer_str)
                layer_groups[layer_num].append(file)
            except Exception as e:
                # raise ValueError(f"Could not extract layer number from filename: {file}") from e
                print(f"Could not extract layer number from filename: {file}")
                continue

        sorted_layers = sorted(layer_groups.items())        
        concatenated_full = []
        for i, (layer_num, files) in enumerate(sorted_layers):
            print(f"Processing Layer {layer_num} with {len(files)} run(s)")
            all_time_series = []

            for file in files:
                file_path = os.path.join(self.data_dir, file)
                time_series = np.load(file_path)
                all_time_series.append(time_series)

            concatenated = np.concatenate(all_time_series, axis=1)
            concatenated_full.append(concatenated)
            print(f"Concatenated shape: {concatenated.shape}")

        all_series_array = np.concatenate(concatenated_full, axis=0)
        full_corr = np.corrcoef(all_series_array)
        # full_corr = self.fisher_z(np.nan_to_num(full_corr, nan=0))
        np.fill_diagonal(full_corr, 0)
        full_corr = np.nan_to_num(full_corr, nan=0)
        threshold = np.percentile(np.abs(full_corr), self.setThresh)
        adj_full = np.where(np.abs(full_corr) >= threshold, full_corr, 0)
        
        return np.abs(adj_full), all_series_array, full_corr


    def get_adj_matrix_singleLayer(self, layerNum):

        print("Working on file: ", self.npy_files[layerNum])

        file_path = os.path.join(self.data_dir, self.npy_files[layerNum])

        time_series = np.load(file_path)
        corr_matrix = np.corrcoef(time_series)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0)
        np.fill_diagonal(corr_matrix, 1)

        threshold = np.percentile(np.abs(corr_matrix), self.setThresh)
        adj_matrix = np.where(np.abs(corr_matrix) >= threshold, corr_matrix, 0)
        
        return np.abs(adj_matrix)


    def runLaplacianEmbedding(self, M, name, num_components=10, epsilon = 1e-10, convert_to_binary=True, full=False, addName='', vMax=1):
        
        self.num_components = num_components
        self.addName = addName
        os.makedirs(f"{self.data_dir}/{name}", exist_ok=True)  # Create folder for layer-wise maps

        if convert_to_binary:
            M[M != 0] = 1 # Convert to binary matrix
        else:
            M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
            M[M < 0] = 0.0

        plt.figure(figsize=(6, 6))
        plt.imshow(M, cmap="viridis", vmin=0, vmax=vMax)
        plt.colorbar(label="Correlation")
        plt.title(f"{name} Block Matrix")
        plt.savefig(f"{self.data_dir}/{name}/Block_matrix{self.addName}.png", bbox_inches="tight")
        plt.close()

        degree_matrix = np.diag(np.sum(M, axis=1))  # Degree matrix
        laplacian_matrix = degree_matrix - M  # Unnormalized Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(M, axis=1) + epsilon))  # Add small value to avoid division by zero
        L_norm = D_inv_sqrt @ laplacian_matrix @ D_inv_sqrt  # Normalized Laplacian

        if full:
            eigvals, eigvecs = scipy.linalg.eigh(L_norm)
            self.num_components = len(eigvals)
            
        else:
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(L_norm, k=num_components, which='SM')
            self.num_components = num_components


        return eigvals, eigvecs


    def runKMeans(self, eigvecs, name, num_clusters=3, random_state=99, eigvecs_to_plot=[1, 2]):

        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        labels = kmeans.fit_predict(eigvecs)
        eigvecs_str = "".join(map(str, eigvecs_to_plot))

        plt.figure(figsize=(8, 6))
        plt.scatter(eigvecs[:, eigvecs_to_plot[0]], eigvecs[:, eigvecs_to_plot[1]], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.xlabel(f'Eigenvector {eigvecs_to_plot[0]+1}')
        plt.ylabel(f'Eigenvector {eigvecs_to_plot[1]+1}')
        plt.title("KMeans Clustering")
        plt.colorbar(label="Cluster")
        plt.savefig(f"{self.data_dir}/{name}/KMeans_laplacian_embedding_{eigvecs_str}.png", bbox_inches="tight")
        plt.close()


    def plotTwoDimEmbedding(self,
                            eigvecs,
                            name,
                            eigvecs_to_plot=(0, 1),
                            layer_labels=None,
                            network_labels=None,
                            x_label="Emb1",
                            y_label="Emb2",
                            network_cmap='tab20',
                            # NEW:
                            atlas='schaefer',  # 'schaefer' (7-net) or 'custom'
                            schaefer_label_L="/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",                   # path to Schaefer*.L.label.gii (fs_LR 32k)
                            schaefer_label_R="/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii"                    # path to Schaefer*.R.label.gii (fs_LR 32k)
                            ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import nibabel as nib

        def _load_label_gii(path):
            g = nib.load(path)
            labs = np.asarray(g.agg_data(), dtype=int).squeeze()
            # build key -> name map from the label table
            lt = g.labeltable
            key_to_name = {lab.key: lab.label for lab in lt.labels}
            return labs, key_to_name

        def _schaefer7_from_name(name: str) -> int:
            """Map Schaefer 7-network label name to an index 0..6 (Yeo-7 order)."""
            n = name.lower()
            # typical substrings in Schaefer 7N names: Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default
            if 'vis' in n:
                return 0  # Visual
            if 'som' in n or 'sommot' in n:
                return 1  # Somatomotor
            if 'dorsattn' in n or ('dors' in n and 'attn' in n):
                return 2  # Dorsal Attention
            if 'ventattn' in n or 'salventattn' in n or ('vent' in n and 'attn' in n) or 'sal' in n:
                return 3  # Ventral/Salience
            if 'limbic' in n:
                return 4  # Limbic
            if 'cont' in n or 'control' in n or 'frontoparietal' in n or 'fp' in n:
                return 5  # Control/Frontoparietal
            if 'default' in n:
                return 6  # Default
            raise ValueError(f"Unrecognized Schaefer-7 network in label name: {name}")

        # ---------- build per-parcel network vector ----------
        if atlas.lower() == 'schaefer':
            if schaefer_label_L is None or schaefer_label_R is None:
                raise ValueError("Provide schaefer_label_L and schaefer_label_R (.label.gii on fs_LR 32k).")

            L_lab, L_map = _load_label_gii(schaefer_label_L)
            R_lab, R_map = _load_label_gii(schaefer_label_R)

            # Unique parcel keys per hemisphere (skip 0 = medial wall), sorted → defines parcel order
            uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
            uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))

            # Per-parcel network indices in order [LH parcels..., RH parcels...]
            networks0 = []
            for k in uL:
                networks0.append(_schaefer7_from_name(L_map[k]))
            for k in uR:
                networks0.append(_schaefer7_from_name(R_map[k]))
            networks0 = np.asarray(networks0, dtype=int)

            N = networks0.size  # parcels total (LH+RH)
            # default labels for 7 networks (only if not provided)
            if network_labels is None:
                network_labels = ['Visual', 'Somatomotor', 'Dorsal Attn',
                                'Ventral/Salience', 'Limbic', 'Control', 'Default']
        else:
            # Fallback: use your existing text file (expects 1..K coded networks)
            cats0 = np.loadtxt('cortex_parcel_network_assignments.txt', dtype=int)
            N = cats0.shape[0]
            networks0 = cats0 - 1
            if network_labels is None:
                # keep your 12-network defaults
                network_labels = [
                    "Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular",
                    "Dorsal-Attentional", "Language", "Frontoparietal", "Auditory",
                    "Default", "Posterior-Multimodal", "Ventral-Multimodal", "Orbito-Affective"
                ]

        # ---------- infer mode from eigvecs vs N ----------
        nrows, ndims = eigvecs.shape
        if nrows == 3 * N:
            mode = "multilayer"
        elif nrows == N:
            mode = "single"
        else:
            if nrows % 3 == 0 and nrows // 3 == N:
                mode = "multilayer"
            else:
                raise ValueError(f"eigvecs has {nrows} rows, but atlas implies N={N} parcels "
                                f"(expected {N} or {3*N} rows).")

        # ---------- pick dims (supports 1- or 0-based tuple) ----------
        x_dim, y_dim = eigvecs_to_plot
        if x_dim >= ndims or y_dim >= ndims:
            x_dim = max(0, x_dim - 1)
            y_dim = max(0, y_dim - 1)
        if not (0 <= x_dim < ndims and 0 <= y_dim < ndims):
            raise ValueError(f"Requested dims {eigvecs_to_plot} not in [0..{ndims-1}]")

        # ---------- expand networks to multilayer if needed ----------
        if mode == "multilayer":
            networks = np.tile(networks0, 3)       # (3N,)
            layers = np.repeat([0, 1, 2], N)       # Deep/Middle/Superficial (or your order)
        else:
            networks = networks0
            layers = np.zeros(N, dtype=int)

        # ---------- labels/colors ----------
        if isinstance(layer_labels, str):
            layer_labels = [layer_labels]
        if layer_labels is None:
            layer_labels = ['Superficial', 'Middle', 'Deep'] if mode == "multilayer" else ['AcrossLayers']
        elif mode == "single" and len(layer_labels) != 1:
            layer_labels = [layer_labels[0]]

        base_cmap = plt.get_cmap(network_cmap, len(network_labels))
        network_colors = [base_cmap(i) for i in range(len(network_labels))]
        shapes = ['o', 's', '^'] if mode == "multilayer" else ['o']

        # ---------- plot ----------
        fig, ax = plt.subplots(figsize=(7, 7))
        unique_layers = np.unique(layers)
        unique_nets = np.unique(networks)

        for lyr in unique_layers:
            for net in unique_nets:
                mask = (layers == lyr) & (networks == net)
                if not np.any(mask):
                    continue
                ax.scatter(
                    eigvecs[mask, x_dim],
                    eigvecs[mask, y_dim],
                    s=10,
                    marker=shapes[int(lyr if mode == "multilayer" else 0)],
                    facecolor=network_colors[int(net)],
                    edgecolor='k',
                    linewidths=0.2,
                    alpha=0.85
                )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('Embedding colored by Schaefer-7 RSN; shapes = layers' if mode == "multilayer"
                    else 'Embedding colored by Schaefer-7 RSN')
        ax.set_aspect('equal', adjustable='box')

        # Legends
        if mode == "multilayer":
            layer_handles = [
                Line2D([0], [0], marker=shapes[i], color='w', markeredgecolor='k',
                    markersize=9, label=layer_labels[int(lyr)])
                for i, lyr in enumerate(unique_layers)
            ]
            leg1 = ax.legend(handles=layer_handles, title='Layer', loc='upper right')
            ax.add_artist(leg1)

        network_handles = [
            Line2D([0], [0], marker='o', color='w',
                markerfacecolor=network_colors[i], markeredgecolor='k',
                markersize=9, label=network_labels[i])
            for i in unique_nets
        ]
        ax.legend(handles=network_handles, title='RSN',
                bbox_to_anchor=(1.32, 1), loc='upper left')

        # Save
        plt.tight_layout()
        eigstr = f"{x_dim}{y_dim}"
        outdir = f"{self.data_dir}/{name}"
        os.makedirs(outdir, exist_ok=True)
        suffix = "_multi" if mode == "multilayer" else "_single"
        outpath = f"{outdir}/Embedding_withNetworks_{eigstr}{suffix}.png"
        fig.savefig(outpath, bbox_inches='tight', dpi=500)
        plt.close(fig)
        print("Saved embedding plot to:", outpath)


    def plotScatterWithGlobalCorrelation(self,
                                        eigvecs,
                                        name,
                                        eigvecs_to_plot=(0, 1),
                                        layer_labels=None,
                                        network_labels=None,
                                        x_label="Emb1",
                                        y_label="Emb2",
                                        fname=None,
                                        network_cmap="tab20",
                                        dot_size=40,
                                        # NEW:
                                        atlas='schaefer',               # 'schaefer' (7-net) or 'custom'
                                        schaefer_label_L="/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
                                        schaefer_label_R="/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
                                        # --- marginal histogram controls ---
                                        show_marginal_hists=True,
                                        hist_bins=30,
                                        hist_size=0.1,   # fraction of figure for each marginal axis
                                        hist_pad=0.02,   # gap between scatter and hist axes
                                        hist_alpha=0.6,  # opacity for bars
                                        # --- NEW regression uncertainty controls ---
                                        show_ci_band=True,
                                        ci_level=0.95,                 # e.g., 0.95 for 95% CI
                                        band_kind="confidence",        # 'confidence' or 'prediction'
                                        ci_band_alpha=0.2,             # shading opacity
                                        return_stats=False):
        """
        eigvecs : (N×d) or (3N×d) array.
                If 3N, rows ordered as [Deep; Middle; Superficial] blocks of size N.
        When atlas='schaefer', parcels are assumed ordered [LH parcels..., RH parcels...]
        with each hemi ordered by **sorted label keys** from the .label.gii files.

        Also plots marginal histograms of X and Y when show_marginal_hists=True.

        NEW:
        - Computes OLS line of best fit and uncertainty:
            * slope/intercept ± SE and 95% CI
            * Optional shaded 95% band along the line ('confidence' for mean response, or 'prediction' for new obs)
        - Optionally returns a dict of regression stats when return_stats=True.
        """
        import os, numpy as np, matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from scipy.stats import pearsonr, t as t_dist
        import nibabel as nib

        # ---------------- helpers ----------------
        def _load_label_gii(path):
            g = nib.load(path)
            labs = np.asarray(g.agg_data(), dtype=int).squeeze()
            lt = g.labeltable
            key_to_name = {lab.key: lab.label for lab in lt.labels}
            return labs, key_to_name

        def _schaefer7_from_name(name: str) -> int:
            n = name.lower()
            if 'vis' in n: return 0                  # Visual
            if 'som' in n or 'sommot' in n: return 1 # Somatomotor
            if 'dorsattn' in n or ('dors' in n and 'attn' in n): return 2  # Dorsal Attn
            if 'ventattn' in n or 'salventattn' in n or ('vent' in n and 'attn' in n) or 'sal' in n: return 3  # Ventral/Salience
            if 'limbic' in n: return 4
            if 'cont' in n or 'control' in n or 'frontoparietal' in n or 'fp' in n: return 5  # Control/FP
            if 'default' in n: return 6
            raise ValueError(f"Unrecognized Schaefer-7 network in label name: {name}")

        # ---------------- build per-parcel network vector ----------------
        if atlas.lower() == 'schaefer':
            if schaefer_label_L is None or schaefer_label_R is None:
                raise ValueError("For atlas='schaefer', provide schaefer_label_L and schaefer_label_R (.label.gii on fs_LR 32k).")

            L_lab, L_map = _load_label_gii(schaefer_label_L)
            R_lab, R_map = _load_label_gii(schaefer_label_R)
            uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
            uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))

            # networks0 in parcel order [LH..., RH...]
            networks0 = []
            for k in uL:
                networks0.append(_schaefer7_from_name(L_map[k]))
            for k in uR:
                networks0.append(_schaefer7_from_name(R_map[k]))
            networks0 = np.asarray(networks0, dtype=int)
            N = networks0.size

            if network_labels is None:
                network_labels = ['Visual', 'Somatomotor', 'Dorsal Attn',
                                'Ventral/Salience', 'Limbic', 'Control', 'Default']
        else:
            cats0 = np.loadtxt("cortex_parcel_network_assignments.txt", dtype=int)
            networks0 = cats0 - 1
            N = networks0.size
            if network_labels is None:
                network_labels = [
                    "Visual1","Visual2","Somatomotor","Cingulo-Opercular",
                    "Dorsal-Attentional","Language","Frontoparietal","Auditory",
                    "Default","Posterior-Multimodal","Ventral-Multimodal","Orbito-Affective"
                ]

        # ---------------- infer rows / mode ----------------
        nrows, ndims = eigvecs.shape
        if nrows == 3 * N:
            mode = "multilayer"
        elif nrows == N:
            mode = "single"
        else:
            if nrows % 3 == 0 and (nrows // 3) == N:
                mode = "multilayer"
            else:
                raise ValueError(f"eigvecs has {nrows} rows, but atlas implies N={N} parcels (expected {N} or {3*N}).")

        # ---------------- parse dims (accept 1-based) ----------------
        x_dim, y_dim = eigvecs_to_plot
        if x_dim >= ndims or y_dim >= ndims:
            x_dim, y_dim = x_dim - 1, y_dim - 1
        if not (0 <= x_dim < ndims and 0 <= y_dim < ndims):
            raise ValueError(f"Requested dims {eigvecs_to_plot} not in [0..{ndims-1}]")

        # ---------------- expand networks to multilayer if needed ----------------
        if mode == "multilayer":
            nets = np.tile(networks0, 3)           # (3N,)
            layers = np.repeat([0, 1, 2], N)       # Deep/Middle/Superficial (your order)
            shapes = ['o', 's', '^']
            if layer_labels is None: layer_labels = ["Deep", "Middle", "Superficial"]
        else:
            nets = networks0
            layers = np.zeros(N, dtype=int)
            shapes = ['o']
            if layer_labels is None: layer_labels = ["AcrossLayers"]

        # ---------------- colours ----------------
        import matplotlib
        seq_cmap = matplotlib.cm.get_cmap('tab20')

        # how many networks actually appear in the data
        n_nets = int(nets.max()) + 1

        if atlas.lower() == 'schaefer':
            # keep your special Schaefer-7 ordering if you like
            if n_nets != 7:
                raise ValueError(f"Expected 7 networks for Schaefer-7, found {n_nets}")

            shades = [seq_cmap(x) for x in np.linspace(0.2, 0.9, n_nets)]
            order_idx = [1, 0, 3, 2, 6, 5, 4]   # your custom order
            net_colours = [None] * n_nets
            for rank, net_idx in enumerate(order_idx):
                net_colours[net_idx] = shades[rank]
        else:
            # generic mapping: one colour per network
            if n_nets < 1:
                raise ValueError("No networks found in 'nets'.")
            net_colours = [
                seq_cmap(i / max(n_nets - 1, 1)) for i in range(n_nets)
            ]
        # ---------------- figure + axes (scatter + optional marginals) ----------------
        if show_marginal_hists:
            fig = plt.figure(figsize=(7.5, 7.5))
            left, bottom = 0.10, 0.10
            width, height = 0.64, 0.64
            rect_scatter = [left, bottom, width, height]
            rect_histx = [left, bottom + height + hist_pad, width, hist_size]
            rect_histy = [left + width + hist_pad, bottom, hist_size, height]

            ax = fig.add_axes(rect_scatter)
            ax_histx = fig.add_axes(rect_histx, sharex=ax)
            ax_histy = fig.add_axes(rect_histy, sharey=ax)
        else:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax_histx, ax_histy = None, None

        # ---------------- scatter ----------------
        for lyr in np.unique(layers):
            for net in np.unique(nets):
                m = (layers == lyr) & (nets == net)
                if not m.any(): continue
                ax.scatter(eigvecs[m, x_dim], eigvecs[m, y_dim],
                        s=dot_size,
                        marker=shapes[int(lyr if mode == "multilayer" else 0)],
                        facecolor=net_colours[int(net)],
                        edgecolor='k', linewidths=0.25, alpha=0.8)

        # ---------------- global correlation & regression with uncertainty ----------------
        x = eigvecs[:, x_dim].astype(float)
        y = eigvecs[:, y_dim].astype(float)
        n = x.size
        if n < 3:
            raise ValueError("Need at least 3 points for regression uncertainty.")

        # Pearson r and two-sided p
        r, p = pearsonr(x, y)

        # OLS fit (equivalent to np.polyfit for deg=1)
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        resid = y - yhat

        # Residual standard error (sigma-hat), Sxx, and t critical value
        x_bar = x.mean()
        Sxx = np.sum((x - x_bar) ** 2)
        if Sxx <= 0:
            raise ValueError("All x-values are identical; cannot fit a line.")
        df = n - 2
        sigma_hat = np.sqrt(np.sum(resid ** 2) / df)
        tcrit = t_dist.ppf(1 - (1 - ci_level) / 2, df)

        # Standard errors and CIs for slope/intercept
        slope_se = sigma_hat / np.sqrt(Sxx)
        intercept_se = sigma_hat * np.sqrt(1/n + (x_bar**2) / Sxx)
        slope_ci = (slope - tcrit * slope_se, slope + tcrit * slope_se)
        intercept_ci = (intercept - tcrit * intercept_se, intercept + tcrit * intercept_se)

        # Fix axes range before drawing line/bands (your plots are normalized 0..1)
        ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        ax.set_title("Embedding colored by Schaefer-7 RSN" + (" (layers as shapes)" if mode == "multilayer" else ""))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        # Best-fit line
        xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
        line_y = slope * xs + intercept
        ax.plot(xs, line_y, color='k', ls='--', lw=1, zorder=5)

        # Confidence or prediction band
        if show_ci_band:
            # SE of mean response vs prediction at each xs
            if band_kind.lower().startswith('pred'):
                mult = 1.0  # adds the "+1" under the sqrt below
            else:
                mult = 0.0
            se_line = sigma_hat * np.sqrt(mult + (1/n) + ((xs - x_bar) ** 2) / Sxx)
            upper = line_y + tcrit * se_line
            lower = line_y - tcrit * se_line
            ax.fill_between(xs, lower, upper, alpha=ci_band_alpha, edgecolor='none', facecolor='gray', zorder=4)

        # Regression summary textbox
        txt = (
            f"r = {r:.3f}\np = {p:.3g}\n"
            f"slope = {slope:.3f} [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]\n"
            f"intercept = {intercept:.3f} [{intercept_ci[0]:.3f}, {intercept_ci[1]:.3f}]"
        )
        bbox_props = dict(boxstyle="round,pad=0.25", fc="w", ec="k", lw=0.4)
        ax.text(-0.05, 1.06, txt, ha="right", va="bottom",
                transform=ax.transAxes, fontsize=5, bbox=bbox_props)

        # ---------------- marginal histograms ----------------
        if show_marginal_hists:
            ax_histx.hist(x, bins=hist_bins, range=(0, 1), edgecolor='k', alpha=hist_alpha)
            ax_histx.tick_params(axis='x', labelbottom=False); ax_histx.tick_params(axis='y', labelleft=False)
            for spine in ('right', 'top'): ax_histx.spines[spine].set_visible(False)

            ax_histy.hist(y, bins=hist_bins, range=(0, 1), orientation='horizontal', edgecolor='k', alpha=hist_alpha)
            ax_histy.tick_params(axis='y', labelleft=False); ax_histy.tick_params(axis='x', labelbottom=False)
            for spine in ('right', 'top'): ax_histy.spines[spine].set_visible(False)

        # ensure we have at least one label per network
        if len(network_labels) < n_nets:
            network_labels = list(network_labels) + [
                f"Net{i}" for i in range(len(network_labels), n_nets)
            ]

        # ---------------- legends ----------------
        net_handles = [
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=net_colours[int(i)],
                markeredgecolor='k',
                markersize=8,
                label=network_labels[int(i)],
            )
            for i in np.unique(nets)
        ]

                    
        ax.legend(handles=net_handles, title="RSN",
                bbox_to_anchor=(1.32, 1), loc='upper left')

        if mode == "multilayer":
            lyr_handles = [Line2D([0],[0], marker=shapes[i], color='w',
                                markeredgecolor='k', markersize=8, label=layer_labels[i])
                        for i in np.unique(layers)]
            ax.add_artist(ax.legend(handles=lyr_handles, title="Layer", loc='upper right'))

        # ---------------- save ----------------
        outdir = os.path.join(self.data_dir, name)
        os.makedirs(outdir, exist_ok=True)
        if fname is None:
            fname = f"ScatterCorr_d{x_dim+1}{y_dim+1}_{'multi' if mode=='multilayer' else 'single'}.png"
        fig.tight_layout()
        outpath = os.path.join(outdir, fname)
        fig.savefig(outpath, dpi=500, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", outpath)

        if return_stats:
            return {
                "n": int(n),
                "df": int(df),
                "pearson_r": float(r),
                "pearson_p": float(p),
                "slope": float(slope),
                "intercept": float(intercept),
                "sigma_hat": float(sigma_hat),
                "slope_se": float(slope_se),
                "intercept_se": float(intercept_se),
                "slope_ci": tuple(map(float, slope_ci)),
                "intercept_ci": tuple(map(float, intercept_ci)),
                "ci_level": float(ci_level),
                "band_kind": band_kind.lower(),
            }



    def plotScatter3DWithPlane(self,
                            X,
                            Y=None,
                            Z=None,
                            name="Scatter3D",
                            dims_to_plot=(0, 1, 2),
                            layer_labels=None,
                            network_labels=None,
                            x_label="Emb1",
                            y_label="Emb2",
                            z_label="Emb3",
                            fname=None,
                            network_cmap="tab20",
                            dot_size=30,
                            show_plane=False,
                            # NEW: de-squish controls
                            equalize_axes=True,     # put data in a cube so scales match
                            cube_pad=0.06,          # 6% padding around the cube
                            proj_type="ortho",      # 'ortho' to remove perspective, 'persp' for default
                            plane_alpha=0.18,
                            # Atlas options
                            atlas='schaefer',
                            schaefer_label_L="/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
                            schaefer_label_R="/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
                            # NEW: marginals
                            show_marginals=True,
                            hist_bins=20,
                            ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from scipy.stats import pearsonr
        import nibabel as nib

        # ---------- helpers ----------
        def _load_label_gii(path):
            g = nib.load(path)
            labs = np.asarray(g.agg_data(), dtype=int).squeeze()
            lt = g.labeltable
            key_to_name = {lab.key: lab.label for lab in lt.labels}
            return labs, key_to_name

        def _schaefer7_from_name(name: str) -> int:
            # 0=Visual, 1=Somatomotor, 2=DorsAttn, 3=Ventral/Sal, 4=Limbic, 5=Control, 6=Default
            n = name.lower()
            if 'vis' in n: return 0
            if 'som' in n or 'sommot' in n: return 1
            if 'dorsattn' in n or ('dors' in n and 'attn' in n): return 2
            if 'ventattn' in n or 'salventattn' in n or ('vent' in n and 'attn' in n) or 'sal' in n: return 3
            if 'limbic' in n: return 4
            if 'cont' in n or 'control' in n or 'frontoparietal' in n or 'fp' in n: return 5
            if 'default' in n: return 6
            raise ValueError(f"Unrecognized Schaefer-7 network in label name: {name}")

        # ---------- networks / N ----------
        if atlas.lower() == 'schaefer':
            if schaefer_label_L is None or schaefer_label_R is None:
                raise ValueError("For atlas='schaefer', provide schaefer_label_L and schaefer_label_R.")
            L_lab, L_map = _load_label_gii(schaefer_label_L)
            R_lab, R_map = _load_label_gii(schaefer_label_R)
            uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
            uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))
            nets0 = []
            for k in uL: nets0.append(_schaefer7_from_name(L_map[k]))
            for k in uR: nets0.append(_schaefer7_from_name(R_map[k]))
            networks0 = np.asarray(nets0, int)
            N = networks0.size
            if network_labels is None:
                network_labels = ['Visual','Somatomotor','Dorsal Attn',
                                'Ventral/Salience','Limbic','Control','Default']
        else:
            cats0 = np.loadtxt("cortex_parcel_network_assignments.txt", dtype=int)
            networks0 = cats0 - 1
            N = networks0.size
            if network_labels is None:
                network_labels = [
                    "Visual1","Visual2","Somatomotor","Cingulo-Opercular",
                    "Dorsal-Attentional","Language","Frontoparietal","Auditory",
                    "Default","Posterior-Multimodal","Ventral-Multimodal","Orbito-Affective"
                ]

        # ---------- get x,y,z ----------
        if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
            nrows, ndims = X.shape
            i, j, k = dims_to_plot
            if i >= ndims or j >= ndims or k >= ndims:  # accept 1-based
                i, j, k = i-1, j-1, k-1
            if nrows in (N, 3*N):
                mode = "multilayer" if nrows == 3*N else "single"
            elif nrows % 3 == 0 and (nrows // 3) == N:
                mode = "multilayer"
            else:
                raise ValueError(f"Data has {nrows} rows, but atlas implies N={N} (expected {N} or {3*N}).")
            x, y, z = X[:, i].astype(float), X[:, j].astype(float), X[:, k].astype(float)
        else:
            if Y is None or Z is None:
                raise ValueError("Provide either a 2D matrix X with dims_to_plot, or explicit vectors X, Y, Z.")
            x = np.asarray(X, float).squeeze()
            y = np.asarray(Y, float).squeeze()
            z = np.asarray(Z, float).squeeze()
            nrows = x.size
            if nrows in (N, 3*N):
                mode = "multilayer" if nrows == 3*N else "single"
            elif nrows % 3 == 0 and (nrows // 3) == N:
                mode = "multilayer"
            else:
                raise ValueError(f"Vector length {nrows} not compatible with N={N} or 3N={3*N}.")

        # ---------- expand networks to multilayer ----------
        if mode == "multilayer":
            nets = np.tile(networks0, 3)
            layers = np.repeat([0, 1, 2], N)
            shapes = ['o', 's', '^']
            if layer_labels is None: layer_labels = ["Deep", "Middle", "Superficial"]
        else:
            nets = networks0
            layers = np.zeros(N, int)
            shapes = ['o']
            if layer_labels is None: layer_labels = ["AcrossLayers"]

        # ---------- colors ----------
        if len(network_labels) == 7:
            try:
                seq_cmap = plt.get_cmap(network_cmap)
            except Exception:
                seq_cmap = plt.get_cmap('tab20')
            shades = [seq_cmap(x_) for x_ in np.linspace(0.2, 0.9, 7)]
            order_idx = [1, 0, 3, 2, 6, 5, 4]
            net_colours = [None] * 7
            for rank, net_idx in enumerate(order_idx):
                net_colours[net_idx] = shades[rank]
        else:
            cmap = plt.get_cmap(network_cmap, len(network_labels))
            net_colours = [cmap(i) for i in range(len(network_labels))]

        # ---------- figure / axes ----------
        fig = plt.figure(figsize=(8.8, 8.0))

        if show_marginals:
            # manual layout: main 3D axis + 3 marginal axes
            ax = fig.add_axes([0.08, 0.08, 0.6, 0.7], projection='3d')
            ax_histx = fig.add_axes([0.08, 0.80, 0.6, 0.16])                # top, x distribution
            ax_histy = fig.add_axes([0.70, 0.08, 0.16, 0.7])                # right, y distribution
            ax_histz = fig.add_axes([0.70, 0.80, 0.16, 0.16])               # top-right, z distribution
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax_histx = ax_histy = ax_histz = None

        try:
            ax.set_proj_type(proj_type)
        except Exception:
            pass

        # ---------- scatter ----------
        for lyr in np.unique(layers):
            for net in np.unique(nets):
                m = (layers == lyr) & (nets == net)
                if not np.any(m):
                    continue
                ax.scatter(x[m], y[m], z[m],
                        s=dot_size,
                        marker=shapes[int(lyr if mode == "multilayer" else 0)],
                        facecolor=net_colours[int(net)],
                        edgecolor='k', linewidths=0.25, alpha=0.9)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.set_title("3D Embedding colored by Schaefer-7 RSN"
                    + (" (layers as shapes)" if mode == "multilayer" else ""))

        # ---------- cube limits ----------
        def _safe_range(lo, hi):
            if not np.isfinite(lo) or not np.isfinite(hi):
                return -1.0, 1.0
            if hi == lo:
                return lo - 0.5, hi + 0.5
            return lo, hi

        xlo, xhi = _safe_range(np.nanmin(x), np.nanmax(x))
        ylo, yhi = _safe_range(np.nanmin(y), np.nanmax(y))
        zlo, zhi = _safe_range(np.nanmin(z), np.nanmax(z))

        if equalize_axes:
            xm, ym, zm = (xlo + xhi)/2.0, (ylo + yhi)/2.0, (zlo + zhi)/2.0
            r = max(xhi - xlo, yhi - ylo, zhi - zlo) * 0.5
            r *= (1.0 + float(cube_pad))
            xl, yl, zl = xm - r, ym - r, zm - r
            xh, yh, zh = xm + r, ym + r, zm + r
        else:
            xl, xh = xlo, xhi
            yl, yh = ylo, yhi
            zl, zh = zlo, zhi

        # ---------- best-fit plane ----------
        r_xy, p_xy = pearsonr(x, y)
        r_xz, p_xz = pearsonr(x, z)
        r_yz, p_yz = pearsonr(y, z)
        r2 = np.nan

        if show_plane:
            Xmat = np.column_stack([x, y, np.ones_like(x)])
            a, b, c = np.linalg.lstsq(Xmat, z, rcond=None)[0]
            z_hat = a * x + b * y + c
            ss_res = np.sum((z - z_hat) ** 2)
            ss_tot = np.sum((z - np.mean(z)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            xx = np.linspace(xl, xh, 45)
            yy = np.linspace(yl, yh, 45)
            XX, YY = np.meshgrid(xx, yy)
            ZZ = a * XX + b * YY + c
            ax.plot_surface(XX, YY, ZZ, alpha=plane_alpha, linewidth=0, antialiased=True)

        # ---------- axis limits / aspect ----------
        ax.set_xlim(xl, xh)
        ax.set_ylim(yl, yh)
        ax.set_zlim(zl, zh)
        if equalize_axes:
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass
        ax.view_init(elev=22, azim=38)

        # ---------- marginals (histograms by network) ----------
        if show_marginals:
            bins_x = np.linspace(xl, xh, hist_bins + 1)
            bins_y = np.linspace(yl, yh, hist_bins + 1)
            bins_z = np.linspace(zl, zh, hist_bins + 1)

            uniq_nets = np.unique(nets)

            # X histogram (top)
            for net in uniq_nets:
                m = (nets == net)
                ax_histx.hist(x[m], bins=bins_x,
                            color=net_colours[int(net)], alpha=0.4,
                            density=False, label=network_labels[int(net)])
            ax_histx.set_xlim(xl, xh)
            ax_histx.set_xticklabels([])
            ax_histx.set_ylabel("count", fontsize=7)
            ax_histx.tick_params(axis='y', labelsize=7)

            # Y histogram (right; horizontal)
            for net in uniq_nets:
                m = (nets == net)
                ax_histy.hist(y[m], bins=bins_y,
                            orientation='horizontal',
                            color=net_colours[int(net)], alpha=0.4,
                            density=False)
            ax_histy.set_ylim(yl, yh)
            ax_histy.set_yticklabels([])
            ax_histy.set_xlabel("count", fontsize=7)
            ax_histy.tick_params(axis='x', labelsize=7)

            # Z histogram (small top-right)
            for net in uniq_nets:
                m = (nets == net)
                ax_histz.hist(z[m], bins=bins_z,
                            color=net_colours[int(net)], alpha=0.4,
                            density=False)
            ax_histz.set_xlabel(z_label, fontsize=7)
            ax_histz.set_ylabel("count", fontsize=7)
            ax_histz.tick_params(axis='both', labelsize=7)

        # ---------- stats textbox ----------
        stats_txt = (f"r_xy={r_xy:.3f} (p={p_xy:.2g})\n"
                    f"r_xz={r_xz:.3f} (p={p_xz:.2g})\n"
                    f"r_yz={r_yz:.3f} (p={p_yz:.2g})")
        if show_plane and np.isfinite(r2):
            stats_txt += f"\nPlane $R^2$={r2:.3f}"
        bbox_props = dict(boxstyle="round,pad=0.25", fc="w", ec="k", lw=0.4)
        ax.text2D(0.02, 0.98, stats_txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=7, bbox=bbox_props)

        # ---------- legends ----------
        if len(network_labels) == 7:
            present = set(int(i) for i in np.unique(nets))
            legend_order = [i for i in [1, 0, 3, 2, 6, 5, 4] if i in present]
        else:
            legend_order = list(int(i) for i in np.unique(nets))

        net_handles = [Line2D([0],[0], marker='o', color='w',
                            markerfacecolor=net_colours[i], markeredgecolor='k',
                            markersize=8, label=network_labels[i])
                    for i in legend_order]
        leg1 = ax.legend(handles=net_handles, title="RSN",
                        bbox_to_anchor=(1.32, 1), loc='upper left')
        ax.add_artist(leg1)

        if mode == "multilayer":
            lyr_handles = [Line2D([0],[0], marker=shapes[i], color='w',
                                markeredgecolor='k', markersize=8, label=layer_labels[i])
                        for i in np.unique(layers)]
            ax.legend(handles=lyr_handles, title="Layer", loc='upper right')

        # ---------- save ----------
        outdir = os.path.join(self.data_dir, name)
        os.makedirs(outdir, exist_ok=True)
        if fname is None:
            if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
                i, j, k = dims_to_plot
                fname = f"Scatter3D_d{i+1}{j+1}{k+1}_{'multi' if mode=='multilayer' else 'single'}.png"
            else:
                fname = f"Scatter3D_{'multi' if mode=='multilayer' else 'single'}.png"

        if not show_marginals:
            fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=500, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", os.path.join(outdir, fname))




    def plotScatterCentroids(self,
                            eigvecs,
                            name,
                            eigvecs_to_plot=(0, 1),
                            layer_labels=None,
                            network_labels=None,
                            x_label="Emb1",
                            y_label="Emb2",
                            fname=None,
                            network_cmap="tab20",
                            dot_size=60,
                            annotate=False,
                            # atlas options
                            atlas='schaefer',  # 'schaefer' (7-net) or 'custom'
                            schaefer_label_L="/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
                            schaefer_label_R="/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii"
                            ):
        """
        Plot 2D centroids of RSNs (and layers if present) for the selected embedding dims.
        Axes are fixed to [0, 1] on both x and y.
        """
        import os, numpy as np, matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import nibabel as nib

        # ---------------- helpers ----------------
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

        # ---------------- per-parcel RSN vector ----------------
        if atlas.lower() == 'schaefer':
            if schaefer_label_L is None or schaefer_label_R is None:
                raise ValueError("For atlas='schaefer', provide schaefer_label_L and schaefer_label_R (.label.gii on fs_LR 32k).")
            L_lab, L_map = _load_label_gii(schaefer_label_L)
            R_lab, R_map = _load_label_gii(schaefer_label_R)
            uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
            uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))

            networks0 = []
            for k in uL: networks0.append(_schaefer7_from_name(L_map[k]))
            for k in uR: networks0.append(_schaefer7_from_name(R_map[k]))
            networks0 = np.asarray(networks0, dtype=int)
            N = networks0.size

            if network_labels is None:
                network_labels = ['Visual', 'Somatomotor', 'Dorsal Attn',
                                'Ventral/Salience', 'Limbic', 'Control', 'Default']
        else:
            cats0 = np.loadtxt("cortex_parcel_network_assignments.txt", dtype=int)  # values 1..K
            networks0 = cats0 - 1
            N = networks0.size
            if network_labels is None:
                network_labels = [
                    "Visual1","Visual2","Somatomotor","Cingulo-Opercular",
                    "Dorsal-Attentional","Language","Frontoparietal","Auditory",
                    "Default","Posterior-Multimodal","Ventral-Multimodal","Orbito-Affective"
                ]

        # ---------------- infer rows / mode ----------------
        nrows, ndims = eigvecs.shape
        if nrows == 3 * N:
            mode = "multilayer"
        elif nrows == N:
            mode = "single"
        else:
            if nrows % 3 == 0 and (nrows // 3) == N:
                mode = "multilayer"
            else:
                raise ValueError(f"eigvecs has {nrows} rows, but atlas implies N={N} parcels (expected {N} or {3*N}).")

        # ---------------- parse dims (accept 1-based) ----------------
        x_dim, y_dim = eigvecs_to_plot
        if x_dim >= ndims or y_dim >= ndims:
            x_dim, y_dim = x_dim - 1, y_dim - 1
        if not (0 <= x_dim < ndims and 0 <= y_dim < ndims):
            raise ValueError(f"Requested dims {eigvecs_to_plot} not in [0..{ndims-1}]")

        # ---------------- expand RSNs / layers ----------------
        if mode == "multilayer":
            nets   = np.tile(networks0, 3)           # (3N,)
            layers = np.repeat([0, 1, 2], N)         # Deep/Middle/Superficial
            shapes = ['o', 's', '^']
            if layer_labels is None: layer_labels = ["Deep", "Middle", "Superficial"]
        else:
            nets   = networks0
            layers = np.zeros(N, dtype=int)
            shapes = ['o']
            if layer_labels is None: layer_labels = ["AcrossLayers"]

        # ---------------- colours ----------------
        import matplotlib
        seq_cmap = matplotlib.cm.get_cmap('tab20')

        # how many networks actually appear in the data
        n_nets = int(nets.max()) + 1

        if atlas.lower() == 'schaefer':
            # keep your special Schaefer-7 ordering if you like
            if n_nets != 7:
                raise ValueError(f"Expected 7 networks for Schaefer-7, found {n_nets}")

            shades = [seq_cmap(x) for x in np.linspace(0.2, 0.9, n_nets)]
            order_idx = [1, 0, 3, 2, 6, 5, 4]   # your custom order
            net_colours = [None] * n_nets
            for rank, net_idx in enumerate(order_idx):
                net_colours[net_idx] = shades[rank]
        else:
            # generic mapping: one colour per network
            if n_nets < 1:
                raise ValueError("No networks found in 'nets'.")
            net_colours = [
                seq_cmap(i / max(n_nets - 1, 1)) for i in range(n_nets)
            ]
        # ---------------- compute centroids ----------------
        uniq_layers = np.unique(layers)
        uniq_nets   = np.unique(nets)
        centroids = []  # (layer, net, xmean, ymean, count)

        for lyr in uniq_layers:
            for net in uniq_nets:
                m = (layers == lyr) & (nets == net)
                if not np.any(m): continue
                xy = eigvecs[m][:, [x_dim, y_dim]]
                if xy.size == 0: continue
                centroids.append((int(lyr), int(net),
                                float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1])),
                                int(xy.shape[0])))

        # ---------------- plot ----------------
        fig, ax = plt.subplots(figsize=(7, 7))
        for lyr, net, xm, ym, cnt in centroids:
            ax.scatter(xm, ym,
                    s=dot_size,
                    marker=shapes[int(lyr if mode == "multilayer" else 0)],
                    facecolor=net_colours[int(net)],
                    edgecolor='k', linewidths=0.6, alpha=0.95)
            if annotate:
                ax.text(xm, ym, f" {network_labels[net]}", va='center', ha='left', fontsize=7, alpha=0.9)

        # fixed unit axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(alpha=0.2, linestyle=':')

        # labels/title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("Centroids by RSN" + (" (layers as shapes)" if mode == "multilayer" else ""))

        # ---------------- legends ----------------
        net_handles = [Line2D([0],[0], marker='o', color='w',
                            markerfacecolor=net_colours[i], markeredgecolor='k',
                            markersize=8, label=network_labels[i])
                    for i in np.unique(nets)]
        ax.legend(handles=net_handles, title="RSN",
                bbox_to_anchor=(1.32, 1), loc='upper left')

        if mode == "multilayer":
            lyr_handles = [Line2D([0],[0], marker=shapes[i], color='w',
                                markeredgecolor='k', markersize=8, label=layer_labels[i])
                        for i in np.unique(layers)]
            ax.add_artist(ax.legend(handles=lyr_handles, title="Layer", loc='upper right'))

        # ---------------- save ----------------
        outdir = os.path.join(self.data_dir, name)
        os.makedirs(outdir, exist_ok=True)
        if fname is None:
            fname = f"ScatterCentroids_d{x_dim+1}{y_dim+1}_{'multi' if mode=='multilayer' else 'single'}_unitaxes.png"
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=500, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", os.path.join(outdir, fname))


    def plotNetworkCentroids3D(self,
                            X,
                            Y=None,
                            Z=None,
                            name="Scatter3D_NetCentroids",
                            dims_to_plot=(0, 1, 2),
                            x_label="Emb1",
                            y_label="Emb2",
                            z_label="Emb3",
                            network_labels=None,
                            fname=None,
                            network_cmap="tab20",
                            centroid_size=200,
                            line_alpha=0.6,
                            equalize_axes=True,
                            cube_pad=0.08,
                            proj_type="ortho",
                            annotate=True,
                            # coordinate export
                            write_coords=True,
                            coords_fname=None,
                            print_coords=True,
                            return_coords=True,
                            # Atlas options
                            atlas='schaefer',
                            schaefer_label_L="/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
                            schaefer_label_R="/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii"
                            ):
        """
        3D plot with ONE centroid per network (parcel-mean; if 3N, first averages across layers).
        Draws a translucent gray, NON-CROSSING cycle so each centroid has exactly two connections.
        No plane is plotted. Also exports/prints/returns centroid coordinates.

        Strategy for the polyline:
        - Compute centroids in 3D.
        - Project centroids to a 2D PCA plane (no plotting of the plane).
        - Greedily add the shortest 3D edges while:
                * keeping degree <= 2,
                * avoiding early cycles (until the last edge),
                * forbidding intersections in the 2D projection.
            If this cannot finish, fall back to angle-sorted cycle in the PCA plane (non-crossing).
        """
        import os, csv
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib.lines import Line2D
        import nibabel as nib

        # ---------- helpers ----------
        def _load_label_gii(path):
            g = nib.load(path)
            labs = np.asarray(g.agg_data(), dtype=int).squeeze()
            lt = g.labeltable
            key_to_name = {lab.key: lab.label for lab in lt.labels}
            return labs, key_to_name

        def _schaefer7_from_name(name: str) -> int:
            # 0=Visual, 1=Somatomotor, 2=DorsAttn, 3=Ventral/Sal, 4=Limbic, 5=Control, 6=Default
            n = name.lower()
            if 'vis' in n: return 0
            if 'som' in n or 'sommot' in n: return 1
            if 'dorsattn' in n or ('dors' in n and 'attn' in n): return 2
            if 'ventattn' in n or 'salventattn' in n or ('vent' in n and 'attn' in n) or 'sal' in n: return 3
            if 'limbic' in n: return 4
            if 'cont' in n or 'control' in n or 'frontoparietal' in n or 'fp' in n: return 5
            if 'default' in n: return 6
            raise ValueError(f"Unrecognized Schaefer-7 network label: {name}")

        # ---------- networks / N ----------
        if atlas.lower() == 'schaefer':
            if schaefer_label_L is None or schaefer_label_R is None:
                raise ValueError("For atlas='schaefer', provide schaefer_label_L and schaefer_label_R.")
            L_lab, L_map = _load_label_gii(schaefer_label_L)
            R_lab, R_map = _load_label_gii(schaefer_label_R)
            uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
            uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))
            nets0 = []
            for k in uL: nets0.append(_schaefer7_from_name(L_map[k]))
            for k in uR: nets0.append(_schaefer7_from_name(R_map[k]))
            networks0 = np.asarray(nets0, int)
            N = networks0.size
            if network_labels is None:
                network_labels = ['Visual','Somatomotor','Dorsal Attn',
                                'Ventral/Salience','Limbic','Control','Default']
            net_abbr = ['VIS','SOM','DAN','VAN','LIM','CON','DMN']
        else:
            cats0 = np.loadtxt("cortex_parcel_network_assignments.txt", dtype=int)
            networks0 = cats0 - 1
            N = networks0.size
            if network_labels is None:
                network_labels = [
                    "Visual1","Visual2","Somatomotor","Cingulo-Opercular",
                    "Dorsal-Attentional","Language","Frontoparietal","Auditory",
                    "Default","Posterior-Multimodal","Ventral-Multimodal","Orbito-Affective"
                ]
            net_abbr = [lab.split()[0][:3].upper() for lab in network_labels]

        # ---------- get x,y,z ----------
        def _from_matrix(M, dims):
            nrows, ndims = M.shape
            i, j, k = dims
            if i >= ndims or j >= ndims or k >= ndims:  # accept 1-based
                i, j, k = i-1, j-1, k-1
            if not (0 <= i < ndims and 0 <= j < ndims and 0 <= k < ndims):
                raise ValueError(f"dims_to_plot {dims} not in [0..{ndims-1}]")
            return M[:, i].astype(float), M[:, j].astype(float), M[:, k].astype(float), nrows

        if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
            x, y, z, nrows = _from_matrix(X, dims_to_plot)
        else:
            if Y is None or Z is None:
                raise ValueError("Provide either a 2D matrix X and dims_to_plot, or explicit vectors X, Y, Z.")
            x = np.asarray(X, float).squeeze()
            y = np.asarray(Y, float).squeeze()
            z = np.asarray(Z, float).squeeze()
            nrows = x.size

        if nrows == 3*N:
            mode = "multilayer"
        elif nrows == N:
            mode = "single"
        elif nrows % 3 == 0 and (nrows // 3) == N:
            mode = "multilayer"
        else:
            raise ValueError(f"Data has {nrows} rows, but atlas implies N={N} (expected {N} or {3*N}).")

        # ---------- average to per-parcel, then per-network ----------
        if mode == "multilayer":
            x_par = (x[0:N] + x[N:2*N] + x[2*N:3*N]) / 3.0
            y_par = (y[0:N] + y[N:2*N] + y[2*N:3*N]) / 3.0
            z_par = (z[0:N] + z[N:2*N] + z[2*N:3*N]) / 3.0
        else:
            x_par, y_par, z_par = x, y, z

        kvals = np.array(sorted(np.unique(networks0)))
        K = len(kvals)
        cx = np.zeros(K); cy = np.zeros(K); cz = np.zeros(K)
        n_parcels = np.zeros(K, dtype=int)
        labels_out = []
        for idx, k in enumerate(kvals):
            sel = (networks0 == k)
            n_parcels[idx] = int(sel.sum())
            if n_parcels[idx] == 0:
                cx[idx] = cy[idx] = cz[idx] = np.nan
            else:
                cx[idx] = float(np.nanmean(x_par[sel]))
                cy[idx] = float(np.nanmean(y_par[sel]))
                cz[idx] = float(np.nanmean(z_par[sel]))
            labels_out.append(network_labels[k] if 0 <= k < len(network_labels) else f"Net {int(k)}")

        # ---------- build a non-crossing 2-regular graph (cycle) ----------
        finite = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cz)
        idx_map = np.where(finite)[0]
        if idx_map.size < 3:
            raise ValueError("Fewer than 3 valid centroids — cannot form non-crossing cycle.")
        CX, CY, CZ = cx[finite], cy[finite], cz[finite]
        Kf = idx_map.size

        # PCA projection to 2D for planarity checks (no plane drawn)
        P = np.vstack([CX, CY, CZ]).T  # (Kf, 3)
        P -= P.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(P, full_matrices=False)
        B = U[:, :2] * S[:2]  # (Kf, 2) scores in PCA plane

        # Distances in 3D (we’ll minimize these), but intersection checks in 2D (B-plane)
        def _pairwise_dists_3d(Q):
            d = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1)
            return d
        D3 = _pairwise_dists_3d(P)

        # segment intersection test in 2D (proper intersection; ignores shared endpoints)
        def _intersect_2d(a, b, c, d, eps=1e-12):
            def orient(p, q, r):
                return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
            def on_seg(p, q, r):
                return (min(p[0], r[0])-eps <= q[0] <= max(p[0], r[0])+eps and
                        min(p[1], r[1])-eps <= q[1] <= max(p[1], r[1])+eps)
            o1 = orient(a, b, c); o2 = orient(a, b, d)
            o3 = orient(c, d, a); o4 = orient(c, d, b)
            # general case
            if (o1*o2 < 0) and (o3*o4 < 0): return True
            # colinear cases
            if abs(o1) < eps and on_seg(a, c, b): return True
            if abs(o2) < eps and on_seg(a, d, b): return True
            if abs(o3) < eps and on_seg(c, a, d): return True
            if abs(o4) < eps and on_seg(c, b, d): return True
            return False

        # Disjoint-set (union-find) to avoid early cycles
        parent = list(range(Kf))
        rank = [0]*Kf
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb: return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        # Candidate edges sorted by 3D length
        cand = []
        for i in range(Kf):
            for j in range(i+1, Kf):
                cand.append((D3[i, j], i, j))
        cand.sort(key=lambda t: t[0])

        deg = np.zeros(Kf, dtype=int)
        edges = []

        def _crosses_any(i, j):
            p1, p2 = B[i], B[j]
            for (a, b) in edges:
                if len({i, j, a, b}) < 4:  # shares endpoint -> allowed
                    continue
                if _intersect_2d(p1, p2, B[a], B[b]):
                    return True
            return False

        for w, i, j in cand:
            if deg[i] == 2 or deg[j] == 2:
                continue
            # avoid intersections
            if _crosses_any(i, j):
                continue
            # avoid early cycles (unless final edge)
            same_comp = (find(i) == find(j))
            if same_comp and len(edges) != Kf - 1:
                continue
            # final edge still must not intersect
            if same_comp and _crosses_any(i, j):
                continue
            # accept
            edges.append((i, j))
            union(i, j)
            deg[i] += 1
            deg[j] += 1
            if len(edges) == Kf:
                break

        # Fallback: if we couldn't make a single 2-regular cycle, use angle order (non-crossing)
        if not (len(edges) == Kf and np.all(deg == 2)):
            edges = []
            deg[:] = 0
            # order by angle around centroid in PCA plane
            ctr = B.mean(axis=0)
            ang = np.arctan2(B[:,1]-ctr[1], B[:,0]-ctr[0])
            order = np.argsort(ang)
            for u, v in zip(order, np.roll(order, -1)):
                edges.append((u, v))
            deg += 2  # each node has two edges in a cycle

        # ---------- figure ----------
        fig = plt.figure(figsize=(8.4, 7.8))
        ax = fig.add_subplot(111, projection='3d')
        try: ax.set_proj_type(proj_type)
        except Exception: pass

        # ---------- colors ----------
        # We want discrete shades going light→dark in the order:
        # Somatomotor (1), Visual (0), Vent/Sal (3), DorsAttn (2), Default (6), Control (5), Limbic (4)
        if len(network_labels) == 7:
            # Use a sequential cmap if provided; if a qualitative cmap is passed, the ramp may not be light→dark.
            try:
                seq_cmap = plt.get_cmap(network_cmap)
            except Exception:
                seq_cmap = plt.get_cmap('tab20')

            shades = [seq_cmap(x) for x in np.linspace(0.2, 0.9, 7)]  # avoid extremes
            order_idx = [1, 0, 3, 2, 6, 5, 4]  # desired light→dark order
            net_colours = [None] * 7
            for rank, net_idx in enumerate(order_idx):
                net_colours[net_idx] = shades[rank]
        else:
            # Fallback for non-7-network sets: categorical palette
            cmap = plt.get_cmap(network_cmap, len(network_labels))
            net_colours = [cmap(i) for i in range(len(network_labels))]

        # centroids
        for local_idx, g_idx in enumerate(idx_map):
            net_idx = int(kvals[g_idx])
            if len(network_labels) == 7 and 0 <= net_idx < 7:
                col = net_colours[net_idx]
            else:
                col = net_colours[net_idx % len(net_colours)]
            ax.scatter(CX[local_idx], CY[local_idx], CZ[local_idx],
                    s=centroid_size, marker='o',
                    facecolor=col, edgecolor='k', linewidths=0.6, alpha=0.95)

        # labels
        if annotate:
            for local_idx, g_idx in enumerate(idx_map):
                kidx = int(kvals[g_idx])
                short = (net_abbr[kidx] if 0 <= kidx < len(net_abbr)
                        else labels_out[g_idx][:3].upper())
                ax.text(CX[local_idx], CY[local_idx], CZ[local_idx],
                        f"  {short}", fontsize=8, va='center')

        # non-crossing gray polyline (each centroid degree 2, single cycle)
        for (i, j) in edges:
            ax.plot([CX[i], CX[j]], [CY[i], CY[j]], [CZ[i], CZ[j]],
                    color='gray', alpha=line_alpha, lw=2.0)

        # axes & look
        ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_zlabel(z_label)
        ax.set_title("Network centroids (non-crossing 2-regular cycle)")

        def _cube_limits(xs, ys, zs):
            xlo, xhi = np.nanmin(xs), np.nanmax(xs)
            ylo, yhi = np.nanmin(ys), np.nanmax(ys)
            zlo, zhi = np.nanmin(zs), np.nanmax(zs)
            xm, ym, zm = (xlo + xhi)/2.0, (ylo + yhi)/2.0, (zlo + zhi)/2.0
            r = max(xhi - xlo, yhi - ylo, zhi - zlo) * 0.5
            r = r if np.isfinite(r) and r > 0 else 1.0
            r *= (1.0 + float(cube_pad))
            return (xm - r, xm + r), (ym - r, ym + r), (zm - r, zm + r)

        if equalize_axes:
            (xl, xh), (yl, yh), (zl, zh) = _cube_limits(CX, CY, CZ)
            ax.set_xlim(xl, xh); ax.set_ylim(yl, yh); ax.set_zlim(zl, zh)
            try: ax.set_box_aspect((1, 1, 1))
            except Exception: pass

        ax.view_init(elev=22, azim=38)

        # ---------- legend (present networks, in light→dark order) ----------
        if len(network_labels) == 7:
            present = set(int(kvals[g_idx]) for g_idx in idx_map)
            legend_order = [i for i in [1, 0, 3, 2, 6, 5, 4] if i in present]
        else:
            present = [int(kvals[g_idx]) for g_idx in idx_map]
            legend_order = [k for k in range(len(network_labels)) if k in present]

        handles = [Line2D([0],[0], marker='o', color='w',
                        markerfacecolor=(net_colours[k] if len(network_labels) == 7
                                        else net_colours[k % len(net_colours)]),
                        markeredgecolor='k', markersize=8,
                        label=(network_labels[k] if 0 <= k < len(network_labels) else f"Net {k}"))
                for k in legend_order]
        ax.legend(handles=handles, title="RSN", loc='upper right')

        # ---------- save fig ----------
        outdir = os.path.join(self.data_dir, name)
        os.makedirs(outdir, exist_ok=True)
        if fname is None:
            if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
                i, j, k = dims_to_plot
                fname = f"NetCentroids3D_d{i+1}{j+1}{k+1}.png"
            else:
                fname = "NetCentroids3D.png"
        fig.tight_layout()
        fpath = os.path.join(outdir, fname)
        fig.savefig(fpath, dpi=500, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", fpath)

        # ---------- export/print/return coordinates ----------
        rows = []
        for idx, k in enumerate(kvals):
            rows.append({
                "network_index": int(k),
                "network_name": labels_out[idx],
                "x": float(cx[idx]),
                "y": float(cy[idx]),
                "z": float(cz[idx]),
                "n_parcels": int(n_parcels[idx]),
            })

        if write_coords:
            if coords_fname is None:
                if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
                    i, j, k = dims_to_plot
                    coords_fname = f"NetCentroids3D_d{i+1}{j+1}{k+1}_coords.csv"
                else:
                    coords_fname = "NetCentroids3D_coords.csv"
            cpath = os.path.join(outdir, coords_fname)
            with open(cpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["network_index", "network_name", "x", "y", "z", "n_parcels"])
                for r in rows:
                    w.writerow([r["network_index"], r["network_name"],
                                f"{r['x']:.6f}", f"{r['y']:.6f}", f"{r['z']:.6f}", r["n_parcels"]])
            print("Saved centroid coordinates:", cpath)

        if print_coords:
            print("Centroid coordinates (network_index, name, x, y, z, n_parcels):")
            for r in rows:
                print(f"{r['network_index']:>2}  {r['network_name']:<16}  "
                    f"{r['x']: .6f}  {r['y']: .6f}  {r['z']: .6f}   {r['n_parcels']}")

        return rows if return_coords else None


    def run_ff_fb_models(self,
        layers,                     # [Sup, Middle, Deep] (each n_parcels,)
        y_send,                     # efferent rDCM gradient (n_parcels,)
        y_recv,                     # afferent rDCM gradient (n_parcels,)
        outdir,                     # REQUIRED: directory to save outputs
        fname,                      # REQUIRED: filename for the bar plot (e.g., "ff_fb_partial_bars.svg")
        robust_se="HC3",
        fdr_alpha=0.05,
        orthogonalize=True,
        xlim=(-0.6, 0.6),
        dpi=500,
    ):
        """
        Builds FB & SD contrasts from layer maps, runs the same regression for:
            send, recv, and send-recv (z-scored difference).
        ALWAYS saves a 3-panel horizontal bar plot and a CSV next to it.

        Bars = partial correlations (unique effects)
        Labels = p(FDR) and ΔR² (unique variance)
        Returns: (df, fig_path, csv_path)
        """
        import os, numpy as np, pandas as pd, statsmodels.api as sm
        import matplotlib.pyplot as plt
        from scipy.stats import zscore
        from statsmodels.stats.multitest import multipletests

        # --- unpack, clean
        Sup, Mid, Deep = [np.asarray(v).ravel() for v in layers]
        y_send = np.asarray(y_send).ravel()
        y_recv = np.asarray(y_recv).ravel()
        M = np.column_stack([Sup, Mid, Deep, y_send, y_recv])
        valid = np.all(np.isfinite(M), axis=1)
        Sup, Mid, Deep, y_send, y_recv = [v[valid] for v in (Sup, Mid, Deep, y_send, y_recv)]

        # --- z-score layer maps
        Sup_z, Mid_z, Deep_z = [zscore(v, ddof=1) for v in (Sup, Mid, Deep)]

        # Planned contrasts
        C_FB = -0.5*Sup_z + 1.0*Mid_z - 0.5*Deep_z           # Middle - (Sup+Deep)/2
        C_SD = 0.5*Sup_z + 0.0*Mid_z - 0.5*Deep_z           # (Sup - Deep)/2

        # Optional orthogonalization: SD ⟂ FB
        if orthogonalize:
            proj = np.dot(C_SD, C_FB) / np.dot(C_FB, C_FB)
            C_SD = C_SD - proj * C_FB

        # Final z-scaling of predictors
        C_FB = zscore(C_FB, ddof=1)
        C_SD = zscore(C_SD, ddof=1)

        def fit_one(y):
            y_z = zscore(y, ddof=1)
            X = np.column_stack([C_FB, C_SD])
            X = sm.add_constant(X)
            res = sm.OLS(y_z, X).fit(cov_type=robust_se) if robust_se else sm.OLS(y_z, X).fit()
            betas = res.params[1:]              # standardized betas
            tvals = res.tvalues[1:]
            pvals = res.pvalues[1:]
            df = res.df_resid
            # partial r from t
            pr = np.sign(tvals) * np.sqrt((tvals**2) / (tvals**2 + df))
            # unique ΔR² via drop-one
            R2_full = res.rsquared
            dR2 = []
            for k in (0,1):
                cols = [i for i in (0,1) if i != k]
                X_red = np.column_stack([C_FB, C_SD])[:, cols]
                X_red = sm.add_constant(X_red)
                r2_red = sm.OLS(y_z, X_red).fit(cov_type=robust_se).rsquared if robust_se \
                        else sm.OLS(y_z, X_red).fit().rsquared
                dR2.append(R2_full - r2_red)
            return betas, pr, np.asarray(pvals), np.asarray(dR2), R2_full

        # Fit 3 outcomes
        b_send, pr_send, p_send, dR2_send, R2_send = fit_one(y_send)
        b_recv, pr_recv, p_recv, dR2_recv, R2_recv = fit_one(y_recv)
        y_diff = zscore(y_send, ddof=1) - zscore(y_recv, ddof=1)
        b_diff, pr_diff, p_diff, dR2_diff, R2_diff = fit_one(y_diff)

        # Assemble tidy table
        rows = []
        for outcome, bs, prs, ps, dR2, R2 in [
            ("send",  b_send, pr_send, p_send, dR2_send, R2_send),
            ("recv",  b_recv, pr_recv, p_recv, dR2_recv, R2_recv),
            ("diff",  b_diff, pr_diff, p_diff, dR2_diff, R2_diff),
        ]:
            rows += [
                {"outcome": outcome, "predictor": "FB", "beta_std": bs[0], "partial_r": prs[0], "p": ps[0],
                "unique_R2": dR2[0], "R2_full": R2},
                {"outcome": outcome, "predictor": "SD", "beta_std": bs[1], "partial_r": prs[1], "p": ps[1],
                "unique_R2": dR2[1], "R2_full": R2},
            ]
        df = pd.DataFrame(rows)

        # BH–FDR across 6 tests
        df["p_FDR"] = multipletests(df["p"].values, alpha=fdr_alpha, method="fdr_bh")[1]
        df["sig_FDR"] = df["p_FDR"] < fdr_alpha

        # --- Make & SAVE plot (always)
        import matplotlib.pyplot as plt
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
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # annotate p(FDR) & ΔR²
            for i, row in sub.iterrows():
                mark = "★" if row["sig_FDR"] else ""
                ax.text(xlim[1], i, f" {mark} p(FDR)={row['p_FDR']:.3g} · ΔR²={row['unique_R2']:.3f}",
                        va="center", ha="left", fontsize=8)

        fig.suptitle("Partial correlations of FF/FB contrasts with rDCM outcomes")
        fig.supxlabel("Partial correlation (unique effect)")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # ensure directory and save both files
        os.makedirs(outdir, exist_ok=True)
        fig_path = os.path.join(outdir, fname)
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        base = os.path.splitext(fname)[0]
        csv_path = os.path.join(outdir, f"{base}.csv")
        df.to_csv(csv_path, index=False)

        return df, fig_path, csv_path


    def plot_horizontal_correlation_bar(
        self,
        layers,
        gradient,
        outdir,
        fname,
        layer_names=None,
        title="Effective connectivity vs. laminar indices",
        xlabel="Association with send/receive gradient",
        xlim=(-0.8, 0.8),
        save_path=None,
        alpha=0.05,
        robust_se="HC3",        # None for classic OLS SEs
        do_fdr=True,
        ridge_alpha=None,       # e.g., 1.0 to add a ridge sensitivity
    ):
        """
        Bars = partial correlations from a joint model y ~ superficial + middle + deep
        Dots = marginal Pearson r (your original numbers)
        Also saves a CSV with betas, partial r, marginal r, ΔR², VIF, FDR p.
        """
        import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
        from scipy.stats import pearsonr, zscore
        import statsmodels.api as sm
        from statsmodels.stats.multitest import multipletests
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X_list = [np.asarray(l).ravel() for l in layers]
        y = np.asarray(gradient).ravel()
        if any(len(l) != len(y) for l in X_list):
            raise ValueError("All layer vectors must have the same length as the gradient.")
        p = len(X_list)
        if layer_names is None:
            layer_names = [f"Layer {i+1}" for i in range(p)]

        # stack & clean
        X = np.column_stack(X_list)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[valid, :], y[valid]

        # --- marginal correlations (your original numbers)
        marg_r, marg_p = zip(*[pearsonr(y, X[:, i]) for i in range(p)])
        marg_r, marg_p = np.asarray(marg_r), np.asarray(marg_p)

        # --- standardize for comparable betas
        y_z = zscore(y, ddof=1)
        X_z = zscore(X, axis=0, ddof=1)

        # multicollinearity diagnostic
        vifs = [variance_inflation_factor(X_z, i) for i in range(p)]

        # --- OLS partial effects
        X_fit = sm.add_constant(X_z)
        ols = sm.OLS(y_z, X_fit)
        res = ols.fit(cov_type=robust_se) if robust_se else ols.fit()
        betas = res.params[1:]; tvals = res.tvalues[1:]; pvals = res.pvalues[1:]
        df_resid = int(res.df_resid); R2_full = float(res.rsquared)

        # partial r from t
        partial_r = np.sign(tvals) * np.sqrt((tvals**2) / (tvals**2 + df_resid))

        # unique variance (ΔR²)
        unique_r2 = []
        for k in range(p):
            cols = [i for i in range(p) if i != k]
            res_red = sm.OLS(y_z, sm.add_constant(X_z[:, cols])).fit(cov_type=robust_se) if robust_se \
                    else sm.OLS(y_z, sm.add_constant(X_z[:, cols])).fit()
            unique_r2.append(R2_full - float(res_red.rsquared))
        unique_r2 = np.asarray(unique_r2)

        # optional ridge sensitivity (doesn't affect bars; just logged)
        ridge_info = {}
        if ridge_alpha is not None:
            # closed-form ridge on standardized X: beta = (X'X + λI)^-1 X'y
            lam = float(ridge_alpha)
            XtX = X_z.T @ X_z
            beta_ridge = np.linalg.solve(XtX + lam * np.eye(p), X_z.T @ y_z)
            ridge_info = {"ridge_alpha": lam, "beta_ridge_std": beta_ridge}

        # FDR across the three predictors (widen this family if you combine send+receive)
        if do_fdr:
            rej, p_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        else:
            rej, p_fdr = np.array([p < alpha for p in pvals]), pvals

        # --- plot: bars = partial r; dots = marginal r
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        y_pos = np.arange(p)
        ax.barh(y_pos, partial_r)
        ax.plot(marg_r, y_pos, 'o', markersize=5)  # dots for marginal r
        ax.axvline(0, lw=1)
        ax.set_yticks(y_pos, labels=layer_names)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)

        # annotate p(FDR) and ΔR²
        for i, (pf, ur2, sig) in enumerate(zip(p_fdr, unique_r2, rej)):
            mark = "★" if sig else ""
            ax.text(xlim[1], i, f" {mark} p(FDR)={pf:.3g} · ΔR²={ur2:.3f}",
                    va="center", ha="left", fontsize=8)

        fig.tight_layout()
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=500, bbox_inches="tight")

        # table for methods/results
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

        csv_path = os.path.join(outdir, os.path.splitext(fname)[0] + ".csv")
        df.to_csv(csv_path, index=False)

        # console hints
        print(df)
        if any(v > 5 for v in vifs):
            print(f"[warn] High collinearity (VIF>5): {vifs}. Consider ridge_alpha=1.0 sensitivity or contrasts.")

        return df, res


    def plotTwoDimEmbedding_byNetwork(self,
                                    eigvecs,
                                    name,
                                    eigvecs_to_plot=(1, 2),   # 0-based dims
                                    layer_labels=None,
                                    network_labels=None,
                                    network_cmap='tab20'):
        # infer dims & parcel count
        x_dim, y_dim = eigvecs_to_plot
        P3, _ = eigvecs.shape
        P = P3 // 3

        # load & tile networks
        cats0 = np.loadtxt('cortex_parcel_network_assignments.txt', dtype=int)
        networks0 = cats0 - 1
        networks = np.tile(networks0, 3)            # length 3P

        # build layers
        layers = np.repeat([0,1,2], P)             # length 3P

        # defaults
        if layer_labels is None:
            layer_labels = ['Superficial','Middle','Deep']
        if network_labels is None:
            network_labels = [
                "Visual1","Visual2","Somatomotor","Cingulo-Opercular",
                "Dorsal-Attentional","Language","Frontoparietal","Auditory",
                "Default","Posterior-Multimodal","Ventral-Multimodal","Orbito-Affective"
            ]

        # get 12 colors
        base_cmap = plt.get_cmap(network_cmap, len(network_labels))
        network_colors = [base_cmap(i) for i in range(len(network_labels))]

        coords2d = eigvecs[:, [x_dim, y_dim]]
        
        # set up subplots
        fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)
        axes = axes.flatten()

        shapes = ['o','s','^']
        for net in range(len(network_labels)):
            ax = axes[net]
            mask_net = (networks == net)

            net_coords = coords2d[mask_net]
            net_layers = layers[mask_net]

            # compute silhouette score if possible
            if len(np.unique(net_layers)) > 1:
                sil_score = silhouette_score(net_coords, net_layers)
            else:
                sil_score = np.nan

            for lyr in [0,1,2]:
                mask = mask_net & (layers == lyr)
                if not mask.any():
                    continue
                ax.scatter(
                    eigvecs[mask, x_dim],
                    eigvecs[mask, y_dim],
                    marker=shapes[lyr],
                    facecolor=network_colors[net],
                    edgecolor='k',
                    alpha=0.7,
                    label=layer_labels[lyr]
                )

            title = f"{network_labels[net]}\nSilhouette = {sil_score:.2f}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(f'EV {x_dim+1}')
            ax.set_ylabel(f'EV {y_dim+1}')

            if net == 0:
                ax.legend(title='Layer', loc='best')

        for ax in axes[len(network_labels):]:
            ax.axis('off')

        plt.tight_layout()
        eigstr = f"{x_dim}{y_dim}"
        outpath = f"{self.data_dir}/{name}/Laplacian_embedding_byNetwork_{eigstr}.png"
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()




    def eigvecs_to_nifti(self, eigvecs, name, hcp_atlas=True, force_run=True, scaleEigVecs=False, saveNifti=False):
        
        if scaleEigVecs:
            M = np.max(np.abs(eigvecs), axis=0)  # Find max absolute value per eigenvector
            eigvecs_scaled = eigvecs / M  # Normalize eigenvectors by their max absolute value
            M_max = np.max(np.abs(eigvecs_scaled))  # Rescale to the same max absolute value
            eigvecs = eigvecs_scaled * M_max  # Rescale back to the target max absolute value

        parcel_atlas_img = nib.load(self.atlas_dir)
        parcel_atlas = parcel_atlas_img.get_fdata()
        unique_parcels = np.unique(parcel_atlas)
        
        if hcp_atlas:
            warnings.warn("Selecting cortex parcels of the HCP-MMP1.0 atlas. Modify the code to use a different atlas.")
            unique_parcels = unique_parcels[(unique_parcels >= 1001) & (unique_parcels <= 3000) & (unique_parcels != 2000)]  
        else:
            unique_parcels = unique_parcels[(unique_parcels >0)]  

        print(f"Unique parcels: {len(unique_parcels)}")

        total_regions = eigvecs.shape[0]  # Total number of nodes
        num_components = eigvecs.shape[1] # Number of eigenvectors
        threshold = 40  

        if num_components > threshold:
            indices = list(range(20)) + list(range(num_components - 20, num_components))
        else:
            indices = list(range(num_components))

        if total_regions % self.num_layers != 0:
            raise ValueError("Total regions must be evenly divisible by number of layers.")

        print(f"Mapping {total_regions} nodes into {self.num_layers} layers of {self.N} regions each.")

        # Split eigvecs into layers dynamically
        eig_layers = np.split(eigvecs, self.num_layers, axis=0)

        # Loop through eigenvector dimensions
        for i in indices:  
            if force_run or not os.path.exists(f"{self.data_dir}/{name}/eigenvector_layers"):
                
                try:
                    os.makedirs(f"{self.data_dir}/{name}/eigenvector_layers{self.addName}", exist_ok=True)  # Create folder for layer-wise maps
                except:
                    os.makedirs(f"{self.data_dir}/{name}/eigenvector_layers", exist_ok=True)                
                
                layer_imgs = []

                for layer_idx, layer_data in enumerate(eig_layers):  
                            
                    map_3D = np.zeros_like(parcel_atlas)

                    for roi_idx, parcel in enumerate(unique_parcels):
                        parcel_mask = np.zeros(parcel_atlas.shape)
                        parcel_mask[parcel_atlas == parcel] = 1
                        parcel_mask = np.array(parcel_mask, dtype=bool)
                        final_mask = parcel_mask
                        map_3D[final_mask] = layer_data[roi_idx, i]

                    layer_img = nib.Nifti1Image(map_3D, affine=parcel_atlas_img.affine)
                    if saveNifti:
                        nib.save(layer_img, f"{self.data_dir}/{name}/eigenvector_layers/eigenvector_{i+1}_layer_{layer_idx+1}.nii.gz")
                    layer_imgs.append(layer_img)  # Store for later plotting
                #self.__plot_on_volume__(layer_imgs, i+1, name)

            Xp_layers = []  
            for layer_idx in range(self.num_layers):
                Xp_layers.append(eig_layers[layer_idx][:, i])
            Xp_layers = np.array(Xp_layers)

            if hcp_atlas:
                self.__plot_on_mmhcp_surface_multipleLayers__(Xp_layers.T, i+1, name)
            else:
                self.__plot_on_volume__(layer_imgs, i+1, name)

        print("All brain maps saved successfully!")

    def __plot_on_mmhcp_surface_multipleLayers__(
        self,
        Xp,
        eigValue,
        name,
        vmin=None,
        vmax=None,
        cm="cividis",
        noSubcortical=True,
        titles=["Deep", "Middle", "Superficial", "Average"],
        folder_name="eigenvector_layers",
        # NEW:
        atlas="schaefer",                             # "mmp" (Glasser) or "schaefer"
        schaefer_label_L="/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",                   # path to Schaefer*.L.label.gii (fs_LR 32k)
        schaefer_label_R="/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii"                    # path to Schaefer*.R.label.gii (fs_LR 32k)
    ):
        """
        Xp: (n_parcels, n_layers) array, ordered [LH parcels..., RH parcels...].
            For Schaefer, order must follow sorted label keys per hemi.
        atlas:
            - "mmp": uses hcp.mmp + hcp.unparcellate (your current setup)
            - "schaefer": uses L/R .label.gii provided via schaefer_label_L/R
        """
        import os
        import numpy as np
        import nibabel as nib
        import matplotlib.pyplot as plt
        from nilearn import plotting

        os.makedirs(f"{self.data_dir}/{name}/{folder_name}", exist_ok=True)

        # -------- helpers --------
        def _load_label_gii(path):
            g = nib.load(path)
            # label.gii -> integer keys per vertex; 0 = medial wall
            return np.asarray(g.agg_data(), dtype=int).squeeze()

        def _build_rank_map(keys):
            u = np.unique(keys[keys > 0])
            return {k: i for i, k in enumerate(sorted(u))}, len(u)

        def _map_parcels_to_vertices_schaefer(vals_lr, L_lab, R_lab, L_rank, R_rank, n_hemi):
            """vals_lr has length 2*n_hemi in order [LH..., RH...]"""
            # Left
            left = np.full(L_lab.shape, np.nan, float)
            mL = L_lab > 0
            if np.any(mL):
                # map each vertex's parcel key -> index -> value
                idxL = np.array([L_rank[k] for k in L_lab[mL]])
                left[mL] = vals_lr[idxL]
            # Right
            right = np.full(R_lab.shape, np.nan, float)
            mR = R_lab > 0
            if np.any(mR):
                idxR = np.array([R_rank[k] for k in R_lab[mR]])
                right[mR] = vals_lr[n_hemi + idxR]
            return left, right

        # -------- load/prepare atlas mapping --------
        if atlas.lower() == "mmp":
            # MMP: we can use your existing utilities
            mmp_labels = hcp.mmp.labels
            n_parcels_target = len(mmp_labels)
            # function to map one layer (1D parcel vector) -> (left_vertices, right_vertices)
            def map_layer(vals_lr):
                vtx_both = hcp.cortex_data(hcp.unparcellate(vals_lr, hcp.mmp))
                nL = len(vtx_both) // 2
                return vtx_both[:nL], vtx_both[nL:]

        elif atlas.lower() == "schaefer":
            if schaefer_label_L is None or schaefer_label_R is None:
                raise ValueError("For atlas='schaefer', provide schaefer_label_L and schaefer_label_R (.label.gii on fs_LR 32k).")
            L_lab = _load_label_gii(schaefer_label_L)
            R_lab = _load_label_gii(schaefer_label_R)
            L_rank, nL_parcels = _build_rank_map(L_lab)
            R_rank, nR_parcels = _build_rank_map(R_lab)
            assert nL_parcels == nR_parcels, f"Unequal parcels per hemi: L={nL_parcels}, R={nR_parcels}"
            n_parcels_target = 2 * nL_parcels
            def map_layer(vals_lr):
                return _map_parcels_to_vertices_schaefer(vals_lr, L_lab, R_lab, L_rank, R_rank, nL_parcels)

        else:
            raise ValueError("atlas must be 'mmp' or 'schaefer'.")

        # -------- pad/truncate parcels if requested --------
        Xp = np.asarray(Xp)
        if Xp.ndim != 2:
            raise ValueError("Xp must be a 2D array of shape (n_parcels, n_layers).")

        current_length = Xp.shape[0]
        if noSubcortical:
            zeros_to_add = n_parcels_target - current_length
            if zeros_to_add > 0:
                Xp = np.concatenate((Xp, np.zeros((zeros_to_add, Xp.shape[1]))), axis=0)
            elif zeros_to_add < 0:
                raise ValueError(f"Xp has {current_length} rows but atlas expects {n_parcels_target} (LH+RH).")
        else:
            if current_length != n_parcels_target:
                raise ValueError(f"Xp has {current_length} rows but atlas expects {n_parcels_target} (LH+RH).")

        # -------- compute global vmin/vmax across all layers --------
        left_right_layers = []
        for i in range(Xp.shape[1]):
            left_i, right_i = map_layer(Xp[:, i])
            left_right_layers.append((left_i, right_i))

        # stack all vertex data to choose robust limits
        all_data = np.hstack([np.concatenate((L, R)) for (L, R) in left_right_layers])
        if vmin is None or vmax is None:
            # robust 2–98% range
            finite = np.isfinite(all_data)
            if not np.any(finite):
                raise ValueError("All mapped values are NaN.")
            vmin, vmax = np.nanpercentile(all_data[finite], [2, 98])
            if vmin == vmax:
                vmin -= 1e-6
                vmax += 1e-6

        # -------- figure & plotting --------
        orientations = ["lateral", "medial", "medial", "lateral"]
        fig, axes = plt.subplots(
            Xp.shape[1], len(orientations),
            figsize=(20, 5 * Xp.shape[1]),
            subplot_kw={"projection": "3d"}
        )

        for i in range(Xp.shape[1]):
            left_i, right_i = left_right_layers[i]
            row_title = titles[i] if (titles is not None and i < len(titles)) else f"Layer {i+1}"

            for j, view in enumerate(orientations):
                try:
                    ax = axes[i, j]
                except Exception:
                    ax = axes[j]

                if j in (0, 1):  # left hemi views
                    plotting.plot_surf_stat_map(
                        hcp.mesh.inflated_left,
                        left_i,
                        view=view,
                        colorbar=False,
                        bg_map=hcp.mesh.sulc_left,
                        bg_on_data=True,
                        darkness=0.3,
                        axes=ax,
                        figure=fig,
                        cmap=cm,
                        vmin=vmin, vmax=vmax,
                        symmetric_cbar=False,
                    )
                else:  # right hemi views
                    plotting.plot_surf_stat_map(
                        hcp.mesh.inflated_right,
                        right_i,
                        view=view,
                        colorbar=False,
                        bg_map=hcp.mesh.sulc_right,
                        bg_on_data=True,
                        darkness=0.3,
                        axes=ax,
                        figure=fig,
                        cmap=cm,
                        vmin=vmin, vmax=vmax,
                        symmetric_cbar=False,
                    )

                ax.set_title(f"{row_title} - {orientations[j].capitalize()}", fontsize=14)

        # shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cm), norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

        plt.suptitle(f"Eigenvector {eigValue}", fontsize=16)
        out_path = f"{self.data_dir}/{name}/{folder_name}/eigenvectorSurface_{eigValue}_twoHem.png"
        print(out_path)
        plt.savefig(out_path, facecolor="white", dpi=300)
        plt.close()

        return out_path

    def plotScree(self, eigvals, name, sort=False):
            
        if sort:
            eigvals_sorted = np.sort(eigvals)[::-1]
        else:
            eigvals_sorted = eigvals
        
        # Compute cumulative explained variance (normalized to 100%)
        eigvals_cumsum = np.cumsum(eigvals_sorted) / np.sum(eigvals_sorted) * 100

        num_components = np.size(eigvals)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(range(1, num_components + 1), eigvals_sorted, marker='o', linestyle='-', color='b', label="Eigenvalues")
        ax1.set_xlabel('Component Number')
        ax1.set_ylabel('Eigenvalue', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create second y-axis for cumulative percentage
        ax2 = ax1.twinx()
        ax2.plot(range(1, num_components + 1), eigvals_cumsum, marker='s', linestyle='--', color='r', label="Cumulative Sum")
        ax2.set_ylabel('Cumulative Sum (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Title and grid
        plt.title('Scree Plot with Cumulative Sum')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Show plot
        plt.savefig(f"{self.data_dir}/{name}/screePlot.png", bbox_inches="tight")
        plt.close()

    def run_plot_zeroCrossings(self, W, U, name):

        n_ROI = U.shape[0]  # Number of regions (nodes)
        wZC = np.zeros(U.shape[1])  # Initialize zero-crossing count array
        for u in range(U.shape[1]):  # Loop through each eigenvector
            summ = 0  # Initialize sum            
            for i in range(n_ROI - 1):  # Loop through each connection
                for j in range(i + 1, n_ROI):
                    if U[i,u] * U[j,u] < 0:  # Check if signs are opposite
                        summ += (W[i, j] >= 1)  # Increment if connection exists
            
            wZC[u] = summ  # Store result
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(wZC) + 1), wZC, marker='o', linestyle='-', color='b')
        plt.xlabel('Eigenvector')
        plt.ylabel('Zero Crossings')
        plt.title('Zero Crossings for Laplacian Eigenvectors')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{self.data_dir}/{name}/Crossings.png", bbox_inches="tight")
        plt.close()
        
        return wZC
    
    def fisher_z(self,r):
        with np.errstate(divide='ignore', invalid='ignore'):
            z = 0.5 * np.log((1 + r) / (1 - r))
            z[np.isinf(z)] = 0
        return z
    

    def plotConnectogram(self, connectivity_matrix, name, layer, color="red", n=360, percent=20):

        os.makedirs(f"{self.data_dir}/{name}/Connectogram", exist_ok=True)  # Create folder for layer-wise maps

        labels = hcp.mmp.labels #self.getLabels()

        # Create a graph just to use the circular layout
        G = nx.Graph()
        G.add_nodes_from(range(n))
        pos = nx.circular_layout(G)

        # Assign edges per layer
        edges = self.get_top_percent_edges(connectivity_matrix, percent=percent)

        # Initialize plot
        fig = go.Figure()

        # Add layers
        self.plot_edges(fig, edges, pos, color=color, name=layer)

        # Add nodes
        for node, (x, y) in pos.items():
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                text=labels[node],
                mode='markers+text',
                textposition='top center',
                textfont=dict(size=8),  
                marker=dict(size=1, color='gray'),
                showlegend=False
            ))

        fig.update_layout(
            width=1200, 
            height=1200,
            showlegend=True,
            title="Multi-layer Connectogram",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        fig.write_image(f"{self.data_dir}/{name}/Connectogram/Crossings_{layer}.png")

    
    def plotConnectogram_allInOne(self, layer1, layer2, layer3, name, percent=20, n=360):

        os.makedirs(f"{self.data_dir}/{name}/Connectogram", exist_ok=True)  # Create folder for layer-wise maps

        labels = hcp.mmp.labels #self.getLabels()

        # Create a graph just to use the circular layout
        G = nx.Graph()
        G.add_nodes_from(range(n))
        pos = nx.circular_layout(G)

        # Assign edges per layer
        edges1 = self.get_top_percent_edges(layer1, percent=percent)
        edges2 = self.get_top_percent_edges(layer2, percent=percent)
        edges3 = self.get_top_percent_edges(layer3, percent=percent)

        # Initialize plot
        fig = go.Figure()

        # Add layers
        self.plot_edges(fig, edges1, pos, 'red', 'Superficial')
        self.plot_edges(fig, edges2, pos, 'green', 'Middle')
        self.plot_edges(fig, edges3, pos, 'blue', 'Deep')

        # Add nodes
        for node, (x, y) in pos.items():
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                text=labels[node],
                mode='markers+text',
                textposition='top center',
                textfont=dict(size=8),  
                marker=dict(size=1, color='gray'),
                showlegend=False
            ))

        # Final layout
        fig.update_layout(
            width=1200, 
            height=1200,
            showlegend=True,
            title="Multi-layer Connectogram",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        fig.write_image(f"{self.data_dir}/{name}/Connectogram/Crossings_All.png")

    def plot_edges(self,fig, edges, pos, color, name):
        edge_x, edge_y = [], []
        for u, v in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.1, color=color),
            mode='lines',
            name=name,
            hoverinfo='none'
        ))

    def get_top_percent_edges(self, mat, percent=20):

        assert 0 < percent <= 100, "Percent must be between 0 and 100."

        # Get upper triangle values (excluding diagonal)
        triu_indices = np.triu_indices_from(mat, k=1)
        edge_weights = mat[triu_indices]

        # Compute threshold for top `percent` strongest connections
        num_edges = len(edge_weights)
        k = int(np.ceil(num_edges * percent / 100.0))
        if k == 0:
            return []

        # Get indices of top-k weights
        top_k_indices = np.argpartition(edge_weights, -k)[-k:]

        # Map back to matrix indices
        edges = [(triu_indices[0][i], triu_indices[1][i]) for i in top_k_indices]
        return edges


    def rich_club_sweep(self,
                        connectivity_matrix: np.ndarray,
                        deg_cutoff_percentile: float = 95,
                        normalized: bool = True,
                        seed: int = 33):
        """
        Build a weighted graph,
        compute (optionally normalized) rich-club coefficients φ(k), then
        summarize them via AUC and pick the top-degree nodes.
        
        Returns
        -------
        phi_auc : float
            Area under the φ(k) vs. k curve for this threshold.
        rich_club_nodes : List[int]
            Node indices whose degree ≥ the deg_cutoff_percentile of the degree distribution.
        """

        G = nx.from_numpy_array(connectivity_matrix)
        phi_raw = nx.rich_club_coefficient(G, normalized=False)
        
        if normalized:
            # compute null model φ_rand(k)
            phi_rand = nx.rich_club_coefficient(G, normalized=False, seed=seed)
            # safe normalization
            phi = {}
            for k, v in phi_raw.items():
                denom = phi_rand.get(k, 0.0)
                phi[k] = (v / denom) if denom > 0 else np.nan
        else:
            phi = phi_raw

        ks   = np.array(sorted(phi))
        phis = np.array([phi[k] for k in ks])
        valid = ~np.isnan(phis)
        phi_auc = np.trapz(phis[valid], x=ks[valid])

        degrees = np.array([d for _, d in G.degree()])
        deg_cut = np.percentile(degrees, deg_cutoff_percentile)
        rich_club_nodes = [n for n, d in G.degree() if d >= deg_cut]

        return phi_auc, rich_club_nodes
    
    def most_common_members(self, members_list, N, min_frac=0.8):

        T = len(members_list)
        # build membership matrix
        membership = np.zeros((N, T), dtype=int)
        for t, mids in enumerate(members_list):
            membership[mids, t] = 1
        freq = membership.sum(axis=1) / T  # fraction of thresholds
        # pick stable members
        stable = np.where(freq >= min_frac)[0]
        
        return stable, freq


    def plotRichClub(self, layer1, layer2, layer3, name, n=360):

        rich_club_node_lists = [layer1, layer2, layer3]
        Xp = np.zeros((n, 3))

        for i, rc_nodes in enumerate(rich_club_node_lists):
            Xp[rc_nodes, i] = 1

        self.__plot_on_mmhcp_surface_multipleLayers__(
            Xp, f"Rich", name, folder_name="NetworkMeasures"
        )

    def run_plot_FstatComp(self, eigvecs, name, thresh=2.5, target=1.0, k=10):
        n_rows, n_cols = eigvecs.shape
        if n_rows != 1080:
            raise ValueError(f"Expected 1080 rows; got {n_rows}")

        avg_corrs  = np.empty(n_cols)
        dissimilar = np.empty(n_cols)

        for i in range(n_cols):
            col = eigvecs[:, i]

            segs = [col[j*360:(j+1)*360] for j in range(3)]

            # pairwise Pearson
            r01 = np.corrcoef(segs[0], segs[1])[0, 1]
            r02 = np.corrcoef(segs[0], segs[2])[0, 1]
            r12 = np.corrcoef(segs[1], segs[2])[0, 1]

            zs = np.arctanh([r01, r02, r12])
            z_bar = zs.mean()
            r_bar = np.tanh(z_bar)
            avg_corrs[i] = r_bar
            dissimilar[i] = 1 - r_bar


        # mu   = dissimilar.mean()
        # sigma = dissimilar.std(ddof=0)
        # z    = (dissimilar - mu) / sigma
        
        # out_low  = np.where(z < -thresh)[0]
        # out_high = np.where(z >  thresh)[0]

        outliers, diffs, neigh_mean = self.__detect_local_outliers__(dissimilar, k_neighbors=2, method='zscore')

        x = np.arange(1, len(avg_corrs) + 1)

        fig, ax1 = plt.subplots()
        ax1.plot(x, dissimilar, marker='o', label='Avg Pearson r')
        ax1.set_xlabel('Eigenvector Number')
        ax1.set_ylabel('Average Dissimilarity (1-r)')
        fig.suptitle('Difference Metrics per Eigenvector')
        ax1.grid(True)
        fig.tight_layout()
        fig.savefig(f"{self.data_dir}/{name}/DifferenceInEigvecs.png", bbox_inches="tight")
        plt.close()

        # diffs = np.abs(dissimilar - target)
        # closest_idxs = np.argsort(diffs)[:k]
            
        with open(f"{self.data_dir}/{name}/OutlierEigenvecs.txt", 'w') as f:
            # f.write(f"Z-score outliers (threshold = ±{thresh}σ)\n\n")
            # f.write("Low outliers (z < -{0}):\n".format(thresh))
            # for idx in out_low:
            #     f.write(f"{idx}\t{dissimilar[idx]:.6f}\n")
            # f.write("\nHigh outliers (z > {0}):\n".format(thresh))
            # for idx in out_high:
            #     f.write(f"{idx}\t{dissimilar[idx]:.6f}\n")
            # f.write("\nOrthogonal (k = {0}):\n".format(k))
            # for idx in closest_idxs:
            #     f.write(f"{idx}\t{dissimilar[idx]:.6f}\n")
            if len(outliers) == 0:
                f.write("No outliers detected.\n")
                return
            f.write("Index\tValue\tNeighborMean\tDiff\n")
            for i in outliers:
                f.write(f"{i+1}\t{dissimilar[i]:.6f}\t{neigh_mean[i]:.6f}\t{diffs[i]:.6f}\n")


    def __detect_local_outliers__(self,
                            vals: np.ndarray,
                            k_neighbors: int = 1,
                            method: str = 'zscore',
                            thresh: float = 2.0):
        """
        Identify indices i where vals[i] deviates from the average of its
        k_neighbors on each side by more than thresh (either in SD units or
        absolute units).
        
        Parameters
        ----------
        vals        : 1D array of length N (e.g. your dissimilarity vector)
        k_neighbors : how many neighbors to include on each side (default 1)
        method      : 'zscore' to threshold on z = (diff - μ)/σ,
                    'abs'    to threshold on |diff|
        thresh      : threshold in SDs (if method='zscore') or in units (if 'abs')
        
        Returns
        -------
        outlier_idxs : 1D array of indices in vals flagged as local outliers
        diffs        : 1D array of length N of vals[i] - mean(neighbors)
        neigh_mean   : 1D array of length N of the neighbor means
        """
        N = len(vals)
        diffs = np.empty(N, dtype=float)
        neigh_mean = np.empty(N, dtype=float)

        for i in range(N):
            lo = max(0, i - k_neighbors)
            hi = min(N, i + k_neighbors + 1)
            # all indices in [lo, hi) except i itself
            nbrs = [j for j in range(lo, hi) if j != i]
            if not nbrs:
                # if N=1, we can’t compare—set diff=0
                neigh_mean[i] = 0.0
                diffs[i] = 0.0
            else:
                m = vals[nbrs].mean()
                neigh_mean[i] = m
                diffs[i] = vals[i] - m

        if method == 'zscore':
            mu, sigma = diffs.mean(), diffs.std(ddof=0)
            z = (diffs - mu) / sigma
            outlier_idxs = np.where(np.abs(z) > thresh)[0]
        elif method == 'abs':
            outlier_idxs = np.where(np.abs(diffs) > thresh)[0]
        else:
            raise ValueError("method must be 'zscore' or 'abs'")

        return outlier_idxs, diffs, neigh_mean
    

    def plotEigenvectorCorrelation(self, eigvecs_orig, name, limit=40, end_num=40):
        
        orig_X = eigvecs_orig.shape[1]
        end = orig_X - end_num
        eigvecs = np.hstack([
            eigvecs_orig[:, :limit],   # cols 0 … limit-1
            eigvecs_orig[:, end:]      # cols end … M-1
            ])
        n_rows, n_cols = eigvecs.shape
        if n_rows != 1080:
            raise ValueError(f"Expected 1080 rows; got {n_rows}")

        layers = [eigvecs[i*360:(i+1)*360, :] for i in range(3)]
        corr_mats = {}

        for i in range(3):
            for j in range(i, 3):
                A = layers[i]
                B = layers[j]

                if i == j:
                    # within-layer: correlation among columns of A
                    # yields X×X matrix
                    C = np.corrcoef(A, rowvar=False)
                else:
                    # between-layer: build a 360×(2X) array [A | B], then
                    # corrcoef(..., rowvar=False) gives 2X×2X block matrix:
                    #   [ Corr(A,A)    Corr(A,B) ]
                    #   [ Corr(B,A)    Corr(B,B) ]
                    # we want the top-right block
                    M = np.concatenate([A, B], axis=1)             # 360×(2X)
                    bigC = np.corrcoef(M, rowvar=False)            # 2X×2X
                    C = bigC[:n_cols, n_cols:2*n_cols]            # X×X

                corr_mats[(i, j)] = C

        pairs = [(0,0), (1,1), (2,2),
                (0,1), (0,2),
                (1,2)]
        tick_labels = list(range(1, limit+1)) + list(range(end+1, orig_X+1))    
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for ax, (i, j) in zip(axes.flat, pairs):
            C = corr_mats[(i, j)]
            im = ax.imshow(C, vmin=-1, vmax=1, cmap='cividis')
            ax.set_title(f"Layer {i} vs Layer {j}")
            # ax.set_xticklabels(tick_labels)
            # ax.set_yticklabels(tick_labels)

            ax.set_xlabel('Eigenvector index')
            ax.set_ylabel('Eigenvector index')
        
        fig.suptitle(f"{name} Layer Correlations", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # shared colorbar
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label('Pearson r')
        
        plt.savefig(f"{self.data_dir}/{name}/CorrelationEigVecsMatrices_First{limit}_Last{end_num}.png", bbox_inches="tight")
        plt.close()


    def zero_lag_fc(self, X):
        """Pearson FC of X (n_parcels×T) at lag 0."""
        Xz = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
        return (Xz @ Xz.T) / (X.shape[1] - 1)

    def lagged_corr(self, X, Y, t):
        """
        Pearson corr of X(t) with Y(t+t).
        X, Y shape = (n_parcels, T)
        t > 0: X leads Y
        t < 0: Y leads X
        """
        n, T = X.shape
        if t >= 0:
            Xtr, Ytr = X[:, :T-t], Y[:, t:]
        else:
            Xtr, Ytr = X[:, -t:],  Y[:, :T+t]
        Xz = (Xtr - Xtr.mean(axis=1, keepdims=True)) / Xtr.std(axis=1, keepdims=True)
        Yz = (Ytr - Ytr.mean(axis=1, keepdims=True)) / Ytr.std(axis=1, keepdims=True)
        return (Xz @ Yz.T) / (Xtr.shape[1] - 1)

    def lagged_multilayer_fc(self, t=2):
        """
        data: np.ndarray, shape (360, 3, T)
        t: integer lag in timepoints
        returns: FC matrix shape (1080, 1080)
        """

        layer_groups = defaultdict(list)
        
        for ifile, file in enumerate(self.npy_files):
            try:
                layer_str = file.split('_')[-1].replace('.npy', '')
                layer_num = int(layer_str)
                layer_groups[layer_num].append(file)
            except Exception as e:
                raise ValueError(f"Could not extract layer number from filename: {file}") from e

        sorted_layers = sorted(layer_groups.items())
        data = np.empty((self.N, self.num_layers, ((ifile+1)//3)*125))
        print(f"Ugly hard coding of T")

        for i, (layer_num, files) in enumerate(sorted_layers):
            all_time_series = []

            for file in files:
                file_path = os.path.join(self.data_dir, file)
                time_series = np.load(file_path)
                all_time_series.append(time_series)

            concatenated = np.concatenate(all_time_series, axis=1)
            data[:, i, :] = concatenated


        n_parcels, n_layers, T = data.shape
        N = n_parcels * n_layers
        M = np.zeros((N, N))

        # 1) fill diagonal blocks with zero‐lag FC, for each layer
        for t in range(n_layers):
            block = self.zero_lag_fc(data[:, t, :])
            i0 = t * n_parcels
            M[i0:i0+n_parcels, i0:i0+n_parcels] = block

        # 2) fill off‐diagonal:
        #    upper‐triangle blocks (ℓ1 < ℓ2) get corr at +τ (layer ℓ1 leads ℓ2)
        #    lower‐triangle gets the opposite, i.e. corr at –τ (layer ℓ2 leads ℓ1)
        for t1 in range(n_layers):
            for t2 in range(t1+1, n_layers):
                X = data[:,t1, :]
                Y = data[:, t2, :]
                C_pos = self.lagged_corr(X, Y, +t)
                C_neg = self.lagged_corr(X, Y, -t)  # equivalently C_pos.T
                i1, i2 = t1*n_parcels, t2*n_parcels

                # ℓ1→ℓ2 block (upper block)  
                M[i1:i1+n_parcels, i2:i2+n_parcels] = C_pos

                # ℓ2→ℓ1 block (lower block)
                M[i2:i2+n_parcels, i1:i1+n_parcels] = C_neg

        return M



    def eigenvector_centrality_calc(self,
                                    adj_matrix,
                                    weight=None): 
        
        G = nx.from_numpy_array(adj_matrix)
        centrality = nx.eigenvector_centrality(G, max_iter=1000, weight=weight)
        centrality_arr = np.array([centrality[i] for i in range(G.number_of_nodes())])

        mat = centrality_arr.reshape(360, 3)
        one_hot_centrality = np.zeros_like(mat, dtype=int)

        # # for each row, find the index of the max and set that position to 1
        idx_max = np.argmax(mat, axis=1)
        one_hot_centrality[np.arange(360), idx_max] = 1 

        return centrality_arr, one_hot_centrality
        

    def eigenvector_centrality_plot(self,
                                    centrality,
                                    one_hot_centrality,
                                    name,
                                    additionalName=''
                                    ):
        cats = np.loadtxt(
            'cortex_parcel_network_assignments.txt', 
            dtype=int
        )
        subs = centrality.shape[-1]

        counts = np.zeros((12, 3,subs), dtype=int)
        averages = np.zeros((12, 3, subs), dtype=float)
        
        for s in range(subs):
            mat = centrality[:,s].reshape(360, 3)
            curr_one_hot = one_hot_centrality[:, :, s]
            for k in range(1, 13):
                mask = (cats == k)
                counts[k-1, :, s] = curr_one_hot[mask, :].sum(axis=0)
                data = mat[mask, :]
                averages[k-1, :, s] = data.mean(axis=0)


        # plotting
        tick_labels = [
            "Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular",
            "Dorsal-Attentional", "Language", "Frontoparietal", "Auditory",
            "Default", "Posterior-Multimodal", "Ventral-Multimodal", "Orbito-Affective"
        ]

        
        fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharey=True)
        axes = axes.flatten()
        for idx, ax in enumerate(axes):
            # bar positions for the 3 layers
            x      = np.arange(1, 4)
            heights= np.mean(counts[idx, :, :],axis=-1) 
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
                edgecolor='black'
            )

            sig = "*" if pval < 0.05 else ""
            ax.text(
                0.5, 0.95,
                f"p = {pval:.3f}{sig}",
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=10
            )

            ax.set_title(tick_labels[idx])
            ax.set_xticks(x)
            ax.set_xticklabels(["Deep", "Middle", "Superficial"])
            ax.set_ylabel("Parcel count - eigenvector centrality \n(± SEM)")

            # annotate each bar with its mean value
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + errors[list(bars).index(bar)] + 0.01*h,  # place above error‐bar
                    f"{h:.4f}",
                    ha="center",
                    va="bottom"
                )

        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/{name}/EigvecsBelongingToEachRSN_EigCentrality_AcrossSubs{additionalName}.png", bbox_inches="tight")
        plt.close(fig)



        fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharey=True)
        axes = axes.flatten()
        for idx, ax in enumerate(axes):
            # bar positions for the 3 layers
            x      = np.arange(1, 4)
            heights= np.mean(averages[idx, :, :],axis=-1) 
            errors = np.std(averages[idx, :, :], axis=-1)/np.sqrt(subs)  

            stat, pval = f_oneway(
                averages[idx, 0, :],
                averages[idx, 1, :],
                averages[idx, 2, :]
            )


            # draw bars with error‐bars
            bars = ax.bar(
                x, heights,
                yerr=errors,
                capsize=5,
                edgecolor='black'
            )

            sig = "*" if pval < 0.05 else ""
            ax.text(
                0.5, 0.95,
                f"p = {pval:.3f}{sig}",
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=10
            )

            ax.set_title(tick_labels[idx])
            ax.set_xticks(x)
            ax.set_xticklabels(["Deep", "Middle", "Superficial"])
            ax.set_ylabel("Mean Centrality\n(± SEM)")

            # annotate each bar with its mean value
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + errors[list(bars).index(bar)] + 0.01*h,  # place above error‐bar
                    f"{h:.4f}",
                    ha="center",
                    va="bottom"
                )

        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/{name}/EigvecsBelongingToEachRSN_EigCentrality_MeanSEM_AcrossSubs{additionalName}.png",
                    bbox_inches="tight")
        plt.close(fig)

    def eigenvector_centrality_plot_avg(self,
                                    centrality,
                                    one_hot_centrality,
                                    name,
                                    additionalName=''
                                    ):
        cats = np.loadtxt(
            'cortex_parcel_network_assignments.txt', 
            dtype=int
        )

        counts = np.zeros((12, 3), dtype=int)
        averages = np.zeros((12, 3), dtype=float)
        sem = np.zeros((12, 3), dtype=float)
        
        mat = centrality.reshape(360, 3)
        for k in range(1, 13):
            mask = (cats == k)
            counts[k-1, :] = one_hot_centrality[mask, :].sum(axis=0)
            data = mat[mask, :]
            averages[k-1, :] = data.mean(axis=0)
            sem[k-1,:] = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])


        # plotting
        tick_labels = [
            "Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular",
            "Dorsal-Attentional", "Language", "Frontoparietal", "Auditory",
            "Default", "Posterior-Multimodal", "Ventral-Multimodal", "Orbito-Affective"
        ]

        
        fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharey=True)
        axes = axes.flatten()
        for idx, ax in enumerate(axes):
            # bar positions for the 3 layers
            x      = np.arange(1, 4)
            heights= counts[idx, :] 

            bars = ax.bar(
                x, heights,
                capsize=5,
                edgecolor='black'
            )

            ax.set_title(tick_labels[idx])
            ax.set_xticks(x)
            ax.set_xticklabels(["Deep", "Middle", "Superficial"])
            ax.set_ylabel("Parcel count - eigenvector centrality \n(± SEM)")

            # annotate each bar with its mean value
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01*h,  # place above error‐bar
                    f"{h:.4f}",
                    ha="center",
                    va="bottom"
                )

        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/{name}/EigvecsBelongingToEachRSN_EigCentrality_AverageAdj{additionalName}.png", bbox_inches="tight")
        plt.close(fig)



        fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharey=True)
        axes = axes.flatten()
        for idx, ax in enumerate(axes):

            x      = np.arange(1, 4)
            heights= averages[idx, :]
            errors = sem[idx, :]

            # stat, pval = f_oneway(
            #     averages[idx, 0, :],
            #     averages[idx, 1, :],
            #     averages[idx, 2, :]
            # )

            # draw bars with error‐bars
            bars = ax.bar(
                x, heights,
                yerr=errors,
                capsize=5,
                edgecolor='black'
            )

            # sig = "*" if pval < 0.05 else ""
            # ax.text(
            #     0.5, 0.95,
            #     f"p = {pval:.3f}{sig}",
            #     transform=ax.transAxes,
            #     ha='center',
            #     va='top',
            #     fontsize=10
            # )

            ax.set_title(tick_labels[idx])
            ax.set_xticks(x)
            ax.set_xticklabels(["Deep", "Middle", "Superficial"])
            ax.set_ylabel("Mean Centrality\n(± SEM)")

            # annotate each bar with its mean value
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01*h,  # place above error‐bar
                    f"{h:.4f}",
                    ha="center",
                    va="bottom"
                )

        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/{name}/EigvecsBelongingToEachRSN_EigCentrality_Count_AverageAdj{additionalName}.png",
                    bbox_inches="tight")
        plt.close(fig)



    def runDegreeDistribution(self, M, name, layerName):

        G = nx.from_numpy_array(M)

        hist = nx.degree_histogram(G)
        counts  = np.array(hist)

        # 3. Convert to PMF and then strict CCDF = P(K > k)
        # ------------------------------------------------
        N = G.number_of_nodes()
        p_k = counts / N                     # PMF: P(K = k)
        F   = np.cumsum(p_k)                 # CDF: F(k) = P(K <= k)
        ccdf = 1 - F                         # strict CCDF: P(K > k)
        degrees = np.arange(len(hist))       # 0, 1, 2, …, k_max
        k = np.arange(len(ccdf)-1)

        comps = nx.number_connected_components(G)
        print(f"{comps} connected component(s)")

        # 4. Plot on log–log axes
        # ------------------------------------------------
        plt.figure(figsize=(6,4))
        plt.step(k, ccdf[:-1], where='post', marker='o')
        # plt.step(degrees, ccdf, where='post', marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Degree $k$ (log scale)')
        plt.ylabel(r'$1 - F(k) = P(K > k)$ (log scale)')
        plt.title('Node‐Degree CCDF (NetworkX)')
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/{name}/DegreePlot_{layerName}.png", bbox_inches="tight")
        plt.close()

    def modularity(self, A):

        labels = np.loadtxt(
            'cortex_parcel_network_assignments.txt', 
            dtype=int
        )

        k = A.sum(axis=1)
        m2 = k.sum()
        Q = 0.0

        for c in np.unique(labels):
            idx = np.where(labels==c)[0]
            lc = A[np.ix_(idx, idx)].sum()
            kc = k[idx].sum()
            Q += (lc/m2) - (kc/m2)**2

        return Q