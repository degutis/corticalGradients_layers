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


class LaminarRestingState:
    def __init__(self, data_dir, N, setThresh, num_layers = 3, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-01/HCP-MM1_in-func.nii"):
        
        self.data_dir = data_dir
        self.N = N
        self.setThresh = setThresh
        self.atlas_dir = atlas_dir
        self.num_layers = num_layers
        self.npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

    # def plotReliability(self, TR=3.2, reference_minutes=7, min_test_minutes=2, n_iterations=500):
    #     """
    #     Computes within‐subject reliability (Laumann‐style) of resting‐state FC
    #     as a function of total scan length, separately for each gray‐matter layer.
    #     """
    #     # volumes per minute, rounded to nearest integer
    #     volumes_per_minute = int(round(60.0 / TR))

    #     # Gather .npy files
    #     npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]
    #     layer_groups = defaultdict(list)

    #     for fname in npy_files:
    #         try:
    #             layer_str = fname.split('_')[-1].replace('.npy', '')
    #             layer_num = int(layer_str)
    #         except Exception as e:
    #             raise ValueError(f"Could not extract layer number from filename: {fname}") from e
    #         layer_groups[layer_num].append(os.path.join(self.data_dir, fname))

    #     reliability_results = {}

    #     for layer_num in sorted(layer_groups):
    #         files = layer_groups[layer_num]
    #         print(f"\nProcessing Layer {layer_num} with {len(files)} run(s)")

    #         # Load and concatenate
    #         runs = [np.load(fp) for fp in files]  # each: [n_voxels, n_timepoints]
    #         data = np.concatenate(runs, axis=1)    # [n_voxels, total_timepoints]
    #         n_parcels, total_volumes = data.shape
    #         total_minutes = total_volumes // volumes_per_minute

    #         print(f"  total_volumes = {total_volumes}, total_minutes = {total_minutes}")

    #         # if too short, skip
    #         if total_minutes <= reference_minutes + min_test_minutes - 1:
    #             print(f"  Skipping layer {layer_num}: not enough data for reference={reference_minutes} and test>= {min_test_minutes} minutes.")
    #             continue

    #         test_minutes = range(min_test_minutes, total_minutes - reference_minutes + 1)
    #         layer_curve = []

    #         # pre‐compute upper‐tri indices for speed
    #         iu = np.triu_indices(n_parcels, k=1)

    #         for tmin in tqdm(test_minutes, desc=f"Layer {layer_num}"):
    #             test_vols = tmin * volumes_per_minute
    #             ref_vols  = reference_minutes * volumes_per_minute
    #             corrs = []

    #             for _ in range(n_iterations):
    #                 # --- reference segment ---
    #                 start_ref = np.random.randint(0, total_volumes - ref_vols + 1)
    #                 ref_seg = data[:, start_ref : start_ref + ref_vols]

    #                 # --- test segment ---
    #                 start_test = np.random.randint(0, total_volumes - test_vols + 1)
    #                 test_seg = data[:, start_test : start_test + test_vols]

    #                 # if lengths mismatch, up/downsample test → ref length
    #                 if test_seg.shape[1] != ref_seg.shape[1]:
    #                     test_seg = resample(test_seg, ref_seg.shape[1], axis=1)

    #                 # compute FC matrices [n_parcels×n_parcels]
    #                 ref_fc  = np.corrcoef(ref_seg)
    #                 test_fc = np.corrcoef(test_seg)

    #                 # flatten upper‐triangle
    #                 ref_vals  = ref_fc[iu]
    #                 test_vals = test_fc[iu]

    #                 # mask out NaNs
    #                 valid = ~np.isnan(ref_vals) & ~np.isnan(test_vals)
    #                 if valid.sum() > 1:  # need at least two valid points
    #                     r = np.corrcoef(ref_vals[valid], test_vals[valid])[0, 1]
    #                     corrs.append(r)

    #             mean_r = np.nanmean(corrs)
    #             layer_curve.append((tmin, mean_r))

    #         reliability_results[layer_num] = layer_curve

    #     # plot
    #     plt.figure(figsize=(10, 6))
    #     for layer_num, curve in sorted(reliability_results.items()):
    #         minutes, rs = zip(*curve)
    #         plt.plot(minutes, rs, marker='o', label=f'Layer {layer_num}')

    #     plt.xlabel('Randomly sampled minutes (test window)')
    #     plt.ylabel(f'Mean correlation (ref={reference_minutes} min)')
    #     plt.title('Within-subject reliability vs. time per layer')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     outpath = os.path.join(self.data_dir, 'Reliability.png')
    #     plt.savefig(outpath)  
      
    def plotReliability(self, TR=3.2, reference_minutes=4, min_test_minutes=2, n_iterations=500):
        """
        Computes within‐subject FC reliability per layer AND across ALL parcels from ALL layers.
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

        # --- 2) per‐layer curves (as before) ---
        for layer_num in sorted(layer_groups):
            files = layer_groups[layer_num]
            print(f"\nLayer {layer_num}: {len(files)} run(s)")

            # load+concat runs → data: [360 parcels, total_timepoints]
            runs = [np.load(fp) for fp in files]
            data = np.concatenate(runs, axis=1)
            n_parcels, total_vols = data.shape
            total_mins = total_vols // volumes_per_minute
            print(f"  total_vols={total_vols}, total_mins={total_mins}")

            if total_mins < reference_minutes + min_test_minutes:
                print(f"  skip (need ≥{reference_minutes+min_test_minutes} min)")
                continue

            # pre‐compute FC upper‐triangle indices
            iu = np.triu_indices(n_parcels, k=1)
            layer_curve = []

            for tmin in tqdm(range(min_test_minutes, total_mins - reference_minutes +1),
                            desc=f"Layer {layer_num}"):
                test_vols = tmin * volumes_per_minute
                ref_vols  = reference_minutes * volumes_per_minute
                corrs = []

                for _ in range(n_iterations):
                    # sample reference segment
                    sr = np.random.randint(0, total_vols - ref_vols +1)
                    ref_seg = data[:, sr:sr+ref_vols]
                    # sample test segment
                    st = np.random.randint(0, total_vols - test_vols +1)
                    test_seg = data[:, st:st+test_vols]

                    # resample if lengths differ
                    if test_seg.shape[1] != ref_seg.shape[1]:
                        test_seg = resample(test_seg, ref_seg.shape[1], axis=1)

                    # compute FCs and flatten
                    ref_fc  = np.corrcoef(ref_seg)
                    test_fc = np.corrcoef(test_seg)
                    rvals = ref_fc[iu]
                    tvals = test_fc[iu]

                    # drop NaNs
                    valid = ~np.isnan(rvals)&~np.isnan(tvals)
                    if valid.sum() < 2:
                        continue

                    # corr of the two FC‐vectors
                    r = np.corrcoef(rvals[valid], tvals[valid])[0,1]
                    corrs.append(r)

                layer_curve.append((tmin, np.nan if not corrs else np.mean(corrs)))

            reliability_results[layer_num] = layer_curve

        # --- 3) ALL‐layers curve ---
        # group by run‐ID so that within‐run time‐axes align
        run_groups = defaultdict(list)
        for files in layer_groups.values():
            for fp in files:
                run_id = os.path.basename(fp).split('_')[1]   # e.g. 'run2'
                run_groups[run_id].append(fp)

        data_runs = []
        for run_id, fps in run_groups.items():
            # sort layers by layer‐number to keep row‐order consistent
            fps_sorted = sorted(fps, key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.npy','')))
            arrs = [np.load(fp) for fp in fps_sorted]
            # check all have same timepoints
            T0 = arrs[0].shape[1]
            if any(a.shape[1]!=T0 for a in arrs):
                raise ValueError(f"Run {run_id} has mismatched timepoints across layers")
            # stack parcels from every layer → shape [360 × n_layers, T0]
            data_runs.append(np.vstack(arrs))

        # now concat across runs → data_all [360 × n_layers, total_vols_all]
        data_all = np.concatenate(data_runs, axis=1)
        n_all, total_vols_all = data_all.shape
        total_mins_all = total_vols_all // volumes_per_minute
        print(f"\nALL‐LAYERS: parcels={n_all}, total_vols={total_vols_all}, total_mins={total_mins_all}")

        if total_mins_all >= reference_minutes + min_test_minutes:
            iu_all = np.triu_indices(n_all, k=1)
            all_curve = []

            for tmin in tqdm(range(min_test_minutes, total_mins_all - reference_minutes +1),
                            desc="ALL‐LAYERS"):
                test_vols = tmin * volumes_per_minute
                ref_vols  = reference_minutes * volumes_per_minute
                corrs = []

                for _ in range(n_iterations):
                    # sample segments
                    sr = np.random.randint(0, total_vols_all - ref_vols +1)
                    ref_seg = data_all[:, sr:sr+ref_vols]
                    st = np.random.randint(0, total_vols_all - test_vols +1)
                    test_seg = data_all[:, st:st+test_vols]

                    if test_seg.shape[1] != ref_seg.shape[1]:
                        test_seg = resample(test_seg, ref_seg.shape[1], axis=1)

                    ref_fc  = np.corrcoef(ref_seg)
                    test_fc = np.corrcoef(test_seg)
                    rv, tv = ref_fc[iu_all], test_fc[iu_all]
                    valid = ~np.isnan(rv)&~np.isnan(tv)
                    if valid.sum() < 2:
                        continue
                    r = np.corrcoef(rv[valid], tv[valid])[0,1]
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
        plt.xlabel('Test window length (minutes)')
        plt.ylabel(f'Mean FC‐matrix corr (ref={reference_minutes} min)')
        plt.title('Within‐subject FC reliability vs. time')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        outpath = os.path.join(self.data_dir, 'Reliability_FC_all.png')
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
        print(f"Saved combined plot to {outpath}")

    def get_adj_matrix_withinLayers(self):
        
        adj_matrix_within = np.empty((self.N,self.N,self.num_layers))

        for i, file in enumerate(self.npy_files):

            print("Working on file: ", file)

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
            print(file)
            try:
                layer_str = file.split('_')[-1].replace('.npy', '')
                layer_num = int(layer_str)
                layer_groups[layer_num].append(file)
            except Exception as e:
                raise ValueError(f"Could not extract layer number from filename: {file}") from e

        sorted_layers = sorted(layer_groups.items())
        adj_matrix_within = np.empty((self.N, self.N, self.num_layers))
        adj_matrix_within_noThresh = np.empty((self.N,self.N,self.num_layers))

        for i, (layer_num, files) in enumerate(sorted_layers):
            print(f"Processing Layer {layer_num} with {len(files)} run(s)")
            all_time_series = []

            for file in files:
                file_path = os.path.join(self.data_dir, file)
                time_series = np.load(file_path)
                all_time_series.append(time_series)

            concatenated = np.concatenate(all_time_series, axis=1)
            print(f"Concatenated shape: {concatenated.shape}")

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
            print(file)
            try:
                layer_str = file.split('_')[-1].replace('.npy', '')
                layer_num = int(layer_str)
                layer_groups[layer_num].append(file)
            except Exception as e:
                raise ValueError(f"Could not extract layer number from filename: {file}") from e

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
        full_corr = self.fisher_z(np.nan_to_num(full_corr, nan=0))
        np.fill_diagonal(full_corr, 0)
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


    def runLaplacianEmbedding(self, M, name, num_components=10, epsilon = 1e-10, convert_to_binary=True, full=False, addName='', vMax=0.4):
        
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
            #self.__validate_eigendecomposition__(self, L_norm, eigvals, eigvecs)
            
        else:
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(L_norm, k=num_components, which='SM')
            self.num_components = num_components


        return eigvals, eigvecs


    def __validate_eigendecomposition__(self, L_norm, eigvals, eigvecs, tol=1e-5):
        if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
            raise ValueError("Eigenvalues contain NaNs or Infs.")
        if np.any(np.isnan(eigvecs)) or np.any(np.isinf(eigvecs)):
            raise ValueError("Eigenvectors contain NaNs or Infs.")

        # Reconstruction residual
        L_reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        residual = np.linalg.norm(L_norm - L_reconstructed, ord='fro')
        if residual > tol:
            raise ValueError(f"Reconstruction residual too high: {residual}")

        # Orthogonality check
        I_approx = eigvecs.T @ eigvecs
        ortho_residual = np.linalg.norm(I_approx - np.eye(eigvecs.shape[1]), ord='fro')
        if ortho_residual > tol:
            raise ValueError(f"Eigenvectors not orthonormal: residual = {ortho_residual}")

        # Value range (specific to normalized Laplacians)
        if np.min(eigvals) < -tol or np.max(eigvals) > 2 + tol:
            raise ValueError("Eigenvalues outside [0, 2] range.")

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


    def plotTwoDimEmbedding(self, eigvecs, name, eigvecs_to_plot=[1, 2]):

        colors = np.repeat([0, 1, 2], self.N)
        cmap = ListedColormap(['red', 'orange', 'purple'])
        categories = np.unique(colors)
        eigvecs_str = "".join(map(str, eigvecs_to_plot))

        plt.figure(figsize=(8, 6))
        for cat in categories:
            category_points = eigvecs[colors == cat]
            plt.scatter(category_points[:, eigvecs_to_plot[0]], category_points[:, eigvecs_to_plot[1]], color=cmap(cat), edgecolors='k', alpha=0.5)

            slope, intercept = np.polyfit(category_points[:, eigvecs_to_plot[0]], category_points[:, eigvecs_to_plot[1]], 1)
            plt.plot(category_points[:, eigvecs_to_plot[0]], slope * category_points[:, eigvecs_to_plot[0]] + intercept, color=cmap(cat), linewidth=2)

        plt.xlabel(f'Eigenvector {eigvecs_to_plot[0]+1}')
        plt.ylabel(f'Eigenvector {eigvecs_to_plot[1]+1}')
        plt.title('Laplacian Embedding (Normalized)')
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) for i in range(3)]
        plt.legend(handles, ['Superficial Layer', 'Middle Layer', 'Deep Layer'], title='Brain Parcel Index')
        plt.savefig(f"{self.data_dir}/{name}/Laplacian_embedding_{eigvecs_str}.png", bbox_inches="tight")
        plt.close()


    def eigvecs_to_nifti(self, eigvecs, name, hcp_atlas=True, force_run=True, scaleEigVecs=False):
        
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
        threshold = 80  

        if num_components > threshold:
            indices = list(range(60)) + list(range(num_components - 20, num_components))
        else:
            indices = list(range(num_components))

        if total_regions % self.num_layers != 0:
            raise ValueError("Total regions must be evenly divisible by number of layers.")

        print(f"Mapping {total_regions} nodes into {self.num_layers} layers of {self.N} regions each.")

        # Split eigvecs into layers dynamically
        eig_layers = np.split(eigvecs, self.num_layers, axis=0)

        # Loop through eigenvector dimensions
        for i in indices:  
            print(i)
            if force_run or not os.path.exists(f"{self.data_dir}/{name}/eigenvector_layers"):

                os.makedirs(f"{self.data_dir}/{name}/eigenvector_layers{self.addName}", exist_ok=True)  # Create folder for layer-wise maps
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

    def __plot_on_mmhcp_surface_multipleLayers__(self, Xp, eigValue, name, cm = "RdBu", noSubcortical=True, titles=None, folder_name="eigenvector_layers"):

        os.makedirs(f"{self.data_dir}/{name}/{folder_name}", exist_ok=True)  # Create folder for layer-wise maps

        mmp_labels = hcp.mmp.labels  # mmp = Glasser parcellation
        
        if noSubcortical:
            current_length = len(Xp[:, 0])  # Get the number of parcels (rows)
            target_length = len(mmp_labels)  # Target length is the number of regions (parcels)
            zeros_to_add = target_length - current_length
            Xp = np.concatenate((Xp, np.zeros((zeros_to_add, Xp.shape[1]))), axis=0)    

        orientations = ["lateral", "medial", "medial", "lateral"]

        # Determine the global min and max values across all layers
        all_data = np.hstack([hcp.cortex_data(hcp.unparcellate(Xp[:, i], hcp.mmp)) for i in range(Xp.shape[1])])
        vmin, vmax = np.percentile(all_data, [2, 98])  #np.min(all_data), np.max(all_data)

        # Create a figure with multiple rows and shared colorbar
        fig, axes = plt.subplots(
            Xp.shape[1], len(orientations),
            figsize=(20, 5 * Xp.shape[1]),
            subplot_kw={"projection": "3d"}
        )

        # Loop over the layers (rows)
        for i in range(Xp.shape[1]):
            layer_data = hcp.cortex_data(hcp.unparcellate(Xp[:, i], hcp.mmp))

            # titles = [["Layer 1 Lateral L", "Layer 1 Medial L", "Layer 1 Lateral R",  "Layer 1 Medial R"], 
            #             ["Layer 2 Lateral L", "Layer 2 Medial L", "Layer 2 Lateral R",  "Layer 2 Medial R"],
            #             ["Layer 3 Lateral L", "Layer 3 Medial L", "Layer 3 Lateral R",  "Layer 3 Medial R"]]
            
            if titles is not None and i < len(titles):
                row_title = titles[i]
            else:
                row_title = f"Layer {i+1}"

                
            # Loop over the views (columns)
            for j, view in enumerate(orientations):
                try:
                    ax = axes[i, j]
                except:
                    ax = axes[j]
                
                if j==0 or j==1:
                    plotting.plot_surf_stat_map(
                        hcp.mesh.inflated_left,
                        layer_data[:len(layer_data) // 2],
                        view=view,
                        colorbar=False,  # Suppress individual colorbars
                        bg_map=hcp.mesh.sulc_left,
                        bg_on_data=True,
                        darkness=0.3,
                        axes=ax,
                        figure=fig,
                        cmap=cm,
                        vmin=vmin, vmax=vmax,  # Ensure consistent color scale
                        symmetric_cbar=False,
                    )
                else:
                    plotting.plot_surf_stat_map(
                        hcp.mesh.inflated_right,
                        layer_data[len(layer_data) // 2:],
                        view=view,
                        colorbar=False,  # Suppress individual colorbars
                        bg_map=hcp.mesh.sulc_right,
                        bg_on_data=True,
                        darkness=0.3,
                        axes=ax,
                        figure=fig,
                        cmap=cm,
                        vmin=vmin, vmax=vmax,  # Ensure consistent color scale
                        symmetric_cbar=False,
                    )

                # ax.set_title(titles[i][j], fontsize=14)
                ax.set_title(f"{row_title} - {orientations[j].capitalize()}", fontsize=14)

        # Add a single colorbar
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Positioning of colorbar
        norm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(norm, cax=cbar_ax)

        plt.suptitle(f"Eigenvector {eigValue}", fontsize=16)
        plt.savefig(f"{self.data_dir}/{name}/{folder_name}/eigenvectorSurface_{eigValue}_twoHem.png", facecolor="white")
        plt.close()


    def __plot_on_volume__(self, layer_imgs, eigValue, name):
        
        fig, axes = plt.subplots(1, self.num_layers, figsize=(15, 5))
        combined_data = np.concatenate([img.get_fdata().flatten() for img in layer_imgs])
        vmin, vmax = np.percentile(combined_data, [2, 98])  # Robust scaling

        ref_img = layer_imgs[0]
        ref_shape = ref_img.shape
        
        warnings.warn("Need to implement the plotting of the middle slice in all dimensions.")
        mid_cut_coords = (ref_shape[0] // 2, ref_shape[1] // 2, ref_shape[2] // 2)  # Middle slice in (x, y, z)

        warnings.warn("Hard coded anatomical image.")

        for layer_idx, layer_img in enumerate(layer_imgs):
            plotting.plot_stat_map(
                layer_img,
                bg_img="../../highRes_resting/derivatives/ref_anat/sub-01/fs_t1_in-func.nii",
                cmap="coolwarm",
                threshold=None,
                vmin=vmin, vmax=vmax,
                axes=axes[layer_idx],
                colorbar=(layer_idx == self.num_layers - 1),
                #cut_coords=mid_cut_coords
            )
            axes[layer_idx].set_title(f"Layer {layer_idx + 1}")

        plt.suptitle(f"Eigenvector {eigValue}")
        plt.savefig(f"{self.data_dir}/{name}/eigenvector_layers/eigenvector_{eigValue}.png", dpi=500)
        plt.close()


    def plotScree(self, eigvals, name, sort=False):
            
        if sort:
            eigvals_sorted = np.sort(eigvals)[::-1]
        else:
            eigvals_sorted = eigvals
        
        # Compute cumulative explained variance (normalized to 100%)
        eigvals_cumsum = np.cumsum(eigvals_sorted) / np.sum(eigvals_sorted) * 100

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(range(1, self.num_components + 1), eigvals_sorted, marker='o', linestyle='-', color='b', label="Eigenvalues")
        ax1.set_xlabel('Component Number')
        ax1.set_ylabel('Eigenvalue', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create second y-axis for cumulative percentage
        ax2 = ax1.twinx()
        ax2.plot(range(1, self.num_components + 1), eigvals_cumsum, marker='s', linestyle='--', color='r', label="Cumulative Sum")
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
        print(n_ROI)
        print(U.shape[1])
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
    
    def getLabels(self):
        
        labels = [
                'V1_R', 'MST_R', 'V6_R', 'V2_R', 'V3_R', 'V4_R', 'V8_R', '4_R', '3b_R', 'FEF_R',
                'PEF_R', '55b_R', 'V3A_R', 'RSC_R', 'POS2_R', 'V7_R', 'IPS1_R', 'FFC_R', 'V3B_R', 'LO1_R',
                'LO2_R', 'PIT_R', 'MT_R', 'A1_R', 'PSL_R', 'SFL_R', 'PCV_R', 'STV_R', '7Pm_R', '7m_R',
                'POS1_R', '23d_R', 'v23ab_R', 'd23ab_R', '31pv_R', '5m_R', '5mv_R', '23c_R', '5L_R', '24dd_R',
                '24dv_R', '7AL_R', 'SCEF_R', '6ma_R', '7Am_R', '7Pl_R', '7PC_R', 'LIPv_R', 'VIP_R', 'MIP_R',
                '1_R', '2_R', '3a_R', '6d_R', '6mp_R', '6v_R', 'p24pr_R', '33pr_R', 'a24pr_R', 'p32pr_R',
                'a24_R', 'd32_R', '8BM_R', 'p32_R', '10r_R', '47m_R', '8Av_R', '8Ad_R', '9m_R', '8BL_R',
                '9p_R', '10d_R', '8C_R', '44_R', '45_R', '47l_R', 'a47r_R', '6r_R', 'IFJa_R', 'IFJp_R',
                'IFSp_R', 'IFSa_R', 'p9-46v_R', '46_R', 'a9-46v_R', '9-46d_R', '9a_R', '10v_R', 'a10p_R', '10pp_R',
                '11l_R', '13l_R', 'OFC_R', '47s_R', 'LIPd_R', '6a_R', 'i6-8_R', 's6-8_R', '43_R', 'OP4_R',
                'OP1_R', 'OP2-3_R', '52_R', 'RI_R', 'PFcm_R', 'PoI2_R', 'TA2_R', 'FOP4_R', 'MI_R', 'Pir_R',
                'AVI_R', 'AAIC_R', 'FOP1_R', 'FOP3_R', 'FOP2_R', 'PFt_R', 'AIP_R', 'EC_R', 'PreS_R', 'H_R',
                'ProS_R', 'PeEc_R', 'STGa_R', 'PBelt_R', 'A5_R', 'PHA1_R', 'PHA3_R', 'STSda_R', 'STSdp_R', 'STSvp_R',
                'TGd_R', 'TE1a_R', 'TE1p_R', 'TE2a_R', 'TF_R', 'TE2p_R', 'PHT_R', 'PH_R', 'TPOJ1_R', 'TPOJ2_R',
                'TPOJ3_R', 'DVT_R', 'PGp_R', 'IP2_R', 'IP1_R', 'IP0_R', 'PFop_R', 'PF_R', 'PFm_R', 'PGi_R',
                'PGs_R', 'V6A_R', 'VMV1_R', 'VMV3_R', 'PHA2_R', 'V4t_R', 'FST_R', 'V3CD_R', 'LO3_R', 'VMV2_R',
                '31pd_R', '31a_R', 'VVC_R', '25_R', 's32_R', 'pOFC_R', 'PoI1_R', 'Ig_R', 'FOP5_R', 'p10p_R',
                'p47r_R', 'TGv_R', 'MBelt_R', 'LBelt_R', 'A4_R', 'STSva_R', 'TE1m_R', 'PI_R', 'a32pr_R', 'p24_R',
                
                'V1_L', 'MST_L', 'V6_L', 'V2_L', 'V3_L', 'V4_L', 'V8_L', '4_L', '3b_L', 'FEF_L',
                'PEF_L', '55b_L', 'V3A_L', 'RSC_L', 'POS2_L', 'V7_L', 'IPS1_L', 'FFC_L', 'V3B_L', 'LO1_L',
                'LO2_L', 'PIT_L', 'MT_L', 'A1_L', 'PSL_L', 'SFL_L', 'PCV_L', 'STV_L', '7Pm_L', '7m_L',
                'POS1_L', '23d_L', 'v23ab_L', 'd23ab_L', '31pv_L', '5m_L', '5mv_L', '23c_L', '5L_L', '24dd_L',
                '24dv_L', '7AL_L', 'SCEF_L', '6ma_L', '7Am_L', '7Pl_L', '7PC_L', 'LIPv_L', 'VIP_L', 'MIP_L',
                '1_L', '2_L', '3a_L', '6d_L', '6mp_L', '6v_L', 'p24pr_L', '33pr_L', 'a24pr_L', 'p32pr_L',
                'a24_L', 'd32_L', '8BM_L', 'p32_L', '10r_L', '47m_L', '8Av_L', '8Ad_L', '9m_L', '8BL_L',
                '9p_L', '10d_L', '8C_L', '44_L', '45_L', '47l_L', 'a47r_L', '6r_L', 'IFJa_L', 'IFJp_L',
                'IFSp_L', 'IFSa_L', 'p9-46v_L', '46_L', 'a9-46v_L', '9-46d_L', '9a_L', '10v_L', 'a10p_L', '10pp_L',
                '11l_L', '13l_L', 'OFC_L', '47s_L', 'LIPd_L', '6a_L', 'i6-8_L', 's6-8_L', '43_L', 'OP4_L',
                'OP1_L', 'OP2-3_L', '52_L', 'RI_L', 'PFcm_L', 'PoI2_L', 'TA2_L', 'FOP4_L', 'MI_L', 'Pir_L',
                'AVI_L', 'AAIC_L', 'FOP1_L', 'FOP3_L', 'FOP2_L', 'PFt_L', 'AIP_L', 'EC_L', 'PreS_L', 'H_L',
                'ProS_L', 'PeEc_L', 'STGa_L', 'PBelt_L', 'A5_L', 'PHA1_L', 'PHA3_L', 'STSda_L', 'STSdp_L', 'STSvp_L',
                'TGd_L', 'TE1a_L', 'TE1p_L', 'TE2a_L', 'TF_L', 'TE2p_L', 'PHT_L', 'PH_L', 'TPOJ1_L', 'TPOJ2_L',
                'TPOJ3_L', 'DVT_L', 'PGp_L', 'IP2_L', 'IP1_L', 'IP0_L', 'PFop_L', 'PF_L', 'PFm_L', 'PGi_L',
                'PGs_L', 'V6A_L', 'VMV1_L', 'VMV3_L', 'PHA2_L', 'V4t_L', 'FST_L', 'V3CD_L', 'LO3_L', 'VMV2_L',
                '31pd_L', '31a_L', 'VVC_L', '25_L', 's32_L', 'pOFC_L', 'PoI1_L', 'Ig_L', 'FOP5_L', 'p10p_L',
                'p47r_L', 'TGv_L', 'MBelt_L', 'LBelt_L', 'A4_L', 'STSva_L', 'TE1m_L', 'PI_L', 'a32pr_L', 'p24_L'
            ]
        
        return labels


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

    def calculateRichClub(self, connectivity_matrix, name, layer, threshold=95):

        os.makedirs(f"{self.data_dir}/{name}/NetworkMeasures", exist_ok=True)  # Create folder for layer-wise maps

        threshold = np.percentile(np.abs(connectivity_matrix), threshold)
        abs_corr_thresh = np.where(np.abs(connectivity_matrix) >= threshold, np.abs(connectivity_matrix), 0)
        
        G = nx.from_numpy_array(abs_corr_thresh)

        rich_club_coeffs = nx.rich_club_coefficient(G, normalized=False, seed=33)

        plt.plot(list(rich_club_coeffs.keys()), list(rich_club_coeffs.values()))
        plt.xlabel('Degree k')
        plt.ylabel('Rich Club Coefficient φ(k)')
        plt.title('Rich Club Coefficient Curve (Weighted)')
        plt.grid(True)
        plt.savefig(f"{self.data_dir}/{name}/NetworkMeasures/RichClub_{layer}.png", bbox_inches="tight")
        plt.close()

        # Optional: Identify high-degree nodes (e.g., top 5%) as potential rich club members
        degrees = dict(G.degree())
        cutoff = np.percentile(list(degrees.values()), 95)
        rich_club_nodes = [n for n, deg in degrees.items() if deg >= cutoff]

        # Save to a text file
        with open(f"{self.data_dir}/{name}/NetworkMeasures/RichClubNodes_{layer}.txt", 'w') as f:
            for node in rich_club_nodes:
                f.write(f"{node}\n")

        return rich_club_nodes

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
        print(eigvecs.shape)
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
            im = ax.imshow(C, vmin=-1, vmax=1, cmap='RdBu_r')
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


    def identifyEigvecActivityPartOfRS(self, eigvecs_orig, name, ignoreFirst=4, limit=25, end_num=0, thresh=3, adjustSize=True, excludeNone=True):
        
        if adjustSize:
            orig_X = eigvecs_orig.shape[1]
            end = orig_X - end_num

            if end_num>0:
                eigvecs = np.hstack([
                    eigvecs_orig[:, :limit],   # cols 0 … limit-1
                    eigvecs_orig[:, end:]      # cols end … M-1
                    ])
            else:
                eigvecs = eigvecs_orig[:, ignoreFirst:limit] # ignore the firs

        else:
            eigvecs = eigvecs_orig

        eigvecs_reshaped = eigvecs.reshape(360, 3, eigvecs.shape[1])
        mu    = eigvecs_reshaped.mean(axis=0)
        sigma = eigvecs_reshaped.std(axis=0, ddof=0)
        z = (eigvecs_reshaped - mu[None, :, :]) / sigma[None, :, :]
        bins = ((z < -thresh) | (z > thresh)).astype(int)

        cats = np.loadtxt('cortex_parcel_network_assignments.txt', dtype=int)   # shape (360,), values 1…12
        cats_exp = cats[:, None, None]
        counts = np.zeros((12, bins.shape[1], bins.shape[2]), dtype=int)

        # Vectorized count over the first axis (360)
        for k in range(1, 13):
            mask = (bins == 1) & (cats_exp == k)
            counts[k-1] = mask.sum(axis=0)

        # counts_dict = {
        #     k: counts[k-1]
        #     for k in range(1, 13)
        # }

        # layer_totals = counts.sum(axis=0)

        # layer_sums = {
        #     layer_idx + 1: layer_totals[layer_idx]
        #     for layer_idx in range(layer_totals.shape[0])
        # }

        cat_counts = counts.sum(axis=2)

        tick_labels = ["Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular", 
                       "Dorsal-Attentional", "Language", "Frontoparietal", "Auditory", 
                       "Default", "Posterior-Multimodal","Ventral-Multimodal", "Orbito-Affective"]

        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        for layer in range(3):
            bars = axs[layer].bar(np.arange(1, 13), cat_counts[:, layer])
            axs[layer].bar(np.arange(1, 13), cat_counts[:, layer])
            axs[layer].set_title(f'Layer {layer+1}')
            axs[layer].set_ylabel('Outlier Count')
            # annotate each bar with its height
            for bar in bars:
                height = bar.get_height()
                axs[layer].text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{int(height)}',
                    ha='center',
                    va='bottom'
        )

        axs[-1].set_xlabel('Resting State Networks')
        plt.xticks(np.arange(1,13), tick_labels, rotation=90)
        plt.tight_layout()
        if adjustSize:
            plt.savefig(f"{self.data_dir}/{name}/EigvecsBelongingToEachRSN_First{limit}_Last{end_num}.png", bbox_inches="tight")
        else:
            plt.savefig(f"{self.data_dir}/{name}/EigvecsBelongingToEachRSN.png", bbox_inches="tight")
        plt.close()

        ## Plotting the combinations of networks
        num_layers, num_samples = counts.shape[1], counts.shape[2]

        layer_counters = []
        for layer in range(num_layers):
            labels = []
            for sample in range(num_samples):
                present = [
                    tick_labels[k]
                    for k in range(len(tick_labels))
                    if counts[k, layer, sample] > 1
                ]
                label = "+".join(present) if present else "None"
                labels.append(label)
            layer_counters.append(Counter(labels))
        
        if excludeNone:
            all_combos = sorted(
                set().union(*(c.keys() for c in layer_counters)) 
                - {"None"}
                )
        else:
            all_combos = sorted(set().union(*(c.keys() for c in layer_counters)))

        freqs = np.zeros((num_layers, len(all_combos)), dtype=int)
        for i, ctr in enumerate(layer_counters):
            for j, combo in enumerate(all_combos):
                freqs[i, j] = ctr.get(combo, 0)

        fig, axs = plt.subplots(num_layers, 1, figsize=(12, 8), sharex=True)

        for layer in range(num_layers):
            axs[layer].bar(np.arange(len(all_combos)), freqs[layer])
            axs[layer].set_title(f'Layer {layer+1}')
            axs[layer].set_ylabel('Frequency')

        # only bottom subplot gets the x‐labels
        axs[-1].set_xticks(np.arange(len(all_combos)))
        axs[-1].set_xticklabels(all_combos, rotation=90)
        axs[-1].set_xlabel('Network Combination Label')

        plt.tight_layout()
        if adjustSize:
            plt.savefig(f"{self.data_dir}/{name}/EigvecsComboRSN_First{limit}_Last{end_num}.png", bbox_inches="tight")
        else:
            plt.savefig(f"{self.data_dir}/{name}/EigvecsComboRSN.png", bbox_inches="tight")
        plt.close()
