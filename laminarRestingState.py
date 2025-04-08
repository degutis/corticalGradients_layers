import numpy as np
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, image
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import hcp_utils as hcp
import warnings
from collections import defaultdict
import re




class LaminarRestingState:
    def __init__(self, data_dir, N, setThresh, num_layers = 3, atlas_dir = "../highRes_resting/derivatives/ref_anat/sub-01/HCP-MM1_in-func.nii"):
        
        self.data_dir = data_dir
        self.N = N
        self.setThresh = setThresh
        self.atlas_dir = atlas_dir
        self.num_layers = num_layers
        self.npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])


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
    
    def get_adj_matrix_withinLayers_multRuns(self):
        
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
            np.fill_diagonal(corr_matrix, 1)

            # Threshold
            threshold = np.percentile(np.abs(corr_matrix), self.setThresh)
            adj_matrix = np.where(np.abs(corr_matrix) >= threshold, corr_matrix, 0)
            adj_matrix_within[:, :, i] = np.abs(adj_matrix)

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
        
        return adj_matrix_full


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
        adj_matrix_within = np.empty((self.N, self.N, self.num_layers))
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
        full_corr = np.nan_to_num(full_corr, nan=0)
        np.fill_diagonal(full_corr, 1)
        threshold = np.percentile(np.abs(full_corr), self.setThresh)
        adj_full = np.where(np.abs(full_corr) >= threshold, full_corr, 0)
        
        return np.abs(adj_full), all_series_array


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


    def runLaplacianEmbedding(self, M, name, num_components=10, epsilon = 1e-10, convert_to_binary=True, full=False):
        
        self.num_components = num_components
        os.makedirs(f"{self.data_dir}/{name}", exist_ok=True)  # Create folder for layer-wise maps

        if convert_to_binary:
            M[M != 0] = 1 # Convert to binary matrix
        else:
            pass

        plt.figure(figsize=(6, 6))
        plt.imshow(M, cmap="viridis")
        plt.colorbar(label="Correlation")
        plt.title(f"{name} Block Matrix")
        plt.savefig(f"{self.data_dir}/{name}/Block_matrix.png", bbox_inches="tight")

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
            print(i)
            if force_run or not os.path.exists(f"{self.data_dir}/{name}/eigenvector_layers"):

                os.makedirs(f"{self.data_dir}/{name}/eigenvector_layers", exist_ok=True)  # Create folder for layer-wise maps
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

    def __plot_on_mmhcp_surface_multipleLayers__(self, Xp, eigValue, name, cm = "RdBu", noSubcortical=True):

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

            titles = [["Layer 1 Lateral L", "Layer 1 Medial L", "Layer 1 Lateral R",  "Layer 1 Medial R"], 
                        ["Layer 2 Lateral L", "Layer 2 Medial L", "Layer 2 Lateral R",  "Layer 2 Medial R"],
                        ["Layer 3 Lateral L", "Layer 3 Medial L", "Layer 3 Lateral R",  "Layer 3 Medial R"]]
            
            # Loop over the views (columns)
            for j, view in enumerate(orientations):
                ax = axes[i, j]
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

                ax.set_title(titles[i][j], fontsize=14)

        # Add a single colorbar
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Positioning of colorbar
        norm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(norm, cax=cbar_ax)

        plt.suptitle(f"Eigenvector {eigValue}", fontsize=16)
        plt.savefig(f"{self.data_dir}/{name}/eigenvector_layers/eigenvectorSurface_{eigValue}_twoHem.png", facecolor="white")
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
                bg_img="../highRes_resting/derivatives/ref_anat/sub-01/fs_t1_in-func.nii",
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