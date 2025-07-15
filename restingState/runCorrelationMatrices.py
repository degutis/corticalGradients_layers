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




data_dir = '../highRes_resting/derivatives/correlations/sub-50/SmallGap'
npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
N = 360
setThresh = 90
adj_matrix_full = np.empty((N,N,3))

all_series = []
epsilon = 1e-10

for i, file in enumerate(npy_files):
    file_path = os.path.join(data_dir, file)
    time_series = np.load(file_path)

    all_series.append(time_series)

    corr_matrix = np.corrcoef(time_series)
    # Make sure no NaNs are present
    corr_matrix = np.nan_to_num(corr_matrix, nan=0)
    np.fill_diagonal(corr_matrix, 1)

    threshold = np.percentile(np.abs(corr_matrix), setThresh)
    adj_matrix = np.where(np.abs(corr_matrix) >= threshold, corr_matrix, 0)
    adj_matrix_full[:,:,i] = np.abs(adj_matrix)

# Block matrix for within-layer connections
I_N = np.eye(N)
A = np.block([
    [adj_matrix_full[:,:,0], I_N, I_N],
    [I_N, adj_matrix_full[:,:,1], I_N],
    [I_N, I_N, adj_matrix_full[:,:,2]]
])

# Matrix for within and between layer connections 
all_series_array = np.concatenate(all_series, axis=0)
full_corr = np.corrcoef(all_series_array)
# Make sure no NaNs are present
full_corr = np.nan_to_num(full_corr, nan=0)
np.fill_diagonal(full_corr, 1)
threshold = np.percentile(np.abs(full_corr), setThresh)
adj_full = np.where(np.abs(full_corr) >= threshold, full_corr, 0)
adj_full_abs = np.abs(adj_full)


#Single layers
A1 = np.nan_to_num(adj_matrix_full[:,:,0], nan=0) 
np.fill_diagonal(A1, 1)
threshold_A1 = np.percentile(np.abs(A1), setThresh)
adj_A1 = np.where(np.abs(A1) >= threshold, A1, 0)
adj_A1_abs = np.abs(adj_A1)

A2 = np.nan_to_num(adj_matrix_full[:,:,1], nan=0) 
np.fill_diagonal(A2, 1)
threshold_A2 = np.percentile(np.abs(A2), setThresh)
adj_A2 = np.where(np.abs(A2) >= threshold, A2, 0)
adj_A2_abs = np.abs(adj_A2)

A3 = np.nan_to_num(adj_matrix_full[:,:,2], nan=0) 
np.fill_diagonal(A3, 1)
threshold_A3 = np.percentile(np.abs(A3), setThresh)
adj_A3 = np.where(np.abs(A3) >= threshold, A3, 0)
adj_A3_abs = np.abs(adj_A3)


def runLaplacianEmbedding(M, name, num_components=10):

    M[M != 0] = 1 # Convert to binary matrix

    plt.figure(figsize=(6, 6))
    plt.imshow(M, cmap="viridis")
    plt.colorbar(label="Correlation")
    plt.title(f"{name} Block Matrix")
    plt.savefig(f"{data_dir}/{name}_block_matrix.png", bbox_inches="tight")

    degree_matrix = np.diag(np.sum(M, axis=1))  # Degree matrix
    laplacian_matrix = degree_matrix - M  # Unnormalized Laplacian

    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(M, axis=1) + epsilon))  # Add small value to avoid division by zero
    L_norm = D_inv_sqrt @ laplacian_matrix @ D_inv_sqrt  # Normalized Laplacian

    eigvals, eigvecs = scipy.sparse.linalg.eigsh(L_norm, k=num_components, which='SM')

    plotScree(eigvals, num_components, f"{data_dir}/{name}_")
    eigvecs_to_nifti(eigvecs, "../highRes_resting/derivatives/ref_anat/sub-01/HCP-MM1_in-func.nii", "../highRes_resting/derivatives/ref_anat/sub-01/ln_depths_equivol.nii", 3, output_prefix=f"{data_dir}/{name}_eigenvector")

    #num_clusters = 3  # Choose number of clusters
    #node_embeddings = eigvecs  # Use the eigenvectors as node features

    #kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    #labels = kmeans.fit_predict(node_embeddings)

    # Plot using the first two non-trivial eigenvectors
    #plt.figure(figsize=(8, 6))
    #plt.scatter(eigvecs[:, 1], eigvecs[:, 2], c=labels, cmap='viridis', edgecolor='k', s=50)
    #plt.xlabel("Eigenvector 1")
    #plt.ylabel("Eigenvector 2")
    #plt.title("Spectral Clustering Visualization")
    #plt.colorbar(label="Cluster")
    #plt.savefig(f"{data_dir}/{name}_KMeans_laplacian_embedding.png", bbox_inches="tight")


    try:
        colors = np.repeat([0, 1, 2], N)  # Assigns 0 to first 100, 1 to next 100, and 2 to last 100
        cmap = ListedColormap(['red', 'orange', 'purple'])  # Assign colors for each category
        categories = np.unique(colors)

        plt.figure(figsize=(8, 6))
        for cat in categories:
            category_points = eigvecs[colors == cat]
            #plt.scatter(category_points[:, 1], category_points[:, 2], c=[cat], cmap=cmap, edgecolors='k', alpha=0.5)
            plt.scatter(category_points[:, 1], category_points[:, 2], color=cmap(cat), edgecolors='k', alpha=0.5)

            slope, intercept = np.polyfit(category_points[:, 1], category_points[:, 2], 1)
            plt.plot(category_points[:, 1], slope * category_points[:, 1] + intercept, color=cmap(cat), linewidth=2)

        #plt.scatter(eigvecs[:, 1], eigvecs[:, 2], c=colors, cmap=cmap, edgecolors='k',alpha=0.5)
        plt.xlabel('Eigenvector 1')
        plt.ylabel('Eigenvector 2')
        plt.title('Laplacian Embedding (Normalized)')
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) for i in range(3)]
        plt.legend(handles, ['Superficial Layer', 'Middle Layer', 'Deep Layer'], title='Brain Parcel Index')
        plt.savefig(f"{data_dir}/{name}_laplacian_embedding.png", bbox_inches="tight")
        plt.close()
    except:
        pass


def eigvecs_to_nifti(eigvecs, parcel_atlas_path, layer_atlas_path, num_layers, 
                      output_prefix="eigenvector"):
    """
    Maps eigenvectors back into brain space using both a parcel atlas and a layer atlas.
    
    Parameters:
    - eigvecs: np.ndarray (N_regions * num_layers, num_components)  
        Eigenvectors (num_regions * num_layers x num_components)
    - parcel_atlas_path: str  
        Path to the parcel atlas NIfTI file
    - layer_atlas_path: str  
        Path to the layer atlas NIfTI file
    - num_layers: int  
        Number of layers in the brain representation
    - output_prefix: str  
        Prefix for saved NIfTI files
    - save_per_layer: bool  
        If True, saves separate NIfTIs for each layer per eigenvector.  
        If False, saves a single NIfTI per eigenvector.
    """

    M = np.max(np.abs(eigvecs), axis=0)  # Find max absolute value per eigenvector
    eigvecs_scaled = eigvecs / M  # Normalize eigenvectors by their max absolute value
    M_max = np.max(np.abs(eigvecs_scaled))  # Rescale to the same max absolute value
    eigvecs_scaled = eigvecs_scaled * M_max  # Rescale back to the target max absolute value

    # Load reference atlases
    parcel_atlas_img = nib.load(parcel_atlas_path)
    parcel_atlas = parcel_atlas_img.get_fdata()
    
    unique_parcels = np.unique(parcel_atlas)
    unique_parcels = unique_parcels[(unique_parcels >= 1001) & (unique_parcels <= 3000) & (unique_parcels != 2000)]  

    layer_atlas_img = nib.load(layer_atlas_path)
    layer_atlas = layer_atlas_img.get_fdata()

    layer_binary = np.zeros_like(layer_atlas)
    layer_binary[(layer_atlas > 0) & (layer_atlas <= 0.2)] = 1
    layer_binary[(layer_atlas > 0.4) & (layer_atlas <= 0.6)] = 2
    layer_binary[(layer_atlas > 0.8) & (layer_atlas <= 0.999)] = 3

    total_regions = eigvecs.shape[0]  # Total number of nodes
    num_components = eigvecs.shape[1] # Number of eigenvectors
    regions_per_layer = total_regions // num_layers  # Compute regions per layer

    if total_regions % num_layers != 0:
        raise ValueError("Total regions must be evenly divisible by number of layers.")

    print(f"Mapping {total_regions} nodes into {num_layers} layers of {regions_per_layer} regions each.")

    # Split eigvecs into layers dynamically
    #eig_layers = np.split(eigvecs_scaled, num_layers, axis=0)
    eig_layers = np.split(eigvecs, num_layers, axis=0)

    # Loop through eigenvector dimensions
    for i in range(num_components):  
        os.makedirs(f"{output_prefix}_layers", exist_ok=True)  # Create folder for layer-wise maps
        layer_imgs = []
        #map_3D = np.zeros_like(parcel_atlas)  # Initialize 3D map

        for layer_idx, layer_data in enumerate(eig_layers):  
                    
            # Create layer mask
            map_3D = np.zeros_like(parcel_atlas)
            layer_mask = np.zeros(layer_binary.shape)
            layer_mask[layer_binary == layer_idx+1] = 1

            for roi_idx, parcel in enumerate(unique_parcels):
                parcel_mask = np.zeros(parcel_atlas.shape)
                parcel_mask[parcel_atlas == parcel] = 1

                layer_mask = np.array(layer_mask, dtype=bool)
                parcel_mask = np.array(parcel_mask, dtype=bool)
                #final_mask = layer_mask & parcel_mask  
                final_mask = parcel_mask
                map_3D[final_mask] = layer_data[roi_idx, i]

            layer_img = nib.Nifti1Image(map_3D, affine=parcel_atlas_img.affine)
            nib.save(layer_img, f"{output_prefix}_layers/eigenvector_{i+1}_layer_{layer_idx+1}.nii.gz")
            layer_imgs.append(layer_img)  # Store for later plotting

        Xp_layers = []  
        for layer_idx in range(num_layers):
            Xp_layers.append(eig_layers[layer_idx][:, i])
        Xp_layers = np.array(Xp_layers)

        plot_on_mmhcp_surface_multipleLayers(Xp_layers.T, output_prefix, i+1)
        plot_on_volume(layer_imgs, num_layers, output_prefix, i+1)

    print("All brain maps saved successfully!")


def plotScree(eigvals, num_components, outdir):
    
    # Sort eigenvalues in ascending order
    eigvals_sorted = np.sort(eigvals)[::-1]
    # Compute cumulative explained variance (normalized to 100%)
    eigvals_cumsum = np.cumsum(eigvals_sorted) / np.sum(eigvals_sorted) * 100

    # Create scree plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot eigenvalues
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
    plt.savefig(f"{outdir}_screePlot.png", bbox_inches="tight")
    plt.close()


def plot_on_mmhcp_surface_multipleLayers(Xp, output_prefix, eigValue, noSubcortical=True):

    # Get the Glasser MMP labels
    mmp_labels = hcp.mmp.labels  # mmp = Glasser parcellation
    
    if noSubcortical:
        current_length = len(Xp[:, 0])  # Get the number of parcels (rows)
        target_length = len(mmp_labels)  # Target length is the number of regions (parcels)
        zeros_to_add = target_length - current_length
        #Xp = np.concatenate((np.zeros((1, Xp.shape[1])), Xp, np.zeros((zeros_to_add, Xp.shape[1]))), axis=0)    
        Xp = np.concatenate((Xp, np.zeros((zeros_to_add, Xp.shape[1]))), axis=0)    

    cm = "RdBu"  # Color map
    # Define orientations
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
    #plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar        # Save the figure
    plt.savefig(f"{output_prefix}_layers/eigenvectorSurface_{eigValue}_twoHem.png", facecolor="white")
    plt.close()


def plot_on_volume(layer_imgs, num_layers, output_prefix, eigValue):
    fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))
    combined_data = np.concatenate([img.get_fdata().flatten() for img in layer_imgs])
    vmin, vmax = np.percentile(combined_data, [2, 98])  # Robust scaling

    ref_img = layer_imgs[0]
    ref_shape = ref_img.shape
    mid_cut_coords = (ref_shape[0] // 2, ref_shape[1] // 2, ref_shape[2] // 2)  # Middle slice in (x, y, z)

    for layer_idx, layer_img in enumerate(layer_imgs):
        plotting.plot_stat_map(
            layer_img,
            bg_img="../highRes_resting/derivatives/ref_anat/sub-01/fs_t1_in-func.nii",
            cmap="coolwarm",
            threshold=None,
            vmin=vmin, vmax=vmax,
            axes=axes[layer_idx],
            colorbar=(layer_idx == num_layers - 1),
            #cut_coords=mid_cut_coords
        )
        axes[layer_idx].set_title(f"Layer {layer_idx + 1}")

    plt.suptitle(f"Eigenvector {eigValue}")
    plt.savefig(f"{output_prefix}_layers/eigenvector_{eigValue}.png", dpi=500)
    plt.close()




runLaplacianEmbedding(A, "onlyWithinLayer")
runLaplacianEmbedding(adj_full_abs, "betweenLayer_and_withinLayer")
#runLaplacianEmbedding(A1, "Layer1")
#runLaplacianEmbedding(A2, "Layer2")
#runLaplacianEmbedding(A3, "Layer3")


def old():
    cm = "cold_hot"  # Color map
    min_thresh = 0
    max_thresh = 0.1

    # Loop over the layers (columns of Xp)
    for i in range(Xp.shape[1]):  # Iterate through layers (columns)
        fig = plt.figure(figsize=[20, 10])

        # Generate different views for each layer
        orientations = ["anterior", "lateral", "medial", "posterior"]  # List of views
        for j, view in enumerate(orientations):
            ax = fig.add_subplot(1, 4, j+1, projection="3d")

            # Plot the surface for the current layer and orientation
            plotting.plot_surf_stat_map(
                hcp.mesh.inflated,
                hcp.cortex_data(hcp.unparcellate(Xp[:, i], hcp.mmp)),  # Use the i-th column for current layer
                view=view,
                colorbar=True,
                bg_map=hcp.mesh.sulc,
                bg_on_data=True,
                darkness=0.3,
                axes=ax,
                figure=fig,
                cmap=cm,
                symmetric_cbar=True,
            )

        # Add title for the current eigenvector (layer)
        fig.suptitle(f"Eigenvector {eigValue}", fontsize=16)



def plot_on_mmhcp_surface(Xp, output_prefix, layerNumber, noSubcortical=True):
    
    """Xp is a 1D Vector same size as hcp.mmp.labels."""
    mmp_labels = hcp.mmp.labels  # mmp = Glasser parcellation

    if noSubcortical==True:
        current_length = len(Xp)
        target_length = len(mmp_labels)
        zeros_to_add = target_length - current_length - 1  # -1 because we add 1 zero at the front
        Xp = np.concatenate(([0], Xp, np.zeros(zeros_to_add)))

    cm = "cold_hot"
    min_thresh = 0
    max_thresh = 0.1

    # 2D plot – I also detail here with an example how you can add subplots…
    fig = plt.figure(figsize=[20, 10])
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    plotting.plot_surf_stat_map(
        hcp.mesh.inflated,
        hcp.cortex_data(hcp.unparcellate(Xp, hcp.mmp)),
        view="anterior",
        colorbar=True,
        bg_map=hcp.mesh.sulc,
        bg_on_data=True,
        darkness=0.3,
        axes=ax,
        figure=fig,
        cmap=cm,
        symmetric_cbar=True,
    )

    ax = fig.add_subplot(1, 4, 2, projection="3d")
    plotting.plot_surf_stat_map(
        hcp.mesh.inflated,
        hcp.cortex_data(hcp.unparcellate(Xp, hcp.mmp)),
        view="lateral",
        colorbar=True,
        bg_map=hcp.mesh.sulc,
        bg_on_data=True,
        darkness=0.3,
        axes=ax,
        figure=fig,
        cmap=cm,
        symmetric_cbar=True,
    )

    fig.suptitle("title", fontsize=16)
    plt.savefig(f"{output_prefix}_layers/eigenvectorSurface_{layerNumber}.png", facecolor="white")
