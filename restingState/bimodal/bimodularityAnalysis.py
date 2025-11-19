import numpy as np
import matplotlib.pyplot as plt
import hcp_utils as hcp
import dgsp
import os
import nibabel as nib
from nilearn import plotting
from matplotlib.colors import ListedColormap
import bimod_plots as bplot



class BimodularityAnalysis:
    def __init__(self, M, data_dir, N, setThresh, analysis, num_layers = 3, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-01/HCP-MM1_in-func.nii"):
        
        self.data_dir = data_dir
        self.N = N
        self.setThresh = setThresh
        self.atlas_dir = atlas_dir
        self.num_layers = num_layers

        M_output_subtracted = self.__subtractAverage2D__(M, block_size=N)
        M_output = self.__thresh_and_binarize__(M_output_subtracted, setThresh=setThresh, binarize=True, block_size=N)
        M_output[N:N*2, :N] = 0
        M_output[N*2:, N:N*2] = 0
        M_output[:N, N*2:] = 0

        # M_output[:N, :N] = 0
        # M_output[N:N*2, N:N*2] = 0
        # M_output[N*2:, N*2:] = 0

        self.adjMatrix = M_output

        plt.imshow(self.adjMatrix, cmap="gray", interpolation="none", origin='lower')
        outpath = os.path.join(self.data_dir, f'{analysis}/NonSymMatrix_forBimod.png')
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()


    def __thresh_and_binarize__(self, adj, setThresh=90, binarize=True, block_size=360):

        n = adj.shape[0]
        assert n % block_size == 0, "block_size must divide matrix size exactly"
        num_blocks = n // block_size
        
        out = np.zeros_like(adj, dtype=float if not binarize else int)
        
        for bi in range(num_blocks):
            for bj in range(num_blocks):
                # coordinates of this block
                r0, r1 = bi*block_size, (bi+1)*block_size
                c0, c1 = bj*block_size, (bj+1)*block_size
                
                block = adj[r0:r1, c0:c1]
                cutoff = np.percentile(block, setThresh)
                
                if binarize:
                    processed = (block > cutoff).astype(int)
                else:
                    processed = np.where(block > cutoff, block, 0)
                
                out[r0:r1, c0:c1] = processed
        
        return out
    

    def __subtractAverage2D__(self, adj, block_size=360):

        n = adj.shape[0]
        assert n % block_size == 0, "block_size must divide matrix size exactly"
        B = n // block_size
        
        # 1) Extract the three on-diagonal blocks
        diag_blocks = {}
        for b in range(B):
            r0, r1 = b*block_size, (b+1)*block_size
            diag_blocks[b] = adj[r0:r1, r0:r1].astype(float)
        
        # 2) Compute the element-wise mean of the on-diagonal blocks
        avg_diag_all = sum(diag_blocks[b] for b in range(B)) / B
        
        # Prepare output
        out = np.zeros_like(adj, dtype=float)
        
        # 3) Process each block
        for i in range(B):
            for j in range(B):
                r0, r1 = i*block_size, (i+1)*block_size
                c0, c1 = j*block_size, (j+1)*block_size
                
                block = adj[r0:r1, c0:c1].astype(float)
                
                if i == j:
                    # on-diagonal: subtract the mean of all on-diagonal blocks
                    block = block - avg_diag_all
                else:
                    # off-diagonal: subtract the mean of the two corresponding diagonal blocks
                    block = block - 0.5*(diag_blocks[i] + diag_blocks[j])
                
                # clip negatives to zero
                out[r0:r1, c0:c1] = np.where(block > 0, block, 0.0)
        
        return out 


    
    def runBimod(self,analysis,vector_id_max=4, n_kmeans=10, startFrom=0):

        # wiring_mod = dgsp.modularity_matrix(self.adjMatrix, null_model="outin")
        # print(f"Asymmetric wiring matrix has shape {self.adjMatrix.shape}")
        # plt.imshow(wiring_mod, cmap="gray", interpolation="none")

        # d_mat = np.diag(self.adjMatrix.sum(axis=1))
        # print(f"Degree matrix is {d_mat}")

        # sort_idx = np.flip(np.argsort(S))
        # S = S[sort_idx]
        # U = U[:, sort_idx]
        # V = V[:, sort_idx]

        graph = self.adjMatrix

        U, S, Vh = dgsp.sorted_SVD(dgsp.modularity_matrix(graph, null_model="outin"))
        V = Vh.T

        if startFrom > 0:
            vector_id_max += startFrom


        plt.plot(S[:40], "o-")
        outpath = os.path.join(self.data_dir, f'{analysis}/singularValuesSpectrum.png')
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()

        for i in range(n_kmeans):

            bplot.plot_graph_embedding(graph, vector_id=i, write_label=True, label_lw=3,
                                                use_cmap=True, cmap="silver", node_clusers=np.ones(self.N*3),
                                                outpath = os.path.join(self.data_dir, f'{analysis}/Embedding_{i}.png'))
            print(f"Embedding {i} plotted.")

        edge_clusters, edge_clusters_mat = dgsp.edge_bicommunities(graph, U[:, startFrom:], V[:, startFrom:], vector_id_max, method="kmeans",
                                                                scale_S = S[startFrom:vector_id_max], n_kmeans=n_kmeans, verbose=True, max_k=10)


        # assume edge_clusters_mat has values in {0,1,2,3,4,5}
        base = plt.cm.tab20.colors   # a tuple of 10 RGBA colors
        my_colors = ['black'] + list(base[1:len(np.unique(edge_clusters_mat))])
        cmap = ListedColormap(my_colors)

        plt.imshow(edge_clusters_mat, cmap=cmap, interpolation="none", origin='lower')
        outpath = os.path.join(self.data_dir, f'{analysis}/EdgeClusterMat.png')
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()

        sending_communities, receiving_communities = dgsp.get_node_clusters(edge_clusters, edge_clusters_mat, method="bimodularity")

        bimod_quad = dgsp.bimod_index_nodes(graph, sending_communities, receiving_communities, scale=True)
        #bimod_quad = bimod_quad**sum_power/np.sum(bimod_quad**sum_power)
        sorted_by_quad = np.flip(np.argsort(bimod_quad))

        return sending_communities.T, receiving_communities.T, bimod_quad, sorted_by_quad
    

    def plotBicoms(self, communities, name, extraName):

        num_components = communities.shape[1] # Number of eigenvectors

        indices = list(range(num_components))
        eig_layers = np.split(communities, self.num_layers, axis=0)

        for i in indices:  

            Xp_layers = []  
            for layer_idx in range(self.num_layers):
                Xp_layers.append(eig_layers[layer_idx][:, i])
            Xp_layers = np.array(Xp_layers)
            self.__plot_on_mmhcp_surface_multipleLayers__(Xp_layers.T, i+1, name, extraName=extraName)


    def __plot_on_mmhcp_surface_multipleLayers__(self, Xp, eigValue, name, vmin=None, vmax=None, cm = "RdBu_r", noSubcortical=True, titles=["Deep","Middle","Superficial","Average"], folder_name="Bicommunities", extraName=""):

        os.makedirs(f"{self.data_dir}/{name}/{folder_name}", exist_ok=True)  # Create folder for layer-wise maps

        mmp_labels = hcp.mmp.labels  # mmp = Glasser parcellation
        
        if noSubcortical:
            current_length = len(Xp[:, 0])  # Get the number of parcels (rows)
            print(f'Currend length/num of parcels: {current_length}')
            target_length = len(mmp_labels)  # Target length is the number of regions (parcels)
            zeros_to_add = target_length - current_length
            print(f'Zeros to add: {zeros_to_add}')
            Xp = np.concatenate((Xp, np.zeros((zeros_to_add, Xp.shape[1]))), axis=0)    

        orientations = ["lateral", "medial", "medial", "lateral"]

        # Determine the global min and max values across all layers
        all_data = np.hstack([hcp.cortex_data(hcp.unparcellate(Xp[:, i], hcp.mmp)) for i in range(Xp.shape[1])])
        if vmin is None or vmax is None:
            vmin, vmax = np.nanpercentile(all_data, [2, 98])  #np.min(all_data), np.max(all_data)
        
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

        plt.suptitle(f"Bicommunity {extraName} {eigValue}", fontsize=16)
        plt.savefig(f"{self.data_dir}/{name}/{folder_name}/Bicommunity_{eigValue}_{extraName}.png", facecolor="white")
        plt.close()


