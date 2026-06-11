# laminar_rs/__init__.py
"""
Laminar resting-state analysis toolkit (functional API).

Public surface:
- LaminarConfig
- IO utilities
- Connectivity & adjacency builders
- Embedding & diagnostics
- Embedding-space plots and brain maps
- Graph metrics & connectograms
- Statistical models
- Reliability analyses
- Gradient-based analyses & flatmaps
"""

from .config import LaminarConfig

# IO utilities
from .io_utils import (
    group_files_by_layer,
    group_files_by_run,
    load_and_concat_layer,
    load_all_layers_concat_by_layer,
)

# Connectivity & FC / adjacency construction
from .connectivity import (
    thresh_and_binarize,
    fisher_z_to_r,
    build_multiplex_adjacency,
    within_layer_block_matrix,
    full_adjacency_multirun,
)

# Embedding (Laplacian, scree, diagnostics)
from .embedding import (
    run_laplacian_embedding,
    plot_scree,
    run_plot_zero_crossings,
    run_plot_FstatComp,
    plot_eigenvector_correlation,
)

# Embedding-space plots & brain maps
from .plots_embedding import (
    plot_scatter_with_global_correlation,
    plot_scatter3D_with_plane,
    plot_scatter_centroids,
    plot_network_centroids3D,
    plot_rsn_distributions_by_network,
)

# Graph metrics & network visualisations
from .graph_metrics import (
    rich_club_sweep,
    most_common_members,
    get_top_percent_edges,
    plot_connectogram,
    plot_connectogram_allInOne,
    run_degree_distribution,
    modularity,
    eigenvector_centrality_calc,
    eigenvector_centrality_plot,
    eigenvector_centrality_plot_avg,
)

# Statistical models / regressions
from .models import (
    plot_horizontal_correlation_bar,
    plot_horizontal_correlation_bar_partial,
)

# Reliability analyses
from .reliability import (
    compute_reliability,
    plot_reliability_curves,
)

# Gradient-based analyses
from .gradients import (
    run_gradient_analysis,
    run_gradient_analysis_auto,
    inter_areal_dissimilarity,
    intra_areal_dissimilarity,
)

# Flatmaps & matrix plotting
from .flatmaps import (
    plotMatrix,
    plotFlatMap,
)

# Surface Map Plotting
from .surface_maps import (
    plotSurfaceMap,
    plotSurfaceMap_LH_gradients,
)

# Schaefer-400 / RSN-7 network stats & spatial nulls
from .schaefer_stats import (
    RSN7_NAMES,
    build_schaefer400_bookkeeping,
    parcel_to_vertices,
    vertices_to_parcels,
    anova_F_by_network,
    p_spin_corr_schaefer400,
    p_spin_partial_corr_schaefer400,
    layerwise_network_anova,
    network_layer_interaction_general,
    save_layerwise_results_csv,
    save_interaction_to_csv,
)

# Also expose submodules for direct access
from . import (
    io_utils,
    connectivity,
    embedding,
    plots_embedding,
    graph_metrics,
    models,
    reliability,
    gradients,
    flatmaps,
    schaefer_stats,
)

__all__ = [
    # Core config
    "LaminarConfig",

    # Submodules
    "io_utils",
    "connectivity",
    "embedding",
    "plots_embedding",
    "graph_metrics",
    "models",
    "reliability",
    "gradients",
    "flatmaps",

    # IO helpers
    "group_files_by_layer",
    "group_files_by_run",
    "load_and_concat_layer",
    "load_all_layers_concat_by_layer",

    # Connectivity
    "thresh_and_binarize",
    "fisher_z_to_r",
    "build_multiplex_adjacency",
    "within_layer_block_matrix",
    "full_adjacency_multirun",

    # Embedding
    "run_laplacian_embedding",
    "plot_scree",
    "run_plot_zero_crossings",
    "run_plot_FstatComp",
    "plot_eigenvector_correlation",

    # Embedding plots & brain maps
    "plot_scatter_with_global_correlation",
    "plot_scatter3D_with_plane",
    "plot_scatter_centroids",
    "plot_network_centroids3D",
    "plot_rsn_distributions_by_network",

    # Graph metrics
    "rich_club_sweep",
    "most_common_members",
    "get_top_percent_edges",
    "plot_connectogram",
    "plot_connectogram_allInOne",
    "run_degree_distribution",
    "modularity",
    "eigenvector_centrality_calc",
    "eigenvector_centrality_plot",
    "eigenvector_centrality_plot_avg",

    # Models
    "plot_horizontal_correlation_bar",
    "plot_horizontal_correlation_bar_partial",

    # Reliability
    "compute_reliability",
    "plot_reliability_curves",

    # Gradients
    "run_gradient_analysis",
    "run_gradient_analysis_auto",
    "inter_areal_dissimilarity",
    "intra_areal_dissimilarity",

    # Flatmaps
    "plotMatrix",
    "plotFlatMap",

    # Surface Maps
    "plotSurfaceMap",
    "plotSurfaceMap_LH_gradients",

    # Schaefer / RSN-7 stats
    "RSN7_NAMES",
    "build_schaefer400_bookkeeping",
    "parcel_to_vertices",
    "vertices_to_parcels",
    "anova_F_by_network",
    "p_spin_corr_schaefer400",
    "p_spin_partial_corr_schaefer400",
    "layerwise_network_anova",
    "network_layer_interaction_general",
    "save_layerwise_results_csv",
    "save_interaction_to_csv",
]

__version__ = "0.1.0"