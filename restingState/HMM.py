from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
import os

output_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/HMM"
os.makedirs(output_dir, exist_ok=True)

data = Data([
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM001/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM002/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM003/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM004/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM005/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM006/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM009/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM011/Layer_run1_parcels_all_layers.npy" 
])

print(data)

# If anatomical ROIs: full-rank PCA then standardize
# data.prepare({"pca": {"n_pca_components": data.n_channels}, "standardize": {}})
data.prepare({"standardize": {}})

config = Config(
    n_states=6,   
    n_channels=data.n_channels,
    learn_means=True,     
    learn_covariances=True,
    sequence_length=100,
    batch_size=22,
    learning_rate=0.005,
    n_epochs=40,
)

model = Model(config)
model.random_state_time_course_initialization(data, n_epochs=1, n_init=3)  # optional init
history = model.fit(data)
model.save(output_dir)