import os
import numpy as np
from sklearn.decomposition import PCA, FastICA
from osl_dynamics.data import Data
import laminarRestingState as lrs


output_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/HMM"
os.makedirs(output_dir, exist_ok=True)


paths = [
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM001/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM002/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM003/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM004/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM005/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM006/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM009/Layer_run1_parcels_all_layers.npy",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM011/Layer_run1_parcels_all_layers.npy",
]

arr_list = []
for p in paths:
    X = np.load(p)
    arr_list.append(np.asarray(X, dtype=np.float64))

n_subj = len(arr_list)
T = arr_list[0].shape[0]
print(T)
P = arr_list[0].shape[1]
print(P)
print(f"{n_subj} subjects, T={T}, P={P}")

# Sanity checks
assert all(A.shape[0] == T for A in arr_list), "Different T across subjects."
assert all(A.shape[1] == P for A in arr_list), "Different #channels across subjects."
assert P == 1080, f"Expected 1080 channels, got {P}."

# ---------- 2) Layer indexing (adjust if your order differs) ----------
layers = {
    "L1": slice(0, 360),
    "L2": slice(360, 720),
    "L3": slice(720, 1080),
}

# ---------- 3) Utilities: common mask + per-layer group ICA ----------
def common_valid_mask_same_shape(Xs_layer):
    P = Xs_layer[0].shape[1]
    valid = np.ones(P, dtype=bool)
    for X in Xs_layer:
        assert X.shape[1] == P, "All subjects must have the same #parcels per layer"
        finite = np.isfinite(X).all(axis=0)
        nonconst = np.nanstd(X, axis=0) > 0
        valid &= finite & nonconst
    return valid  # length P (e.g., 360)

def zscore_time(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def group_ica_per_layer(Xs_layer, K=20, pca_max=100, random_state=0):
    # 1) common mask across subjects (fixed length = 360)
    keep_mask = common_valid_mask_same_shape(Xs_layer)
    idx_keep = np.where(keep_mask)[0]
    Xs_clean = [zscore_time(X[:, keep_mask]) for X in Xs_layer]  # all same width

    # 2) concatenate in time, PCA then ICA
    X_concat = np.concatenate(Xs_clean, axis=0)
    n_pca = int(min(pca_max, X_concat.shape[1], X_concat.shape[0]-1))
    pca = PCA(n_components=n_pca, whiten=True, random_state=random_state)
    Xw = pca.fit_transform(X_concat)

    ica = FastICA(n_components=K, whiten='arbitrary-variance',
                  max_iter=5000, tol=5e-4, random_state=random_state)
    S_concat = ica.fit_transform(Xw)

    # 3) split back to subjects
    lens = [X.shape[0] for X in Xs_clean]
    S_subjects = np.split(S_concat, np.cumsum(lens)[:-1], axis=0)

    # 4) (optional) parcel-space IC maps, padded back to 360
    # component map in kept-parcel space:
    A_kept = pca.components_.T @ ica.mixing_          # (P_kept, K)
    P_full = Xs_layer[0].shape[1]
    A_full = np.zeros((P_full, A_kept.shape[1]))       # (360, K)
    A_full[idx_keep, :] = A_kept                       # zero where parcels were dropped

    return S_subjects, pca, ica, idx_keep, A_full


# ---------- 4) Run group-ICA per layer ----------
K = 5  
S_by_layer = {}      # dict[layer] -> list of (T, K) per subject
models_by_layer = {} # dict[layer] -> (pca, ica)
idx_keep_by_layer = {}
A_full_by_layer = {}

for lname, sl in layers.items():
    Xs_layer = [X[:, sl] for X in arr_list]
    S_subjects, pca, ica, idx_keep, A_full = group_ica_per_layer(Xs_layer, K=K, pca_max=100, random_state=0)    
    S_by_layer[lname] = S_subjects
    models_by_layer[lname] = (pca, ica)
    idx_keep_by_layer[lname] = idx_keep
    A_full_by_layer[lname] = A_full

    print(f"{lname}: got {len(S_subjects)} subjects of IC time courses with shape {S_subjects[0].shape}")


A_full_L1 = A_full_by_layer["L1"]   # shape: (360, K)
A_full_L2 = A_full_by_layer["L2"]   # shape: (360, K)
A_full_L3 = A_full_by_layer["L3"]   # shape: (360, K)
A_stack = np.vstack([A_full_L1, A_full_L2, A_full_L3])  # (1080, K)

print(A_stack.shape)
restStateSub = lrs.LaminarRestingState(output_dir, 360, 1, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
restStateSub.eigvecs_to_nifti(A_stack, "ICAplots")

# # A_full_L1, A_full_L2, A_full_L3 are each (360, K)
# A_stack = np.vstack([A_full_L1, A_full_L2, A_full_L3])   # (1080, K)

# # Reuse your existing function to plot/save
# self.eigvecs_to_nifti(
#     eigvecs=A_stack,
#     name="ICA",
#     hcp_atlas=True,
#     force_run=True,
#     scaleEigVecs=True,   # optional: normalize component scales for prettier maps
#     saveNifti=True
# )




# ---------- 5) Build HMM inputs: concat ICs from all 3 layers per subject ----------
Y_subject_list = []
for s in range(n_subj):
    # order layers deterministically
    Y_s = np.concatenate([S_by_layer["L1"][s], S_by_layer["L2"][s], S_by_layer["L3"][s]], axis=1)
    # Optionally re-zscore channels (ICs) per subject
    Y_s = zscore_time(Y_s)
    Y_subject_list.append(Y_s)

print("Example subject HMM input shape:", Y_subject_list[0].shape)  # (T, 3*K)

# ---------- 6) Hand off to OSL-Dynamics ----------
data = Data(Y_subject_list)            # each item is (time, 3*K)
data.prepare({"standardize": {}})      # usually sufficient after ICA
print("n_channels for HMM:", data.n_channels)  # should be 3*K

seq_len = 100
step_size = 10
# Make sure batches will have enough time points for the initializer: need >= 2 * n_channels
min_seqs = int(np.ceil((2 * data.n_channels) / seq_len))
batch_size = max(min_seqs, 8)

train_ds = data.dataset(sequence_length=seq_len,
                        batch_size=batch_size,
                        step_size=step_size,
                        shuffle=True,
                        concatenate=True)

# ---------- 7) Configure & train HMM ----------
from osl_dynamics.models.hmm import Config, Model

config = Config(
    n_states=6,
    n_channels=data.n_channels,      # 3*K
    learn_means=True,
    learn_covariances=True,
    sequence_length=seq_len,
    batch_size=batch_size,
    learning_rate=0.05,
    n_epochs=100,
)

model = Model(config)

# Use time-course initialization on the dataset (now satisfies the batch requirement)
model.random_state_time_course_initialization(train_ds, n_epochs=1, n_init=3)
history = model.fit(train_ds)
model.save("results/model")

# ---------- 8) Dual estimation & (later) plotting ----------
means, covs = model.dual_estimation(data)
np.save(os.path.join(output_dir,"means.npy"), means)
np.save(os.path.join(output_dir, "covs.npy"), covs)
print("Saved dual-estimated means/covs.")