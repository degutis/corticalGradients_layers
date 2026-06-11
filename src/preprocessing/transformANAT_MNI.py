import ants

# Load the MNI template and subject anatomical image
mni_template = ants.image_read("/Users/karolis/Desktop/repos/ciftify_lib/MNI152_T1_0.5mm_brain.nii.gz")
subject_anat = ants.image_read("/Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/anatomical/ANAT.nii")

glasser_parcels = ants.image_read("/Users/karolis/Desktop/repos/ciftify_lib/MMP_in_MNI_corr.nii")

# Compute the transformation from MNI to subject space
tx = ants.registration(fixed=subject_anat, moving=mni_template, type_of_transform="SyN", reg_iterations = [100,100,20])

#glasserInEPI = ants.apply_transforms(fixed=subject_anat, 
#                                   moving=glasser_parcels, 
#                                    transformlist=tx, 
#                                    interpolator='nearestNeighbor')

glasserInEPI = ants.apply_transforms( fixed = subject_anat, 
                                       moving = glasser_parcels , 
                                       transformlist = tx['fwdtransforms'], 
                                       interpolator  = 'nearestNeighbor')

ants.image_write(glasserInEPI, "/Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/atlas/GlasserInEPI_kd.nii")
