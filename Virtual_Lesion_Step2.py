
# coding: utf-8

# # Virtual Lesion
# This series of notebooks is used to determine the evidence for a pathway connecting two ROIs following those described in Pestilli et al.<sup>[1]</sup> and Leong et al.<sup>[2]</sup>.
# 
# Calculation steps done in this notebook:
# 
# <ol>
#     <li>Calculate the neighborhood of the candidate streamlines (Voxels where the streamlines pass through)</li>
#     <li>Find fibers in the whole brain connectome that pass through the neibghorhood</li>
#     <li>Combine whole brain connectome with candidate streamlines into one steamline set</li>
#     <li>Run the LiFE optimization on this combined streamline set and retrieve RMSE between the original data set and the LiFE prediction</li>
#     <li>Run the LiFE otimization and RMSE calculation again on the whole brain, but this time without the candidate streamlines</li>
# </ol>
# 
# **<small>Do not run the parallization on all cores. This notebook is parallalized by subjects. The limiting factor is the memory usage per process.</small>
# 
# <sup>[1]</sup> <i>PMID: 25194848</i> <br/>
# <sup>[2]</sup> <i>PMID: 26748088</i> <br/>

# In[1]:

from dipy.tracking.vox2track import streamline_mapping
import os
import numpy as np
from nibabel import trackvis as tv
from utilities import *
import nibabel as nib
import dipy.tracking.life as life
from dipy.tracking import utils
import copy
from parallelization import *


# In[2]:

# Library of Files
path = '/hcp/'
path_saveing = '/data/hcp/data/'

subjects = os.listdir(path_saveing)

subjects_sorted = sorted(subjects)
subjects_sorted.remove('.nii.gz')  


# In[4]:

def run_LiFE(subject):
    print 'Process subject ' + subject

    if os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_LIFE_started2.txt')) == False:
        print "LiFE Files do not exist for this subject, start calculation."

        if os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_clustered.trk')) == True and os.path.isfile(os.path.join(path_saveing, subject, '2M_SIFT.trk')) == True:
            print "All neccessary files there, continue ..."

            print "Show other processes that this subject is processed"
            done = np.array([1])
            np.savetxt(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_LIFE_started2.txt'), done, delimiter=',')
            
            try: 
                directory_output = os.path.join(path_saveing, subject)

                print "Start calculation for subject %s" % subject
                f_streamlines = os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_clustered.trk')
                f_in_nifti = os.path.join(path, subject, 'T1w/Diffusion/data.nii.gz')

                streams, hdr = tv.read(f_streamlines, points_space='voxel')
                streamlines = [i[0] for i in streams]

                data, affine, gtab, header, shell_mask = load_hcp_data(path, subject)
                dim = header['dim'][1:4]
                
                # Otherwise all weights are NaN
                data[data <= 0.0] = 1.0

                print "Calculating neighborhood with LiFE"
                fiber_model_neighborhood = life.FiberModel(gtab)
                fiber_fit_neighborhood = fiber_model_neighborhood.fit(data, streamlines, affine=np.eye(4))
                indices_neighborhood = fiber_fit_neighborhood.vox_coords

                neighborhood  = np.zeros(dim, dtype=bool)
                for i in range(indices_neighborhood.shape[0]):
                    neighborhood[indices_neighborhood[i][0], indices_neighborhood[i][1], indices_neighborhood[i][2]] = 1

                save_as_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_neighborhood", neighborhood.astype(np.int), affine)
                
                print 'Find fiber that pass through neighborhood'
                f_streamlines_whole_brain = path_saveing + subject + "/2M_SIFT.trk"
                streams_whole_brain, hdr_whole_brain = tv.read(f_streamlines_whole_brain, points_space='voxel')
                streamlines_whole_brain = [i[0] for i in streams_whole_brain]
                neighborhood_streamlines = utils.target(streamlines_whole_brain, neighborhood, affine=np.eye(4))

                neighborhood_streamlines = list(neighborhood_streamlines)

                strm = ((sl, None, None) for sl in neighborhood_streamlines)
                tv.write(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_2M_SIFT_without_path.trk", strm,  hdr_mapping=hdr_whole_brain, points_space='voxel')

                print "Combine streamlines"
                streamlines_together = neighborhood_streamlines + streamlines
                strm_together = ((sl, None, None) for sl in streamlines_together)
                tv.write(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_2M_SIFT_with_path.trk", strm_together,  hdr_mapping=hdr_whole_brain, points_space='voxel')

                print "Start LiFE optimization with new path"
                fiber_model_together = life.FiberModel(gtab)
                fiber_fit_together = fiber_model_together.fit(data, streamlines_together, affine=np.eye(4))
                model_predict_together = fiber_fit_together.predict()
                indices_together = fiber_fit_together.vox_coords

                mask_with  = np.zeros(dim, dtype=bool)
                whole_brain_together = np.zeros(header['dim'][1:5])
                for i in range(indices_together.shape[0]):
                    whole_brain_together[indices_together[i][0], indices_together[i][1], indices_together[i][2]] = model_predict_together[i]
                    mask_with[indices_together[i][0], indices_together[i][1], indices_together[i][2]] = 1

                save_as_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_prediction_with_path", whole_brain_together, affine)
                save_as_matlab_file(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_betas_with_path", beta = fiber_fit_together.beta)
                save_as_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_mask_with_path", mask_with.astype(np.int), affine)

                print "Calculate RMSE with"
                model_error_together = model_predict_together - fiber_fit_together.data
                model_rmse_together = np.sqrt(np.mean(model_error_together[..., ~gtab.b0s_mask] ** 2, -1))

                whole_brain_rmse_together = np.zeros(dim)
                for i in range(indices_together.shape[0]):
                    whole_brain_rmse_together[indices_together[i][0], indices_together[i][1], indices_together[i][2]] = model_rmse_together[i]

                save_as_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_rmse_with_path", whole_brain_rmse_together, affine)
                
                print "Start LiFE optimization without new path"
                fiber_fit = copy.deepcopy(fiber_fit_together)
                fiber_fit.beta[-len(streamlines):] = 0
                model_predict = fiber_fit.predict()
                indices = fiber_fit.vox_coords

                whole_brain = np.zeros(header['dim'][1:5])
                mask_without  = np.zeros(dim, dtype=bool)
                for i in range(indices.shape[0]):
                    whole_brain[indices[i][0], indices[i][1], indices[i][2]] = model_predict[i]
                    mask_without[indices[i][0], indices[i][1], indices[i][2]] = 1

                save_as_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_prediction_without_path", whole_brain, affine)
                save_as_matlab_file(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_betas_without_path", beta = fiber_fit.beta)
                save_as_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_mask_without_path", mask_without.astype(np.int), affine)

                print "Calculate RMSE without"
                model_error = model_predict - fiber_fit.data
                model_rmse = np.sqrt(np.mean(model_error[..., ~gtab.b0s_mask] ** 2, -1))

                whole_brain_rmse = np.zeros(dim)
                for i in range(indices.shape[0]):
                    whole_brain_rmse[indices[i][0], indices[i][1], indices[i][2]] = model_rmse[i]

                save_as_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_rmse_without_path", whole_brain_rmse, affine)
                
                print "All done"
            except:
                print "An error occured while computing LiFE. Skip this subject."
        else:
            print "Some input files are missing, skip this subject."
    else:
        print "LiFE Files exist already for this subject, skip calculation."
    
    return 0


# In[ ]:

subjects_for_life = subjects_sorted
p = parallelization(maximum_number_of_cores=7, display=False)
return_values = p.start(run_LiFE, len(subjects_for_life), subjects_for_life)


# In[ ]:



