
# coding: utf-8

# # Virtual Lesion

# This series of Notebooks (Step 3 to 3) calculate the likeliness of the existens of a fiber bundle with LiFEs Virtual Lesion approach. The probability will be difined based on the Strength of Evidence.

# In[6]:

import os
import numpy as np
from nibabel import trackvis as tv
from utilities import *
import nibabel as nib
from dipy.tracking import utils


# In[7]:

plot = False


# In[8]:

if plot == True:
    import matplotlib.pyplot as plt
    get_ipython().magic(u'matplotlib inline')


# In[9]:

path_saveing = '/data/hcp/data/'
path = '/hcp/'


# In[10]:

subjects = os.listdir(path_saveing)
subjects_sorted = sorted(subjects)
subjects_sorted.remove('.nii.gz')

for subject in subjects_sorted:
    print 'Process subject ' + subject

    if os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_Strength_of_Evidence3.txt')) == False:
        print "    Strength of Evidence does not exist, start calculation"
        
        if os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_neighborhood.nii.gz')) == True and os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_LiFE_rmse_with_path.nii.gz')) == True and os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_LiFE_rmse_without_path.nii.gz')) == True and os.path.isfile(os.path.join(path, subject, 'T1w/Diffusion/data.nii.gz')) == True:
            print "    All necessary files exist, continue"
            
            directory_output = os.path.join(path_saveing, subject)

            #mask = load_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_neighborhood")
            #mask = mask.astype(bool)

            #with_path = load_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_rmse_with_path")
            #without_path = load_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_rmse_without_path")

            #data, affine, gtab, header, shell_mask = load_hcp_data(path, subject)

            #noise = data[..., gtab.b0s_mask].std(axis=3, ddof=1) 

            #print "    Total Error divided by STD before adding: %.2f" % np.sum(without_path[mask]/noise[mask])
            #print "    Total Error divided by STD after adding : %.2f" % np.sum(with_path[mask]/noise[mask])
            #strength = np.sum(without_path[mask]/noise[mask]) - np.sum(with_path[mask]/noise[mask])
            #print "    Strength of Evidence                    : %.2f" % strength

            #np.savetxt(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_Strength_of_Evidence2.txt'), np.array([strength]), delimiter=',')

            
            print "    Calculate optimized Fibers"
            beta_with = load_matlab_file(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_betas_with_path")['beta']

            streams, hdr = tv.read(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_clustered.trk")#, points_space='voxel'
            streamlines = [i[0] for i in streams]

            weights_with = np.squeeze(beta_with[:, -len(streamlines):])
            optimized_with = streamlines[-len(streamlines):]
            
            try:
                optimized_sl = list(np.array(optimized_with)[np.where(weights_with > 0)])

                optimized_save = ((sl, None, None) for sl in optimized_sl)
                tv.write(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_optimized.trk'), optimized_save,  hdr_mapping=hdr)
            except:
                print "Could not save streamlines, 0 streamlines detected"
        
            """
            beta_without = load_matlab_file(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_betas_without_path")['beta']

            streams_neighbourhood, hdr_neighbourhood = tv.read(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_2M_SIFT_with_path.trk", points_space='voxel')
            streamlines_neighbourhood = [i[0] for i in streams_neighbourhood]

            bins_RMSE = np.linspace(50.0,500,100)
            hist_with = np.histogram(with_path[mask], bins=bins_RMSE)
            hist_without = np.histogram(without_path[mask], bins=bins_RMSE)

            if plot == True:
                red = plt.plot(bins_RMSE[:-1], hist_with[0]*1.0/np.sum(mask), 'r')
                blue = plt.plot(bins_RMSE[:-1], hist_without[0]*1.0/np.sum(mask), 'b')
                plt.xlabel('RMSE of Diffusion Weighted Signal only')
                plt.ylabel('Probability')
                plt.legend((red[0], blue[0]), ('With Path', 'Without Path'))
                plt.title('Histogram of RMSE')

            weights_with = np.squeeze(beta_with[:, -len(streamlines):])

            mask_beta = beta_with != 0
            non_zero_beta = beta_with[mask_beta]

            mask_beta_artifically = weights_with != 0
            non_zero_beta_artifically = weights_with[mask_beta_artifically]

            if plot == True:
                bins_beta = np.linspace(0.0,0.007,100)
                hist_beta = np.histogram(non_zero_beta, bins=bins_beta)
                percentage = hist_beta[0]*1.0/non_zero_beta.shape[0]
                bins_plot = bins_beta[:-1]
                red = plt.plot(bins_plot, percentage, 'r')

                for value in non_zero_beta_artifically:
                    idx = (np.abs(bins_plot - value)).argmin()
                    blue = plt.plot(bins_plot[idx], percentage[idx], 'bo')

                plt.legend((red[0], blue[0]), ('Hole Neighborhood', 'Artificially added Fibers'))
                plt.title('Histogram of Betas with Path')
                plt.xlabel('Beta per Fiber calculated by LiFE')
                plt.ylabel('Probability')

            density = utils.density_map(streamlines, hdr['dim'], [1, 1, 1])

            mask_density = density != 0
            non_zero_density = density[mask_density]

            density_neighbourhood = utils.density_map(streamlines_neighbourhood, hdr_neighbourhood['dim'], [1, 1, 1])

            mask_density_neighbourhood = density_neighbourhood != 0
            non_zero_density_neighbourhood = density_neighbourhood[mask_density_neighbourhood]

            bins_density = np.linspace(0.0,30,100)
            hist_denisty = np.histogram(non_zero_density, bins=bins_density)
            hist_denisty_neighbourhood = np.histogram(non_zero_density_neighbourhood, bins=bins_density)

            if plot == True:
                red = plt.plot(bins_density[:-1], hist_denisty_neighbourhood[0]*1.0/np.max(hist_denisty_neighbourhood[0]), 'r')
                blue = plt.plot(bins_density[:-1], hist_denisty[0]*1.0/np.max(hist_denisty[0]), 'b')
                plt.legend((red[0], blue[0]), ('Neighborhood', 'Artificially added Fibers'))
                plt.xlabel('Fiber Density per Voxel')
                plt.ylabel('Relative Quantity')
                plt.title('Histogram of Fiber Density')
            """
        else:
            print "    Not all necessary files exist, skip subject"
            
    else:
        print "Strength of Evidence Exists already, skip subject"


# In[ ]:



