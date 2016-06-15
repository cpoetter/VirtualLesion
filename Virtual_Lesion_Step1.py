
# coding: utf-8

# # Virtual Lesion

# This series of Notebooks (Step 1 to 3) calculate the likeliness of the existens of a fiber bundle with LiFEs Virtual Lesion approach. The probability will be difined based on the Strength of Evidence.

# In[3]:

import numpy as np
from nibabel import trackvis as tv
from dipy.segment.clustering import QuickBundles
import utilities
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
import itertools
import MRTrix2TrackVis
import os


# In[11]:

# Library of Files
path = '/hcp/'
path_saveing = '/data/hcp/data/'

subjects = os.listdir(path_saveing)

subjects_sorted = sorted(subjects)
subjects_sorted.remove('.nii.gz')

for subject in subjects_sorted:
    print 'Process subject ' + subject
    
    if os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_clustered2.trk')) == False:
        print "    Clustered File does not exist for this subject, start calculation."
    
        if os.path.isfile(os.path.join(path_saveing, subject, 'FOD.mif')) == True and os.path.isfile(os.path.join(path_saveing, subject, 'ROI_RVLPFC_15mm_54_27_12.nii.gz')) == True and os.path.isfile(os.path.join(path_saveing, subject, 'ROI_FIRSTamyg_bigRight.nii.gz')) == True:
            print "    All neccessary files there, continue ..."
    
            directory_output = os.path.join(path_saveing, subject)

            if os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_combined.tck')) == False:
                print '    Fiber Tracks do not exist, start First Fiber Fracking'
                cmd = "tckgen " + directory_output + "/FOD.mif " + directory_output + "/RVLPFC2FIRSTamyg_bigRight1.tck -number 2500 -seed_image " + directory_output + "/ROI_RVLPFC_15mm_54_27_12.nii.gz  -include " + directory_output + "/ROI_FIRSTamyg_bigRight.nii.gz -force -maxnum 500000000 -act " + directory_output + "/5TT.mif -backtrack -crop_at_gmwmi -maxlength 250"
                os.system(cmd)

                print '    Start Second Fiber Fracking'
                cmd = "tckgen " + directory_output + "/FOD.mif " + directory_output + "/RVLPFC2FIRSTamyg_bigRight2.tck -number 2500 -seed_image " + directory_output + "/ROI_FIRSTamyg_bigRight.nii.gz -include " + directory_output + "/ROI_RVLPFC_15mm_54_27_12.nii.gz -force -maxnum 500000000 -act " + directory_output + "/5TT.mif -backtrack -crop_at_gmwmi -maxlength 250"
                os.system(cmd)

                print '    First step to remove too long fiber from the first streamlines'
                cmd = "tckedit " + directory_output + "/RVLPFC2FIRSTamyg_bigRight1.tck " + directory_output + "/RVLPFC2FIRSTamyg_bigRight1_cut.tck -include " + directory_output + "/ROI_FIRSTamyg_bigRight.nii.gz -test_ends_only -force"
                os.system(cmd)

                print '    Second step to remove too long fiber from the first streamlines'
                cmd = "tckedit " + directory_output + "/RVLPFC2FIRSTamyg_bigRight1_cut.tck " + directory_output + "/RVLPFC2FIRSTamyg_bigRight1_cut_cut.tck -include " + directory_output + "/ROI_RVLPFC_15mm_54_27_12.nii.gz -test_ends_only -force"
                os.system(cmd)

                print '    First step to remove too long fiber from the second streamlines'
                cmd = "tckedit " + directory_output + "/RVLPFC2FIRSTamyg_bigRight2.tck " + directory_output + "/RVLPFC2FIRSTamyg_bigRight2_cut.tck -include " + directory_output + "/ROI_RVLPFC_15mm_54_27_12.nii.gz -test_ends_only -force"
                os.system(cmd)

                print '    Second step to remove too long fiber from the second streamlines'
                cmd = "tckedit " + directory_output + "/RVLPFC2FIRSTamyg_bigRight2_cut.tck " + directory_output + "/RVLPFC2FIRSTamyg_bigRight2_cut_cut.tck -include " + directory_output + "/ROI_FIRSTamyg_bigRight.nii.gz -test_ends_only -force"
                os.system(cmd)

                print '    Combine resulting streamlines'
                cmd = "tckedit " + directory_output + "/RVLPFC2FIRSTamyg_bigRight1_cut_cut.tck " + directory_output + "/RVLPFC2FIRSTamyg_bigRight2_cut_cut.tck " + directory_output + "/RVLPFC2FIRSTamyg_bigRight_combined.tck  -force"
                os.system(cmd)
                
            else:
                f_in_nifti = os.path.join(path, subject, 'T1w/Diffusion/data.nii.gz')
                f_in_stream = os.path.join(directory_output, 'RVLPFC2FIRSTamyg_bigRight_combined.tck')
                f_out_converted = os.path.join(directory_output, 'RVLPFC2FIRSTamyg_bigRight_combined.trk')
                f_out_clustered = os.path.join(directory_output, 'RVLPFC2FIRSTamyg_bigRight_clustered.trk')
                f_out_centroids = os.path.join(directory_output, 'RVLPFC2FIRSTamyg_bigRight_centroids.trk')

                if os.path.isfile(f_in_nifti) == True:
                    print "    Can access raw nifti data, start conversion and clustering."

                    print '    Convert MRTrix streams to TrackVis'
                    try: 
                        MRTrix2TrackVis.convert_tck2trk(f_in_stream, f_in_nifti, f_out_converted)
                    except:
                        print 'Could not convert .tck to .trk'

                    print '    Cluster Steams'
                    try: 
                        streams, hdr = tv.read(f_out_converted)
                        streamlines = [i[0] for i in streams]

                        feature = ResampleFeature(nb_points=50)
                        metric = AveragePointwiseEuclideanMetric(feature=feature)
                        qb = QuickBundles(threshold=10., metric=metric)
                        clusters = qb.cluster(streamlines)

                        major_cluster = clusters > 60
                        major_path = []
                        #centroids = []
                        for j in range(len(clusters)):
                            if major_cluster[j] == True:
                                major_path.append([streamlines[i] for i in clusters[j].indices])
                        #        centroids.append(clusters[j].centroid)  
                        major_streams = list(itertools.chain(*major_path))

                        strm = ((sl, None, None) for sl in major_streams)
                        tv.write(f_out_clustered, strm,  hdr_mapping=hdr)
                        
                        #strm_centroids = ((sl, None, None) for sl in centroids)
                        #tv.write(f_out_centroids, strm_centroids,  hdr_mapping=hdr)
                        print '    All done'
                        
                    except:
                        print '    Could not Cluster streams'
                else:
                    print "    Could not load raw diffusion data, skip conversion and clustering."
        else:
            print "    Some input files are missing, skip this subject."
    else:
        print "    Clustered File exists already for this subject, skip calculation."


# In[ ]:



