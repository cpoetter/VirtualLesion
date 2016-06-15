
# coding: utf-8

# # MrTrix to TrackVis Fiber converter

# This function takes a .tck file generates with MrTrix and converts it to a .trk file viewable with TrackVis.

# In[1]:

import nipype.interfaces.mrtrix as mrt


# In[2]:

def convert_tck2trk(streamline_file, image_file, output_file):
    tck2trk = mrt.MRTrix2TrackVis()
    tck2trk.inputs.in_file = streamline_file
    tck2trk.inputs.image_file = image_file
    tck2trk.inputs.out_filename  = output_file
    tck2trk.run()

