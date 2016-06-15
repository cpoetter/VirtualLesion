
# coding: utf-8

# In[1]:

import numpy as np
import nibabel as nib
from dipy.viz import fvtk
import scipy.stats as average
import dipy.core.sphere as sphere
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io
from dipy.tracking import utils
from dipy.tracking.local import LocalTracking
from dipy.tracking.eudx import EuDX
import vtk  
from dipy.core.gradients import gradient_table


# In[2]:

def load_hcp_data(path, subject):
    '''Load data, affine and gradients from the HCP data set'''
    
    dirname_praefix = '/T1w/Diffusion/'
    dirname = path + subject + dirname_praefix
    
    nii_filename = dirname + 'data.nii.gz'
    bval_filename = dirname + 'bvals'
    bvec_filename = dirname + 'bvecs'

    print "Loading data"
    img = nib.load(nii_filename)
    bvals = np.loadtxt(bval_filename)
    bvecs = np.loadtxt(bvec_filename).T
    
    print "Correcting data"
    shell_mask = np.round(bvals/100.0, 0)*100
    bvals[bvals < 0.01*np.max(bvals)] = 0 # set bval of b0 meassurements to 0
    
    print "Creating shells for LiFE"
    mask_for_life = np.logical_or(bvals > 0.98*np.max(bvals), bvals == 0.0)
    bvals = bvals[mask_for_life]
    bvecs = bvecs[mask_for_life]
    
    print "Retrieving data, affine and gradients"
    data = img.get_data()
    affine = img.get_affine()
    header = img.get_header()
    gtab = gradient_table(bvals,bvecs)
    
    print "Masking data for LiFE"
    data = data[..., mask_for_life]
    header['dim'][4] = data.shape[-1]
    
    return data, affine, gtab, header, shell_mask


# In[3]:

def save_trk(streamlines, header, filename):
    '''Save tractography to a .trk file'''
    
    print "Save tracks in %s" % filename
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] =  header.get_zooms()[:3]
    hdr['voxel_order'] = 'LAS' #'RAS'
    hdr['dim'] = header['dim'][1:4]
    
    ## Move streamlines to "trackvis space"
    #trackvis_point_space = utils.affine_for_trackvis(voxel_size)
    #lr_sf_trk = utils.move_streamlines(streamlines, trackvis_point_space, input_space=np.eye(4))
    #lr_sf_trk = list(lr_sf_trk)
    
    strm = ((sl, None, None) for sl in streamlines)
    nib.trackvis.write(filename, strm, hdr, points_space='voxel')
    
def fiber_tracking(peaks, mask):
    print "Start Fibertracking"
    seeds = utils.seeds_from_mask(mask, density=[2, 2, 2])
    streamline_generator = EuDX(peaks.peak_values, peaks.peak_indices, odf_vertices=peaks.sphere.vertices, a_low=.05, step_sz=.5, seeds=seeds)
    streamlines = list(streamline_generator)
    return streamlines


# In[4]:

def vtk_show(renderer, w=750, h=750):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(w, h)
    renderWindow.Render()
     
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()
     
    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = str(buffer(writer.GetResult()))
    
    from IPython.display import Image
    return Image(data)


# In[5]:

def save_as_matlab_file(path, **kwargs):
    # Saves single numpy variable as matlab file
    # Exp.: a = np.array([1, 2, 3]) 
    # save_as_matlab_file('file_name', variable_name = a)
    print 'Save %s' %path
    scipy.io.savemat(path, mdict=kwargs)
    return True
    
def load_matlab_file(path):
    # Loads variable stored in matlab file and returns it as numpy array
    #variable_dict = scipy.io.loadmat(path)
    #variable_names = [x for x in variable_dict.keys() if "__" not in x]
    #content = [variable_dict[x] for x in variable_names]
    variable_dict = scipy.io.loadmat(path)
    variable_names = [x for x in variable_dict.keys() if "__" in x]
    for i in variable_names:
        del variable_dict[i]
    return variable_dict

def save_as_nifti(path, data, affine):
    # Saves data as a nifti file
    print 'Save %s' %path
    image = nib.Nifti1Image(data, affine)
    filename = path + '.nii.gz'
    nib.save(image, filename)
    return True

def load_nifti(path):
    # Loads a nifti file and returns the data
    filename = path  + '.nii.gz'
    image = nib.load(filename)
    return image.get_data()

# Loading nifti image, data, affine and gtabs
def load_nifti_image(path, nifti_name, bvals_name, bvecs_name):
    nifti_path = path + nifti_name
    bvals_path = path + bvals_name
    bvecs_path = path + bvecs_name

    img = nib.load(nifti_path)
    data = img.get_data()
    affine = img.affine
    header = img.header
    
    bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
    gtab = gradient_table(bvals, bvecs)
    
    print 'Image %s loaded' % nifti_name
    return img, data, affine, header, gtab

def save_bvecs(bvecs, path, name):
    with open(path + name, 'w') as fp:
        fp.write(' '.join(['%0.4f' % bv for bv in np.squeeze(np.asarray(bvecs[:, 0]))]) + '\n')
        fp.write(' '.join(['%0.4f' % bv for bv in np.squeeze(np.asarray(bvecs[:, 1]))]) + '\n')
        fp.write(' '.join(['%0.4f' % bv for bv in np.squeeze(np.asarray(bvecs[:, 2]))]) + '\n')
    print 'bvecs %s successfully saved' % name


# In[6]:

# Calculates the Correlation Coefficient
# Faster then the build in functions for correlation
def CC(data1, data2, predicted_data1, predicted_data2, white_matter):
    predicted_data1_masked_demean = predicted_data1_masked - np.mean(predicted_data1_masked, -1)[..., np.newaxis]
    predicted_data2_masked_demean = predicted_data2_masked - np.mean(predicted_data2_masked, -1)[..., np.newaxis]
    data1_masked_demean = data1_masked - np.mean(data1_masked, -1)[..., np.newaxis]
    data2_masked_demean = data2_masked - np.mean(data2_masked, -1)[..., np.newaxis]

    predicted_data1_masked_demean_xx = np.sum(predicted_data1_masked_demean ** 2, -1)
    predicted_data2_masked_demean_xx = np.sum(predicted_data2_masked_demean ** 2, -1)
    data1_masked_demean_xx = np.sum(data1_masked_demean ** 2, -1)
    data2_masked_demean_xx = np.sum(data2_masked_demean ** 2, -1)

    CC_M1_D2_masked = np.sum(predicted_data1_masked_demean * data2_masked_demean, -1)*1.0/np.sqrt(predicted_data1_masked_demean_xx * data2_masked_demean_xx)
    CC_M2_D1_masked = np.sum(predicted_data2_masked_demean * data1_masked_demean, -1)*1.0/np.sqrt(predicted_data2_masked_demean_xx * data1_masked_demean_xx)
    CC_D1_D2_masked = np.sum(data1_masked_demean * data2_masked_demean, -1)*1.0/np.sqrt(data1_masked_demean_xx * data2_masked_demean_xx)

    # Unmasking Data
    CC_M1_D2 = np.zeros(data1.shape[:-1])
    CC_M2_D1 = np.zeros(data1.shape[:-1])
    CC_D1_D2 = np.zeros(data1.shape[:-1])

    CC_M1_D2[white_matter] = CC_M1_D2_masked
    CC_M2_D1[white_matter] = CC_M2_D1_masked
    CC_D1_D2[white_matter] = CC_D1_D2_masked
    
    return CC_M1_D2, CC_M2_D1, CC_D1_D2


# In[7]:

# Calculates rRMSE between two data sets
def rRMSE(data1, data2, predicted_data1, predicted_data2, white_matter):
    data1_masked = data1[white_matter]
    data2_masked = data2[white_matter]
    predicted_data1_masked = predicted_data1[white_matter]
    predicted_data2_masked = predicted_data2[white_matter]

    #Calculation rRMSE
    RMSE_M1_D2_masked = np.sqrt(np.average((predicted_data1_masked-data2_masked)**2, axis=-1, weights=data2_masked.astype(bool)))
    RMSE_M2_D1_masked = np.sqrt(np.average((predicted_data2_masked-data1_masked)**2, axis=-1, weights=data1_masked.astype(bool)))
    RMSE_D1_D2_masked = np.sqrt(np.average((data1_masked-data2_masked)**2, axis=-1, weights=data1_masked.astype(bool)))

    all_zero = (RMSE_D1_D2_masked == 0).astype(int)
    rRMSE_masked = (RMSE_M1_D2_masked + RMSE_M2_D1_masked)/(2 * RMSE_D1_D2_masked + all_zero)

    # Unmasking Data
    rRMSE = np.zeros(data1.shape[:-1])
    rRMSE[white_matter] = rRMSE_masked
        
    return rRMSE


# In[8]:

def draw_histogramm(rRMSE_masked):
    bins_rRMSE = np.linspace(0.6,1.1,100)
    hist_rRMSE = np.histogram(rRMSE_masked, bins=bins_rRMSE)
    plt.plot(bins_rRMSE[:-1], hist_rRMSE[0], 'r')
    plt.xlim(xmin=0.6)
    plt.xlim(xmax=1.1)
    plt.ylim(ymax=max(hist_rRMSE[0])+100)
    plt.xlabel('rRMSE')
    plt.ylabel('Quantity')
    plt.show()

    print 'Most Appearing rRMSE:        ', bins_rRMSE[:-1][np.argmax(hist_rRMSE[0])]
    print 'Mean:                        ', np.mean(bins_rRMSE[:-1])
    print 'Total rRMSE Sum:             ', np.sum(hist_rRMSE[0]*bins_rRMSE[:-1])
    
    for i in range(hist_rRMSE[0].shape[0]):
        print '(', "{0:.3f}".format(bins_rRMSE[i]), ',', hist_rRMSE[0][i], ')'   


# In[9]:

def compare_histogramm(rRMSE_masked, rRMSE_masked2):
    bins_rRMSE = np.linspace(0.6,1.1,100)
    hist_rRMSE = np.histogram(rRMSE_masked, bins=bins_rRMSE)
    hist_rRMSE2 = np.histogram(rRMSE_masked2, bins=bins_rRMSE)
    red = plt.plot(bins_rRMSE[:-1], hist_rRMSE[0], 'r')
    blue = plt.plot(bins_rRMSE[:-1], hist_rRMSE2[0], 'b')
    plt.xlim(xmin=0.6)
    plt.xlim(xmax=1.1)
    plt.ylim(ymax=max(hist_rRMSE[0])+100)
    plt.xlabel('rRMSE')
    plt.ylabel('Quantity')
    plt.legend((red[0], blue[0]), ('Dataset 1', 'Dataset 2') )
    plt.show()


# In[10]:

def draw_p(p, x):
    ren = fvtk.ren()
    raw_data = x * np.array([p, p, p])
    #point = fvtk.point(raw_data, fvtk.colors.red, point_radius=0.00001)
    #fvtk.add(ren, point)
    #fvtk.show(ren)
    
    new_sphere = sphere.Sphere(xyz=x)
    sf1 = fvtk.sphere_funcs(raw_data, new_sphere, colormap=None, norm=False, scale=0)
    sf1.GetProperty().SetOpacity(0.35)
    fvtk.add(ren, sf1)


# In[11]:

def draw_odf(data, gtab, odf, sphere_odf):

    ren = fvtk.ren()
    bvecs = gtab.bvecs
    raw_data = bvecs * np.array([data, data, data]).T

    # Draw Raw Data as points, Red means outliers    
    point = fvtk.point(raw_data[~gtab.b0s_mask, :], fvtk.colors.red, point_radius=0.05)
    fvtk.add(ren, point)

    sf1 = fvtk.sphere_funcs(odf, sphere_odf, colormap=None, norm=False, scale=0)
    sf1.GetProperty().SetOpacity(0.35)
    fvtk.add(ren, sf1)
    
    fvtk.show(ren)

def draw_points(data, gtab, predicted_data):
    ren = fvtk.ren()
    bvecs = gtab.bvecs
    raw_points = bvecs * np.array([data, data, data]).T
    predicted_points = bvecs * np.array([predicted_data, predicted_data, predicted_data]).T

    # Draw Raw Data as points, Red means outliers    
    point = fvtk.point(raw_points[~gtab.b0s_mask, :], fvtk.colors.red, point_radius=0.02)
    fvtk.add(ren, point)

    new_sphere = sphere.Sphere(xyz=bvecs[~gtab.b0s_mask, :])
    sf1 = fvtk.sphere_funcs(predicted_data[~gtab.b0s_mask], new_sphere, colormap=None, norm=False, scale=0)
    sf1.GetProperty().SetOpacity(0.35)
    fvtk.add(ren, sf1)
    
    fvtk.show(ren)

def draw_ellipsoid(data, gtab, outliers, data_without_noise):
    
    bvecs = gtab.bvecs
    raw_data = bvecs * np.array([data, data, data]).T
    
    ren = fvtk.ren()
    
    # Draw Sphere of predicted data
    new_sphere = sphere.Sphere(xyz=bvecs[~gtab.b0s_mask, :])
    sf1 = fvtk.sphere_funcs(data_without_noise[~gtab.b0s_mask], new_sphere, colormap=None, norm=False, scale=1)
    sf1.GetProperty().SetOpacity(0.35)
    fvtk.add(ren, sf1)
    
    # Draw Raw Data as points, Red means outliers
    good_values = [index for index, voxel in enumerate(outliers) if voxel == 0]
    bad_values = [index for index, voxel in enumerate(outliers) if voxel == 1]
    point_actor_good = fvtk.point(raw_data[good_values, :], fvtk.colors.yellow, point_radius=.05)
    point_actor_bad = fvtk.point(raw_data[bad_values, :], fvtk.colors.red, point_radius=0.05)
    fvtk.add(ren, point_actor_good)
    fvtk.add(ren, point_actor_bad)

    fvtk.show(ren)
    
def draw_adc(D_noisy, D, threeD = False):
    
    # 3D Plot
    if threeD == True:
        
        amound = 50
        alpha = np.linspace(0, 2*np.pi, amound)
        theta = np.linspace(0, 2*np.pi, amound)
        vector = np.empty((amound**2, 3))
        vector[:, 0] = (np.outer(np.sin(theta), np.cos(alpha))).reshape((-1,1))[:,0]
        vector[:, 1] = (np.outer(np.sin(theta), np.sin(alpha))).reshape((-1,1))[:,0]
        vector[:, 2] = (np.outer(np.cos(theta), np.ones(amound))).reshape((-1,1))[:,0]
    
        adc_noisy = np.empty((vector.shape[0],3))
        shape_noisy = np.empty((vector.shape[0],1))
        shape = np.empty((vector.shape[0],1))

        for i in range(vector.shape[0]):
            adc_noisy[i] = np.dot(vector[i], np.dot(vector[i], np.dot(D_noisy, vector[i].T)))
            shape_noisy[i] = np.dot(vector[i], np.dot(D_noisy, vector[i].T))
            shape[i] = np.dot(vector[i], np.dot(D, vector[i].T))

        ren = fvtk.ren() 

        # noisy sphere
        new_sphere = sphere.Sphere(xyz=vector)
        sf1 = fvtk.sphere_funcs(shape_noisy[:, 0], new_sphere, colormap=None, norm=False, scale=0.0001)
        sf1.GetProperty().SetOpacity(0.35)
        sf1.GetProperty().SetColor((1, 0, 0))
        fvtk.add(ren, sf1)

        # ideal sphere
        sf2 = fvtk.sphere_funcs(shape[:, 0], new_sphere, colormap=None, norm=False, scale=0.0001)
        sf2.GetProperty().SetOpacity(0.35)
        fvtk.add(ren, sf2)
        #point_actor_bad = fvtk.point(adc_noisy, fvtk.colors.red, point_radius=0.00003)
        #fvtk.add(ren, point_actor_bad)

        fvtk.show(ren)
        
    # 2D Plot in XY-Plane
    alpha = np.linspace(0, 2*np.pi, 100)
    vector = np.empty((100, 3))
    vector[:, 0] = np.cos(alpha)
    vector[:, 1] = np.sin(alpha)
    vector[:, 2] = 0.0
    adc_noisy_2d = np.empty((vector.shape[0],3))
    adc_2d = np.empty((vector.shape[0],3))
    for i in range(vector.shape[0]):
        adc_noisy_2d[i] = np.dot(vector[i], np.dot(vector[i], np.dot(D_noisy, vector[i].T)))
        adc_2d[i] = np.dot(vector[i], np.dot(vector[i], np.dot(D, vector[i].T)))
    
    # Change Axis so that there is 20% room on each side of the plot
    x = np.concatenate((adc_noisy_2d, adc_2d), axis=0)
    minimum = np.min(x, axis=0)    
    maximum = np.max(x, axis=0)
    plt.plot(adc_noisy_2d[:,0],adc_noisy_2d[:,1], 'r')
    plt.plot(adc_2d[:,0],adc_2d[:,1],'b')
    plt.axis([minimum[0]*1.2, maximum[0]*1.2, minimum[1]*1.2, maximum[1]*1.2])
    plt.xlabel('ADC (mm2*s-1)')
    plt.ylabel('ADC (mm2*s-1)')
    red_patch = mpatches.Patch(color='red', label='Noisy ADC')
    blue_patch = mpatches.Patch(color='blue', label='Ideal ADC')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()


# In[12]:

def draw_p(p, x, new_sphere):
    ren = fvtk.ren()
    raw_data = x * np.array([p, p, p])

    sf1 = fvtk.sphere_funcs(raw_data, new_sphere, colormap=None, norm=False, scale=0)
    sf1.GetProperty().SetOpacity(0.35)
    fvtk.add(ren, sf1)
    
    fvtk.show(ren)
    
def draw_p_2D(p, x):
    raw_data = x * np.array([p, p, p])
    plt.plot(raw_data[0, :] ,raw_data[2, :], 'ro', ms=2.0)
    plt.show()


# In[ ]:



