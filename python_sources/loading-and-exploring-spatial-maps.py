#!/usr/bin/env python
# coding: utf-8

# # Loading and Exploring Spatial Maps
# 
# In this notebook, we show how to load and display subject spatial map information for fMRI spatial maps. 
# In general the spatial maps are saved as 4-D tensors
# 
# $$\mathcal{X}_i \in \mathbb{R}^{X \times Y \times Z \times K}$$
# 
# where $X$, $Y$, and $Z$ are the three spatial dimensions of the volume, and $K$ is the number of independent 
# components.
# 
# ## File Format
# 
# The subject spatial maps have been saved in `.mat` files using the `v7.3` flag, so they must be loaded as `h5py` datasets,
# and a nifti file must be used to set the headers for display purposes. We have included the `load_subject` function, which 
# takes a subject `.mat` filename, and the loaded nilearn image to use for setting the headers.

# In[ ]:


# Download the ch2better template image for display
get_ipython().system('wget https://github.com/Chaogan-Yan/DPABI/raw/master/Templates/ch2better.nii')


# In[ ]:


"""
    Load and display a subject's spatial map
"""

import numpy as np # linear algebra
import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
import h5py
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
mask_filename = '../input/fmri-mask/fMRI_mask.nii'
subject_filename = '../input/trends-assessment-prediction/fMRI_train/10004.mat'
smri_filename = 'ch2better.nii'
mask_niimg = nl.image.load_img(mask_filename)

def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with
        the version 7.3 flag. Return the subject
        niimg, using a mask niimg as a template
        for nifti headers.
        
    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(subject_filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    return subject_niimg
subject_niimg = load_subject(subject_filename, mask_niimg)
print("Image shape is %s" % (str(subject_niimg.shape)))
num_components = subject_niimg.shape[-1]
print("Detected {num_components} spatial maps".format(num_components=num_components))


# ## Displaying all Components in a Probability Atlas
# First, we will display the 53 spatial maps in one complete atlas using the nilearn `plot_prob_atlas` function. These 
# maps will be overlaid on a structural MRI template. 

# In[ ]:


nlplt.plot_prob_atlas(subject_niimg, bg_img=smri_filename, view_type='filled_contours', draw_cross=False, title='All %d spatial maps' % num_components, threshold='auto')


# ## Displaying Individual Component Maps
# 
# Additionally, we can separately display each of the 53 maps to get a more complete view
# of individual component structure.

# In[ ]:


grid_size = int(np.ceil(np.sqrt(num_components)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*10, grid_size*10))
[axi.set_axis_off() for axi in axes.ravel()]
row = -1
for i, cur_img in enumerate(nl.image.iter_img(subject_niimg)):
    col = i % grid_size
    if col == 0:
        row += 1
    nlplt.plot_stat_map(cur_img, bg_img=smri_filename, title="IC %d" % i, axes=axes[row, col], threshold=3, colorbar=False)

