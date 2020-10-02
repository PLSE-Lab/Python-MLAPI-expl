#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook is setup to reproduce some of the results published by MIT in https://arxiv.org/pdf/1802.02604.pdf and made popular in media sites like: https://www.healthdatamanagement.com/news/mit-algorithm-speeds-process-of-image-registration
# 
# The challenge here is to see how the results look on their test datasets and see how the performance is using K80 GPU's provided by Kaggle. Furthre notebooks could go and compare these results to those from standard approaches using ITK/SimpleITK and even OpenCV-based image registration

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os, sys
import glob
# image code
import SimpleITK as sitk
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import axes3d
# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
def meshgridnd_like(in_img,
                    rng_func=range):
    new_shape = list(in_img.shape)
    all_range = [rng_func(i_len) for i_len in new_shape]
    return tuple([x_arr.swapaxes(0, 1) for x_arr in np.meshgrid(*all_range)])


# In[2]:


vm_dir = '../input/voxelmorph-master/voxelmorph-master'
sys.path.append(os.path.join(vm_dir, 'src')) # add source
sys.path.append(os.path.join(vm_dir, 'ext', 'medipy-lib'))
import medipy
import networks
from medipy.metrics import dice
import datagenerators


# # Setup the Network
# The network is designed for specific dimensions and in particular the pretrained values expect these dimensions and kernel count. If we want to use it without retrainined we are forced to stick to these sizes / depths

# In[3]:


nf_enc=[16,32,32,32]
nf_dec=[32,32,32,32,32,16,16,3]
vol_size=(160,192,224) # old size for brain MRI
vol_size = (192, 256, 256) # use more appropriate dimensions for CT scans
# generate some grids based on these values
xx = np.arange(vol_size[1])
yy = np.arange(vol_size[0])
zz = np.arange(vol_size[2])
grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)


# # Load Atlas Data
# Here we load the atlas data so we can try and make our CT data have a similar distribution

# In[4]:


atlas = np.load(os.path.join(vm_dir, 'data', 'atlas_norm.npz'))
atlas_vol = atlas['vol']
print(atlas_vol.shape, atlas_vol.min(), atlas_vol[atlas_vol>0].mean(), atlas_vol[atlas_vol>0].std(), atlas_vol.max())
plt.hist(atlas_vol[atlas_vol>0], 100);


# # Load CT Data
# Here we load the DICOMs as full 3D volumes using SimpleITK and we pick the first as the reference volume and the second as the moving volume

# In[26]:


dicom_dir = '../input'
all_ct_folders = sorted([path for path, _, files in os.walk(dicom_dir) 
                  if any([('.dcm' in c_file) or 
                          c_file.startswith('IM') for c_file in files])])
print(len(all_ct_folders), 'scans found', all_ct_folders)


# In[6]:


def load_dicom_stack(in_folder):
    """read and make isotropic"""
    series_reader = sitk.ImageSeriesReader()
    cur_paths = series_reader.GetGDCMSeriesFileNames(in_folder)
    print(cur_paths[0])
    c_img = sitk.ReadImage(cur_paths)
    print([c_img.GetMetaData(k) for k in c_img.GetMetaDataKeys()])
    c_vox_size = np.array(c_img.GetSpacing()[::-1])
    c_arr = sitk.GetArrayFromImage(c_img)
    print(c_vox_size, c_arr.shape)
    n_vox_size = np.mean(c_vox_size)
    n_arr = zoom(c_arr, c_vox_size/n_vox_size)
    print(n_vox_size, n_arr.shape)
    return n_arr[::-1]


# In[27]:


# load all the data
all_vols = {c_folder: load_dicom_stack(c_folder) 
             for c_folder in all_ct_folders}


# In[30]:


fixed_vol, moving_vol, *_ = list(all_vols.values())
fig, ax1 = plt.subplots(1,1,figsize = (8, 8))
ax1.hist(fixed_vol.ravel(), 100, log = True, label = 'Fixed');
ax1.hist(moving_vol.ravel(), 100, log = True, label = 'Moving', alpha = 0.5);
ax.legend()


# In[31]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))
ax1.imshow(np.max(fixed_vol, 0), cmap = 'bone')
ax2.imshow(np.max(fixed_vol, 1), cmap = 'bone')
ax3.imshow(np.max(fixed_vol, 2), cmap = 'bone')


# In[32]:


print(fixed_vol.shape, fixed_vol.min(), fixed_vol.mean(), fixed_vol.max())
fig, ax1 = plt.subplots(1,1,figsize = (8, 8))
ax1.hist(fixed_vol.ravel(), 100, log = True);


# ## Matching Distributions
# Neural Networks are much happier when distributions are similar, so we try and match the mean and standard deviation values as a starting point

# In[33]:


min_ct, max_ct = -1100, 1000
mean_ct, std_ct = -644, 495
mean_atlas, std_atlas = 0.29250547, 0.118224226
def rescale_vol(in_vol):
    """rescale to atlas intensity and make dimensions correct"""
    n_vol = (((np.clip(in_vol, min_ct, max_ct)-mean_ct)/std_ct)*std_atlas+mean_atlas)
    old_shape = np.array(n_vol.shape)
    new_shape = np.array(vol_size)
    n_vol = zoom(n_vol, new_shape/old_shape)
    return np.expand_dims(np.expand_dims(n_vol, 0), -1)


# In[34]:


fig, ax1 = plt.subplots(1,1,figsize = (6, 6))
n_fixed_vol = rescale_vol(fixed_vol)
n_moving_vol = rescale_vol(moving_vol)
print(n_fixed_vol.shape, n_fixed_vol.min(), n_fixed_vol.mean(), n_fixed_vol.std(), n_fixed_vol.max())
out_vals = ax1.hist(atlas_vol[atlas_vol>0], 100, label = 'Atlas MRI', alpha = 0.5, normed=True, log = True)
ax1.hist(n_fixed_vol.ravel(), out_vals[1], label = 'Rescaled CT-Fix', alpha = 0.5, normed=True, log = True)
ax1.hist(n_moving_vol.ravel(), out_vals[1], label = 'Rescaled CT-Moving', alpha = 0.5, normed=True, log = True)
ax1.legend()


# # Load the Model
# Here we create the model and load in the pretrained weights

# In[13]:


net = networks.unet(vol_size, nf_enc, nf_dec)
net.load_weights(os.path.join(vm_dir, 'models',  'vm2_cc.h5'))
net.summary()


# # Gut Check
# Shift a dataset a few voxels and see if the results make sense

# In[36]:


Y_vol = np.roll(np.roll(n_fixed_vol, shift = -3, axis = 2), shift = +5, axis = 3)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))
ax1.imshow(np.max(Y_vol[0, :, :, :, 0], 0), cmap = 'bone')
ax2.imshow(np.max(Y_vol[0, :, :, :, 0], 1), cmap = 'bone')
ax3.imshow(np.max(Y_vol[0, :, :, :, 0], 2), cmap = 'bone')


# In[37]:


get_ipython().run_cell_magic('time', '', 'pred = net.predict([Y_vol, n_fixed_vol])')


# In[38]:


flow = pred[1][0, :, :, :, :]
sample = flow+grid
sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
warp_vol = interpn((yy, xx, zz), Y_vol[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)


# In[39]:


DS_FACTOR = 16
c_xx, c_yy, c_zz = [x.flatten()
                    for x in 
                    meshgridnd_like(flow[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, 0])]

get_flow = lambda i: flow[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, i].flatten()


# In[40]:


fig = plt.figure(figsize = (10, 10))
ax = fig.gca(projection='3d')

ax.quiver(c_xx,
          c_yy,
          c_zz,
          get_flow(0),
          get_flow(1), 
          get_flow(2), 
          length=0.5, 
          normalize=True)


# In[41]:


fig, m_axs = plt.subplots(2, 3, figsize = (20, 10))
mid_slice = n_fixed_vol.shape[2]//2
max_diff = np.max(np.abs(n_fixed_vol - Y_vol))
for (ax1, ax2), c_vol, c_label in zip(
    m_axs.T, 
    [Y_vol, n_fixed_vol, np.expand_dims(np.expand_dims(warp_vol, 0), -1)], 
    ['Input', 'Atlas', 'Warped Input']
):
    ax2.imshow(c_vol[0, :, mid_slice, :, 0] - n_fixed_vol[0, :, mid_slice, :, 0], 
               cmap = 'RdBu', vmin = -max_diff, vmax = max_diff)
    ax1.set_title(c_label)
    ax1.imshow(c_vol[0, :, mid_slice, :, 0], cmap = 'bone')
    ax2.set_title('${}-Ref$'.format(c_label))


# # Load Test Data
# The test data are what we try to register to the original image. In this case another brain image with the same resolution and dimensions. 

# In[42]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))
ax1.imshow(np.max(n_moving_vol[0, :, :, :, 0], 0), cmap = 'bone')
ax2.imshow(np.max(n_moving_vol[0, :, :, :, 0], 1), cmap = 'bone')
ax3.imshow(np.max(n_moving_vol[0, :, :, :, 0], 2), cmap = 'bone')


# In[43]:


get_ipython().run_cell_magic('time', '', 'pred = net.predict([n_moving_vol, n_fixed_vol])')


# # Show Displacement Maps

# In[44]:


# Warp segments with flow
flow = pred[1][0, :, :, :, :]
flow_sd = np.std(flow)
v_args = dict(cmap = 'RdBu', vmin = -flow_sd, vmax = +flow_sd)
fig, m_axs = plt.subplots(3, 3, figsize = (20, 10))
for i, (ax1, ax2, ax3) in enumerate(m_axs):
    ax1.imshow(np.mean(flow[:, :, :, i], 0), **v_args)
    ax1.set_title('xyz'[i]+' flow')
    ax2.imshow(np.mean(flow[:, :, :, i], 1), **v_args)
    ax3.imshow(np.mean(flow[:, :, :, i], 2), **v_args)


# ## Show Flow Field
# Here we show the flow field as a quiver map

# In[45]:


fig = plt.figure(figsize = (10, 10))
ax = fig.gca(projection='3d')

ax.quiver(c_xx,
          c_yy,
          c_zz,
          get_flow(0),
          get_flow(1), 
          get_flow(2), 
          length=0.9,
          normalize=True)


# In[46]:


sample = flow+grid
sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
warp_vol = interpn((yy, xx, zz), n_moving_vol[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)


# In[47]:


fig, m_axs = plt.subplots(5, 3, figsize = (10, 20))
mid_slice = n_moving_vol.shape[1]//2
diff_max = np.std(np.abs(n_moving_vol[0, mid_slice, :, :, 0]-n_fixed_vol[0, mid_slice, :, :, 0]))
for (ax1, ax2, ax3, ax4, ax5), c_vol, c_label in zip(
    m_axs.T, 
    [n_moving_vol, 
     n_fixed_vol, 
     np.expand_dims(np.expand_dims(warp_vol, 0), -1)], 
    ['Input', 'Ref', 'Warped Input']
):
    ax1.imshow(np.max(c_vol[0, :, :, :, 0], 0), cmap = 'bone')
    ax1.set_title(c_label)
    ax2.imshow(np.max(c_vol[0, :, :, :, 0], 1), cmap = 'bone')
    ax2.set_title(c_label)
    ax3.imshow(np.max(c_vol[0, :, :, :, 0], 2), cmap = 'bone')
    ax3.set_title(c_label)
    ax4.imshow(c_vol[0, mid_slice, :, :, 0], cmap = 'bone')
    ax4.set_title(c_label)
    
    ax5.imshow(c_vol[0, mid_slice, :, :, 0]-n_fixed_vol[0, mid_slice, :, :, 0], 
               cmap = 'RdBu', vmin = -diff_max, vmax = diff_max)
    ax5.set_title('${}-Ref$'.format(c_label))


# # Study the Warped Image
# Here we look at the warped image in more detail to see what was done. It looks like tissue was spread into the lung to make the more inflated lungs match better with the less inflated (before and after respectively)

# In[52]:


from skimage.util.montage import montage2d
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage2d(warp_vol[50:-50:4]), cmap = 'bone')


# In[ ]:




