#!/usr/bin/env python
# coding: utf-8

# # Overview
# The goal of the kernel is to compare the NMC datasets and in particular try to identify the structural changes that occur between the 0 and 2000bar images. We apply a number of basic techniques to the images to segment and quantify the structures inside. The 3D renderings are very low resolution representations of the data and serve to show the large scale differences and hint at the best kinds of quantitative metrics to extract. 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
from skimage.io import imread
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, ball
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_cdt # much faster than euclidean
from skimage.morphology import label
from skimage.feature import peak_local_max
from skimage.segmentation import mark_boundaries, watershed
from skimage.util.montage import montage2d
montage_pad = lambda x: montage2d(np.pad(x, [(0,0), (10, 10), (10, 10)], mode = 'constant', constant_values = 0))
import gc # since memory gets tight very quickly
gc.enable()
base_dir = os.path.join('..', 'input')


# In[2]:


all_tiffs = glob(os.path.join(base_dir, 'nmc*/*/grayscale/*'))
tiff_df = pd.DataFrame(dict(path = all_tiffs))
tiff_df['frame'] = tiff_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
tiff_df['experiment'] = tiff_df['frame'].map(lambda x: '_'.join(x.split('_')[0:-1]))
tiff_df['slice'] = tiff_df['frame'].map(lambda x: int(x.split('_')[-1]))
print('Images Found:', tiff_df.shape[0])
tiff_df.sample(3)


# In[6]:


from tqdm import tqdm_notebook
out_vols = {}
for c_group, c_df in tqdm_notebook(tiff_df.groupby('experiment'), desc = 'Experiment'):
    vol_stack = np.stack(c_df.sort_values('slice')['path'].map(lambda x: imread(x)[500:-500, 500:-500]).values, 0)
    out_vols[c_group] = vol_stack


# In[4]:


for k,v in out_vols.items():
    print(k, v.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(montage_pad(v[::10]), cmap = 'bone')
    ax1.set_title('Axial Slices - {}'.format(k))
    ax2.imshow(montage_pad(v.swapaxes(0,1)[::30]), cmap = 'bone')
    ax2.set_title('Sagittal Slices - {}'.format(k))
    ax2.set_aspect(4)
    fig.savefig('{}_slices.png'.format(k))


# # Filter and Downsample
# We can apply a filter to remove some of the noise and then downsample the image.

# In[7]:


from skimage.filters import gaussian
from scipy.ndimage import zoom
for k,v in tqdm_notebook(out_vols.items()):
    out_vols[k] = zoom(gaussian(v, 3), 0.5, order = 3)


# In[8]:


for k,v in out_vols.items():
    print(k, v.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(montage_pad(v[::10]), cmap = 'bone')
    ax1.set_title('Axial Slices - {}'.format(k))
    ax2.imshow(montage_pad(v.swapaxes(0,1)[::30]), cmap = 'bone')
    ax2.set_title('Sagittal Slices - {}'.format(k))
    ax2.set_aspect(4)
    fig.savefig('{}_slices.png'.format(k))


# In[9]:


get_ipython().run_cell_magic('time', '', 'out_segs = {}\nfor k,v in tqdm_notebook(out_vols.items()):\n    thresh_img = v > threshold_otsu(v)\n    bw_seg_img = closing(\n            opening(thresh_img, ball(2)),\n            ball(1)\n        )\n    out_segs[k] = bw_seg_img')


# In[11]:


for k,v in out_segs.items():
    print(k, v.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(montage_pad(v[::5]), cmap = 'bone')
    ax1.set_title('Axial Slices - {}'.format(k))
    ax2.imshow(montage_pad(v.swapaxes(0,1)[::15]), cmap = 'bone')
    ax2.set_title('Sagittal Slices - {}'.format(k))
    ax2.set_aspect(4)
    fig.savefig('{}_slices.png'.format(k))


# In[12]:


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
from scipy.ndimage import zoom
from skimage import measure
py.init_notebook_mode()


# In[13]:


for k,v in out_segs.items():
    smooth_pt_img = zoom(v[20:-20, 200:-200, 200:-200], (0.5, 0.25, 0.25), order = 3)
    print(k, smooth_pt_img.shape)
    verts, faces, _, _ = measure.marching_cubes_lewiner(
        smooth_pt_img, # you can make it bigger but the file-size gets HUUUEGE 
        smooth_pt_img.mean())
    x, y, z = zip(*verts)
    ff_fig = FF.create_trisurf(x=x, y=y, z=z,
                               simplices=faces,
                               title="Segmentation {}".format(k),
                               aspectratio=dict(x=1, y=1, z=1),
                               plot_edges=False)
    c_mesh = ff_fig['data'][0]
    c_mesh.update(lighting=dict(ambient=0.18,
                                diffuse=1,
                                fresnel=0.1,
                                specular=1,
                                roughness=0.1,
                                facenormalsepsilon=1e-6,
                                vertexnormalsepsilon=1e-12))
    c_mesh.update(flatshading=False)
    py.iplot(ff_fig)


# In[14]:


from skimage.io import imsave
for k,v in out_segs.items():
    imsave('{}_seg.tif'.format(k), v.astype(np.uint8))


# In[15]:


get_ipython().run_cell_magic('time', '', 'out_dm = {}\nfor k,v in out_segs.items():\n    out_dm[k] = distance_transform_cdt(v)')


# In[17]:


for k,v in out_dm.items():
    print(k, v.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(montage_pad(v[::5]), cmap = 'nipy_spectral')
    ax1.set_title('Axial Slices - {}'.format(k))
    ax2.imshow(montage_pad(v.swapaxes(0,1)[::20]), cmap = 'nipy_spectral')
    ax2.set_title('Sagittal Slices - {}'.format(k))
    ax2.set_aspect(4)
    fig.savefig('{}_slices.png'.format(k))


# In[18]:


from skimage.io import imsave
for k,v in out_dm.items():
    imsave('{}_dmap.tif'.format(k), v.astype(np.uint8))


# In[19]:


del out_dm
gc.collect() # force garbage collection


# In[20]:


get_ipython().run_cell_magic('time', '', 'out_label = {}\nfor k,v in out_segs.items():\n    out_label[k] = label(v)')


# In[36]:


from skimage.segmentation import mark_boundaries
for k,v in out_label.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    lab_mtg = montage_pad(v[10:-10:25]).astype(int)
    vol_mtg = montage_pad(out_vols[k][10:-10:25]).astype(np.float32)
    
    color_mtg = plt.cm.bone(np.clip(vol_mtg/vol_mtg.max(), 0,1))[:,:,:3]
    ax1.imshow(mark_boundaries(image = color_mtg, label_img = lab_mtg) , cmap = 'gist_earth')
    ax1.set_title('Axial Slices - {}'.format(k))
    lab_mtg = montage_pad(v.swapaxes(0,1)[10:-10:60]).astype(int)
    vol_mtg = montage_pad(out_vols[k].swapaxes(0,1)[10:-10:60]).astype(np.float32)
    color_mtg = plt.cm.autumn(np.clip(vol_mtg/vol_mtg.max(), 0,1))[:,:,:3]

    ax2.imshow(mark_boundaries(image = color_mtg, label_img = lab_mtg, color = (0,0,1)))
    ax2.set_title('Sagittal Slices - {}'.format(k))
    ax2.set_aspect(4)
    fig.savefig('{}_labels.png'.format(k))


# In[ ]:


from skimage.segmentation import mark_boundaries
middle_slice = lambda x: x[x.shape[0]//2]
for k,v in out_label.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    lab_mtg = middle_slice(v[10:-10:25]).astype(int)
    vol_mtg = middle_slice(out_vols[k][10:-10:25]).astype(np.float32)
    
    color_mtg = plt.cm.bone(np.clip(vol_mtg/vol_mtg.max(), 0,1))[:,:,:3]
    ax1.imshow(mark_boundaries(image = color_mtg, label_img = lab_mtg) , cmap = 'gist_earth')
    ax1.set_title('Axial Slice - {}'.format(k))
    lab_mtg = middle_slice(v.swapaxes(0,1)[10:-10:60]).astype(int)
    vol_mtg = middle_slice(out_vols[k].swapaxes(0,1)[10:-10:60]).astype(np.float32)
    color_mtg = plt.cm.autumn(np.clip(vol_mtg/vol_mtg.max(), 0,1))[:,:,:3]

    ax2.imshow(mark_boundaries(image = color_mtg, label_img = lab_mtg, color = (0,0,1)))
    ax2.set_title('Sagittal Slice - {}'.format(k))
    ax2.set_aspect(4)
    fig.savefig('{}_labels_mid.png'.format(k))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




