#!/usr/bin/env python
# coding: utf-8

# # Overview
# The goal of the kernel is to compare the NMC datasets and in particular try to identify the structural changes that occur between the 0 and 2000bar images. We apply a number of basic techniques to the images to segment and quantify the structures inside. The 3D renderings are very low resolution representations of the data and serve to show the large scale differences and hint at the best kinds of quantitative metrics to extract. 

# In[ ]:


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
from skimage.util import montage as montage2d
from multiprocessing.pool import ThreadPool
from IPython.display import display
import h5py
from bokeh.io import output_notebook
from bokeh.resources import CDN
output_notebook(CDN, hide_banner=True)
montage_pad = lambda x: montage2d(np.pad(x, [(0,0), (10, 10), (10, 10)], mode = 'constant', constant_values = 0))
import gc # since memory gets tight very quickly
gc.enable()
base_dir = os.path.join('..', 'input')


# In[ ]:


all_tiffs = glob(os.path.join(base_dir, 'nmc*/*/grayscale/*'))
tiff_df = pd.DataFrame(dict(path = all_tiffs))
tiff_df['frame'] = tiff_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
tiff_df['experiment'] = tiff_df['frame'].map(lambda x: '_'.join(x.split('_')[0:-1]))
tiff_df['slice'] = tiff_df['frame'].map(lambda x: int(x.split('_')[-1]))
print('Images Found:', tiff_df.shape[0])
tiff_df.sample(3)


# In[ ]:


import skimage.io
import dask.array as da
import dask
import dask.diagnostics as diag

def dask_read_paths(filenames):
    dask_imread = dask.delayed(skimage.io.imread, pure=True)  # Lazy version of imread
    lazy_images = [dask_imread(url) for url in filenames]     # Lazily evaluate imread on each url
    sample = lazy_images[0].compute()
    arrays = [da.from_delayed(lazy_image,           # Construct a small Dask array
                              dtype=sample.dtype,   # for every lazy value
                              shape=sample.shape)
              for lazy_image in lazy_images]

    return da.stack(arrays, axis=0)                # Stack all small Dask arrays into one


# In[ ]:


from tqdm import tqdm_notebook
out_vols = {}
for c_group, c_df in tiff_df.groupby('experiment'):
    vol_stack = dask_read_paths(c_df.sort_values('slice')['path'])
    out_vols[c_group] = vol_stack.rechunk((15, 2048, 2048))
    print(c_group)
    display(out_vols[c_group])


# In[ ]:


for k,v in out_vols.items():
    print(k, v.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(montage_pad(v[::10].compute(num_workers=4)), cmap = 'bone')
    ax1.set_title('Axial Slices - {}'.format(k))
    ax2.imshow(montage_pad(v[:, ::30].swapaxes(0,1).compute(num_workers=4)), cmap = 'bone')
    ax2.set_title('Sagittal Slices - {}'.format(k))
    ax2.set_aspect(4)
    fig.savefig('{}_slices.png'.format(k))


# # Filter and Downsample
# We can apply a filter to remove some of the noise and then downsample the image.

# In[ ]:


from skimage.filters import gaussian
def dask_gauss(x, sigma_tuple):
    chunksize = tuple(c[0] for c in v.chunks)
    out_sigma = tuple(max(1, int(2*sigma)) for sigma in sigma_tuple)
    if any([a<b for a,b in zip(chunksize, out_sigma)]):
        new_chunksize = [max(a,b) for a,b in zip(chunksize, out_sigma)]
        print('Rechunking:', new_chunksize)
        x = x.rechunk(new_chunksize)
        
    return x.map_overlap(lambda y: gaussian(y, sigma_tuple).astype(np.float32), depth = out_sigma, boundary = 'reflect')
for k,v in tqdm_notebook(out_vols.items()):
    out_vols[k] = dask_gauss(v, (1.0, 3.0, 3.0))[:, 500:-500:2, 500:-500:2]
    print(k)
    display(out_vols[k])


# In[ ]:


out_vols['NMC_90wt_0bar'][50].visualize(optimize_graph = True)


# In[ ]:


with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof:
    for k,v in out_vols.items():
        print(k, v.shape)
        with dask.config.set(pool = ThreadPool(4)):
            v.to_hdf5('{}_filtered.h5'.format(k), '/images', compression = 'lzf')
get_ipython().system('ls -lh *.h5')


# In[ ]:


diag.visualize([prof, rprof])


# In[ ]:


c_keys = list(out_vols.keys())
for k in c_keys:
    v = h5py.File('{}_filtered.h5'.format(k), 'r')['images']
    print(k,v)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(montage_pad(v[::10]), cmap = 'bone')
    ax1.set_title('Axial Slices - {}'.format(k))
    ax2.imshow(montage_pad(v[:, ::30].swapaxes(0,1)), cmap = 'bone')
    ax2.set_title('Sagittal Slices - {}'.format(k))
    ax2.set_aspect(4)
    fig.savefig('{}_filtered.png'.format(k))
    # update the dask array to the checkpoint
    out_vols[k] = da.from_array(v, chunks = (50, 150, 150))


# In[ ]:


get_ipython().run_cell_magic('time', '', "out_segs = {}\ndef dask_segment(in_vol):\n    def bw_seg_func(in_block):\n        try:\n            thresh_img = in_block > threshold_otsu(in_block)\n        except:\n            thresh_img = in_block > 0\n        clean_img = closing(\n                opening(thresh_img, ball(4)),\n                ball(1)\n            )\n        return distance_transform_cdt(clean_img).astype(np.uint16)\n    return in_vol.rechunk((20, 100, 100)).map_overlap(bw_seg_func, depth = (5, 10, 10), boundary = 'reflect')\nfor k,v in tqdm_notebook(out_vols.items()):\n    \n    out_segs[k] = dask_segment(v)\n    print(k)\n    display(out_segs[k])")


# In[ ]:


with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof:
    for k,v in out_segs.items():
        print(k, v.shape)
        with dask.config.set(pool = ThreadPool(4)):
            v.to_hdf5('{}_segmented.h5'.format(k), '/images', compression = 'lzf')
        # dont need to delete if we use fp16 and uint16 
        # os.remove('{}_filtered.h5'.format(k))
get_ipython().system('ls -lh *.h5')


# In[ ]:


diag.visualize([prof, rprof])


# In[ ]:


c_keys = list(out_segs.keys())
for k in c_keys:
    v = h5py.File('{}_segmented.h5'.format(k), 'r')['images']
    print(k,v)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(montage_pad(v[::10]), cmap = 'nipy_spectral')
    ax1.set_title('Axial Slices - {}'.format(k))
    ax2.imshow(montage_pad(v[:, ::30].swapaxes(0,1)), cmap = 'nipy_spectral')
    ax2.set_title('Sagittal Slices - {}'.format(k))
    ax2.set_aspect(4)
    fig.savefig('{}_dmap.png'.format(k))
    # update the dask array to the checkpoint
    out_segs[k] = da.from_array(v, chunks = (50, 200, 200))


# # Questions
# 1. Why might labeling be a more challenging problem? How is it different from filtering and distance maps?
# 1. What effect does chunk size have on the analysis? How can this best be chosen?
# 1. In order to do the analysis with multiple nodes what needs to be taken into account?
# 
# # Challenges
# 1. Adapt the code to perform watershed analysis to segment the bubbles
