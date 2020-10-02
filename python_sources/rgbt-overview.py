#!/usr/bin/env python
# coding: utf-8

# # Overview
# A number of images taken from a fixed RGBD camera recording a scene

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from skimage.io import imread
from glob import glob
import yaml
import ipywidgets as ipw
from mpl_toolkits.mplot3d import Axes3D
try:
    from skimage.util.montage import montage2d
except ImportError:
    from skimage.util import montage as montage2d
base_dir = '../input/hallway_rgbds'


# ## YAML Data
# ### Calibration Format
# 
# Available in `calibration.yaml` or `calibration.mat` files, with the
# following fields:
# 
# * K - camera matrix
# * Pwc - world-to-camera matrix
# * Pcw - camera-to-world matrix
# * dist - distortion coefficients

# In[ ]:


all_yaml = {'/'.join(p.split('/')[-2:]): yaml.load(open(p, 'r')) 
            for p in glob(os.path.join(base_dir, '*','*.yaml'))}


# In[ ]:


all_images_df = pd.DataFrame({'path': glob(os.path.join(base_dir, '*','*', '*.png'))})


# In[ ]:


all_images_df['file_id'] = all_images_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
all_images_df['file_prefix'] = all_images_df['file_id'].map(lambda x: ''.join([c for c in x if c.isalpha()]))
all_images_df['file_idx'] = all_images_df['file_id'].map(lambda x: ''.join([c for c in x if c.isnumeric()]))
all_images_df['experiment'] = all_images_df['path'].map(lambda x: x.split('/')[-2])
all_images_df['series'] = all_images_df['path'].map(lambda x: x.split('/')[-3])
all_images_df.sample(3)


# In[ ]:


image_pairs_df = all_images_df.pivot_table(values='path', 
                          index=['series', 'experiment', 'file_idx'], 
                          columns='file_prefix', 
                          aggfunc='first').reset_index()
image_pairs_df.sample(3)


# In[ ]:


fig, m_axs = plt.subplots(3, 2, figsize = (20, 10))
for (ax1, ax2), (_, i_row) in zip(m_axs, 
                                  image_pairs_df.sample(len(m_axs)).iterrows()):
    ax1.imshow(imread(i_row['rgb']))
    ax1.set_title('RGB')
    ax2.imshow(imread(i_row['depth']))
    ax2.set_title('Depth Map')


# In[ ]:


exp_list = list(image_pairs_df.groupby(['series', 'experiment']))
print(len(exp_list), 'experiments')


# In[ ]:


(series, exp), t_rows = exp_list[-1]
print((series, exp))
t_rows = t_rows.copy()
print(t_rows.shape[0], 'rows to process')
t_rows['rgb'] = t_rows['rgb'].map(imread)
t_rows['depth'] = t_rows['depth'].map(imread)
all_depth = np.stack(t_rows['depth'].values)


# In[ ]:


@ipw.interact()
def show_scene_figure(index=(0, t_rows.shape[0])):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
    ax1.imshow(t_rows['rgb'].iloc[index])
    ax2.imshow(t_rows['depth'].iloc[index])


# In[ ]:


fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 15))
vmin, vmax = np.min(all_depth), np.max(all_depth)
ax1.imshow(np.mean(all_depth, 0), vmin=vmin, vmax=vmax)
ax1.set_title('Average')
ax2.imshow(np.median(all_depth, 0), vmin=vmin, vmax=vmax)
ax2.set_title('Median')
ax3.imshow(np.std(all_depth, 0), vmin=vmin, vmax=vmax)
ax3.set_title('Std')
ax4.imshow(np.min(all_depth, 0), vmin=vmin, vmax=vmax)
ax4.set_title('Min')
ax5.imshow(np.max(all_depth, 0), vmin=vmin, vmax=vmax)
ax5.set_title('Max')
ax6.imshow(np.max(all_depth, 0)-np.min(all_depth, 0), vmin=0, vmax=vmax-vmin)
ax6.set_title('Range')


# In[ ]:


na_depth = all_depth.astype('float32')
na_depth[na_depth==0] = np.NAN
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 15))
vmin, vmax = np.min(all_depth), np.max(all_depth)
ax1.imshow(np.nanmean(na_depth, 0), vmin=vmin, vmax=vmax)
ax1.set_title('Average')
ax2.imshow(np.nanmedian(na_depth, 0), vmin=vmin, vmax=vmax)
ax2.set_title('Median')
ax3.imshow(np.nanstd(na_depth, 0), vmin=vmin, vmax=vmax)
ax3.set_title('Std')
ax4.imshow(np.nanmin(na_depth, 0), vmin=vmin, vmax=vmax)
ax4.set_title('Min')
ax5.imshow(np.nanmax(na_depth, 0), vmin=vmin, vmax=vmax)
ax5.set_title('Max')
ax6.imshow(np.nanmax(na_depth, 0)-np.nanmin(na_depth, 0), vmin=0, vmax=vmax-vmin)
ax6.set_title('Range')


# # Convert RGBD -> Volume
# Here we convert the RGBD data into a Volume for each time step. The volume will then make it easier to see the regions which change the most

# In[ ]:


(series, exp)
cur_calib_dict = all_yaml['{}/calibration.yaml'.format(series)]
K = np.array(cur_calib_dict['K'])
Pcw = np.array(cur_calib_dict['Pcw'])


# In[ ]:


plt.hist(all_depth[all_depth>0])


# In[ ]:


# hacky point cloud reconstruction using some TUM code
# https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/generate_pointcloud.py
# TODO: replace hard coded focal length and scaling factor with values from Pcw
focalLength = 525.0
centerX = K[0,2]
centerY = K[1,2]
scalingFactor = 5000.0
def slice_to_cloud(in_depth):
    xx, yy = np.meshgrid(range(in_depth.shape[1]), range(in_depth.shape[0]), indexing='xy')
    Z = in_depth.astype('float32') / scalingFactor
    X = (xx - centerX) * Z / focalLength
    Y = (yy - centerY) * Z / focalLength
    return X.ravel(), Y.ravel(), Z.ravel()
def slice_to_dfcloud(in_rgb, in_depth):
    X, Y, Z = slice_to_cloud(in_depth[::-1])
    pc_df = pd.DataFrame({'x': X, 'y': Y, 'z': Z})
    for i,k in enumerate('rgb'):
        pc_df[k] = in_rgb[::-1, :, i].ravel()
    return pc_df.query('z>0')


# In[ ]:


show_scene_figure(0)


# In[ ]:


test_df = slice_to_dfcloud(t_rows['rgb'].iloc[0], 
                           t_rows['depth'].iloc[0]).sample(100000)
fig, m_axs = plt.subplots(1, 3, figsize = (20, 5))
ax_names = 'xyz'
for i, c_ax in enumerate(m_axs.flatten()):
    plot_axes = [x for j, x in enumerate(ax_names) if j!=i]
    c_ax.scatter(test_df[plot_axes[0]],
                test_df[plot_axes[1]],
                c=test_df[['r', 'g', 'b']].values/255, 
                 s=1
                )
    c_ax.set_xlabel(plot_axes[0])
    c_ax.set_ylabel(plot_axes[1])


# In[ ]:


fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')
ax.scatter(test_df['z'], test_df['x'], test_df['y'],
            c=test_df[['r', 'g', 'b']].values/255, s=3)  
ax.view_init(15, -45)


# ## Create a Volume
# We make a low resolution volume and just keep track of the occupancy

# In[ ]:


import sys
from scipy.spatial import KDTree
sys.setrecursionlimit(10000) # kdtree gets hungry (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html)


# In[ ]:


x_sig = 2
y_sig = 2
z_sig = 2
x_steps = 100
y_steps = 40
z_steps = 100
bbox = {}
for c, sigma in zip('xyz', [x_sig,y_sig,z_sig]):
    ax_mean, ax_std = test_df[c].mean(), test_df[c].std()
    bbox[c] = (ax_mean-sigma*ax_std, ax_mean+sigma*ax_std)
xx, yy, zz = np.meshgrid(np.linspace(*bbox['x'], x_steps),
                         np.linspace(*bbox['y'], y_steps),
                         np.linspace(*bbox['z'], z_steps),
                         indexing='ij'
                        )
print(xx.shape)
dx = np.diff(xx[0:2, 0, 0])[0]
dy = np.diff(yy[0, 0:2, 0])[0]
dz = np.diff(zz[0, 0, 0:2])[0]
dr = np.sqrt(dx**2+dy**2+dz**2)
print(dx, dy, dz, dr)


# In[ ]:


test_df = slice_to_dfcloud(t_rows['rgb'].iloc[0], 
                           t_rows['depth'].iloc[0])


# ## Fast Lookup
# Here we transform the test_df coordinates into indices and then set the appropriate indices.

# In[ ]:


get_ipython().run_cell_magic('time', '', "for c_ax, c_xx, c_dx, c_steps in zip('xyz', \n                            [xx, yy, zz], \n                            [dx, dy, dz], \n                            [x_steps, y_steps, z_steps]):\n    test_df['i{}'.format(c_ax)] = (test_df[c_ax]-c_xx.min())/c_dx\n    test_df['i{}'.format(c_ax)] = test_df['i{}'.format(c_ax)].map(lambda x: x if (x>0) and (x<c_steps) else np.NAN)\ntest_idx = test_df[['ix', 'iy', 'iz']].dropna().values.astype(int)\nprint('Valid Points: {}/{}'.format(test_idx.shape[0], test_df.shape[0]))\nout_vol = np.zeros_like(xx)\nout_vol[test_idx[:, 0], test_idx[:, 1], test_idx[:, 2]]+=1")


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))
ax1.imshow(np.sum(out_vol, 0))
ax2.imshow(np.sum(out_vol, 1))
ax3.imshow(np.sum(out_vol, 2))


# ## KDTree Lookup 
# Comparing thousands of points thousands of times is very very inefficient without binary search trees

# In[ ]:


get_ipython().run_cell_magic('time', '', "test_kdtree = KDTree(test_df[['x', 'y', 'z']])")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'out_dist, _ = test_kdtree.query(np.stack([xx, yy, zz], -1), k=1, distance_upper_bound = 1.1*dr)\ndist_vol = np.isinf(out_dist).reshape(xx.shape)==False')


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))
ax1.imshow(np.sum(dist_vol, 0))
ax2.imshow(np.sum(dist_vol, 1))
ax3.imshow(np.sum(dist_vol, 2))


# ## Distance Maps could be more useful
# Here we make a distance map to the nearest point rather than a binary map

# In[ ]:


get_ipython().run_cell_magic('time', '', 'out_dist, _ = test_kdtree.query(np.stack([xx, yy, zz], -1), k=1)\ndist_vol = (out_dist<1.5*dr).reshape(xx.shape)==False')


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))
dist_vol = np.log2(dr/out_dist.reshape(xx.shape))
ax1.imshow(np.sum(dist_vol, 0))
ax2.imshow(np.sum(dist_vol, 1))
ax3.imshow(np.sum(dist_vol, 2))


# In[ ]:


fig, (ax1) = plt.subplots(1, 1, figsize = (15, 15))
ax1.imshow(montage2d(dist_vol.swapaxes(0,1)))


# ## Simple Marching Cubes Isosurface

# In[ ]:


plt.hist(dist_vol.ravel(), 50);


# In[ ]:


fig, (ax1) = plt.subplots(1, 1, figsize = (15, 15))
ax1.imshow(montage2d(dist_vol.swapaxes(0,1)>0.5))


# In[ ]:


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
verts, faces, normals, values = measure.marching_cubes_lewiner(dist_vol, 1.0, spacing=(dx, dy, dz))


# In[ ]:


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 2], verts[:, 0], faces, verts[:, 1],
                cmap=plt.cm.Greens, lw=0.05, edgecolor='k')
ax.view_init(15, -90)


# # Generate all timesteps
# Here we process the rest of the time steps

# In[ ]:


from tqdm import tqdm
import h5py
with h5py.File('time_steps.h5', 'w') as f:
    time_ds = f.create_dataset('volume_time', 
                               shape=(t_rows.shape[0],)+xx.shape,
                               chunks=(1,)+xx.shape, 
                               dtype='int', 
                               compression='gzip')
    for i, (_, c_row) in tqdm(enumerate(t_rows.iterrows())):
        test_df = slice_to_dfcloud(c_row['rgb'], 
                                   c_row['depth'])
        for c_ax, c_xx, c_dx, c_steps in zip('xyz', 
                                    [xx, yy, zz], 
                                    [dx, dy, dz], 
                                    [x_steps, y_steps, z_steps]):
            test_df['i{}'.format(c_ax)] = (test_df[c_ax]-c_xx.min())/c_dx
            test_df['i{}'.format(c_ax)] = test_df['i{}'.format(c_ax)].map(lambda x: x if (x>0) and (x<c_steps) else np.NAN)
        test_idx = test_df[['ix', 'iy', 'iz']].dropna().values.astype(int)
        out_vol = np.zeros_like(xx)
        out_vol[test_idx[:, 0], test_idx[:, 1], test_idx[:, 2]]+=1
        time_ds[i] = out_vol


# In[ ]:


get_ipython().system('ls -lh *.h5')


# In[ ]:




