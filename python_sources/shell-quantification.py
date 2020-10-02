#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook shows how to
# - take an image of nanoparticles
# - segment the individual particles
# - label each particle
# - measure the signed distance from the surface (negative is inside, positive is outside)
# - show the image intensity at each point to quantify the shell thickness

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from IPython.display import Image, display, SVG, clear_output, HTML
plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'gray' # grayscale looks better


# # Load and Preprocess
# Here we load the images, perform some filtering and morphological operations to identify the individual particles

# In[ ]:


import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.filters import threshold_isodata, median
from skimage.morphology import opening, disk


# In[ ]:


image_list = list(Path('../input').glob('**/*.tif'))
fig, m_axs = plt.subplots(len(image_list), 3, figsize=(16, 8))
image_dict = {}
for (c_ax, h_ax, d_ax), c_img_path in zip(m_axs, image_list):
    c_img = imread(c_img_path)
    c_ax.imshow(c_img)
    c_ax.set_title('Raw Image')
    c_ax.axis('off')
    
    d_img = median(c_img)
    d_val = threshold_isodata(d_img)
    image_dict[c_img_path] = {'image': d_img, 'mask': opening(d_img>d_val, disk(4))}
    h_ax.hist(c_img.ravel())
    h_ax.axvline(d_val, color='k')
    h_ax.set_title('Distribution')
    
    d_ax.imshow(image_dict[c_img_path]['mask'])
    d_ax.axis('off')


# ## Use distance maps to get the geometric shells
# - Positive distances are outside the particle
# - Negative distances are inside the particle

# In[ ]:


from scipy.ndimage import distance_transform_edt as distmap
from scipy.ndimage import label


# In[ ]:


fig, m_axs = plt.subplots(len(image_list), 4, figsize=(16, 8))
for (c_ax, d_ax, f_ax, g_ax), c_img_path in zip(m_axs, image_list):
    c_img = image_dict[c_img_path]['image']
    c_mask = image_dict[c_img_path]['mask']
    label_img, _ = label(c_mask)
    c_ax.imshow(label_img, cmap=plt.cm.nipy_spectral)
    c_ax.set_title('Labeled Particles')
    
    c_map, (x_idx, y_idx) = distmap(~c_mask, return_indices=True, return_distances=True)
    rev_c_map = distmap(c_mask, return_indices=False, return_distances=True)
    c_map[c_mask>0] = -1*rev_c_map[c_mask>0] # negative values on the inside
    grow_labels = label_img[x_idx.ravel(), y_idx.ravel()].reshape(label_img.shape)
    d_ax.imshow(grow_labels, cmap=plt.cm.nipy_spectral)
    d_ax.set_title('Territories for each particle')
    shell_region = grow_labels*(c_map<5)*(c_map>0)
    f_ax.imshow(shell_region, cmap=plt.cm.nipy_spectral)
    f_ax.set_title('5 pixel shell')
    
    bound_d_map = c_map.copy()
    bound_d_map[np.abs(bound_d_map)>10] = np.NAN    
    g_ax.imshow(bound_d_map, cmap='RdBu')
    g_ax.set_title('Distance Map')
    
    image_dict[c_img_path]['labels'] = label_img
    image_dict[c_img_path]['dist'] = c_map
    image_dict[c_img_path]['label_regions'] = grow_labels


# In[ ]:


fig, m_axs = plt.subplots(1, len(image_list), figsize=(16, 8))
df_list = []
for (c_ax), c_img_path in zip(m_axs, image_list):
    c_img = image_dict[c_img_path]['image']
    label_img = image_dict[c_img_path]['labels']
    c_dist = image_dict[c_img_path]['dist']
    grow_labels = image_dict[c_img_path]['label_regions']
    for k in np.unique(label_img[label_img>0]):
        x_val = c_dist[grow_labels==k]
        y_val = c_img[grow_labels==k]
        df_list.append(pd.DataFrame({'distance': x_val, 'intensity': y_val, 'particle': k, 'image': c_img_path}))
        c_ax.plot(x_val, y_val, '.', label=str(k), ms=0.2)
        c_ax.set_xlim(-5, 20)
    c_ax.set_xlabel('Distance from Particle Surface')
    c_ax.set_ylabel('Intensity')


# In[ ]:


all_df = pd.concat(df_list)
all_df.to_csv('intensity_decay_curve.csv', index=False)
all_df.sample(3)


# In[ ]:


sns.catplot(x='distance',
            y='intensity',
            hue='particle',
            col='image',
            kind='swarm',
            data=all_df.query('distance>-5').query('distance<5').query('particle==1'))


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
for c_grp, c_rows in all_df.        groupby(['image', 
                 'particle', 
                 (all_df['distance']*2).astype(int).rename('rounded_dist')]).\
        agg('mean').\
        reset_index().\
        query('abs(distance)<20').\
        groupby(['image', 'particle']):
        ax1.plot(c_rows['distance'], c_rows['intensity'], label=str(c_grp))
        ax1.set_title('Raw Values')
        ax2.plot(c_rows['distance'], c_rows['intensity']/c_rows['intensity'].max(), label=str(c_grp))
        ax2.set_title('Normalized')


# # Show Average Curves

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
for c_grp, c_rows in all_df.        groupby(['image', 
                 (all_df['distance']*1).astype(int).rename('rounded_dist')]).\
        agg('mean').\
        reset_index().\
        query('abs(distance)<40').\
        groupby(['image']):
        ax1.plot(c_rows['distance'], c_rows['intensity'], '.-', label=str(c_grp))
        ax1.set_title('Raw Values')
        ax2.plot(c_rows['distance'], c_rows['intensity']/c_rows['intensity'].max(), '.-', label=str(c_grp))
        ax2.set_title('Normalized')
        ax1.legend()
        ax1.axvline(0, color='k')
        ax2.legend()
        ax2.axvline(0, color='k')


# # Cluster Curves
# Grab the curve for each particle and cluster them into groups

# In[ ]:


from scipy.interpolate import interp1d
dist_vals = np.linspace(0, 20, 10)
out_rows = []
for (c_path, c_idx), raw_rows in all_df.groupby(['image', 'particle']):
    # average duplicates and sort
    c_rows = raw_rows.groupby('distance').agg('mean').reset_index().sort_values('distance')
    i_func = interp1d(x=c_rows['distance'].values, 
                      y=c_rows['intensity'].values, 
                      bounds_error=False, 
                      fill_value='extrapolate', 
                      kind='cubic', 
                      assume_sorted=True)
    out_rows.append({'image': c_path, 'particle': c_idx, 'distance': dist_vals, 'intensity': i_func(dist_vals)})
std_df = pd.DataFrame(out_rows)
std_df.sample(2)


# In[ ]:


plt.plot(np.stack(std_df['intensity'], 1));


# In[ ]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters=4)
std_df['cluster'] = np.argmin(km.fit_transform(np.stack(std_df['intensity'], 0)), -1)+1
std_df['cluster'].value_counts()


# ### Show clusters on original images
# We mainly see that one of the groups has a significantly reduced decay from the others

# In[ ]:


from skimage.color import label2rgb
fig, m_axs = plt.subplots(1, len(image_list), figsize=(16, 8))
for (c_ax), c_img_path in zip(m_axs, image_list):
    raw_image = image_dict[c_img_path]['image']
    raw_labels = image_dict[c_img_path]['labels']
    clusters = np.zeros_like(raw_labels)
    for _, c_row in std_df[std_df['image']==c_img_path][['particle', 'cluster']].iterrows():
        clusters[raw_labels==c_row['particle']] = c_row['cluster']
    norm_img = np.clip(raw_image/(raw_image.mean()+2*raw_image.std()), 0, 1)
    n_img = label2rgb(image=norm_img, label=clusters, bg_label=0)
    c_ax.imshow(n_img)


# In[ ]:





# In[ ]:




