#!/usr/bin/env python
# coding: utf-8

# # Overview
# In this notebook we load the data and view different images to get a better idea about the challenge we are facing. This is always a very helpful first step. It is also important that you can see and try to make some of your own predictions about the data. If you cannot see differences between the groups it is going to be difficult for a biomarker to capture that (but not necessarily impossible)

# In[ ]:


import numpy as np # for manipulating 3d images
import pandas as pd # for reading and writing tables
import h5py # for reading the image files
import skimage # for image processing and visualizations
import sklearn # for machine learning and statistical models
import os # help us load files and deal with paths


# ### Plot Setup Code
# Here we setup the defaults to make the plots look a bit nicer for the notebook

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})


# # Load the Training Data
# We start with the training data since we have labels for them and can look in more detail

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head(5) # show the first 5 lines


# In[ ]:


train_df['age_years'].hist(bins=10) # make a histogram of the ages of patients


# We see two groups form in this list young ($<60$) and old ($>60$) and very few in-between. This should make our task a little bit easier since the difference between young and old is probably bigger than between 30 and 35.

# In[ ]:


train_df['age_group'] = train_df['age_years'].map(lambda age: 'old' if age>60 else 'young') 
train_df['age_group'].value_counts() # show how many of each we have


# # Load a Scan
# - the data on kaggle are located in a parent folder called input. 
# - Since the files have been organized into train and test we use the train folder

# In[ ]:


sample_scan = train_df.iloc[0] # just take the first row
print(sample_scan)
# turn the h5_path into the full path
full_scan_path = os.path.join('..', 'input','train', sample_scan['h5_path'])
# load the image using hdf5
with h5py.File(full_scan_path, 'r') as h:
    image_data = h['image'][:][:, :, :, 0] # we read the data from the file
print(image_data.shape, 'loaded')


# # Visualize the Data
# ## Middle Slice

# In[ ]:


# show the middle slice
plt.imshow(image_data[image_data.shape[0]//2, :, :])


# ## Show the middle slice in x, y and z
# We can also change the colormap to gray to make the images look more like MRIs

# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image_data[image_data.shape[0]//2, :, :], cmap='gray')
ax2.imshow(image_data[:, image_data.shape[1]//2, :], cmap='gray')
ax3.imshow(image_data[:, :, image_data.shape[2]//2], cmap='gray')


# ## Show all of the slices
# We can use the `montage` function from skimage to show all the slices

# In[ ]:


brain_montage = skimage.util.montage(image_data)
plt.imshow(brain_montage)


# # Compare Young and Old

# In[ ]:


fig, m_axs = plt.subplots(2, 3, figsize=(15, 10))
for (group_name, c_rows), (ax1, ax2, ax3) in zip(train_df.groupby('age_group'), m_axs):
    full_scan_path = os.path.join('..', 'input','train', c_rows['h5_path'].iloc[0])
    with h5py.File(full_scan_path, 'r') as h:
        cur_image_data = h['image'][:][:, :, :, 0] # we read the data from the file
    ax1.imshow(cur_image_data[cur_image_data.shape[0]//2, :, :], cmap='gray')
    ax1.set_title('{} scan'.format(group_name))
    ax2.imshow(cur_image_data[:, cur_image_data.shape[1]//2, :], cmap='gray')
    ax3.imshow(cur_image_data[:, :, cur_image_data.shape[2]//2], cmap='gray')


# In[ ]:




