#!/usr/bin/env python
# coding: utf-8

# In this notebook I will be following this one: https://www.kaggle.com/robinkraft/getting-started-with-the-data-now-with-docs which is based on some basic image manipulation on Planet data.
# Original Notebook by Jesus Martinez Manso and Benjamin Goldenberg
# 
# (C) Planet 2017

# In[ ]:


import sys
import os
import subprocess

from six import string_types

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from skimage import io
from scipy import ndimage
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system(' ls -lrt ../input')


# ## Setup
#  Lets set up some global variables for file locations and ensure that they exist when we run this notebook
# 
# 

# In[ ]:


PLANET_KAGGLE_ROOT = os.path.abspath('../input/')
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')
assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)            


# ## Inspect Image Labels
# Image labels are in the CSV file called train_v2.csv

# In[ ]:


get_ipython().system('ls -lha /kaggle/input/train_v2.csv')


# In[ ]:


labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
labels_df.head()


# Some of these labels contain more than one feature. Let's create a table that tells us the features that each image has more clearly. First lets extract all the possible labels seen in the labels df into a list

# In[ ]:


label_list = []
for tag_str in labels_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)
label_list
#labels_df['tags']


# We will put the images into whats called a 'one hot' representation, where 1 represents if the image can be described by the label, with all labels as columns. We're essentially parsing the text of the csv file here.

# In[ ]:


for label in label_list:
    labels_df[label] =  labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0 )
  
labels_df.head()


# Now lets visualize the frequency of each label.

# In[ ]:


labels_df[label_list].sum().sort_values().plot.bar()


# In[ ]:


def make_cooccurrence_matrix(labels):
    numeric_df = labels_df[labels]
    c_matrix = numeric_df.T.dot(numeric_df)
    sns.heatmap(c_matrix)
    return c_matrix

make_cooccurrence_matrix(label_list)


# Check that each image has only one weather label.

# In[ ]:


weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
make_cooccurrence_matrix(weather_labels)


# However for land labels, we may have more than one label

# In[ ]:


land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']
make_cooccurrence_matrix(land_labels)


# There is little overlap in the rarer labels.

# In[ ]:


rare_labels = [l for l in label_list if labels_df[label_list].sum()[l] < 2000]
make_cooccurrence_matrix(rare_labels)


# ## Inspect images
# Now lets display and image and plot the pixel values.

# In[ ]:


def sample_images(tags, n=None):
    ''' Randomly sample n images with the specified tags.'''
    condition = True
    if isinstance(tags, string_types):
        raise ValueError('Pass a list of tags, not a single tag')
    for tag in tags:
        condition = condition & labels_df[tag] == 1
    if n is not None:
        return labels_df[condition].sample(n)
    else:
        return labels_df[condition]
    


# In[ ]:


def load_image(filename):
    '''Look through directory tree to find the image specified'''
    for dirname in os.listdir(PLANET_KAGGLE_ROOT):
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))
        if os.path.exists(path):
            print('Found image {}'.format(path))
            return io.imread(path)
        print('Load failed: could not find image {}'.format(path))
        
def sample_to_fname(sample_df, row_idx, suffix='tif'):
    '''Given a dataframe of sampled images, get the corresponding filename'''
    fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')
    return '{}.{}'.format(fname, suffix)


# In[ ]:


def plot_rgbn_hist(r, g, b, n):
    for slice_, name, color in ((r, 'r', 'red'), (g, 'g', 'green'), (b, 'b', 'blue'), (n, 'n', 'magenta')):
        plt.hist(slice_.ravel(), bins=100,
                range=[0,rgb_image.max()],
                label=name, color=color,histtype='step')
    plt.legend()    


# In[ ]:


s = sample_images(['primary', 'water', 'road'], n=1)
fname = sample_to_fname(s, 0)

bgrn_image = load_image(fname)

bgr_image = bgrn_image[:,:,:3]
rgb_image = bgr_image[:, :, [2,1,0]]

# Extract each band
# extract the different bands
b, g, r, nir = bgrn_image[:, :, 0], bgrn_image[:, :, 1], bgrn_image[:, :, 2], bgrn_image[:, :, 3]

# plot a histogram of rgbn values
plot_rgbn_hist(r, g, b, nir)


# In[ ]:


# Plot the bands
fig = plt.figure()
fig.set_size_inches(12, 4)
for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'), (nir, 'near-ir'))):
    a = fig.add_subplot(1, 4, i+1)
    a.set_title(c)
    plt.imshow(x)                       


# # The combined RGB image doesn't look right however...

# In[ ]:


plt.imshow(rgb_image)


# The image has not been colour calibrated and needs to be normalized. We need to use a reference colour curve from a image that has been normalized. Sometimes one would use a third party aerial image of a canopy, but in this case we will use JPEG images in the data set that have already been colour corrected.
# 
# The goal is to transform the (test) image such that its mean and variance match the reference image data. 

# In[ ]:


# Collect a list of 20000 image names
jpg_list = os.listdir(PLANET_KAGGLE_JPEG_DIR)[:20000]

# Pick 100 at random
np.random.shuffle(jpg_list)
jpg_list = jpg_list[:100]

print(jpg_list)


# Read each image (8 bit RGBA) and dump the pixels values to ref_colours, which has buckets for R, G, B

# In[ ]:


ref_colors = [[],[],[]]
for _file in jpg_list:
    #keep only rgb
    _img = mpimg.imread(os.path.join(PLANET_KAGGLE_JPEG_DIR, _file))[:,:,:3]
    #flatten the 2 dimensions to one
    _data = _img.reshape((-1,3))
    # Dump pixel values into correct buckets
    for i in range(3):
        ref_colors[i] = ref_colors[i] + _data[:,i].tolist()
    
ref_colors = np.array(ref_colors)    


# In[ ]:


for i, color in enumerate(['r','g','b']):
    plt.hist(ref_colors[i], bins=30, range=[0,255], label=color, color=color, histtype='step')
plt.legend()
plt.title('Reference colour histogram')
   


# The histogram above represents the response in RGB that we would like to use for our uncorrected image. Compare this histogram to the one plotted above using our function plot_r_g_b_n_hist. While the corrected image is 8 bit and the one above is 16 bit, you can see that the uncorrected image has a different response in each band, covering different pixel value domains. This explains why we see such extreme values in the RGB image above. The dynamic range is very large and does not allow us to view the subtleties in all 3 bands.

# In[ ]:


ref_means = [np.mean(ref_colors[i]) for i in range(3)]
ref_stds = [np.std(ref_colors[i]) for i in range(3)]


# In[ ]:




