#!/usr/bin/env python
# coding: utf-8

# **This is based on the official [notebook](https://www.kaggle.com/robinkraft/getting-started-with-the-data-now-with-docs) from the competition `Planet: Understanding the Amazon from Space`, adjusted to the paths in this subset of the main dataset. **
# 
# # *Planet: Understanding the Amazon from Space* challenge
# 
# This notebook will show you how to do some basic manipulation of the images and label files.

# In[ ]:


import sys
import os
import random
import subprocess
from tqdm import tqdm

from six import string_types

# Make sure you have all of these packages installed, e.g. via pip
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px
import scipy
from skimage import io
from scipy import ndimage
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('ls -lha ../input/planets-dataset/planet/planet')


# ## Setup
# Set `PLANET_KAGGLE_ROOT` to the proper directory where we've got the TIFF and JPEG zip files, and accompanying CSVs.

# In[ ]:


PLANET_KAGGLE_ROOT = os.path.abspath("../input/planets-dataset/planet/planet")
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_classes.csv')
assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)


# ## Inspect image labels
# The labels are in a CSV entitled `train.csv`. Note that each image can be tagged with multiple tags. We'll convert them to a "one hot" style representation where each label is a column:

# In[ ]:


labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
labels_df.head()


# In[ ]:


# Build list with unique labels
label_list = []
for tag_str in labels_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)


# In[ ]:


# Add onehot features for every label
for label in label_list:
    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
# Display head
labels_df.head()


# In[ ]:


# Histogram of label instances
labels_df[label_list].sum().sort_values().plot.bar()


# In[ ]:


def make_cooccurence_matrix(labels):
    numeric_df = labels_df[labels]; 
    c_matrix = numeric_df.T.dot(numeric_df)
    sns.heatmap(c_matrix, cmap ="Blues")
    return c_matrix
    
# Compute the co-ocurrence matrix
make_cooccurence_matrix(label_list)


# Each image should have exactly one weather label:

# In[ ]:


weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
make_cooccurence_matrix(weather_labels)


# But the land labels may overlap:

# In[ ]:


land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']
make_cooccurence_matrix(land_labels)


# The rarer labels have very little overlap:

# In[ ]:


rare_labels = [l for l in label_list if labels_df[label_list].sum()[l] < 2000]
make_cooccurence_matrix(rare_labels)


# ## Labels vs weather 
# **Cloudy** label has no other labels.

# In[ ]:


for w in weather_labels:
    df_weather_subset = labels_df.loc[labels_df[w] == 1, label_list].drop([w], axis=1)
    weather_percent_subset = df_weather_subset.sum(axis =0) / df_weather_subset.shape[0]
    weather_percent_subset = weather_percent_subset[weather_percent_subset >0].sort_values(ascending=False)
    fig = px.bar(x=weather_percent_subset.index, y=weather_percent_subset.values,  
                 labels={'x':'label', 'y':f'Another labels given {w} label'})
    fig.update_layout(title_text=f"Main label: {w}", yaxis_tickformat=',.0%')
    fig.show()


# ## Inspect images
# Let's display an image and visualize the pixel values. Here we will pick an image, load every single single band, then create RGB stack. These raw images are 16-bit (from 0 to 65535), and contain red, green, blue, and [Near infrared (NIR)](https://en.wikipedia.org/wiki/Infrared#Regions_within_the_infrared) channels. In this example, we are discarding the NIR band just to simplify the steps to visualize the image. However, you should probably keep it for ML classification.
# 
# The files can be easily read into numpy arrays with the skimage.

# In[ ]:


def sample_images(tags, n=None):
    """Randomly sample n images with the specified tags."""
    condition = True
    if isinstance(tags, string_types):
        raise ValueError("Pass a list of tags, not a single tag.")
    for tag in tags:
        condition = condition & labels_df[tag] == 1
    if n is not None:
        return labels_df[condition].sample(n)
    else:
        return labels_df[condition]
    

def plot_rgbn_histo(r, g, b):
    for slice_, name, color in ((r,'r', 'red'),(g,'g', 'green'),(b,'b', 'blue')):
        plt.hist(slice_.ravel(), bins=100, 
                 range=[0,rgb_image.max()], 
                 label=name, color=color, histtype='step')
    plt.legend()


# In[ ]:


def load_image(filename):
    '''Look through the directory tree to find the image you specified
    (e.g. train_10.tif vs. train_10.jpg)'''
    for dirname in os.listdir(PLANET_KAGGLE_ROOT):
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))
        if os.path.exists(path):
            #print('Found image {}'.format(path))
            return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))


# Let's look at an individual image. First, we'll plot a histogram of pixel values in each channel. Note how the intensities are distributed in a relatively narrow region of the dynamic range

# In[ ]:


def get_rgb_image(labels=['primary', 'water', 'road'], n_samples=1):
    s = sample_images(labels, n=n_samples)
    fnames = s.loc[:, "image_name"].apply(lambda fname: '{}.{}'.format(fname, "jpg"))
    rgb_images = []
    for name in fnames:
    # find the image in the data directory and load it
        bgr_image = load_image(name)
        rgb_image = bgr_image[:, :, [2,1,0]]
        rgb_images.append(rgb_image)
    return np.array(rgb_images)


def get_r_g_b_channels(rgb_image):
    b, g, r = rgb_image[:, :, 2], rgb_image[:, :, 1], rgb_image[:, :, 0]
    return r, g, b


# In[ ]:


rgb_images= get_rgb_image(labels=['primary', 'water', 'road'], n_samples=5)
rgb_image = rgb_images[0]
r, g, b = get_r_g_b_channels(rgb_image)
# plot a histogram of rgbn values
plot_rgbn_histo(r, g, b)


# We can look at each channel individually:

# In[ ]:


# Plot the bands
fig = plt.figure()
fig.set_size_inches(9, 3)
for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'))):
    a = fig.add_subplot(1, 3, i+1)
    a.set_title(c)
    plt.imshow(x)


# In[ ]:


plt.imshow(rgb_image)


# ## Calibrate the image

# Find the mean for the colors across the entire dataset. It takes agess, **I'm unable to commit it due to the long performance.**

# In[ ]:


all_image_paths = os.listdir(PLANET_KAGGLE_JPEG_DIR)
random.shuffle(all_image_paths)


# In[ ]:


n = 200

ref_colors = [[],[],[]]
for _file in tqdm(all_image_paths[:n]):
    # keep only the first 3 bands, RGB
    _img = mpimg.imread(os.path.join(PLANET_KAGGLE_JPEG_DIR, _file))[:,:,:3]
    # Flatten 2-D to 1-D
    _data = _img.reshape((-1,3))
    # Dump pixel values to aggregation buckets
    for i in range(3): 
        ref_colors[i] = ref_colors[i] + _data[:,i].tolist()
    
ref_colors = np.array(ref_colors)


# In[ ]:


ref_colors = np.array(ref_colors)
ref_color_mean = [np.mean(ref_colors[i]) for i in range(3)]
ref_color_std = [np.std(ref_colors[i]) for i in range(3)]


# In[ ]:


print("ref_color_mean:")
print(ref_color_mean)
print("ref_color_std:")
print(ref_color_std)


# In[ ]:


def calibrate_image(rgb_img):
    calibrated_img = rgb_image.copy().astype('float32')
    for i in range(3):
        calibrated_img[:,:,i] = (rgb_img[:,:,i] -  np.mean(rgb_img[:,:,i])) / np.std(rgb_img[:,:,i])
        calibrated_img[:,:,i] = calibrated_img[:,:,i] * ref_color_std[i] + ref_color_mean[i]
    return calibrated_img.astype('uint8')


# In[ ]:


img = calibrate_image(rgb_image)
plt.imshow(img)


# ## Sample images

# In[ ]:


def display_multiple_images(rgb_images):
    col, row = (1, len(rgb_images)) if len(rgb_images) <=4 else ((len(rgb_images) / 4) + 1, 4)
    fig = plt.figure()
    fig.set_size_inches(12, 3 * col)
    for i, _img in enumerate(rgb_images):
        a = fig.add_subplot(col, row, i+1)
        plt.imshow(calibrate_image(_img))


# In[ ]:


# provide labels to display sample images
labels = ['water']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# In[ ]:


# provide labels to display sample images
labels = ['primary']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# In[ ]:


# provide labels to display sample images
labels = ['agriculture']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# In[ ]:


# provide labels to display sample images
labels = ['cultivation']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# In[ ]:


# provide labels to display sample images
labels = ['habitation']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# In[ ]:


# provide labels to display sample images
labels = ['selective_logging']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# In[ ]:


# provide labels to display sample images
labels = ['slash_burn']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# In[ ]:


# provide labels to display sample images
labels = ['blow_down']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# In[ ]:


# provide labels to display sample images
labels = ['blooming']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# In[ ]:


# provide labels to display sample images
labels = ['conventional_mine']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# In[ ]:


# provide labels to display sample images
labels = ['artisinal_mine']
rgb_images= get_rgb_image(labels=labels, n_samples=4)
display_multiple_images(rgb_images)


# Original Notebook by Jesus Martinez Manso and Benjamin Goldenberg
# 
# (C) Planet 2017
