#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# I have never worked with image data before, so my goals are limited to basic data set exploration.
# 
# **Credits:**
# *  When I was wondering how to work with images and which image packges (scikit-image, Pillow, Matplotlib, OpenCV) to use: 
#     -  Checked out few kernels from past image competitions and the current competition, this kernel -  https://www.kaggle.com/jschnab/exploring-the-human-protein-atlas-images came to the rescue. I decided to go with OpenCV and learn along the way.

# # Imports

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import re
from itertools import product

# matplotlib style
plt.style.use('fivethirtyeight')

# random state
RSTATE=1984


# # Definitions

# In[ ]:


# color hunt palettes
ch_div_palette_1 = ["#288fb4", "#1d556f", "#efddb2", "#fa360a"]
ch_div_palette_2 = ["#ff5335", "#dfe0d4", "#3e92a3", "#353940"]
ch_div_palette_3 = ["#daebee", "#b6d7de", "#fcedda", "#ff5126"]
# matplotlib "fivethirtyeight" style colors
ch_div_palette_4 = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']


# # Helper functions

# In[ ]:


# https://www.kaggle.com/c/human-protein-atlas-image-classification/data
label_names = [
    "Nucleoplasm",
    "Nuclear membrane",
    "Nucleoli",
    "Nucleoli fibrillar center",
    "Nuclear speckles",
    "Nuclear bodies",
    "Endoplasmic reticulum",
    "Golgi apparatus",
    "Peroxisomes",
    "Endosomes",
    "Lysosomes",
    "Intermediate filaments",
    "Actin filaments",
    "Focal adhesion sites",
    "Microtubules",
    "Microtubule ends",
    "Cytokinetic bridge",
    "Mitotic spindle",
    "Microtubule organizing center",
    "Centrosome",
    "Lipid droplets",
    "Plasma membrane",
    "Cell junctions",
    "Mitochondria",
    "Aggresome",
    "Cytosol",
    "Cytoplasmic bodies",
    "Rods & rings",    
]
    
def get_num_labels_for_instance(label_string):
    labels = re.split(r'\s+', label_string)
    return len(labels)

def get_label_presence_func(label):
    def is_label_present(label_string):
        labels = set(re.split(r'\s+', label_string))
        return int(str(label) in labels)
    return is_label_present

def is_single_label(label_string):
    label_ids = re.split(r'\s+', label_string)
    if len(label_ids) > 1:
        return False
    return True
 
def get_label_name_for_label_id_string(label_ids_str):
    label_ids = re.split(r'\s+', label_ids_str)
    label_ids = [int(id) for id in label_ids]
    label = "+".join([label_names[id] for id in label_ids])
    return label

# Returns different bar color for single and multi-labels
def get_bar_color_1(is_single_label):
    if is_single_label:
        return ch_div_palette_1[0]
    return ch_div_palette_1[2]

# Returns different bar color for single and multi-labels
def get_bar_color_2(is_single_label):
    if is_single_label:
        return ch_div_palette_2[0]
    return ch_div_palette_2[2]

# Returns different bar color for single and multi-labels
def get_bar_color_3(is_single_label):
    if is_single_label:
        return ch_div_palette_3[2]
    return ch_div_palette_3[3]

# Returns different bar color for single and multi-labels
def get_bar_color_4(is_single_label):
    if is_single_label:
        return ch_div_palette_4[0]
    return ch_div_palette_4[1]


# # Training labels

# In[ ]:


labels_df = pd.read_csv("../input/train.csv")
print("Shape of the training labels frame (train.csv): ", labels_df.shape)
labels_df.head()


# The target is a multi-labels separated by a space. Our task is to predict those labels using multi-label classification methods.

# ## Breakdown of multi-labels distribution

# In[ ]:


labels_df["num_labels"] = labels_df["Target"].apply(get_num_labels_for_instance)
labels_count_dist = labels_df.groupby("num_labels")["num_labels"].count()
fig, ax = plt.subplots(num=1)
ax.bar(labels_count_dist.index.values, labels_count_dist.values)
ax.set_xlabel("Number of labels per instance")
ax.set_ylabel("Number of instances")
ax.set_title("Labels count distribution")
plt.show()


# Most of the training images have 1 or 2 labels. The number labels per image vary from 1 to 5. 

# ## Distribution of top-50 training labels (single/multi)

# In[ ]:


multi_labels_dist = pd.DataFrame()
tmp = labels_df.groupby("Target")["Target"].count().sort_values(ascending=False)
multi_labels_dist["Target"] = tmp.index.values
multi_labels_dist["Count"] = tmp.values
multi_labels_dist["is_single_label"] = multi_labels_dist["Target"].apply(is_single_label)
multi_labels_dist["Target_str"] = multi_labels_dist["Target"].apply(get_label_name_for_label_id_string)
multi_labels_dist = multi_labels_dist[["Target", "Target_str", "Count", "is_single_label"]]
print("Number of unique labels (single/multi): {}".format(multi_labels_dist.shape[0]))
multi_labels_dist.head()


# In[ ]:


topn = 50
fig, ax = plt.subplots(num=2)
fig.set_figwidth(15)
fig.set_figheight(10)
bar_colors = multi_labels_dist["is_single_label"].apply(get_bar_color_4).head(topn)
ax.bar(multi_labels_dist["Target_str"].head(topn), multi_labels_dist["Count"].head(topn), color=bar_colors)
ax.set_xticks(range(topn))
ax.set_xticklabels(multi_labels_dist["Target_str"].head(topn), rotation = 45, ha="right")
ax.set_title("Distribution of top-{} training labels (single/multi)".format(topn))
plt.show()


# Many top frequency labels are multi-label categories. 

# ## Distribution of single training labels

# In[ ]:


label_columns_df = pd.DataFrame()
for i in range(len(label_names)):
    label_chk_fn = get_label_presence_func(i)
    label_columns_df[label_names[i]] = labels_df["Target"].apply(label_chk_fn)
labels_dist = label_columns_df.sum().sort_values(ascending=False)


# In[ ]:


fig, ax = plt.subplots(num=2)
fig.set_figwidth(15)
fig.set_figheight(10)
ax.bar(labels_dist.index.values, labels_dist.values)
ax.set_xticks(range(len(labels_dist.values)))
ax.set_xticklabels(labels_dist.index.values, rotation = 45, ha="right")
ax.set_title("Distribution of single labels")
plt.show()


# # Look at sample training images

# > Each file represents a different filter on the subcellular protein patterns represented by the sample. The format should be [filename]_[filter color].png for the PNG files, and [filename]_[filter color].tif for the TIFF files.
# 
# > All image samples are represented by four filters (stored as individual files), the protein of interest (green) plus three cellular landmarks: nucleus (blue), microtubules (red), endoplasmic reticulum (yellow). The green filter should hence be used to predict the label, and the other filters are used as references.
# 
# We may want to look at some images to understand how they are labeled. 

# In[ ]:


fig, ax = plt.subplots(num=1, nrows=3, ncols=3)
fig.set_figheight(15)
fig.set_figwidth(15)
for idx, (x, y) in enumerate(product(range(3), range(3))):
    img_blue = cv2.imread("../input/train/" + labels_df.loc[idx, "Id"] + "_blue.png", cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread("../input/train/" + labels_df.loc[idx, "Id"] + "_green.png", cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread("../input/train/" + labels_df.loc[idx, "Id"] + "_red.png", cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.merge((img_blue, img_green, img_red))
    image_label = get_label_name_for_label_id_string(labels_df.loc[idx, "Target"])
    ax[x,y].imshow(img_bgr)
    ax[x,y].set_xticks([])
    ax[x,y].set_yticks([])
    ax[x,y].set_title(image_label, fontdict={"fontsize": 12})
plt.show()


# ## Sample images for top frequency labels
# We have not yet looked at how the images within the same label category (single/multi) look like and how similar or dissimilar they are. We can try looking into few images per label for the top frequency labels. 

# In[ ]:


# top-5
topn = 5
topn_labels = multi_labels_dist.head(topn)["Target"]
multi_labels_dist.head(5)


# In[ ]:


images_per_label = 5
fig, ax = plt.subplots(num=1, nrows=topn, ncols=images_per_label)
fig.set_figheight(21)
fig.set_figwidth(21)
for idx, t in enumerate(topn_labels):
    sample_ids_for_label = labels_df[labels_df["Target"] == t].sample(n=images_per_label, random_state=RSTATE)["Id"].tolist()
    ax[idx, 0].set_ylabel(get_label_name_for_label_id_string(t))
    for idy in range(images_per_label):
        img_blue = cv2.imread("../input/train/" + sample_ids_for_label[idy] + "_blue.png", cv2.IMREAD_GRAYSCALE)
        img_green = cv2.imread("../input/train/" + sample_ids_for_label[idy] + "_green.png", cv2.IMREAD_GRAYSCALE)
        img_red = cv2.imread("../input/train/" + sample_ids_for_label[idy] + "_red.png", cv2.IMREAD_GRAYSCALE)
        img_bgr = cv2.merge((img_blue, img_green, img_red))
        ax[idx,idy].imshow(img_bgr)
        ax[idx,idy].set_xticks([])
        ax[idx,idy].set_yticks([])
        ax[idx,idy].set_title(sample_ids_for_label[idy], fontdict={"fontsize":10})
plt.show()


# ## Sample images for label: "Nucleoplasm" (0)
# We previously looked at few sample images for top-n frequency labels and looked at them together. Let's look into more images for a particular category, for example: "Nucleoplasm". 

# In[ ]:


n_images = 25
ncols = 5
nrows = n_images // ncols
image_ids = labels_df[labels_df["Target"] == "0"].sample(n=n_images, random_state=RSTATE)["Id"].tolist()
fig, ax = plt.subplots(num=1, nrows=nrows, ncols=ncols)
fig.set_figheight(21)
fig.set_figwidth(21)
for idx, (x, y) in enumerate(product(range(nrows), range(ncols))):
    img_blue = cv2.imread("../input/train/" + image_ids[idx] + "_blue.png", cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread("../input/train/" + image_ids[idx] + "_green.png", cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread("../input/train/" + image_ids[idx] + "_red.png", cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.merge((img_blue, img_green, img_red))
    ax[x,y].imshow(img_bgr)
    ax[x,y].set_xticks([])
    ax[x,y].set_yticks([])
    ax[x,y].set_title(image_ids[idx], fontdict={"fontsize": 12})
plt.show()


# ## Sample images for label: "Mitochondria" (23)

# In[ ]:


n_images = 25
ncols = 5
nrows = n_images // ncols
image_ids = labels_df[labels_df["Target"] == "23"].sample(n=n_images, random_state=RSTATE)["Id"].tolist()
fig, ax = plt.subplots(num=1, nrows=nrows, ncols=ncols)
fig.set_figheight(21)
fig.set_figwidth(21)
for idx, (x, y) in enumerate(product(range(nrows), range(ncols))):
    img_blue = cv2.imread("../input/train/" + image_ids[idx] + "_blue.png", cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread("../input/train/" + image_ids[idx] + "_green.png", cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread("../input/train/" + image_ids[idx] + "_red.png", cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.merge((img_blue, img_green, img_red))
    ax[x,y].imshow(img_bgr)
    ax[x,y].set_xticks([])
    ax[x,y].set_yticks([])
    ax[x,y].set_title(image_ids[idx], fontdict={"fontsize": 12})
plt.show()


# ## Sample images for label: "Cytosol" (25)

# In[ ]:


n_images = 25
ncols = 5
nrows = n_images // ncols
image_ids = labels_df[labels_df["Target"] == "25"].sample(n=n_images, random_state=RSTATE)["Id"].tolist()
fig, ax = plt.subplots(num=1, nrows=nrows, ncols=ncols)
fig.set_figheight(21)
fig.set_figwidth(21)
for idx, (x, y) in enumerate(product(range(nrows), range(ncols))):
    img_blue = cv2.imread("../input/train/" + image_ids[idx] + "_blue.png", cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread("../input/train/" + image_ids[idx] + "_green.png", cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread("../input/train/" + image_ids[idx] + "_red.png", cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.merge((img_blue, img_green, img_red))
    ax[x,y].imshow(img_bgr)
    ax[x,y].set_xticks([])
    ax[x,y].set_yticks([])
    ax[x,y].set_title(image_ids[idx], fontdict={"fontsize": 12})
plt.show()


# In[ ]:




