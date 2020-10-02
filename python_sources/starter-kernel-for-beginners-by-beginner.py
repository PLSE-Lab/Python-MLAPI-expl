#!/usr/bin/env python
# coding: utf-8

# This Kernel contains basic analysis of data that includes the following:
# 
# 1) Converting *train.csv*'s "Target" column from string to list [required for one-hot-encoding]
# 
# 2) One Hot Encoding of "Target" column.
# 
# 3) Frequency of protein occurance
# 
# 4) Correlations among protein types.
# 
# Note: This kernal is a work in progress. I will update above description periodically.
# 
# Acknowledgements:
# 1) "Protein Atlas - Exploration and Baseline" Kernel by Allunia
# 

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


# In[ ]:


train_labels = pd.read_csv("../input/train.csv")


# In[ ]:


label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}


# **Converting Target: string to list **

# In[ ]:



train_labels_modified = train_labels.copy()
for i in range(train_labels_modified.shape[0]):
    train_labels_modified.Target[i] = train_labels_modified.Target[i].split()
    for k,j in enumerate(train_labels_modified.Target[i]):
        train_labels_modified.Target[i][k] = label_names[int(j)]
    
train_labels_modified.head()


# **One Hot Encoding Target Column**
# 
# (I am using MultiLabelBinarizer here. For one hot encoding using first principles, please refer Allunia's kernel mentioned in Acknowledgement.)

# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
s = pd.DataFrame(mlb.fit_transform(train_labels_modified.Target),columns=mlb.classes_, index=train_labels_modified.index)
train_labels_one_hot = train_labels_modified.join(s)
train_labels_one_hot


# **Protein Histogram: Frequency of occurence of proteins** 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
target_counts = train_labels_one_hot.drop(["Id", "Target"],axis =1).sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(10,10))
sns.barplot(x = target_counts.values, y=target_counts.index, order=target_counts.index)
plt.xlabel("Number of Occurences in training set")
#plt.ylabel("Protein Type")


# Above Plot tells us that "Nucleoplasm" is the most frequent protein followed by cytosol and so on.
# 
# It is worthwhile to see: In what percentage of training images/cells, a particular protein is identified

# In[ ]:


plt.figure(figsize=(10,10))
# Using the same code as above, but divided by number of training images to get percentage
sns.barplot(x = (target_counts.values)/train_labels_modified.shape[0], y=target_counts.index, order=target_counts.index)

plt.xlabel("Fraction of Occurences in training set")


# Now it is clear that "Nucleoplasm" is present is about 40% of training images,"cytosol" in about 30% and so on.
# 
# So, even without any modeling, we know that predicting "Nucleoplasm" in a cell is an educated guess that will get us right ~40% of time!!

# In[ ]:


print("Number of training images with no protein identified:", train_labels_modified.Target.isnull().sum())
# every image has atleast one protein identified, so we don't have to worry about  missing data.


# In[ ]:


# Distribution of number of proteins per image

occurances  = [len(train_labels_modified.Target[i]) for i in range(train_labels_modified.shape[0])]
plt.hist(occurances, align = "left",range = [0,5])


# **Observations**: Almost half of the training images have one protein and most of other half have 2 proteins.  

# **Are there any proteins that (almost always) occur in pairs ?**

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(train_labels_one_hot.drop(["Id", "Target"],axis=1).corr(),cmap="RdYlBu", vmin=-1, vmax=1)


# 1) *endosomes* and *lysosomes* occur in pairs(most of the time).
# 
# 2)* mitotic spindle* and *cytokinetic bridge* occur in pairs(sometimes)

# In[ ]:


#print(os.listdir("../input/train"))


# **Working on data loading and display**(Incomplete)

# In[ ]:


from skimage.io import imread
def load_image(image_id, path="../input/train/"):
    images = np.zeros((4,512,512))
    images[0,:,:] = imread(path + image_id + "_green" + ".png")
    images[1,:,:] = imread(path + image_id + "_red" + ".png")
    images[2,:,:] = imread(path + image_id + "_blue" + ".png")
    images[3,:,:] = imread(path + image_id + "_yellow" + ".png")
    return images
#load_image(image_id = "000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0")
def display_image_row(image, axis, title):
    axis[0].imshow(image[0,:,:], cmap = "Greens")
    axis[1].imshow(image[1,:,:], cmap = "Reds")
    axis[2].imshow(image[2,:,:], cmap = "Blues")    
    axis[3].imshow(image[3,:,:], cmap = "Oranges")
    axis[1].set_title("microtubules")
    axis[2].set_title("nucleus")
    axis[3].set_title("endoplasmatic reticulum")


# 
