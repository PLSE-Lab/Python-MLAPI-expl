#!/usr/bin/env python
# coding: utf-8

# # Loading Libraries

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import os
from itertools import chain
from collections import Counter
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


os.listdir('../input/')


# # The input directory contains the following files:
# 1. train - This folder contains the training images of size 512x512 in PNG format.
# 1. test -  This folder contains the test images of size 512x512 in PNG format.
# 1. train.csv - This CSV file contains all the filenames and labels for the training set.
# 1. sample_submission.csv - This file contains a sample submission file with two columns Id and Predicted. The predicted column contains all the predicted test classes..

# Let's load the train.csv file.
# It contains two columns:
# 1.     Id - the base filename of the sample. All ids consist of four files - blue, green, red, and yellow.
# 1.     Target - in the training data, this represents the labels assigned to each sample.
# 

# In[ ]:


df = pd.read_csv('../input/train.csv')
def count_target(target_val):
    return len(target_val.split(' '))

df['nclass'] = df.Target.map(count_target)
df.head()


# # Function for plotting the images

# In[ ]:


def plot_img(img_id):
    filt_no = 0
    plt.figure(figsize=(30,60))
    plt.tight_layout()
    for filt in img_filt:
        img_path = train_img_path + img_id + filt
        img = np.array(load_img(img_path))
        plt.subplot(1,4,filt_no+1)
        plt.imshow(img)
        if i==0: plt.title(filt[1:-4]+' filter')
        plt.axis('off')
        filt_no += 1        


# # Label Count Distribribution

# In[ ]:


label_count = []
for i in range(df.nclass.min(),df.nclass.max()+1):
    label_count.append(np.sum(df.nclass==i))
    print('No. of images with',i,'label:',label_count[-1])


# In[ ]:


x = np.arange(len(label_count))+1
plt.bar(x,label_count)
plt.title('Label Count Distribution in Train Set')


# # It seems that most of the images contain only 1 class ! There are very images with 4 or more classes.

# # Lets see the contents of train folder.
# As expected it contains files in the format of** file_id + filter_name + .png**

# In[ ]:


os.listdir('../input/train/')[:10]


# # Visualizing files with 5 targets

# There are only 2 such files !

# In[ ]:


img_filt = ['_green.png','_blue.png','_red.png','_yellow.png']
train_img_path = '../input/train/'
ids = df[df.nclass==5].reset_index()
i = 0; nimg = 2
while True:
    if i == nimg: break
    img_id =  ids.Id[i]   
    plot_img(img_id)
    i += 1


# # Visualizing files with 4 targets

# In[ ]:


ids = df[df.nclass==4].reset_index()
i = 0; nimg = 5
while True:
    if i == nimg: break
    img_id =  ids.Id[i]   
    plot_img(img_id)
    i += 1


# # Visualizing files with 3 targets

# In[ ]:


ids = df[df.nclass==3].reset_index()
i = 0; nimg = 5
while True:
    if i == nimg: break
    img_id =  ids.Id[i]   
    plot_img(img_id)
    i += 1


# # Visualizing files with 2 targets

# In[ ]:


ids = df[df.nclass==2].reset_index()
i = 0; nimg = 5
while True:
    if i == nimg: break
    img_id =  ids.Id[i]   
    plot_img(img_id)
    i += 1


# # Visualizing files with 1 target

# In[ ]:


ids = df[df.nclass==1].reset_index()
i = 0; nimg = 5
while True:
    if i == nimg: break
    img_id =  ids.Id[i]   
    plot_img(img_id)
    i += 1


# # Mapping the labels
# There are in total 28 different labels present in the dataset. 
# 
# The labels are represented as integers that map to the following

# In[ ]:


def mk_list(val):
    return [int(label) for label in val.split(' ')]
df['target_list'] = df['Target'].map(mk_list)
all_labels = list(chain.from_iterable(df['target_list'].values))
label_count = Counter(all_labels)
arr = np.zeros((28,))
for key,value in label_count.items():
    arr[key] = value


# In[ ]:


map_class_labels = {0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6: 'Endoplasmic reticulum',
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',
14: 'Microtubules',
15: 'Microtubule ends',
16: 'Cytokinetic bridge',
17: 'Mitotic spindle',
18: 'Microtubule organizing center',
19: 'Centrosome',
20: 'Lipid droplets',
21: 'Plasma membrane',
22: 'Cell junctions',
23: 'Mitochondria',
24: 'Aggresome',
25: 'Cytosol',
26: 'Cytoplasmic bodies',
27: 'Rods and rings'}


# In[ ]:


plt.figure(figsize=(30,5))
ax = sns.barplot(x=np.arange(28),y=arr)
ax.set_xticklabels(list(map_class_labels.values()), fontsize=15, rotation=40, ha="right")
ax.set(xlabel='Classes', ylabel='Class Counts')
plt.show()


# From the figure it is clear that Nucleoplasm is the most common class followed by Cytosol.
# 
# Nucleoplasm  - the substance of a cell nucleus, especially that not forming part of a nucleolus.
# 
# Cytosol - the aqueous component of the cytoplasm of a cell, within which various organelles and particles are suspended.

# # Visualizing the Nucleoplasm class images

# In[ ]:


ids = []
nucleoplasm_class = [0]
for i,val in enumerate(df.target_list.values):
    if nucleoplasm_class==val:
        ids.append(i)
i = 0; nimg = 3
while True:
    if i == nimg: break
    img_id =  df.Id[ids[i]]
    plot_img(img_id)
    i += 1


# # Visualizing the Cytosol class images

# In[ ]:


ids = []
cytosol_class = [25]
for i,val in enumerate(df.target_list.values):
    if cytosol_class==val:
        ids.append(i)
i = 0; nimg = 3
while True:
    if i == nimg: break
    img_id =  df.Id[ids[i]]
    plot_img(img_id)
    i += 1


# I will keep updating this notebook as i explore further.
# 
# PS - Mitochondria is the power house of the cell :-)
