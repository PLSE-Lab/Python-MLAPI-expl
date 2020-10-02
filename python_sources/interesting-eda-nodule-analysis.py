#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


# Nodular extraction
df_nodules = pd.read_csv('../input/stage_1_train_labels.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nnodule_locations = {}\n\nfor i in range(len(df_nodules)):\n    filename = df_nodules.iloc[i][0]\n    location = df_nodules.iloc[i][1:5]\n    nodule = df_nodules.iloc[i][-1]\n    \n    if nodule == 1:\n        location = [int(float(loc)) for  loc in location]\n        if filename in nodule_locations.keys():\n            nodule_locations[filename].append(location)\n        else:\n            nodule_locations[filename] = [location]\n    ')


# In[ ]:


import random
folder = '../input/stage_1_train_images'
filenames = os.listdir(folder)
random.shuffle(filenames)

n_valid_samples = 2560
train_filenames = filenames[n_valid_samples:]
valid_filenames = filenames[:n_valid_samples]

print("Train Samples :", len(train_filenames))
print("Validation Samples :", len(valid_filenames))
n_train_samples = len(filenames) - n_valid_samples


# In[ ]:


node_samples = [len(i) for i in nodule_locations.values()]
sns.countplot(node_samples)


# In[ ]:


heatmap = np.zeros((1024, 1024))
ws = []
hs = []
for vals in nodule_locations.values():
    for val in vals:
        x, y, w, h = val
        heatmap[y: y+h, x: x+w] += 1
        ws.append(w)
        hs.append(h)
plt.figure(figsize = (10, 10))
plt.title('Nodule location heatmap')
plt.imshow(heatmap, cmap = 'Greys_r')


# In[ ]:


plt.hist(hs, bins = np.linspace(1, 1000, 50))


# In[ ]:




