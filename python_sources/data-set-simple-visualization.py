#!/usr/bin/env python
# coding: utf-8

# **Simple visualisation of each class **

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


# 0 - No DR
# 
# 1 - Mild
# 
# 2 - Moderate
# 
# 3 - Severe
# 
# 4 - Proliferative DR

# In[ ]:



import pandas as pd
from glob import glob
import os
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# In[ ]:


input_path = "../input/"


# Load the annotations and file

# In[ ]:


def load_df(path):    
    def get_filename(image_id):
        return os.path.join(input_path, "train_images", image_id + ".png")

    df_node = pd.read_csv(path)
    df_node["file"] = df_node["id_code"].apply(get_filename)
    df_node = df_node.dropna()
    
    return df_node

df = load_df(os.path.join(input_path, "train.csv"))
len(df)

df.head()


# Plotting retina images

# In[ ]:


import math

def get_filelist(diagnosis=0):
    return df[df['diagnosis'] == diagnosis]['file'].values

def subplots(filelist):
    plt.figure(figsize=(16, 9))
    ncol = 3
    nrow = math.ceil(len(filelist) // ncol)
    
    for i in range(0, len(filelist)):
        plt.subplot(nrow, ncol, i + 1)
        img = cv2.imread(filelist[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)


# 
# Severity 0: No DR
# No abnormalities

# In[ ]:


filelist = get_filelist(diagnosis=0)
subplots(filelist[:9])


# 
# Severity 1: Mild
# 

# In[ ]:


filelist = get_filelist(diagnosis=1)
subplots(filelist[:9])


# 
# Severity 2: Moderate

# In[ ]:


filelist = get_filelist(diagnosis=2)
subplots(filelist[:9])


# Severity 3: Severe

# In[ ]:


filelist = get_filelist(diagnosis=3)
subplots(filelist[:9])


# Severity 4: Proliferative DR

# In[ ]:


filelist = get_filelist(diagnosis=4)
subplots(filelist[:9])


# Class statics 

# In[ ]:


Counter(df['diagnosis'])


# In[ ]:


plt.hist(df['diagnosis'], bins=5)

