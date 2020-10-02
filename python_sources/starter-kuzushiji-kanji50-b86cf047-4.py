#!/usr/bin/env python
# coding: utf-8

# ## Read the data
# 

# In[ ]:


import numpy as np # linear algebra
import os


# In[ ]:


input_dir = "../input"
dataset_dir = 'kuzushiji-kanji50'
base_folder_dir = os.path.join(input_dir, dataset_dir)
os.listdir(base_folder_dir)


# In[ ]:


x_train = np.load(os.path.join(base_folder_dir,'kuzushiji50_train_imgs.npy'))
x_test = np.load(os.path.join(base_folder_dir,'kuzushiji50_test_imgs.npy'))
y_train = np.load(os.path.join(base_folder_dir,'kuzushiji50_train_labels.npy'))
y_test = np.load(os.path.join(base_folder_dir,'kuzushiji50_test_labels.npy'))


# In[ ]:


y_test.shape


# Then go ahead and use those for training and testing your model :-)
