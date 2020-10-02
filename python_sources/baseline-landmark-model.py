#!/usr/bin/env python
# coding: utf-8

# # Baseline Model
# Here we try and build a simple baseline model to have a frame of reference for others

# In[16]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob
# not needed in Kaggle, but required in Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


base_dir = '../input/'
train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
train_df.sample(3)


# # Overview
# We see a few wide outliers and quite a few images with single examples

# In[18]:


train_df['landmark_id'].value_counts().hist()


# In[19]:


submit_df = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
# take the most frequent label
def_guess = train_df['landmark_id'].value_counts()/train_df['landmark_id'].value_counts().sum()
def_guess.index[0], def_guess.values[0]


# In[20]:


submit_df['landmarks'] = '%d %2.2f' % (def_guess.index[0], def_guess.values[0])
submit_df.to_csv('submission.csv', index=False)
submit_df.sample(2)


# # Random Guessing
# Here we do some random guessing normalized by the frequency in the training set

# In[25]:


np.random.seed(2018)
r_idx = lambda : np.random.choice(def_guess.index, p = def_guess.values)


# In[29]:


get_ipython().run_cell_magic('time', '', "r_score = lambda idx: '%d %2.4f' % (def_guess.index[idx], def_guess.values[idx])\nsubmit_df['landmarks'] = submit_df.id.map(lambda _: r_score(r_idx()))\nsubmit_df.to_csv('rand_submission.csv', index=False)\nsubmit_df.sample(2)")


# In[30]:


submit_df.sample(2)


# In[ ]:




