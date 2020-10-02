#!/usr/bin/env python
# coding: utf-8

# In[13]:


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


# > # Loading Files

# In[14]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[15]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[16]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()


# # Create Submission

# In[17]:


# Create a dummy submission that has entries as many as the test set.
y_pred = np.ones(test.shape[0])
sample_submission.loc[:, 'Predicted'] = y_pred.astype(int) # Remember to explicitly cast the type to integer 1s and 0s


# In[18]:


sample_submission.head()


# In[19]:


sample_submission.to_csv('first_entry.csv', header=True, index=False)


# In[ ]:





# In[ ]:





# In[ ]:




