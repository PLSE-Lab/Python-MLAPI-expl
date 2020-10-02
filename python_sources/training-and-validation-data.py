#!/usr/bin/env python
# coding: utf-8

# Template to generate sets of training data ("<code>training</code>") and validation data ("<code>validation</code>") that parallel the contents of the full training set and the test set, shifted by 24 hours.  Based on [this kernel](https://www.kaggle.com/konradb/validation-set) from Konrad Banachewicz, but taking into account what we learned in [this thread](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/51877) from Alexander Firsov (including comments by James Trotman and me).

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }


# ## Training Data

# In[10]:


training = pd.read_csv( "../input/train.csv", 
                        nrows=122071523, 
                        usecols=columns, 
                        dtype=dtypes)


# In[11]:


training.tail()


# Note that the full training set ends at 16:00:00 on 2017-11-09, so to parallel what would be available to predict the test set, I truncate the training data one day earlier.

# ## Validation Data

# There are 3 separate chunks of test data, each extending for 2 hours and 1 second.  Here I read in each analogous chunk separately and concatenate them.

# In[12]:


valid1 = pd.read_csv( "../input/train.csv", 
                      skiprows=range(1,144708153), 
                      nrows=7705357, 
                      usecols=columns, 
                      dtype=dtypes)


# In[13]:


valid1.head()


# In[14]:


valid1.tail()


# In[18]:


valid2 = pd.read_csv( "../input/train.csv", 
                      skiprows=range(1,161974466), 
                      nrows=6291379, 
                      usecols=columns, 
                      dtype=dtypes)


# In[19]:


valid2.head()


# In[20]:


valid2.tail()


# In[21]:


valid2 = pd.concat([valid1, valid2])


# In[22]:


valid2.head()


# In[23]:


valid2.tail()


# In[24]:


del valid1
import gc
gc.collect()


# In[25]:


valid3 = pd.read_csv( "../input/train.csv", 
                      skiprows=range(1,174976527), 
                      nrows=6901686, 
                      usecols=columns, 
                      dtype=dtypes)


# In[26]:


valid3.head()


# In[27]:


valid3.tail()


# In[28]:


valid3 = pd.concat([valid2,valid3])


# In[30]:


valid3.head()


# In[31]:


valid3.tail()


# In[29]:


del valid2
gc.collect()


# In[32]:


validation = valid3
del valid3
gc.collect()
validation.head()


# In[33]:


validation.tail()


# In[ ]:




