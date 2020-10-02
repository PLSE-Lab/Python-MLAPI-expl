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


# Let's import some packages.

# In[ ]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# Let's read data.

# In[ ]:


train = pd.read_csv('../input/training_set.csv')
train_meta = pd.read_csv('../input/training_set_metadata.csv')


# The telescope only works during the night.  
# 
# When is the night in term of mjd?

# In[ ]:


train['mjd_frac'] = np.modf(train.mjd)[0]

_ = plt.hist(train.mjd_frac, bins=100)


# Seems the night starts around 0.95 and ends around 0.45.  Let's shift time to have night in a consecutive integer part.

# In[ ]:


train['mjd_frac'] = np.modf(train.mjd + 0.3)[0]

_ = plt.hist(train.mjd_frac, bins=100)


# Now the night starts at around 0.25 and ends around 0.75. We can define the night oindex accordingly.

# In[ ]:


train['mjd_night'] = np.modf(train.mjd + 0.3)[1]


# Let's now look at wich passbands are used on every night.

# In[ ]:


train = train.groupby(['object_id', 'mjd_night', 'passband']).mjd.mean().to_frame()
train = train.reset_index()
train.head()


# Let's look at the ddf trainings set.

# In[ ]:


train_ddf = train.merge(train_meta, how='left', on='object_id')
train_ddf = train_ddf[train_ddf.ddf == 1]


# Let's look at which nights passband 0 is used and which nights the other passbands are used.[](http://)

# In[ ]:


passband_0 = set(train_ddf[train_ddf.passband == 0].mjd_night.unique())
passband_1_to_5 = set(train_ddf[train_ddf.passband > 0].mjd_night.unique())
len(passband_0), len(passband_1_to_5)


# Ddf set is photographied quite a lot.  
# 
# Here is the peculiar thing.  Passband 0 is never used the same nigh as the other ones for sources in the ddf set:

# In[ ]:


passband_0 & passband_1_to_5


# We don't have the same dichotomy for the non ddf sources.  

# I have not found a way to leverage the above, but Im' interested to hear about any useful use of it.  This is whay I am sharing it here!

# In[ ]:




