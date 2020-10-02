#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Load the training parquet file, this file contains signal measurements, each column contains aa single 800,000 measurement signal

# In[ ]:


train = pq.read_pandas('../input/train.parquet').to_pandas()


# In[ ]:


train.info()


# Plotting first few signals from the training set

# In[ ]:


plt.figure(figsize=(24, 8))
plt.plot(train.iloc[:, :5]);


# Loading the meta files

# In[ ]:


meta = pd.read_csv('../input/metadata_train.csv')


# In[ ]:


meta.info()


# In[ ]:


meta.describe()


# In[ ]:


meta.corr()


# It can be seen that `phase` and `target` are independent of `signal_id` and `id_measurement` and are independent of each other

# In[ ]:


meta.head(10)


# So, the fault meter detects faults on each phase (three: 0, 1, 2) and each detection has a unique id called `id_measurement`.
# 

# Let's plot some positive and negative samples

# In[ ]:


# get positive and negative `id_measurement`s
positive_mid = np.unique(meta.loc[meta.target == 1, 'id_measurement'].values)
negative_mid = np.unique(meta.loc[meta.target == 0, 'id_measurement'].values)


# In[ ]:


# get one positive and one negative signal_id
pid = meta.loc[meta.id_measurement == positive_mid[0], 'signal_id']
nid = meta.loc[meta.id_measurement == negative_mid[0], 'signal_id']


# In[ ]:


positive_sample = train.iloc[:, pid]
negative_sample = train.iloc[:, nid]


# In[ ]:


plt.figure(figsize=(24, 8))
plt.plot(positive_sample);


# In[ ]:


plt.figure(figsize=(24, 8))
plt.plot(negative_sample);


# Analysing class imbalance

# In[ ]:


meta['target'].value_counts().plot(kind='bar');


# In[ ]:


meta['target'].value_counts()


# We can see, there is a huge class imbalance in the data

# In[ ]:


meta['phase'].value_counts()


# In[ ]:





# In[ ]:




