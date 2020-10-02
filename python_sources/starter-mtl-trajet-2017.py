#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[7]:


df = pd.read_csv('../input/mtl_trajet_points.csv', delimiter='\t')
print('There are {} trips.'.format(len(df.id_trip.unique()))) # There are 68777 trips.


# In[25]:


df.drop_duplicates(subset=['id_trip'])['mode'].value_counts() # Trips per mode


# In[35]:


# Points per trip; <4 already dropped from dataset
sns.distplot(df['id_trip'].value_counts().value_counts(), bins=20)


# Let's see that in a table... I don't know if trips with such a low number of points will be useful to give a representative profile.

# In[36]:


df['id_trip'].value_counts().value_counts()

