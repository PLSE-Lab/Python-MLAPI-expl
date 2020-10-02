#!/usr/bin/env python
# coding: utf-8

# Since the format HDF5 is not as popular as other simpler ones like CSV, this notebook contains a short introduction on how to work and manage this kind of files. This includes just a simple exploration using `pandas`, but it aims to be a starting point on understanding the format and how to work with it. The only real dependency we need is `pandas` but a couple more libraries are imported to showcase a simple exploratory example.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# HDF5 is a hierarchical data format. A good abstraction to think about is a Python dictionary, where a set of keys are mapped to a set of values. Just like a dictionary, you can get the dataset associated with a key in an HDF5 using a file descriptor. This file descriptor can be generated with `pandas.HDFStore`, and since HDF5 can be infamously prone to corruption issues when misused we recommend using it in simple operations inside context managers (using the `with` idiom) to ensure that the file descriptor is closed when not used. Here we can see a simple example:

# In[2]:


with pd.HDFStore('../input/madrid.h5') as data:
    df = data['master']
    
df.head()


# This files are lazy and allow precise access to the data: not all data needs to be read from disk when accessing a single dataset. However, you need to know beforehand the key of a dataset to access it. Using the method `pandas.HDFStore.keys()` is possible to get a generator that returns the complete sequence of keys contained in such file. Using it, we can iterate over all datasets and print the list of measures that each station has entries of.

# In[3]:


with pd.HDFStore('../input/madrid.h5') as data:
    for k in data.keys():
        print('{}: {}'.format(k, ', '.join(data[k].columns)))


# Even though the keys are returned with a leading forwardslash, this is not necessary if accessing files manually. The design of the file separates in different datasets measures from different stations, making it easier to access sequential data (which is likely the use case for time series analysis or prediction). Extracting a dataset to a DataFrame is as straightforward as expected:

# In[4]:


with pd.HDFStore('../input/madrid.h5') as data:
    test = data['28079016']

test.rolling(window=24).mean().plot(figsize=(20, 7), alpha=0.8)


# However, this separation in stations can look like a burden if a single DataFrame containing all information is necessary. However, the whole dataset (containing 2.7 million entries) can be generated from the partial datasets by concatenating them all together and adding the HDF5 key as a new column (the `station` columns below).

# In[5]:


partials = list()

with pd.HDFStore('../input/madrid.h5') as data:
    stations = [k[1:] for k in data.keys() if k != '/master']
    for station in stations:
        df = data[station]
        df['station'] = station
        partials.append(df)
            
df = pd.concat(partials, sort=False).sort_index()

df.info()

