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


# Light curves are irregularly spaced on the time axis.  They all exhibit some pretty large gaps. 
# 
# Let's look at these.
# 
# First, let's load some data and packages

# In[ ]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/training_set.csv')
train_meta = pd.read_csv('../input/training_set_metadata.csv')
train = train.merge(train_meta[['object_id', 'ddf', 'ra', 'decl', 'target']], 
                    how='left', on='object_id')


# Al light curves have gaps at regularly spaced intervals, see for instance this curve:

# In[ ]:


object_id = 105869076
print('object_id', object_id)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
df = train[train.object_id == object_id]
ax.scatter(df.mjd, df.flux, c=df.passband, cmap='rainbow', marker='+')
plt.show()


# We see 3 gaps, repating every year.  These are due to the relative position of the source, earth and the sun: the source must be visible during the night from the location of the telescope on earth. 
# 
# Let's analyse this further.

# First, we reuse the time analysis we did in https://www.kaggle.com/cpmpml/some-peculiarity# : when is the night in term of mjd?

# In[ ]:


train['mjd_frac'] = np.modf(train.mjd)[0]

_ = plt.hist(train.mjd_frac, bins=100)


# Seems the night starts around 0.95 and ends around 0.45. Let's shift time to have night in a consecutive integer part.

# In[ ]:


train['mjd_frac'] = np.modf(train.mjd + 0.3)[0]

_ = plt.hist(train.mjd_frac, bins=100)


# Now the night starts at around 0.25 and ends around 0.75. We can define the night index accordingly.

# In[ ]:


train['mjd_night'] = np.modf(train.mjd + 0.3)[1]


# Let's look for the largest of these gaps, and compute its middle date within a year by taking its fraction modulo 365.  We do it for each object_id.

# In[ ]:


def middle_gap(s):
    s = s.values
    s_prev = np.roll(s, 1)
    s_delta = s - s_prev
    s_delta_max = np.argmax(s_delta)
    s_middle_gap = (s[s_delta_max] + s_prev[s_delta_max]) / 2
    s_middle_gap = np.modf(s_middle_gap / 365)[0] * 365
    return s_middle_gap
    
train['middle_gap'] = train.groupby('object_id'
                                   ).mjd_night.transform(middle_gap)


# We can see how the middle gap relates to the location of the source in earth coordinates.

# In[ ]:


fix, ax = plt.subplots(1, 1, figsize=(12, 12))
plt.title('Middle gap')
plt.ylabel('Decl')
plt.xlabel('Ra')
ax.scatter(train.ra, train.decl, 
           cmap='rainbow', c=train.middle_gap, marker='+')


# It seems that the  middle gap is highly correlated with the ra coordinate.

# In[ ]:


fix, ax = plt.subplots(1, 1, figsize=(12, 12))
plt.ylabel('Middle gap')
plt.xlabel('Ra')
ax.scatter(train.ra, train.middle_gap, marker='+')


# We see that the middle gap and ra are proportional to each other, modulo 365 for the middle gap, and modulo 360 for ra.
