#!/usr/bin/env python
# coding: utf-8

# This is cloned from https://www.kaggle.com/xhlulu/liverpool-simple-averaging - my only change was switching from mean to median to account for the asymmetry in the data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# Let's check if open channels have a clear difference in signal:

# In[ ]:


train.open_channels.value_counts().plot(kind='bar')


# In[ ]:


for i in range(11):
    train.query(f'open_channels == {i}').signal.hist(bins=100, label=i)
plt.legend()


# Let's compute the average signal for each class from the training labels:

# In[ ]:


avgs_array = []

for i in range(11):
    avg = train.query(f'open_channels == {i}').signal.median()
    avgs_array.append(avg)

avgs_array = np.array(avgs_array)
avgs_array


# In[ ]:


def shortest_distance_index(v):
    """Return the index (neg or pos) that has the smallest difference from input v"""
    return np.argmin(np.abs(v - avgs_array))


# We will assign a label to an entry depending on which mean it is closer to.

# In[ ]:


sub.open_channels = test.signal.apply(shortest_distance_index)


# Thanks to [this kernel by Bojan for finding the fix](https://www.kaggle.com/tunguz/simple-ion-ridge-regression-starter):

# In[ ]:


sub.time = sub.time.apply(lambda x: '{:.4f}'.format(x))


# Let's check what the output distribution looks like:

# In[ ]:


sub.open_channels.value_counts().plot(kind='bar')


# In[ ]:


sub.to_csv('submission.csv', index=False)


# In[ ]:




