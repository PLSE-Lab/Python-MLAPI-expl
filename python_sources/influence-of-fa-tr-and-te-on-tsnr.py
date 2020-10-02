#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import pylab as plt
df = pd.read_csv('../input/mriqc-data-cleaning/bold.csv')


# In[ ]:


df


# In[ ]:


from json import load
import urllib.request, json 
from pandas.io.json import json_normalize
get_ipython().run_line_magic('matplotlib', 'inline')
import pylab as plt


# In[ ]:


dataset = pd.read_csv('../input/mriqc-data-cleaning/bold.csv',
                        usecols=[
                                'bids_meta.Manufacturer',
                                'bids_meta.MultibandAccelerationFactor',
                                'bids_meta.RepetitionTime',
                                'bids_meta.FlipAngle',
                                'bids_meta.EchoTime',
                                'tsnr'])
dataset.describe()


# In[ ]:


data = dataset.round(2)


# ### Impact of Flip Angle on tSNR

# In[ ]:


plt.figure(figsize=(20,10))
sns.stripplot(x='bids_meta.FlipAngle', y='tsnr', data=data,
              jitter=0.4, alpha=0.3, size=10)
plt.ylim(0, 100)
plt.xlim(0, None)


# ### Impact of Repetiton Time on tSNR

# In[ ]:


plt.figure(figsize=(100,10))
sns.stripplot(x='bids_meta.RepetitionTime', y='tsnr', 
              data=data,  
              jitter=0.4, alpha=0.3, size=10)
plt.ylim(0, 100)
plt.xlim(0.5, None)
plt.show


# ### Impact of Echo Time on tSNR

# In[ ]:


plt.figure(figsize=(100,10))
sns.stripplot(x='bids_meta.EchoTime', y='tsnr', 
              data=data,  
              jitter=0.4, alpha=0.3, size=10)
plt.ylim(0, 100)
plt.show


# 
