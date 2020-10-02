#!/usr/bin/env python
# coding: utf-8

# In each competition I find myself in a need for a feature reference. It's just what you look at when you wonder how important is feature "xyz", is it meaningful to scale it, etc. This is the goal of this script. 
# 
# ATTENTION: To acutally see the final version click on "You are viewing the last successful run of this script. Click here to see the current version with an error." The warning is shown, because the script is creating 3 graphics for each feature, or more than

# In[ ]:


import pandas as pd
import numpy as np # linear algebra
from matplotlib import pyplot as plt
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
import pylab as pl
sns.set(color_codes=True)


# In[ ]:


with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")


# In[ ]:


print('Dataframe shape:', df.shape)
print('Columns', df.columns.values)


# In[ ]:


print('Ids (each id represents a different stock, e.g. Apple, Google, etc.) count:', len(df['id'].unique()))
print('Time frames count:', len(df['timestamp'].unique()))


# If you multiply timestamps and id, you get 1424x1813 =  2,581,712. But we only have 1,710,756 samples of data => At each time frame you have some variable count of ids (stocks) X, where X<=1424. Let's look at how much ids one will find at each time frame:

# In[ ]:


id_count = [len(df[df['timestamp'] == i]['id'].unique()) for i in range(1813)]
plt.figure(figsize=(9,3))
plt.xlabel('Timestamp index')
plt.ylabel('Unique IDs for the timestamp')
plt.plot(range(1813), id_count,'.b')
plt.show()


# That's a bit strange. The number of IDs for each timestamp is increasing over time...  You can also observe a pattern of time periodicity... There is already a script about that: https://www.kaggle.com/chaseos/two-sigma-financial-modeling/understanding-id-and-timestamp

# In[ ]:


features = [feature for feature in df.columns.values if not feature in ['id', 'timestamp']]
for feature in features:
    values = df[feature].values
    nan_count = np.count_nonzero(np.isnan(values))
    values = sorted(values[~np.isnan(values)])
    print('NaN count:', nan_count, 'Unique count:', len(np.unique(values)))
    print('Max:', np.max(values), 'Min:', np.min(values))
    print('Median', np.median(values), 'Mean:', np.mean(values), 'Std:', np.std(values))
    plt.figure(figsize=(8,5))
    plt.title('Values '+feature)
    plt.plot(values,'.b')
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.title('Percentiles 1,5,10...95,99 '+feature)
    percentiles = [1] + list(range(5,100,5)) +[99]
    plt.plot(percentiles, np.percentile(values, percentiles),'.b')
    plt.show()
    
    fit = stats.norm.pdf(values, np.mean(values), np.std(values))  #this is a fitting indeed
    plt.title('Distribution Values '+feature)
    plt.plot(values,fit,'-g')
    plt.hist(values,normed=True, bins=10)      #use this to draw histogram of your data
    plt.show()


# In[ ]:




