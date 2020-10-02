#!/usr/bin/env python
# coding: utf-8
##### Goal: 
* Predict the number of open_channels present, based on electrophysiological signal data. (at each time step and for each time series). 

* IMPORTANT: While the time series appears continuous, the data is from discrete batches of 50 seconds long 10 kHz samples (500,000 rows per batch). In other words, the data from 0.0001 - 50.0000 is a different batch than 50.0001 - 100.0000, and thus discontinuous between 50.0000 and 50.0001.

I'll do: 

* Simple baseline around min/max/average of having X channels opened. 
* Add the groups to segment the data (ech group is a seperate experiment) , for evaluation and further features 
# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import math


# In[ ]:


train = pd.read_csv("../input/liverpool-ion-switching/train.csv")


print(train.shape)

# df  =train.copy()
# data=df.values


# In[ ]:


test = pd.read_csv("../input/liverpool-ion-switching/test.csv")
print(test.shape)


# In[ ]:


n_groups = 100
# df["group"] = 0
# for i in range(n_groups):
#     ids = np.arange(i*50000, (i+1)*50000)
#     df.loc[ids,"group"] = i

train['group'] = 0
train.loc[:500000, 'group'] = 1 
train.loc[500000:500000*2, 'group'] = 2
train.loc[500000*2:500000*3, 'group'] = 3
train.loc[500000*3:500000*4, 'group'] = 4
train.loc[500000*4:500000*5, 'group'] = 5
train.loc[500000*5:500000*6, 'group'] = 6
train.loc[500000*6:500000*7, 'group'] = 7
train.loc[500000*7:500000*8, 'group'] = 8
train.loc[500000*8:500000*9, 'group'] = 9
train.loc[500000*9:500000*10, 'group'] = 10


test['group'] = 0
test.loc[:500000, 'group'] = 11 
test.loc[500000:500000*2, 'group'] = 12
test.loc[500000*2:500000*3, 'group'] = 13
test.loc[500000*3:500000*4, 'group'] = 14


# ## New Times
# * make new running "time" index pergroup = time that has passed, within the batch/group**
# *  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.GroupBy.cumcount.html#pandas.core.groupby.GroupBy.cumcount
# 
# * We can also add a real datetime and add these values to it with `pd.to_TimeDelta()` , for time series methods that expect it, based on this 

# In[ ]:


test.nunique()


# In[ ]:


train.head()


# In[ ]:


### I oerwrite the original time column, to make it easy to "plug and play" into other code that expect the time col 
test["time"] = test.groupby("group").cumcount()+1
train["time"] = train.groupby("group").cumcount()+1

display(test.groupby("group")["time"].head(2))
display(test.groupby("group")["time"].tail(2))


# In[ ]:


test


# In[ ]:


train.describe()


# In[ ]:


# for i in range(n_groups):
#     sub = df[df.group == i]
#     signals = sub.signal.values
#     imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))
#     signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))
#     signals = signals*(imax-imin)
#     df.loc[sub.index,"open_channels"] = [0,] + list(np.array(signals[:-1],np.int))


# ## Features:
# * Z-score per group
# * coiuld add more , e.g. change vs minmax. 
# 
# * e.g. `df.groupby('batch')['signal'].rolling(r).mean().reset_index()['signal']`

# In[ ]:


from scipy.stats import zscore
test["signal_zscore"] = test.groupby(["group"])["signal"].transform(lambda x : zscore(x,ddof=1))
train["signal_zscore"] = train.groupby(["group"])["signal"].transform(lambda x : zscore(x,ddof=1))


# In[ ]:


train.tail()


# ##### add arbitrary start time
# * each step is 0.1 ms (10khz measurement).  We may want to arbitrarily scale up due to worries about precision , although it shouldn;'t be ap roblem.  The real times are important for easy feature engineering using biological prior knowledge

# In[ ]:


arbitrary_time_start = pd.to_datetime(1490195805403502912, unit='ns')
print(arbitrary_time_start)

test["datetime"] = arbitrary_time_start + pd.to_timedelta(test["time"]*10,unit="ms")
train["datetime"] = arbitrary_time_start + pd.to_timedelta(train["time"]*10,unit="ms")

test.tail()


# In[ ]:


train.head()


# In[ ]:


train.to_csv("train_ionChannels.csv.gz",index=False,compression="gzip")
test.to_csv("test_ionChannels.csv.gz",index=False,compression="gzip")


# #### forked code after htis. commented out for now

# In[ ]:


## We are not going to bother with sample submission, let's just test this smoothing method against the labels in the training data. Should be fine.
# print(data[:5,1])
# print(data[:5,2])


# In[ ]:


# prediction = np.array(df.open_channels, np.int)
# print(prediction[:5])


# In[ ]:


# print(prediction.shape)
# prediction.tail()


# Looks good!
# So get the metrics, F1, Kappa, Quadratic Kappa and Accuracy;

# In[ ]:


# #To check I am working the metrics right ;-)
# gd=[1,2,3,4,5,6,7,8,9,0]
# pr=[1,2,3,4,5,6,7,8,8,0]

# from sklearn.metrics import cohen_kappa_score , accuracy_score ,f1_score

# print("Regular Cohen's Kappa", cohen_kappa_score(np.asarray(data[:,2],np.int),np.array(df.open_channels, np.int),weights="quadratic"))
# print("Quadratic Cohen's Kappa", cohen_kappa_score(np.asarray(data[:,2],np.int),np.array(df.open_channels, np.int)))
# print("Accuracy", accuracy_score(data[:,2],np.array(df.open_channels, np.int)))
# print("f1", f1_score(data[:,2],np.array(df.open_channels, np.int)))
# print("test Accuracy", accuracy_score(gd,pr))


# In[ ]:




