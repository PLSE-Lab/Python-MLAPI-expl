#!/usr/bin/env python
# coding: utf-8

# Visualisation (modified from https://www.kaggle.com/anokas/two-sigma-financial-modeling/two-sigma-time-travel-eda )

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
p = sns.color_palette()


# In[ ]:


with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")


# In[ ]:


# Shape of the datase
print('Number of rows: {}, Number of columns: {}'.format(*df.shape))


# In[ ]:


cols = [0, 0, 0]
for c in df.columns:
    if 'derived' in c: cols[0] += 1
    if 'fundamental' in c: cols[1] += 1
    if 'technical' in c: cols[2] += 1
print('Derived columns: {}, Fundamental columns: {}, Technical columns: {}'.format(*cols))
print('\nColumn dtypes:')
print(df.dtypes.value_counts())
print('\nint16 columns:')
print(df.columns[df.dtypes == 'int16'])


# Analysis of the target

# In[ ]:


y = df['y'].values
plt.hist(y, bins=50, color=p[1])
plt.xlabel('Target Value')
plt.ylabel('Count')
plt.title('Distribution of target value')
print('Target value min {0:.3f} max {1:.3f} mean {2:.3f} std {3:.3f}'.format(
                                   np.min(y), np.max(y), np.mean(y), np.std(y)))


# In[ ]:


# Closer look at the tails and head
mask = df['y']< -0.08
y = df['y'][mask].values
print(y)
plt.hist(y, bins=50, color=p[1])
plt.xlabel('Target Value')
plt.ylabel('Count')
plt.title('Distribution of target value')
print('Target value min {0:.3f} max {1:.3f} mean {2:.3f} std {3:.3f}'.format(np.min(y), np.max(y), np.mean(y), np.std(y)))


# In[ ]:


# Closer look at the tails and head
mask = df['y']> 0.08
y = df['y'][mask].values
print(y)
plt.hist(y, bins=50, color=p[1])
plt.xlabel('Target Value')
plt.ylabel('Count')
plt.title('Distribution of target value')
print('Target value min {0:.3f} max {1:.3f} mean {2:.3f} std {3:.3f}'.format(np.min(y), np.max(y), np.mean(y), np.std(y)))


# Target is normally distributed

# Study of the timestamps

# In[ ]:


timestamp = df.timestamp.values
for bins in [100, 250]:
    plt.figure(figsize=(10,5))
    plt.hist(timestamp, bins=bins)
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.title('Histogram of Timestamp - {} bins'.format(bins))


# The target looks fairly irregular despite showing regularities.

# In[ ]:


timestamp = df.timestamp.loc[df.timestamp < 500].values
for bins in [100, 250]:
    plt.figure(figsize=(10, 5))
    plt.hist(timestamp, bins=bins,color=p[2])
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.title('Histogram of Timestamp - {} bins'.format(bins))


# In[ ]:


timediff = df.groupby('timestamp')['timestamp'].count().diff()
plt.figure(figsize=(10, 5))
plt.plot(timediff)
plt.xlabel('Timestamp')
plt.ylabel('Change in count since last timestamp')
plt.title('1st discrete difference of timestamp count')


# In[ ]:


pd.Series(timediff[timediff > 10].index).diff()


# In[ ]:


print(timediff[timediff > 10].index[0])


# Relationship timestep jumps and target 

# In[ ]:


time_targets = df.groupby('timestamp')['y'].mean()
plt.figure(figsize=(8, 5))
plt.plot(time_targets)
plt.xlabel('Timestamp')
plt.ylabel('Mean of target')
plt.title('Change in target over time - Red lines = new timeperiod')
for i in timediff[timediff > 5].index:
    plt.axvline(x=i, linewidth=0.25, color='red')


# We want to slightly change the analysis and focus on the single time stamps

# In[ ]:


for j in df.id.unique():
    if j < 10:
        mask = df.id==j
        time_mini = df.timestamp[mask].values
        for bins in [100, 250]:
            plt.figure(figsize=(15, 5))
            plt.hist(time_mini, bins=bins, color=p[4])
            plt.xlabel('Timestamp')
            plt.ylabel('Count')
            plt.title('Histogram of id = {0} Zoomed-in Timestamp - {1} bins'.format(j,bins))


# As we can see there is much more regularity considering the timestamps separately for IDs, this suggests a different periodicity in the different assets, that summed up in all of the assets gives the irregular result that was previously seen

# Study of NaNs

# In[ ]:


df1 = df[['id','timestamp','derived_1']] #.pivot( columns='derived_1', values='id')


# In[ ]:


df2 = df1.pivot( columns='derived_1', values='id')


# In[ ]:




