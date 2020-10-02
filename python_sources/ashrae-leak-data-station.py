#!/usr/bin/env python
# coding: utf-8

# # All Leak Dataset
# 
# As you already know there are huge data leak in this competition. Until now site-0, site-1, site-2, site-4 and site-15 building meter reading data
# was discovered by great kagglers.
# 
# This Kernel collects all leak data revealed by following great kernels:
# 
# * [ASHRAE - UCF Spider and EDA (Full Test Labels)](https://www.kaggle.com/gunesevitan/ashrae-ucf-spider-and-eda-full-test-labels) v3
# * [UCL: Data Leakage (Episode 2)](https://www.kaggle.com/mpware/ucl-data-leakage-episode-2) v1
# * [ASU train and scraped test data](https://www.kaggle.com/pdnartreb/scrap-asu-data) v7
# * [UCB: Data Leakage (Site 4)](https://www.kaggle.com/serengil/ucb-data-leakage-site-4/output) v15
# * [ASHRAE-site15-cornell](https://www.kaggle.com/pp2file/ashrae-site15-cornell) v3
# 
# Thank you @gunesevitan, @mpware, @poedator, @pdnartreb, @serengil and @pp2file . You are Great Kagglers!!

# In[ ]:


import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt


# In[ ]:


import os
os.listdir('../input/')


# In[ ]:


train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])


# In[ ]:


def plot_meter(train, leak, start=0, n=100, bn=10):
    for bid in leak.building_id.unique()[:bn]:    
        tr = train[train.building_id == bid]
        lk = leak[leak.building_id == bid]
        
        for m in lk.meter.unique():
            plt.figure(figsize=[10,2])
            trm = tr[tr.meter == m]
            lkm = lk[lk.meter == m]
            
            plt.plot(trm.timestamp[start:start+n], trm.meter_reading.values[start:start+n], label='train')    
            plt.plot(lkm.timestamp[start:start+n], lkm.meter_reading.values[start:start+n], '--', label='leak')
            plt.title('bid:{}, meter:{}'.format(bid, m))
            plt.legend()


# # site 0

# In[ ]:


# load site 0 data
ucf_root = Path('../input/ashrae-ucf-spider-and-eda-full-test-labels')
leak0_df = pd.read_pickle(ucf_root/'site0.pkl') 
leak0_df['meter_reading'] = leak0_df.meter_reading_scraped
leak0_df.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)
leak0_df.fillna(0, inplace=True)
leak0_df.loc[leak0_df.meter_reading < 0, 'meter_reading'] = 0
print(len(leak0_df))


# In[ ]:


leak0_df.head()


# In[ ]:


leak0_df.tail()


# In[ ]:


plot_meter(train_df, leak0_df, start=5000)


# # site 1

# In[ ]:


# load site 1 data
ucl_root = Path('../usr/lib/ucl_data_leakage_episode_2')
leak1_df = pd.read_pickle(ucl_root/'site1.pkl') 
leak1_df['meter_reading'] = leak1_df.meter_reading_scraped
leak1_df.drop(['meter_reading_scraped'], axis=1, inplace=True)
leak1_df.fillna(0, inplace=True)
leak1_df.loc[leak1_df.meter_reading < 0, 'meter_reading'] = 0
print(len(leak1_df))


# In[ ]:


leak1_df.head()


# In[ ]:


leak1_df.tail()


# In[ ]:


plot_meter(train_df, leak1_df, start=0)


#  # site 2

# In[ ]:


# load site 2 data
leak2_df = pd.read_csv('/kaggle/input/asu-buildings-energy-consumption/asu_2016-2018.csv')
leak2_df['timestamp'] = pd.to_datetime(leak2_df['timestamp'])

leak2_df.fillna(0, inplace=True)
leak2_df.loc[leak2_df.meter_reading < 0, 'meter_reading'] = 0

leak2_df = leak2_df[leak2_df.building_id!=245] # building 245 is missing now.

#leak2_df = leak2_df[leak2_df.timestamp.dt.year > 2016]
print(len(leak2_df))


# In[ ]:


leak2_df.head()


# In[ ]:


leak2_df.tail()


# In[ ]:


plot_meter(train_df, leak2_df, start=0)


# # site 4

# In[ ]:


# load site 4 data
# its looks better to use threshold ...
leak4_df = pd.read_csv('../input/ucb-data-leakage-site-4/site4.csv')

leak4_df['timestamp'] = pd.to_datetime(leak4_df['timestamp'])
leak4_df.rename(columns={'meter_reading_scraped': 'meter_reading'}, inplace=True)
leak4_df.fillna(0, inplace=True)
leak4_df.loc[leak4_df.meter_reading < 0, 'meter_reading'] = 0
leak4_df['meter'] = 0

print('before remove dupilicate', leak4_df.duplicated(subset=['building_id','timestamp']).sum())
leak4_df.drop_duplicates(subset=['building_id','timestamp'],inplace=True)
print('after remove dupilicate', leak4_df.duplicated(subset=['building_id','timestamp']).sum())
print(len(leak4_df))


# In[ ]:


leak4_df.head()


# In[ ]:


leak4_df.tail() # its include 2019. i will delete them later


# In[ ]:


len(leak4_df.building_id.unique())


# In[ ]:


plot_meter(train_df, leak4_df, start=0)


# In[ ]:


train_df[train_df.building_id == 621].timestamp.min() # some train data is missing


# # site 15

# In[ ]:


# this data does not include 2016.
leak15_df = pd.read_csv('../input/ashrae-site15-cornell/site15_leakage.csv')

leak15_df['timestamp'] = pd.to_datetime(leak15_df['timestamp'])
leak15_df.fillna(0, inplace=True)
leak15_df.loc[leak15_df.meter_reading < 0, 'meter_reading'] = 0

print(leak15_df.duplicated().sum())
print(len(leak15_df))


# In[ ]:


leak15_df.head()


# In[ ]:


leak15_df.tail()


# In[ ]:


df = pd.concat([leak0_df, leak1_df, leak2_df, leak4_df, leak15_df])
df.drop('score', axis=1, inplace=True)
df = df[(df.timestamp.dt.year >= 2016) & (df.timestamp.dt.year < 2019)]
df.reset_index(inplace=True, drop=True)
print(len(df))


# In[ ]:


df.timestamp.min(), df.timestamp.max() 


# In[ ]:


df.head()


# In[ ]:


df.to_feather('leak.feather')


# In[ ]:


leak_df = pd.read_feather('leak.feather')


# In[ ]:


leak_df.head()


# In[ ]:


leak_df.meter.value_counts()


# In[ ]:


len(leak_df.building_id.unique())


# In[ ]:


# Wow!! What do you think ? it's really huge now !! 
len(leak_df) / len(train_df)

