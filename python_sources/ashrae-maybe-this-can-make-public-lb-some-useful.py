#!/usr/bin/env python
# coding: utf-8

# **Warning !**
# This is just a test of leaked data. 
# **Warning !**
# I just make the leaked data downlable !!
# 
# Since @sohier said:
# > We will ensure that any meter readings that can be publicly downloaded are not included in the private set.
# 
# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117357
# 
# It will nothing about the private LB. You can replce the 'submission_base' with your own one. 
# If everyone use this snippet, maybe it can make public LB some useful.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import os
import gc
import pickle
import numpy as np
import pandas as pd
import random as rn
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime


# In[ ]:


test = pd.read_csv('../input/ashrae-energy-prediction/test.csv', index_col=0, parse_dates = ['timestamp'])
building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv', usecols=['site_id', 'building_id'])


# In[ ]:


test = test.merge(building, left_on = "building_id", right_on = "building_id", how = "left")


# I used a public kernel from [@rohanrao](http://www.kaggle.com/rohanrao) as example.
# 
# https://www.kaggle.com/rohanrao/ashrae-half-and-half
# 
# **Replace this part with yours.**

# In[ ]:


submission_base = pd.read_csv('../input/ashrae-half-and-half/submission.csv', index_col=0)


# In[ ]:


submission = submission_base.copy()


# * Site 0
# 
# Thanks [@yamsam](https://www.kaggle.com/yamsam/) and his great kernel.
# 
# https://www.kaggle.com/yamsam/new-ucf-starter-kernel

# In[ ]:


site_0 = pd.read_csv('../input/new-ucf-starter-kernel/submission_ucf_replaced.csv', index_col=0)
# submission.loc[test[test['site_id']==0].index, 'meter_reading'] = site_0['meter_reading']
# del site_0
gc.collect()


# In[ ]:


site_0.to_csv('site0.csv.gz',index=False,compression='gzip', float_format='%.4f')


# In[ ]:


from IPython.display import FileLink
FileLink('../input/new-ucf-starter-kernel/submission_ucf_replaced.csv')


# * Site 1
# 
# Thanks [@mpware](https://www.kaggle.com/mpware) and his great kernel.
# 
# https://www.kaggle.com/mpware/ucl-data-leakage-episode-2
# 

# In[ ]:


with open('../usr/lib/ucl_data_leakage_episode_2/site1.pkl', 'rb') as f:
    site_1 = pickle.load(f)
site_1 = site_1[site_1['timestamp'].dt.year > 2016]


# In[ ]:


site_1.to_csv('site1.csv.gz',index=False,compression='gzip', float_format='%.4f')


# In[ ]:


t = test[['building_id', 'meter', 'timestamp']]
t['row_id'] = t.index
site_1 = site_1.merge(t, left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
site_1 = site_1[['meter_reading_scraped', 'row_id']].set_index('row_id').dropna()
submission.loc[site_1.index, 'meter_reading'] = site_1['meter_reading_scraped']
del site_1
gc.collect()


# * Site 2
# 
# Thanks [@pdnartreb](https://www.kaggle.com/pdnartreb) and his great dataset.
# 
# https://www.kaggle.com/pdnartreb/asu-buildings-energy-consumption

# In[ ]:


site_2 = pd.read_csv('../input/asu-buildings-energy-consumption/asu_2016-2018.csv', parse_dates = ['timestamp'])
site_2 = site_2[site_2['timestamp'].dt.year > 2016]


# In[ ]:


site_2.to_csv('site2.csv.gz',index=False,compression='gzip', float_format='%.4f')


# In[ ]:


t = test[['building_id', 'meter', 'timestamp']]
t['row_id'] = t.index
site_2 = site_2.merge(t, left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
site_2 = site_2[['meter_reading', 'row_id']].set_index('row_id').dropna()
submission.loc[site_2.index, 'meter_reading'] = site_2['meter_reading']
del site_2
gc.collect()


# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:


submission.describe()


# In[ ]:




