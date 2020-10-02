#!/usr/bin/env python
# coding: utf-8

# This is include site0 & site1 & asu & site 4

# **Unfortunately**
# # this is another leaked data
# # Thanks for the https://www.kaggle.com/wuliaokaola/ashrae-maybe-this-can-make-public-lb-some-useful
# # @Jonny Lee,This is where had been forked

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


test.describe()


# In[ ]:


test = test.merge(building, left_on = "building_id", right_on = "building_id", how = "left")


# I used a public kernel from .fit_%.(https://www.kaggle.com/aitude) as example.
# # 
# # https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08
# # 
# # **Replace this part with yours.**

# In[ ]:


submission_base = pd.read_csv('../input/ashrae-kfold-lightgbm-without-leak-1-08/submission.csv', index_col=0)


# In[ ]:


submission = submission_base.copy()


# In[ ]:


submission.describe()


# * Site 0
# # 
# # Thanks [@yamsam](https://www.kaggle.com/yamsam/) and his great kernel.
# # 
# # https://www.kaggle.com/yamsam/new-ucf-starter-kernel

# In[ ]:


site_0 = pd.read_csv('../input/new-ucf-starter-kernel/submission_ucf_replaced.csv', index_col=0)
submission.loc[test[test['site_id']==0].index, 'meter_reading'] = site_0['meter_reading']
del site_0
gc.collect()


# * Site 1
# # 
# # Thanks [@mpware](https://www.kaggle.com/mpware) and his great kernel.
# # 
# # https://www.kaggle.com/mpware/ucl-data-leakage-episode-2
# #

# In[ ]:


with open('../usr/lib/ucl_data_leakage_episode_2/site1.pkl', 'rb') as f:
    site_1 = pickle.load(f)
site_1 = site_1[site_1['timestamp'].dt.year > 2016]


# In[ ]:


t = test[['building_id', 'meter', 'timestamp']]
t['row_id'] = t.index
site_1 = site_1.merge(t, left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
site_1 = site_1[['meter_reading_scraped', 'row_id']].set_index('row_id').dropna()
submission.loc[site_1.index, 'meter_reading'] = site_1['meter_reading_scraped']
del site_1
gc.collect()


# * Site 2
# # 
# # Thanks [@pdnartreb](https://www.kaggle.com/pdnartreb) and his great dataset.
# # 
# # https://www.kaggle.com/pdnartreb/asu-buildings-energy-consumption

# In[ ]:


site_2 = pd.read_csv('../input/asu-buildings-energy-consumption/asu_2016-2018.csv', parse_dates = ['timestamp'])
site_2 = site_2[site_2['timestamp'].dt.year > 2016]


# In[ ]:


t = test[['building_id', 'meter', 'timestamp']]
t['row_id'] = t.index
site_2 = site_2.merge(t, left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
site_2 = site_2[['meter_reading', 'row_id']].set_index('row_id').dropna()
submission.loc[site_2.index, 'meter_reading'] = site_2['meter_reading']
del site_2
gc.collect()


# In[ ]:


site_4 = pd.read_csv('../input/ucb-data-leakage-site-4/site4.csv')


# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:




