#!/usr/bin/env python
# coding: utf-8

# **ASHRAE - Great Energy Predictor III**
# # SUBMISSION VISUALIZATION
# 
# This notebook is a tool to graphically compare traininig and submissions data. Just load 1, 2 or more submissions 

# # imports and loading

# In[ ]:


KAGGLE_MODE = True  # drives file loading


# In[ ]:


import numpy as np
import pandas as pd 

from tqdm import tqdm, tqdm_notebook
import gc
import zipfile
import os
import datetime

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


get_ipython().run_cell_magic('time', '', '# load files for test\nif KAGGLE_MODE:\n    building_df = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")\n#     weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")\n#     weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")\n    train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")\n    test =  pd.read_csv("../input/ashrae-energy-prediction/test.csv")\nelse:\n    zf = zipfile.ZipFile(\'./ashrae-energy-prediction.zip\') \n    building_df = pd.read_csv(zf.open(\'building_metadata.csv\'))\n    train = pd.read_csv(zf.open(\'train.csv\'))\n#     weather_train = pd.read_csv(zf.open(\'weather_train.csv\'))\n#     weather_test = pd.read_csv(zf.open(\'weather_test.csv\'))\n    test =  pd.read_csv(zf.open(\'test.csv\'))')


# In[ ]:


def process_df(df):
    # adding timestamp, building_id and meter to submission data
    if not 'timestamp' in df.columns:
        df = pd.merge (df, test, on='row_id')
        df = df.drop(columns=['row_id'])

    # transforming timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour.astype(np.uint8)
    
    # aggregating data EDIT THIS PART TO GENERATE MORE STATISTICS
    df_daily = df.groupby(['building_id', 'date', 'meter']).agg({'meter_reading':['std','mean','count']}).reset_index()
    df_daily[('meter_reading','mean')] = np.log1p(df_daily[('meter_reading','mean')])
    df_daily[('meter_reading','std')] = np.log1p(df_daily[('meter_reading','std')])
    return df_daily


# In[ ]:


# loading and processing train data
dfs = [process_df(train)]
del train
gc.collect()
dfs[0].shape


# ### Load and augment submission

# Insert your submission names here. Any number of them, but above 3 will be crowded ...
# 
# For this example I used output from 2 of the popular notebooks:
# - baseline from Konstantin Yakovlev (https://www.kaggle.com/kyakovlev/ashrae-baseline-lgbm)
# - higher-scoring Half and Half one from Vopani (https://www.kaggle.com/rohanrao/ashrae-half-and-half)

# In[ ]:


# ls ../input/ -l


# In[ ]:


# Manually edit path, filenames and names. Names should correspond to filenames.
sub_path = '../input/'
sub_filenames = ["ashrae-baseline-lgbm/submission.csv","ashrae-half-and-half/submission.csv" ]
sub_names = ["baseline","half-and-half" ]


# In[ ]:


for sub_filename in sub_filenames:
    print(f'adding submission {sub_path + sub_filename}: ', end='')
    print('loading...', end='')
    sub = pd.read_csv(sub_path + sub_filename)
    print(', processing...')
    dfs.append(process_df(sub))
    del sub
    gc.collect()
print('done!')


# In[ ]:


del test
gc.collect()


# In[ ]:


# function to generate chart data and build charts
def chart_submissions (building_id, meter=0):
    titles = ['2016 train data', *sub_names]
    tmp_df = [df[(df.building_id == building_id) & (df.meter == meter)]                 [['date', 'meter_reading']].set_index('date') for df in dfs]
    if tmp_df[0].shape[0]:

        fig, axes = plt.subplots(nrows=len(tmp_df), figsize=(18, 2+len(tmp_df)*2),)
        fig.suptitle(f'Building {building_id}, meter {meter}', fontsize=18, y = 0.94)
        max_y = np.concatenate([df[('meter_reading','mean')].values for df in tmp_df]).max() *  1.05

        for i in range (3):
#             tmp_df[i][('meter_reading', 'std')].plot(ax=axes[i], label='log_std')
            tmp_df[i][('meter_reading', 'mean')].plot(ax=axes[i], label='log mean')
#             tmp_df[i][('meter_reading', 'count')].plot(ax=axes[i], label='count')
            axes[i].axvline(x=datetime.date(2018, 1, 1),  color='k', linestyle='--')

            axes[i].legend()
            axes[i].set_title(titles[i], fontsize=16, y = 0.8)
            axes[i].set_ylim(0,max_y)
        building_df[building_df.building_id==building_id]
    else:
        print (f"Building_id={building_id}, Meter={meter} combination not present..." )


# Set desired building ID and meter ID in the cell below and see the resulting charts. Repeat as needed.

# In[ ]:


chart_submissions (building_id=0, meter=0)
# some interesting ones: 107, 869, 1001, 0, 888


# The example above shows how the missing data in building 0 was affecting forecast in baseline (score 1.25) and in a better developed kernel (score 1.10).
