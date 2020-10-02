#!/usr/bin/env python
# coding: utf-8

# # Fast no new cases baseline
# 
# Inspired by the [No New Cases Baseline (COVID-19 Week 1, Global)](https://www.kaggle.com/benhamner/no-new-cases-baseline-covid-19-week-1-global) kernel by @benhamner.<br/>
# This is simplified version.

# In[ ]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', "datadir = Path('/kaggle/input/covid19-global-forecasting-week-1')\n\n# Read in the data CSV files\ntrain = pd.read_csv(datadir/'train.csv')\ntest = pd.read_csv(datadir/'test.csv')\nsubmission = pd.read_csv(datadir/'submission.csv')")


# In[ ]:


train


# In[ ]:


test


# In[ ]:


submission


# In[ ]:


train.rename({'Country/Region': 'Country', 'Province/State': 'Province'}, axis=1, inplace=True)
test.rename({'Country/Region': 'Country', 'Province/State': 'Province'}, axis=1, inplace=True)


# In[ ]:


train['country_province'] = train['Country'].fillna('') + '/' + train['Province'].fillna('')
test['country_province'] = test['Country'].fillna('') + '/' + test['Province'].fillna('')


# In[ ]:


train['country_province'].unique()


# # Extract last train date information (no-leak)
# 
# Use proper date to ensure no-leak!

# In[ ]:


public_leaderboard_start_date = "2020-03-12"
last_public_leaderboard_train_date = "2020-03-11"
public_leaderboard_end_date  = "2020-03-26"


# In[ ]:


last_date_train = train.query("Date == @last_public_leaderboard_train_date")


# # Check histogram of last train date...

# In[ ]:


sns.countplot(x='ConfirmedCases', data=last_date_train)


# In[ ]:


last_date_train['ConfirmedCases'].value_counts()


# In[ ]:


len(train['country_province'].unique()), len(test['country_province'].unique())


# # Prediction and submission
# 
# Just by inserting last train date values to test DataFrame, assuming no new cases happen.

# In[ ]:


test = test.merge(last_date_train[['country_province', 'ConfirmedCases', 'Fatalities']],
                  on='country_province')


# In[ ]:


submission


# In[ ]:


this_submission = test[['ForecastId', 'ConfirmedCases', 'Fatalities']]
this_submission.to_csv('submission.csv', index=False)

