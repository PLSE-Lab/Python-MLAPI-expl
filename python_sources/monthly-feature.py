#!/usr/bin/env python
# coding: utf-8

# # Hacking AvSigVersion
# 
# This notebook is slightly different approach to [Chris' kernel](https://www.kaggle.com/cdeotte/private-leaderboard-0-703). This idea is NOT the same, just an alternative. But upon re-exploring my hacky feature, and EDA, I have realized why I got such a high private LB score, despite not having the gumption, chutzpah, or what-have-you, to stay true to avoiding using public LB as a progress gauge and fell prey to greedy desire to land on the leaderboard. Even after countless warning signs from [here](https://www.kaggle.com/tunguz/ms-malware-adversarial-validation), and [here](https://www.kaggle.com/rquintino/2-months-train-1-month-public-1-day-private) as well as in my own analysis below.
# 
# In this notebook I explore, how I used AvSigVersion to improve Private LB while decreasing Public LB. After getting inspired by  [this kernel](https://www.kaggle.com/cdeotte/time-split-validation-malware-0-68).
# 
# My reasoning behind hacking the AvSigVersion through this approach was that the training data only contains a small sample of months in October, November which is heavily comprised of the private dataset that was discovered in [this kernel](https://www.kaggle.com/rquintino/2-months-train-1-month-public-1-day-private). 
# 
# Below is some EDA I had performed around the time the two kernels above were released. Had I stuck to simple feature engineering, avoided overfitting/chasing the LB, I would have scored 0.663 - 0.688 using a simple lightgbm model. I too fell pray to chasing the Public LB and left caution to the wind. I hope some will will find my hindsight in this kernel useful.
# 
# Special thanks to @cdeotte, **Chris Deotte** for your dedication to publishing your kernels throughout this competition and inspiring this kernel submission. I would have never explored the hacky feature if it weren't for your comments on how you didn't drop unstable features in your time-slit-validation kernel.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime 
import numpy as np


# In[ ]:


get_ipython().run_cell_magic('time', '', "dtypes = {}\ndtypes['MachineIdentifier'] = 'str'\ndtypes['AvSigVersion'] = 'category'\ndtypes['HasDetections'] = 'int8'\n\n# LOAD TRAIN & TEST DATA\ntrain = pd.read_csv('../input/microsoft-malware-prediction/train.csv', usecols=list(dtypes.keys()), dtype=dtypes)\ntest = pd.read_csv('../input/microsoft-malware-prediction/test.csv', usecols=list(dtypes.keys())[0:-1], dtype=dtypes)\n\n# Load AvSigVersion Dates\ndates1 = np.load('../input/avgsig/train_AvSigVersion.npy')[()]\ndates2 = np.load('../input/avgsig/train_AvSigVersion2.npy')[()]\ndates3 = np.load('../input/avgsig/test_AvSigVersion.npy')[()]")


# In[ ]:


# process the dates, create a dictionary to store all dates
date = {}
for key, value in zip(dates1.keys(), dates1.values()):
    if key not in date.keys():
        date[key] = value
        
for key, value in zip(dates2.keys(), dates2.values()):
    if key not in date.keys():
        date[key] = value
        
for key, value in zip(dates3.keys(), dates3.values()):
    if key not in date.keys():
        date[key] = value


# In[ ]:


# function for stripping month, year, day, week data. try/except since there are missing dates
def strip_month(feature):
    try:
        return datetime.strptime(feature, '%b %d,%Y %I:%M %p UTC').month
    except: 
        return 0

def strip_year(feature):
    try:
        return datetime.strptime(feature, '%b %d,%Y %I:%M %p UTC').year
    except: 
        return 0

def strip_day(feature):
    try:
        return datetime.strptime(feature, '%b %d,%Y %I:%M %p UTC').day
    except: 
        return 0

def strip_week(feature):
    try:
        # be careful, there is a leap week. apparently there is a 53rd week!
        return datetime.strptime(feature, '%b %d,%Y %I:%M %p UTC').isocalendar()[1]
    except: 
        return 0

# binary featurization
def month11(feature):
    return 1 if feature == 11 else 0

def month10(feature):
    return 1 if feature == 10 else 0


# In[ ]:


get_ipython().run_cell_magic('time', '', "# create a numerical feature that includes only months October/November\ntrain['Month'] = train['AvSigVersion'].map(date).apply(strip_month)\ntest['Month'] = test['AvSigVersion'].map(date).apply(strip_month)\ntrain['Year'] = train['AvSigVersion'].map(date).apply(strip_year)\ntest['Year'] = test['AvSigVersion'].map(date).apply(strip_year)\ntrain['Day'] = train['AvSigVersion'].map(date).apply(strip_day)\ntest['Day'] = test['AvSigVersion'].map(date).apply(strip_day)\ntrain['Week'] = train['AvSigVersion'].map(date).apply(strip_week)\ntest['Week'] = test['AvSigVersion'].map(date).apply(strip_week)\n\n# binary features, specifically used to hack those months (for private LB)\ntrain['AvSigMonth_10'] = train['Month'].apply(month10).astype('int8')\ntest['AvSigMonth_10'] = test['Month'].apply(month10).astype('int8')\ntrain['AvSigMonth_11'] = train['Month'].apply(month11).astype('int8')\ntest['AvSigMonth_11'] = test['Month'].apply(month11).astype('int8')")


# In[ ]:


# Explore Detection Counts by Month
plt.figure(figsize=(16,8))
sns.countplot(train['Month'], hue=train['HasDetections'])


# In[ ]:


plt.figure(figsize=(16,8))
sns.countplot(test['Month'])


# In[ ]:


# Zoom into detection counts for Nov - Dec
subset_train = train[train['Month'] >= 10]
sns.countplot(subset_train['Month'], hue=subset_train['HasDetections'])


# Other thoughts regarding the nature of low sample rate of October, November, December months. I thought about upsampling the months at the time (1.5months ago) which only in hindsight would have been a brilliant idea. However, I was risk adverse and unsure of my wild ideas. Everything in my gut was telling me not to follow the crowd and that true detection rates were much lower than led to believe as indicated by my EDA.
# 
# Below is a screenshot of submissions using the hacky features.
# 
# ![image](https://i.imgur.com/Nf2xE6h.png)
