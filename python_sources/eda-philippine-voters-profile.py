#!/usr/bin/env python
# coding: utf-8

# ## Philippie Voters Profile (2016)
# EXPLORATORY DATA ANALYSIS NOTEBOOK
# 
# Full data description here: https://www.kaggle.com/aldrinl/philippine-voters-profile
# 
# ![Philippines](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2Ff%2Ff6%2FLabelled_map_of_the_Philippines_-_Provinces_and_Regions.png%2F825px-Labelled_map_of_the_Philippines_-_Provinces_and_Regions.png&f=1&nofb=1)

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')


# In[ ]:


voters = pd.read_csv('/kaggle/input/philippine-voters-profile/2016_voters_profile.csv')
voters.head()


# Data Preparations

# In[ ]:


voters['literacy'] = voters['literacy'].str.replace('%','').astype(float)

sex_cols = ['male', 'female']
age_cols = ['17-19', '20-24', '25-29','30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64','65-above']
civil_status_cols = ['single', 'married', 'widow', 'legally_seperated']


# For this version of the notebook. We will first do an analysis at the regional level.

# In[ ]:


per_region = voters.groupby('region').sum()


# ## Voters' Age Distribution per Region

# In[ ]:


nrow=3
ncol=6
fig, axes = plt.subplots(nrow, ncol,figsize=(20,10),sharex=True,sharey=True)

reions = per_region.index
count=0
for r in range(nrow):
    for c in range(ncol):
        if(count==len(reions)):
            break
        col = reions[count]
        per_region.loc[col,age_cols].plot(kind='bar',ax=axes[r,c])
        axes[r,c].set_title(col)
        count = count+1


# ## Voters' Sex proportion per Region

# In[ ]:


nrow=3
ncol=6
fig, axes = plt.subplots(nrow, ncol,figsize=(20,10),sharex=True,sharey=True)

reions = per_region.index
count=0
for r in range(nrow):
    for c in range(ncol):
        if(count==len(reions)):
            break
        col = reions[count]
        per_region.loc[col,sex_cols].plot(kind='bar',ax=axes[r,c],color=['blue','pink'])
        axes[r,c].set_title(col)
        count = count+1


# ## Voters' Civil Status Distribution per Region

# In[ ]:


nrow=3
ncol=6
fig, axes = plt.subplots(nrow, ncol,figsize=(20,10),sharex=True,sharey=True)

reions = per_region.index
count=0
for r in range(nrow):
    for c in range(ncol):
        if(count==len(reions)):
            break
        col = reions[count]
        per_region.loc[col,civil_status_cols].plot(kind='bar',ax=axes[r,c])
        axes[r,c].set_title(col)
        count = count+1


# ## Notebook in progress
# #### Do UPVOTE if this notebook is helpful to you in some way :) <br/>COMMENT below any suggetions that can help improve this notebook. TIA
