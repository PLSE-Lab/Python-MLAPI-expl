#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_df = pd.read_csv('/kaggle/input/sf-crime/train.csv')
test_df = pd.read_csv('/kaggle/input/sf-crime/test.csv')

sample_submission = pd.read_csv('/kaggle/input/sf-crime/sampleSubmission.csv')


# ## Initial Inspection

# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


test_df.head()


# In[ ]:


test_df.shape


# The model will need to generalize crime categories based on location information (PdDistrict, Address, X/Y). The model will use training data that includes descriptions, resolutions, and categories of crimes based on location data. Inputs for prediction will be limited to datetime and location information, and the model will need to predict a category of crime.

# ## Calculate # of Missing Values by Column

# In[ ]:


train_df.isnull().sum()


# ## Explore Features

# ### Dates

# Convert to datetime object

# In[ ]:


import time
from datetime import datetime

train_df['Dates'] = train_df['Dates'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))


# In[ ]:


train_df['Dates'].describe()


# ### DayOfWeek

# In[ ]:


train_df['DayOfWeek'].value_counts()


# In[ ]:


day_count = train_df['DayOfWeek'].value_counts()


# In[ ]:


plt.figure(figsize=(12,8))

plt.title('# of Crimes Observed by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('# of Crimes')

sns.barplot(day_count.index, day_count.values)

plt.show()


# ### Category

# In[ ]:


train_df['Category'].value_counts()


# In[ ]:


train_df['Category'].value_counts().shape


# The Category column is a categorical variable (surprise!) with 39 possible values. 
# 
# Notes:
# 
# * "OTHER OFFENSES" appears to to be a catch-all and is the second most observed value. 
# * "TREA" is Trespassing and Loitering in Industrial Area

# In[ ]:


cat_count = train_df['Category'].value_counts()


# In[ ]:


plt.figure(figsize=(16,8))

plt.title('Top 20 # of Crimes Observed by Crime Category')
plt.xlabel('Category')
plt.ylabel('# of Crimes')
plt.xticks(rotation=45, horizontalalignment='right')

sns.barplot(cat_count.index[:20], cat_count.values[:20])

plt.show()


# ### Descript

# In[ ]:


train_df['Descript'].value_counts().shape


# Descript has 879 potential values. 

# In[ ]:


desc_count = train_df['Descript'].value_counts()


# In[ ]:


plt.figure(figsize=(16,8))

plt.title('Top 20 # of Crimes Observed by Crime Description')
plt.xlabel('Descript')
plt.ylabel('# of Crimes')
plt.xticks(rotation=45, horizontalalignment='right')

sns.barplot(desc_count.index[:20], desc_count.values[:20])

plt.show()


# ### PdDistrict

# In[ ]:


train_df['PdDistrict'].value_counts()


# In[ ]:


pd_count = train_df['PdDistrict'].value_counts()


# In[ ]:


plt.figure(figsize=(16,8))

plt.title('Top 20 # of Crimes Observed by Police District')
plt.xlabel('PdDistrict')
plt.ylabel('# of Crimes')
plt.xticks(rotation=45, horizontalalignment='right')

sns.barplot(pd_count.index[:20], pd_count.values[:20])

plt.show()


# ### Resolution

# In[ ]:


train_df['Resolution'].value_counts()


# In[ ]:


res_count = train_df['Resolution'].value_counts()


# In[ ]:


plt.figure(figsize=(16,8))

plt.title('Top 20 # of Crimes Observed by Crime Resolution')
plt.xlabel('Resolution')
plt.ylabel('# of Crimes')
plt.xticks(rotation=45, horizontalalignment='right')

sns.barplot(res_count.index[:20], res_count.values[:20])

plt.show()


# ### Address

# In[ ]:


train_df['Address'].value_counts()[:50]


# In[ ]:


add_count = train_df['Address'].value_counts()


# In[ ]:


plt.figure(figsize=(16,8))

plt.title('Top 20 # of Crimes Observed by Address')
plt.xlabel('Address')
plt.ylabel('# of Crimes')
plt.xticks(rotation=45, horizontalalignment='right')

sns.barplot(add_count.index[:20], add_count.values[:20])

plt.show()


# There are >23,000 addresses observed, but there are a lot of crimes reported at the top of the list. 
# 
# This column should probably be procesed based on street names. Split on '/' or 'of' if 'Block' in row.
# 
# Explore discussion to see how people handled the two different formats for logging address. There is a potential for multiple address labels applying to the same general location. 
# 
# For example, the 200 block of Jones St. is beteween Turk St. and Eddy St. 

# In[ ]:


train_df[train_df['Address'] == '200 Block of JONES ST']


# In[ ]:


train_df[train_df['Address'] == 'TURK ST / JONES ST']


# In[ ]:


train_df[train_df['Address'] == 'EDDY ST / JONES ST']


# It may be useful to find some actual intersection coordinates and see how far away from an intersection a crime may be observed to be logged at the cross street as opposed to the block number.

# In[ ]:




