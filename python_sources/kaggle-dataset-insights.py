#!/usr/bin/env python
# coding: utf-8

# ## Kaggle Dataset insights
# In this notebook , I have done some basic exploration of meta data of datasets from Kaggle & try to get some insights like which average number of votes per dataset,total datasets, votes to download ratio etc . Please review & provide any other insight you are looking for . I will keep updating .
# 
# Below is summary of all insights . Please check code for more details :
# 
# ### Insights
# #### - There are total 24855 datasets.
# 
# #### - 9.32 average votes per dataset.
# 
# #### - 135444 notebooks for all the datasets , 5.44 is average notebook per kernel ( we can exclude 1 notebook which is starter kernel from kaggle for every dataset).
# 
# #### - Alarming votes to download ratio . Just 2 % . It means most of the folks download without upvoting . I appeal to Kaggle team to upvote whenever someone downloads the dataset.thoughts ?
# 
# #### - Most 10 popular categories or tags of datasets . It accounts for 16.6 K datasets out of approx total 24.8 K datasets.
# 
# - business 3436
# - natural and physical sciences 2124
# - computing 2081
# - arts and entertainment 1887
# - computer science 1545
# - reference 1238
# - statistics 1213
# - socrata 1114
# - internet 1050
# - economics 979 
# 
# #### - 12.8 K users have uploaded datasets . This number is expected to increase rapidly since Kaggle has introduced dataset as progression feature .
# 
# #### - Year on year rise of Kaggle dataset counts:
# 
#  - 2015    10   
#  - 2016    456  
#  - 2017    3402 
#  - 2018    10766
#  - 2019    10221

# In[ ]:


# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
pd.set_option('display.max_colwidth', -1)


# In[ ]:


# Load data

dataset_df = pd.read_csv("../input/meta-kaggle/Datasets.csv")
dataset_votes = pd.read_csv('../input/meta-kaggle/DatasetVotes.csv')
datasources = pd.read_csv('../input/meta-kaggle/Datasources.csv')
datasources_object = pd.read_csv('../input/meta-kaggle/DatasourceObjects.csv')
user = pd.read_csv('../input/meta-kaggle/Users.csv')


# In[ ]:


# lets see total number of datasets.
dataset_df.shape


# In[ ]:


# explore few rows
dataset_df.head(3)


# In[ ]:


# Let's check average of total votes per dataset.
dataset_df['TotalVotes'].mean()


# In[ ]:


dataset_df['TotalKernels'].sum() , dataset_df['TotalKernels'].mean() 


# ### Observations:
# #### There are total 24855 datasets.
# #### 9.32 votes per dataset.
# #### 135444 notebooks for all the datasets , 5.44 is average notebook per kernel ( we can exlude 1 notebook which is starter kernel from kaggle).

# In[ ]:


# Let's find dataset with max votes.

dataset_df[dataset_df['TotalVotes'] == dataset_df['TotalVotes'].max()]


# In[ ]:


# lets check which user has most upvotes in a dataset
user[user['Id'] == 14069]


# In[ ]:


# Lets check which dataset got max votes
datasources_object[datasources_object.DatasourceVersionId == 23502.0]


# #### Credit card fraud detection is the most successful dataset with 4137 votes and 2259 kernels . It is uploaded by Andrea and downloaded 174003 times.

# In[ ]:


# Let's check which tags have most datasets
tags = pd.read_csv("../input/meta-kaggle/Tags.csv")


# In[ ]:


tags.groupby('Name')['DatasetCount'].sum().sort_values(ascending = False).head(10)


# In[ ]:


top_10_tags = tags.groupby('Name')['DatasetCount'].sum().sort_values(ascending = False).head(10).reset_index()


# In[ ]:


# Bar chart using matplotlib

plt.figure(figsize = (20,10))
plt.bar(top_10_tags['Name'],top_10_tags['DatasetCount'])
plt.xlabel('Tag Name',fontsize = 15)
plt.ylabel('Dataset Counts',fontsize = 15)
plt.xticks(top_10_tags['Name'],  fontsize=10, rotation=30)
plt.title(' Dataset Tags Popularity',fontsize = 30)
plt.show()


# ## Observation:
# #### We observed that below 10 tags account for 16.6 K out of 24 K datasets .
# 
# * business 3436
# * natural and physical sciences 2124
# * computing 2081
# * arts and entertainment 1887
# * computer science 1545
# * reference 1238
# * statistics 1213
# * socrata 1114
# * internet 1050
# * economics 979

# In[ ]:





# In[ ]:


### How many users have uploaded dataset
dataset_df['CreatorUserId'].nunique()


# In[ ]:


# Let's explore Kaggle dataset year by year

# change creation date to datetime
dataset_df['CreationDate'] = pd.to_datetime(dataset_df['CreationDate'] )


# In[ ]:


# extract year
dataset_df['Year'] = dataset_df['CreationDate'].dt.year


# In[ ]:


# Year on year count of datasets . It is expected to increase dramatically.
dataset_df.groupby('Year')["Id"].count()


# In[ ]:


year_ds = dataset_df.groupby('Year')["Id"].count().reset_index()
plt.plot(year_ds['Year'],year_ds['Id'])


# In[ ]:


user.head()


# In[ ]:




