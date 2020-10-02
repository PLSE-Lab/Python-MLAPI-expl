#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://www.kaggle.com/donorschoose/io/data

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


# In[3]:


import os
# print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")


# ## Importing CSV's and setting up index columns
# - The idea is to start the analysis by consolidating as much information as possible into one mega dataframe to start running some EDA.
# - The **donations** table is the transactional repository so it will be the starting point.
# - **projects** has a Project Essay column that apparently has a lot of text, so in order to make the mega DF faster, this was dropped (no point in keeping it for now)

# In[5]:


donations = pd.read_csv('../input/Donations.csv')
print(donations.shape)
donations.head()


# In[ ]:


# many 0 donations? 
donations["Donation Amount"].describe()


# In[6]:


pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID',nrows=6)


# In[7]:


pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID",nrows=3)


# In[8]:


pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID",nrows=3)


# In[9]:


pd.read_csv('../input/Teachers.csv', low_memory=False,index_col='Teacher ID',nrows=4)


# ### Resources has many bad lines / escapers
# * unit price/amount not always number - should be cleaned so we can get the sum cost

# In[11]:


resources = pd.read_csv('../input/Resources.csv', index_col="Project ID",error_bad_lines=False,warn_bad_lines = False)
print(resources.shape)
resources["sum_resource_price"] = resources["Resource Quantity"]*resources["Resource Unit Price"]
resources.head()


# In[ ]:


# resources.describe(include="all")


# In[ ]:


# donors = pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID')
# projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID")
# schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID")

# projects_light = projects.drop(columns='Project Essay',axis=1)


# In[14]:


df = donations.join(pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID"),on='Project ID',how='left')
df = df.join(pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID'),on='Donor ID')
df = df.join(pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID"),on='School ID')
df = df.join(pd.read_csv('../input/Teachers.csv', low_memory=False,index_col='Teacher ID'),on='Teacher ID')

# df = df.join(resources,on="Project ID") # multiple resources per project


# In[15]:


print(df.shape, "\n")
df.describe()


# In[26]:


print(df.shape)
df.head()


# ## Filter by minimal amount donated
# * low end is 0. 
# * could filter by higher threshhold. 
# * TODO: investigate the zeros. 

# In[28]:


df = df.loc[df["Donation Amount"]>5]
df.shape


# In[29]:


df["count_teacher"] = df.groupby('Teacher ID')['Project ID'].transform("count")
df["count_donor"] = df.groupby('Donor ID')['Project ID'].transform("count")
df["count_school"] = df.groupby('School ID')['Project ID'].transform("count")

df[["count_teacher","count_donor","count_school"]].describe()


# ### Most donors give only once
# * hard for recomending - cold start.
# * Lets look at a smaller data subset with multiple donations & occurneces. 
#     * Also look at schools that appeared multiple times - making it easier to characterize them , even without random embeddings/cooccurence. 

# In[30]:


df_multidonor = df.loc[df["count_school"]>2]
print(df_multidonor.shape)

df_multidonor = df_multidonor.loc[df_multidonor["count_teacher"]>2]
print(df_multidonor.shape)

df_multidonor = df_multidonor.loc[df_multidonor["count_donor"]>2]
print(df_multidonor.shape)

df_multidonor[["count_teacher","count_donor","count_school"]].describe()


# In[31]:


df_multidonor["Donation Amount"].describe()


# ## Save

# In[38]:


print(resources.shape)
# resources.loc[resources["Project ID"].isin(df_multidonor["Project ID"])].shape
resources.loc[resources.index.isin(df_multidonor["Project ID"])].shape


# In[ ]:


df_multidonor.to_csv("merged_donorsChoose-multiDonor3_v1.csv.gz",compression="gzip")


# In[ ]:


resources.loc[resources.index.isin(df_multidonor["Project ID"])].to_csv("resources_donorsChoose-multiDonor3_v1.csv.gz",compression="gzip")


# In[ ]:


# df.to_csv("merged_donorsChoose_v1.csv.gz",compression="gzip")

