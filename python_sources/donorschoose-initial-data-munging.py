#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")


# ## Importing CSV's and setting up index columns
# - The idea is to start the analysis by consolidating as much information as possible into one mega dataframe to start running some EDA.
# - The **donations** table is the transactional repository so it will be the starting point.
# - **projects** has a Project Essay column that apparently has a lot of text, so in order to make the mega DF faster, this was dropped (no point in keeping it for now)

# In[ ]:


donations = pd.read_csv('../input/Donations.csv')
print(donations.shape)
donations.head()


# In[ ]:


pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID',nrows=6)


# In[ ]:


pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID",nrows=3)


# In[ ]:


pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID",nrows=3)


# In[ ]:


pd.read_csv('../input/Teachers.csv', low_memory=False,index_col='Teacher ID',nrows=4)


# ### Resources has many bad lines / escapers
# * unit price/amount not always number - should be cleaned so we can get the sum cost

# In[ ]:


resources = pd.read_csv('../input/Resources.csv', index_col="Project ID",error_bad_lines=False,warn_bad_lines = False)
print(resources.shape)
# resources["sum_resource_price"] = resources["Resource Quantity"]*resources["Resource Unit Price"]
resources.head()


# In[ ]:


resources[["Resource Quantity","Resource Unit Price"]].dtypes


# In[ ]:


# resources.describe(include="all")


# In[ ]:


# donors = pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID')
# projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID")
# schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID")

# projects_light = projects.drop(columns='Project Essay',axis=1)


# In[ ]:


df = donations.join(pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID"),on='Project ID',how='left')
df = df.join(pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID'),on='Donor ID')
df = df.join(pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID"),on='School ID')
df = df.join(pd.read_csv('../input/Teachers.csv', low_memory=False,index_col='Teacher ID'),on='Teacher ID')

# df = df.join(resources,on="Project ID")


# In[ ]:


print(df.shape)
print(df.describe())
df.head()


# In[ ]:





# In[ ]:


df.to_csv("merged_donorsChoose_v1.csv.gz",compression="gzip")

