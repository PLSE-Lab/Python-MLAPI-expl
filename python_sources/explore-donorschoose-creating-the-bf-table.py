#!/usr/bin/env python
# coding: utf-8

# In[4]:


# https://www.kaggle.com/donorschoose/io/data

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


# ## Importing CSV's and setting up index columns
# - The idea is to start the analysis by consolidating as much information as possible into one mega dataframe to start running some EDA.
# - The **donations** table is the transactional repository so it will be the starting point.
# - **projects** has a Project Essay column that apparently has a lot of text, so in order to make the mega DF faster, this was dropped (no point in keeping it for now)

# In[5]:


donors = pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID')
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID")
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID")

projects_light = projects.drop(columns='Project Essay',axis=1)
donations = pd.read_csv('../input/Donations.csv')


# In[6]:


megadf = donations.join(projects_light,on='Project ID',how='left')
megadf = megadf.join(donors,on='Donor ID')
megadf = megadf.join(schools,on='School ID')


# In[7]:


megadf


# In[ ]:




