#!/usr/bin/env python
# coding: utf-8

# # DonorsChoose - EDA for good
# 
# > Kernal Under construction
# 
# Below kernel will give an overview and idea of data types and columns of the complete dataset.
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import cufflinks as cf

import plotly.offline as py
py.init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objs as go
import plotly.tools as tls 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df_Resources = pd.read_csv("../input/Resources.csv")
df_Schools = pd.read_csv("../input/Schools.csv")
df_Donors = pd.read_csv("../input/Donors.csv")
df_Donations = pd.read_csv("../input/Donations.csv")
df_Teachers = pd.read_csv("../input/Teachers.csv")
df_Projects = pd.read_csv("../input/Projects.csv")
# Any results you write to the current directory are saved as output.


# # Exploring Resources Data

# In[2]:


df_Resources.head()


# In[3]:


df_Resources.info()


# In[15]:


#lets explore the na values
df_Resources.isnull().sum(axis=0)


# # Exploring Donors Data

# In[4]:


df_Donors.head()


# In[5]:


df_Donors.info()


# In[14]:


len(df_Donors['Donor Zip'].unique())


# In[16]:


df_Donors.isnull().sum(axis=0)


# # Exploring Donations Data

# In[6]:


df_Donations.head()


# In[7]:


df_Donations.info()


# In[17]:


df_Donations.isnull().sum(axis=0)


# In[ ]:





# # Exploring Schools Data

# In[8]:


df_Schools.head()


# In[9]:


df_Schools.head()


# In[18]:


df_Schools.isnull().sum(axis=0)


# In[23]:


df_Schools['School State'].unique()


# In[25]:


scCount = df_Schools.groupby(['School State']).count()


# In[ ]:





# In[30]:


scCount[['School ID']].sort_values('School ID').iplot(kind='bar')


# # Exploring Teachers Data

# In[10]:


df_Teachers.head()


# In[11]:


df_Teachers.info()


# In[19]:


df_Teachers.isnull().sum(axis=0)


# In[ ]:





# # Exploring Projects Data

# In[12]:


df_Projects.head()


# In[13]:


df_Projects.info()


# In[20]:


df_Projects.isnull().sum(axis=0)


# In[22]:


df_Projects['Project Resource Category'].unique()


# In[ ]:




