#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('dark_background')

df = pd.read_csv("/kaggle/input/airplane-accidents-severity-dataset/train.csv")

df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# ## Understanding every column

# ### Safety Score

# In[ ]:


sns.distplot(df.Safety_Score, color="white", kde=False)


# In[ ]:


sns.barplot(df.Safety_Score, df.Severity)


# #### Lesser the Safety score higher the damage will be

# ## Days Since Inspection

# In[ ]:


sns.distplot(df.Days_Since_Inspection, kde=False)


# In[ ]:


sns.barplot(df.Days_Since_Inspection, df.Severity)


# #### It is strange to see that lesser the days since inspection is done higher the damage.

# ## Total Safety complaints

# In[ ]:


sns.swarmplot(y = df.Severity, x =  df.Total_Safety_Complaints)


# ## Control Metric

# In[ ]:


sns.barplot(y = df.Severity, x= df.Control_Metric)


#  ## Terbulance in gforce

# In[ ]:


sns.barplot(df.Turbulence_In_gforces, df.Severity)


# ## Terbulance vs Control Metric

# In[ ]:


sns.scatterplot(y = df.Turbulence_In_gforces, x = df.Control_Metric)


#  ## Accident Type code

# In[ ]:


sns.barplot(df.Accident_Type_Code, df.Severity)


# ## Maximum elevation
# 

# In[ ]:


sns.barplot(df.Max_Elevation, df.Severity)


# ## Voilations

# In[ ]:


sns.barplot(df.Violations, df.Severity)


# ## Adverse_Weather_Metric

# In[ ]:


sns.barplot(df.Adverse_Weather_Metric, df.Severity)


# In[ ]:


df = df.drop(['Accident_ID'], axis = "columns")


# In[ ]:


sns.pairplot(df)


# Here is just the visualization of every field, If you like please upvote it.

# In[ ]:




