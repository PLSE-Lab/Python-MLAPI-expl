#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import __version__
import cufflinks as cf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# Let's check whether we are having some missing data or not ?

# In[ ]:


plt.figure(figsize=(12,4))
sns.heatmap(data.isnull(),cmap='Blues')


# Well we are having a little missing data 

# In[ ]:


missing_percent= (data.isnull().sum()/len(data))[(data.isnull().sum()/len(data))>0].sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Percentage':missing_percent*100})
missing_data


# In[ ]:


mis_data= (data.isnull().sum() / len(data)) * 100
mis_data= mis_data.drop(mis_data[mis_data == 0].index).sort_values(ascending=False)
plt.figure(figsize=(16,5))
sns.barplot(x=mis_data.index,y=mis_data)


# In[ ]:


data.describe()


# **Let's start analyzing what we have in our data**

# In[ ]:


data['Job Title'].value_counts()


# In[ ]:


sns.barplot(x=data['Job Title'].value_counts()[0:9],y=data['Job Title'].value_counts()[0:9].index)


# Well from above we can conclude that Job title is related for expanding their business

# In[ ]:


data['Job Salary'].value_counts()


# In[ ]:


sns.barplot(x=data['Job Salary'].value_counts()[0:9],y=data['Job Salary'].value_counts()[0:9].index)


# Well we didn't get a clear info from above plot

# In[ ]:


data['Key Skills']


# In[ ]:


sns.barplot(x=data['Key Skills'].value_counts()[0:10],y=data['Key Skills'].value_counts()[0:10].index, palette="ch:.25")


# In[ ]:


data['Role Category'].value_counts()[0:10]


# In[ ]:


sns.barplot(x=data['Role Category'].value_counts()[0:10],y=data['Role Category'].value_counts()[0:10].index, palette="ch:.25")


# Well from above we can conclude that demand of Programming and design tends to higher than other roles

# Let's see the where the most **IT** companies tends to setup !

# In[ ]:


data['Location'].value_counts()


# In[ ]:


sns.barplot(x=data['Location'].value_counts()[0:10],y=data['Location'].value_counts()[0:10].index)


# Well mostly companies are located in Bengaluru , Mumbai , Pune , Hyderabad and Gurgaon

# In[ ]:


data['Industry']


# In[ ]:


sns.barplot(x=data['Industry'].value_counts()[0:10],y=data['Industry'].value_counts()[0:10].index)


# Well IT companies have been leading for the private jobs in a large quantity and will gonna rule in the future also :)

# Now look at the experience companies expect from the candidates

# In[ ]:


data['Job Experience Required'].value_counts()


# In[ ]:


sns.barplot(x=data['Job Experience Required'].value_counts()[0:10],y=data['Job Experience Required'].value_counts()[0:10].index)


# In[ ]:


data['Functional Area']


# In[ ]:


sns.barplot(x=data['Functional Area'].value_counts()[0:10],y=data['Functional Area'].value_counts()[0:10].index)


# Well Well IT has been leading all across our data

# In[ ]:


data['Role']


# In[ ]:


sns.barplot(x=data['Role'].value_counts()[0:10],y=data['Role'].value_counts()[0:10].index)


# In[ ]:




