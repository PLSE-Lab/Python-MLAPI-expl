#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

Uploading the dataset
# In[ ]:


df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# From the above information we see we have null values in our dataset.
# 
# Check how many null values 
# 

# In[ ]:


df.isnull().sum()


# See this as graphically.
# import the visualization library
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis',cbar=False)


# Lets see in term's of percentage.

# In[ ]:


100*df.isnull().sum()/len(df)


# Remove the null value Row's

# In[ ]:


df=df.dropna()


# Let's see graphically whether we have null value or not

# In[ ]:


sns.heatmap(df.isnull(),cmap='viridis',cbar=False,yticklabels=False)


# Now we does not have null values in our dataset

# From the above dataset we see 'Uniq Id' column is not neccesary for us,so we drop this column.

# In[ ]:


df=df.drop('Uniq Id',axis=1)


# 'Crawl Timestamp' column gives the idea about when the uniq id for naukri.com is created.so, this information is also not neccesary,so we can remove this column. 

# In[ ]:


df=df.drop('Crawl Timestamp',axis=1)


# Let's see the Job Title column and find out the TOP 10 Job's

# In[ ]:


df['Job Title'].value_counts().head(10)


# Let's see the Salary given by the Recruiter.

# In[ ]:


df['Job Salary'].value_counts()


# The salary column does not give the any idea about salary of emplyoee. 

# Let's see the top 10 job location 

# In[ ]:


df['Location'].value_counts().head(10)


# Top 5 job in Bengaluru

# In[ ]:


df[df['Location']=='Bengaluru']['Job Title'].value_counts().head()


# Top 5 industry in Bengaluru

# In[ ]:


df[df['Location']=='Bengaluru']['Industry'].value_counts().head()


# Top 5 Functional area in Bengaluru

# In[ ]:


df[df['Location']=='Bengaluru']['Functional Area'].value_counts().head()


# Bengaluru is also known as IT- HUB city of india

# Let's see the Role column

# In[ ]:


df['Role'].value_counts().head(10)


# The above result show Software Developer is the most popular Role

# Let's see the Functional Area of Software Developer

# In[ ]:


df[df['Role']=='Software Developer']['Functional Area'].value_counts().head()


# Let's see which industry requiers more software developer.

# In[ ]:


df[df['Role']=='Software Developer']['Industry'].value_counts().head()


# See the IT Industry require more Software Developer

# In[ ]:




