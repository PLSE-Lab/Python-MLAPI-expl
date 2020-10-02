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
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


a=pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')


# In[ ]:


# This is my first analysis ; So we will start firt by analysing How much different regions we have and their corresponding country


# In[ ]:


a.head()


# In[ ]:


a['Region'].value_counts()  # Hence we have these regions in our dataset :)


# In[ ]:


# Here's the visualisation of these Regions
plt.figure(figsize=(17,6))
order = a['Region'].value_counts(ascending=False).index
region=sns.countplot(x='Region',data=a,order=order)
region.set_title('Different Regions in the World')


# In[ ]:


# Now we have viewed different regions in the world , let's see the visulation of different regions one by one
a[(a['Region']=='Asia')]['Country'].value_counts()


# In[ ]:


b=a[ (a['Country']=='India') & (a['Year']<2020) ]
b.head(1)


# In[ ]:


plt.figure(figsize=(15,6))
p=sns.lineplot(x=b['Year'],y=b['AvgTemperature'])
p.set_title('Year Vs Avg Temprature')


# In[ ]:


b['City'].unique()


# **Now we will evaluate diiferent cities temprature Vs corresponding year**

# In[ ]:


plt.figure(figsize=(15,6))
sns.pointplot(x=b['Year'],y=b['AvgTemperature'],hue=b['City'])
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
# We can also plot the graph for individual city by setting particular city name as hue :)


# Now we will analyze each city to a much broader view :)

# Let us Analyze our capital city 

# In[ ]:


plt.figure(figsize=(40,10))
i=sns.FacetGrid(data=b,col='City',col_wrap= 2, height= 4, aspect= 3, margin_titles=True)
i.map(sns.pointplot,'Year','AvgTemperature')


# In[ ]:


plt.figure(figsize=(40,10))
i=sns.FacetGrid(data=b,col='City',margin_titles=True)
i.map(sns.distplot,'AvgTemperature')


# In[ ]:




