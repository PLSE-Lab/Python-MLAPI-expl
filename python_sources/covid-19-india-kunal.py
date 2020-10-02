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


ageGroupDetailDS = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
ageGroupDetailDS.head()


# In[ ]:


ageGroupDetailDS.columns


# In[ ]:


hospitalBedsIndiaDS = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')
hospitalBedsIndiaDS.head()


# In[ ]:


hospitalBedsIndiaDS.columns


# In[ ]:


covid19IndiaDS = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
covid19IndiaDS.head()


# In[ ]:


ICMRTestingDetailsDS  = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingDetails.csv')
ICMRTestingDetailsDS.head()


# In[ ]:


population_india_census2011DS = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')
population_india_census2011DS.sample(10)


# In[ ]:


population_india_census2011DS.columns


# In[ ]:


#Lets start with one by one 
ageGroupDetailDS = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
ageGroupDetailDS.head(10)


# In[ ]:


#Confirm the data Types
ageGroupDetailDS.dtypes


# In[ ]:


#get the shape
ageGroupDetailDS.shape


# In[ ]:


#Create a Bar Graph depend upon AgeGroudp vs Total Cases
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
sns.set(style="whitegrid")
ax = sns.barplot(x="AgeGroup", y="TotalCases", data=ageGroupDetailDS)  


# In[ ]:


#Lets Play with Covid 19 India
covid19IndiaDS.info()


# In[ ]:


covid19IndiaDS.describe()


# In[ ]:


covid19IndiaDS.shape


# In[ ]:


covid19IndiaDS.columns


# In[ ]:


#check if any null value present. Otherwise we don't see any null value in Aboave script
covid19IndiaDS.isnull()


# In[ ]:


#We are confirmed that we dont have any null value in current data type


# In[ ]:


#Now check the most cases in our states
covid19IndiaDS.sample(10)


# In[ ]:


covid19IndiaDS['totalIndian'] = (covid19IndiaDS['Cured']+covid19IndiaDS['Deaths']+covid19IndiaDS['Confirmed'])


# In[ ]:


#Total Number of State wise
covid19IndiaDS.groupby('State/UnionTerritory')['totalIndian'].sum()


# In[ ]:


covid19IndiaDS['State/UnionTerritory'].value_counts()


# In[ ]:


covid19IndiaDS['totalIndian'].plot(kind='hist')
plt.title('Histogram of Age')
plt.xlabel('totalIndian')


# In[ ]:


import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
covid19IndiaDS.head()


# In[ ]:


df_covid = covid19IndiaDS[covid19IndiaDS['Date']>'24/02/20']

fig = px.bar(df_covid, x=df_covid['State/UnionTerritory'],y='totalIndian', color='totalIndian', height=600)
py.iplot(fig)


# In[ ]:


sns.set(style="whitegrid")
ax = sns.catplot(x="State/UnionTerritory", y="totalIndian", data=df_covid,kind="bar",height=30, aspect=.9)  


# In[ ]:




