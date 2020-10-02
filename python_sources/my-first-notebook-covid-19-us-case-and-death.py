#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


us_death=pd.read_csv('../input/uncover/USAFacts/confirmed-covid-19-deaths-in-us-by-state-and-county.csv')
us_death.head()


# In[ ]:


us_case=pd.read_csv('../input/uncover/USAFacts/confirmed-covid-19-cases-in-us-by-state-and-county.csv')
us_case.head()


# In[ ]:


us_case.info()


# In[ ]:


us_death.info()


# In[ ]:


us_case['date'] = pd.to_datetime(us_case['date'])
us_death['date'] = pd.to_datetime(us_death['date'])


# In[ ]:


us_death = us_death.dropna()
us_case = us_case.dropna()


# In[ ]:


us_death = us_death.reset_index()
us_death = us_death.drop('index',axis = 1)
us_case = us_case.reset_index()
us_case = us_case.drop('index',axis = 1)


# In[ ]:


us_case=us_case.rename(columns = {'confirmed':'cases'})


# In[ ]:


us_death.head()


# In[ ]:


us_case.head()


# In[ ]:


# Top 5 covid-19 effected states in US
state_case = us_case.groupby(['state_name'])[['cases']].sum().sort_values('cases',ascending=False)
state_case.head()


# In[ ]:


# 500 or more cases in NY
NY_case = us_case[(us_case.state_name=='NY') & (us_case.cases>=500)].sort_values('date')
NY_case


# In[ ]:


NY = NY_case.pivot_table(
    values='cases',
    index='date',
    columns='county_name',
    aggfunc='sum')
NY


# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(NY_case.date, NY_case.cases)
plt.grid(True)
plt.xlabel('Date' , fontsize=20)
plt.ylabel("Count" , fontsize=20)
plt.title("Confirmed cases in NY",fontsize=30)
plt.show()


# In[ ]:


state_death = us_death.groupby(['state_name'])[['deaths']].sum().sort_values('deaths',ascending=False)
state_death.head()


# In[ ]:


county_death = us_death.groupby(['county_fips','county_name','state_name'])[['deaths']].sum().sort_values('deaths',ascending=False)
county_death[county_death.deaths>50]


# In[ ]:


death_date = us_death.groupby('date')[['deaths']].sum().sort_values('deaths',ascending=False).reset_index()
death_date = death_date[death_date.deaths>=1]
death_date


# In[ ]:


plt.figure(figsize=(23,10))
death_date.plot(x='date',y='deaths',kind='line')
plt.xlabel('Date' , fontsize=20)
plt.ylabel("Count" , fontsize=20)
plt.title("Deaths in US",fontsize=30)
plt.show()


# In[ ]:


NY_death=us_death[(us_death.state_name=='NY') & (us_death.deaths>=5)].sort_values('date')
NY_death.head()


# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(NY_death.date, NY_death.deaths,label="deaths")
plt.grid(True)
plt.xlabel('Date',fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.title("Deaths in NY",fontsize=30)
plt.show()

