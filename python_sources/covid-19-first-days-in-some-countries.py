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


# In[ ]:


import pandas as pd
world = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
world['Date'] = pd.to_datetime(world['Date'])


# In[ ]:


# number of days after the initial spread
days=20
# cutoff is number of cases we considerd the spread start growing exponentially
cutoff=50


# In[ ]:


world['Country/Region'].unique()


# In[ ]:


countries = ['Italy', 'Spain', 'France', 'Germany', 'US', 'Hubei', 'Netherlands']


# In[ ]:


df = pd.DataFrame()
pops = {'Italy': 60480, 'Spain': 46660,'France': 66990, 
        'Germany': 82790, 'UK': 66440, 'US': 327200, 
        'Hubei': 58500,
       'Netherlands': 17180}
df = world[world['Country/Region']=='United Kingdom']
df.reset_index(inplace=True)
df = df[df['Confirmed']>cutoff][['Date','Confirmed','Deaths']]
df['Confirmed'] = df['Confirmed']/pops['UK']
df['Deaths'] = df['Deaths']/pops['UK']
df['Mortality_Rate'] = df['Deaths']/df['Confirmed']
df=df[:days+1]
c='United Kingdom'
df.columns=[c + '_Date', c + '_Confirmed', c + '_Deaths', c + '_Mortality_Rate']
df['id'] = [i for i in range(len(df))]
df.set_index('id', inplace=True)

for c in countries:
    if c == 'France':
        cf = world[world['Province/State']=='France']
    elif c=='US':
        cf = world[world['Country/Region'] == 'US']
        cf = cf.groupby('Date').sum()
    elif c == 'Hubei':
        cf = world[world['Province/State'] == 'Hubei']
        cf = cf.groupby('Date').sum()
    else:
        cf = world[world['Country/Region']==c]
        
    cf.reset_index(inplace=True)
    cf = cf[cf['Confirmed']>cutoff][['Date','Confirmed','Deaths']]
    cf['Confirmed'] = cf['Confirmed']/pops[c]
    cf['Deaths'] = cf['Deaths']/pops[c]
    cf['Mortality_Rate'] = cf['Deaths']/cf['Confirmed']
    

    cf.columns=[c + '_Date', c + '_Confirmed', c + '_Deaths', c + '_Mortality_Rate']
    cf = cf[:days+1]
    cf['id'] = [i for i in range(len(cf))]
    cf.set_index('id', inplace=True)
    df = df.merge(cf, how='outer', on='id')
countries.append('United Kingdom')


# In[ ]:


t = [i for i in range(days+1)]

columns = [c + '_Confirmed' for c in countries]
# plt.title('Confirmed Positive Cases x 1000 people since at least 20 cases registerd in country')
# plt.xlabel('Day')
# plt.ylabel('Confirmed case x 1000 people')
ax = df[columns].plot(figsize=(16,8),
                 title= 'Confirmed Positive Covid-19 Cases x 1000 people since at least 20 cases registerd in country')
ax.set_xlabel("Days")
ax.set_ylabel("Confirmed case x 1000 people")
ax.legend(countries);


# In[ ]:


counries = countries.remove('Hubei')
t = [i for i in range(days+1)]

columns = [c + '_Confirmed' for c in countries]
# plt.title('Confirmed Positive Cases x 1000 people since at least 20 cases registerd in country')
# plt.xlabel('Day')
# plt.ylabel('Confirmed case x 1000 people')
ax = df[columns].plot(figsize=(16,8),
                 title= 'Confirmed Positive Covid-19 Cases x 1000 people since at least 20 cases registerd in country without Hubei')
ax.set_xlabel("Days")
ax.set_ylabel("Confirmed case x 1000 people")
ax.legend(countries);


# ### It seems Spain is heading towards a disaster similar the one that struck Italy while other countries seems be able, for the moment, to containt the outbreaks better

# In[ ]:


countries.append('Hubei')


# In[ ]:


t = [i for i in range(days+1)]

columns = [c + '_Deaths' for c in countries]
# plt.title('Confirmed Positive Cases x 1000 people since at least 20 cases registerd in country')
# plt.xlabel('Day')
# plt.ylabel('Confirmed case x 1000 people')
ax = df[columns].plot(figsize=(16,8),
                 title= 'Deaths for Covid-19 Cases x 1000 people since at least 20 positive cases registerd in country')
ax.set_xlabel("Days")
ax.set_ylabel("Deaths x 1000 people")
ax.legend(countries);


# In[ ]:


countries.remove('Hubei')
t = [i for i in range(days+1)]

columns = [c + '_Deaths' for c in countries]
# plt.title('Confirmed Positive Cases x 1000 people since at least 20 cases registerd in country')
# plt.xlabel('Day')
# plt.ylabel('Confirmed case x 1000 people')
ax = df[columns].plot(figsize=(16,8),
         title= 'Deaths for Covid-19 Cases x 1000 people since at least 20 positive cases registerd in country without Hubei')
ax.set_xlabel("Days")
ax.set_ylabel("Deaths x 1000 people")
ax.legend(countries);


# In[ ]:


countries.append('Hubei')


# In[ ]:


t = [i for i in range(days+1)]

columns = [c + '_Mortality_Rate' for c in countries]
# plt.title('Confirmed Positive Cases x 1000 people since at least 20 cases registerd in country')
# plt.xlabel('Day')
# plt.ylabel('Confirmed case x 1000 people')
ax = df[columns].plot(figsize=(16,8),
                 title= 'Mortality rate for Covid-19 Cases x 1000 people since at least 20 positive cases registerd in country')
ax.set_xlabel("Days")
ax.set_ylabel("Mortality Rate")
ax.legend(countries);

### Situation is very uncertain for most of the countries and i guess is normal as most of people infected are unknown.
### Italy is way over what the mortality rate should be and in Germany instead is basically 0
### US might have a lot more people infected then they think
# In[ ]:


world[world['Country/Region'] == 'Germany'].groupby('Date').sum()


# In[ ]:


world[world['Country/Region'] == 'Germany']


# In[ ]:


df[['Germany_Confirmed', 'Italy_Confirmed']].plot()


# In[ ]:




