#!/usr/bin/env python
# coding: utf-8

# # Relation between growth rate, containment and mitigation measures.
# 
# In this notebook, I show how there is a correlation between mitigation mesures, containment and growth rates.
# To do this, we will observe the strategy of each country(China, US, Germany,South Korea, italy, France...) to fight against covid19 and the impact of this strategy on the growth rate. We will need three growth rates.
# 
# > ** growth rate positive cases**
# 
# > ** growth rate recovered**
# 
# > ** growth rate death**
# 
# And the end of this work, I give some conclusion. Let's start.

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


from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Prepare data

# In[ ]:


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 150)
media = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')
covid19 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# In[ ]:


media.head(2)


# In[ ]:


policy = media[['Country','Date Start','Description of measure implemented']].sort_values('Date Start',ascending=True)


# In[ ]:


policy.head(3)


# In[ ]:


# we see different country
policy.Country.unique()


# In[ ]:


covid19.head(3)


# In[ ]:


covid19 = covid19.groupby(['Country/Region', 'ObservationDate'])[['Confirmed', 'Deaths', 'Recovered']].agg('sum')
covid19.head()


# In[ ]:


country = covid19.reset_index()
country.info()


# In[ ]:


country.head(3)


# In[ ]:


country.loc[:, 'ObservationDate'] = pd.to_datetime(country.loc[:, 'ObservationDate'])
country.loc[:, 'Confirmed'] = pd.to_numeric(country.loc[:, 'Confirmed'], errors='coerce')
country.loc[:,'Recovered'] = pd.to_numeric(country.loc[:,'Recovered'], errors='coerce')
country.loc[:,'Deaths'] = pd.to_numeric(country.loc[:,'Deaths'], errors='coerce')
policy.loc[:, 'Date Start'] = pd.to_datetime(policy.loc[:, 'Date Start'])


# In[ ]:


country['Country/Region'].unique()


# In[ ]:


def growth_rate(data=None):
    x = []
    x.append(0)
    for i in range(data.shape[0]-1):
        a = data.iloc[i+1]-data.iloc[i]
        if data.iloc[i] == 0:
            v = 0.0
        else:
            v = a/data.iloc[i]
        v=v*100
        x.append(v)
        
    return np.array(x)


# In[ ]:


def compute_growth_rate(data=None):
    """
        :params data
        
    """
    for c in ['Confirmed', 'Recovered', 'Deaths']:
        r = 'growth_rate_{}(%)'.format(c)
        data.loc[:,r] = growth_rate(data.loc[:,c])
        
    return data.copy()


# # China containment and mitigation mesures

# In[ ]:


china = policy[policy.Country == 'China'].sort_values(by=['Date Start'])
mainland_china = country[country['Country/Region'] == 'Mainland China']


# In[ ]:


# to see date that policy has started at China
china['Date Start'].unique()


# ## before

# In[ ]:


china[china['Date Start'] < '2020-02-22'].style.set_properties(**{'background-color': 'black',
                            'color': 'white',
                            'border-color': 'lawngreen'})


# In[ ]:


gr_china = compute_growth_rate(mainland_china)


# In[ ]:


cols = list(set(gr_china.columns) - set(['Confirmed', 'Deaths', 'Recovered','Country/Region']))
icols = ['ObservationDate', 'Confirmed', 'Recovered', 'Deaths']


# ## After 

# In[ ]:


china[china['Date Start'] >= '2020-02-22'].style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color':'white'})


# ### Growth rate

# In[ ]:


mainland_china[cols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The effect of containment and mitigation mesures on growth rate confirmed, recovered,deaths in China ')
plt.ylabel('growth rate (%)')


# In[ ]:


mainland_china[['ObservationDate','Confirmed','Recovered', 'Deaths']].plot(x='ObservationDate', figsize=(15,5))
plt.title('Control disease state in China')


# # Italy Containment and Mitigation mesures

# In[ ]:


italy = policy[policy['Country'] == 'Italy'].sort_values(by=['Date Start'])
c_italy = country[country['Country/Region'] == 'Italy']


# ## Before

# In[ ]:


italy[italy['Date Start'] < '2020-02-24'].style.set_properties(**{'background-color': 'black',
                            'color': 'white',
                            'border-color': 'lawngreen'})


# In[ ]:


gr_italo = compute_growth_rate(c_italy)


# ## After

# In[ ]:


italy[italy['Date Start'] >= '2020-02-24'].style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color': 'white'})


# In[ ]:


gr_italo[cols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The effect of containment and mitigation mesures on growth rate confirmed,Recovered and death in Italy')
plt.ylabel('growth rate (%)')


# In[ ]:


fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(hspace=0.4, wspace=0.1)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
c_italy.plot(x='ObservationDate', y = 'Confirmed', ax=ax1)
ax1.set_title('Increasing confirmed in Italy')
c_italy[['ObservationDate','Deaths', 'Recovered']].plot(x='ObservationDate', ax=ax2)
ax2.set_title('Increasing recovered and deaths in Italy')


# **To continuous with Italy see** https://www.kaggle.com/lumierebatalong/italy-space-time-spreading-of-covid19

# # Germany containment and mitigation mesures

# In[ ]:


german = policy[policy['Country'] == 'Germany'].sort_values(by=['Date Start'])
germany = country[country['Country/Region'] == 'Germany']


# ## Before

# In[ ]:


german[german['Date Start'] <= '2020-02-28'].style.set_properties(**{'background-color': 'black',
                            'color': 'white',
                            'border-color': 'lawngreen'})


# In[ ]:


gr_german = compute_growth_rate(germany)


# # After

# In[ ]:


german[german['Date Start'] > '2020-02-28'].style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color': 'white'})


# In[ ]:


gr_german[cols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The effect of containment and mitigation mesures on growth rate confirmed, recovered,deaths in Germany')
plt.ylabel('growth rate (%)')


# In[ ]:


fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(hspace=0.4, wspace=0.1)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
germany.plot(x='ObservationDate', y = 'Confirmed', ax=ax1)
ax1.set_title('Increasing confirmed in Germany')
germany[['ObservationDate','Deaths', 'Recovered']].plot(x='ObservationDate', ax=ax2)
ax2.set_title('Increasing recovered and deaths in Germany')


# # France 

# In[ ]:


french = policy[policy['Country'] == 'France'].sort_values(by=['Date Start'])
france = country[country['Country/Region'] == 'France']


# In[ ]:


french.style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color': 'white'})


# In[ ]:


gr_france = compute_growth_rate(france)


# In[ ]:


gr_france[cols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The effect of containment and mitigation mesures on growth rate confirmed, recovered,deaths in France')


# In[ ]:


fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(hspace=0.4, wspace=0.1)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
france.plot(x='ObservationDate', y = 'Confirmed', ax=ax1)
ax1.set_title('Increasing confirmed in France')
france[['ObservationDate','Deaths', 'Recovered']].plot(x='ObservationDate', ax=ax2)
ax2.set_title('Increasing recovered and deaths in France')


# # Iran

# In[ ]:


tehran = policy[policy['Country'] == 'Iran'].sort_values(by=['Date Start'])
iran = country[country['Country/Region'] == 'Iran']


# In[ ]:


tehran.style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color': 'white'})


# In[ ]:


gr_iran = compute_growth_rate(iran)


# In[ ]:


gr_iran[cols].plot(x='ObservationDate',figsize=(15,5))
plt.title('The effect of containment and mitigation mesures on growth rate confirmed, recovered, death in Iran')
plt.ylabel('growth rate (%)')


# In[ ]:


iran[icols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The fight against disease controlled by Iran')


# # Egypt containment and mitigation mesures

# In[ ]:


cairo = policy[policy.Country == 'Egypt']
egypt = country[country['Country/Region'] == 'Egypt']


# In[ ]:


cairo.style.set_properties(**{'background-color': 'black',
                            'color': 'lawngreen',
                            'border-color': 'white'})


# In[ ]:


gr_egypt = compute_growth_rate(egypt)


# In[ ]:


gr_egypt[cols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The effect of containment and mitigation mesure on growth rate confirmed,recovered and deaths in Egypt')
plt.ylabel('growth rate (%)')


# In[ ]:


egypt[icols].plot(x='ObservationDate', figsize=(15,5))
plt.title('The fight against disease controlled by Egypt')


# **You can see the relationship between growth rate and containment, mitigation measures very well.
# we notice that this mitigation measures can be efficient if government and population fight together against this disease. (e.g. China, ..)**

# ## UpNext
