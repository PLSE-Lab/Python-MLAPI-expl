#!/usr/bin/env python
# coding: utf-8

# # Trend Analysis of Patient Recovery from Covid19 Virus

# ## Corona has been instigating fear into the masses for the past few months now.
# 
# * Corona has rapidly spread to all the countries in the world, leaving only 3 countries uninfected as of today.
# * The confirmed cases are growing exponentially, and we have been witnessing thousands of casualties everyday.
# * Looking at the number of recoveries is supposed to bring some assurance, but when our mind inadvertantly compares it to the number of active cases, it results in the exact opposite effect. 
# * The recovery in china has been magical and makes us wish for a similar curve in our countries.
# 
# > **Personally, living in the United States, witnessing it surpass all the countries including China and Italy in the number of confirmed cases, makes me question if there's a light at the end of the tunnel.**
# 
# * The wait for recovery seems too long now.
# * But, China had been dealing with Covid19 long before it made a substantial effect in any other country. 
# * **So, when can we expect for a similar trend in recoveries to begin in our own countries?**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


covid = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv",                   names=['sno','date','state','country','datetime','confirmed','deaths','recovered'],                    parse_dates = ['date','datetime'],                    skiprows=1)
covid['active'] = covid['confirmed']-covid['deaths']-covid['recovered']
covid.tail()


# ### Available data
# 
# We have data available from ***22 January, 2020***, by that time, there were more than 500 cases in china. 
# 
# To compare the available data for other countries with china, we need to bring them to consistent measures.
# 
# Let us just compare the top 10 countries based on the number of confirmed cases.

# ### Data Preparation
# 
# Below we performed few data processing steps to ensure we are extracting the necessary information.
# 
# The data contains cumulative counts of confirmed, deaths and recovered on a day to day basis.
# 
# > Hence we filter for lines for atleast **400 confirmed cases**, the cumulative counting ensures, that the earliest data available for a country is the first time it crossed 400.
# 
# For the resulting data, we can make a **pseudo day field**, that represents the number of days passed after the country has crossed 400 confirmed cases.

# In[ ]:


### Summarise the data to country level
covida = covid.groupby(by=['country','date'],as_index=False).sum()

### Extract the list of top 10 countries based on number of confirmed cases
covid_countries = covida.groupby(by=['country'],as_index=False).last().                    sort_values(by='confirmed',ascending=False).head(10)['country'].to_list()
covid_countries.append('India')

### Filter the data to the extracted 10 countries
covida = covida[covida['country'].isin(covid_countries)]

### Filter for more than 400 confirmed cases
covida = covida[covida['confirmed']>400]

### Assign rank based on date for each country, for the pseudo day
covida['day'] = covida.groupby(by=['country'],as_index=False)['date'].rank()

### Additional ratio fields for analysis
covida['conf_rec'] = covida['recovered']/covida['confirmed']
covida['act_rec'] = covida['recovered']/covida['active']
covida['dea_rec'] = covida['recovered']/covida['deaths']
covida['non_recovered'] = covida['confirmed']-covida['recovered']


# In[ ]:


covida[covida['country']=='Mainland China'].head(1)


# # Recoveries and Deaths on day 1
# 
# **Day 1** typically is the day, the cummulative counts of confirmed cases has crossed 400 for the listed countries. Let us take a look at the recoveries and deaths on that day.

# In[ ]:


covida[covida['day']==1].plot(x='country',y=['deaths','recovered'], kind='bar', figsize=(12,7))
plt.show()


# As you can see, Iran being an expection, all the other countries are almost at the same level.

# ### Recoveries and deaths as per latest information available

# In[ ]:


covida.groupby(by=['country'],as_index=False).last().plot(x='country',y=['deaths','recovered'], kind='bar',figsize=(12,7))
plt.show()


# For the same set of countries, the recoveries and deaths vary a lot. This is quite alarming, if not for the fact that we don't how many days has passed from day 1 to present.

# ### Let us first look at how the cases have grown in these countries with time
# 
# *The number of days, represent the passing from when the country surpassed **400** confirmed cases.*

# In[ ]:


fig, ax = plt.subplots(1,figsize=(16,8))
covida.pivot(index='day',columns='country', values='confirmed').plot(ax=ax)
ax.set(xlabel='Day from 400 confirmed cases', ylabel='confirmed cases',title="Confirmed cases by country")
plt.show()


# United States definitely has a much steeper curve compared to any other countries, closely followed by France. At this rate and the very low recovery rate, it can soon be chaos. But how many days did it actually take China to make the magical recovery? 

# # The ratio of Recoveries to Active cases
# 
# The biggest problem with Covid19 is, even the recovery is slow. After fighting the virus, it takes around 1-2 weeks for total recovery. The actual process of contracting and then recoving is spread out.
# 
# Let us look at how the ratio, that defines the recoveries as a function of number of active cases, changes over time.

# In[ ]:


fig, ax = plt.subplots(1,figsize=(16,8))
covida.pivot(index='day',columns='country', values='act_rec').plot(ax=ax)
ax.set(xlabel='Day from 400 confirmed cases', ylabel='Ratio',title="Ratio of Recoveries to Active cases")
plt.show()


# ### Clearly, it took China around 35-40 days to start the recovery process. Other countries are just not there yet. Hopefully, once they reach that point, the recovery will follow a similar trend. 
# 
# ### Italy, which is next to China in the graph, can be seen to pick up a similar trend. 
# 
# ### All the other countries are nearing this point. The next couple of weeks are going to be very crucial in deciding, if we will be able to flatten the curve effectively.
# 
# # Stay home! Stay safe! SAVE LIVES! Now, more than ever.

# In[ ]:


# covid['country'].value_counts()


# In[ ]:


# covid_latest = covida.groupby(by=['country'],as_index=False).last()
# covid_latest
# covid_countries = covid_latest.sort_values(by='confirmed',ascending=False).head(5)['country'].to_list()
# covid_countries


# In[ ]:


# fig, ax = plt.subplots(1,figsize=(20,10))
# covida.pivot(index='day',columns='country', values='conf_rec').plot(ax=ax)
# ax.set(xlabel='Day from 400 confirmed cases', ylabel='Ratio',title="Ratio of Recoveries to Confirmed")


# In[ ]:


# fig, ax = plt.subplots(1,figsize=(20,10))
# covida.pivot(index='day',columns='country', values='dea_rec').plot(ax=ax)
# ax.set(xlabel='Day from 400 confirmed cases', ylabel='Ratio',title="Ratio of Recoveries to Deaths")

