#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Pakistan may not be rich country with lots of resources, but it has so far held off Corona virus using a number of steps. This dataset is tracking those efforts and their impact towards Corona virus prevention.
# 
# Related Information about COVID-19
# COVID-19 may not be fatal but it spreads faster than other diseases, like common cold. Every virus has Basic Reproduction number (R0) which implies how many people will get the disease from the infected person. As per inital reseach work R0 of COVID-19 is 2.7.
# 
# Currently the goal of all scientists around the world is to "Flatten the Curve". COVID-19 currently has exponential growth rate around the world which we will be seeing in the notebook ahead. Flattening the Curve typically implies even if the number of Confirmed Cases are increasing but the distribution of those cases should be over longer timestamp. To put it in simple words if say suppose COVID-19 is going infect 100K people then those many people should be infected in 1 year but not in a month.
# 
# The sole reason to Flatten the Curve is to reudce the load on the Medical Systems so as to increase the focus of Research to find the Medicine for the disease.
# 
# Every Pandemic has four stages:
# 
# Stage 1: Confirmed Cases come from other countries
# 
# Stage 2: Local Transmission Begins
# 
# Stage 3: Communities impacted with local transimission
# 
# Stage 4: Significant Transmission with no end in sight
# 
# Italy and Korea are the two countries which are currently in Stage 4 While India is in Stage 2.
# 
# Other ways to tackle the disease like Corona other than Travel Ban, Cross-Border shutdown, isolation, closure of some high risk market places, Contact Tracing and Quarantine.
# 
# ### Reference
# https://www.kaggle.com/neelkudu28/covid-19-visualizations-predictions-forecasting

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

plt.style.use('fivethirtyeight')


# Any results you write to the current directory are saved as output.


# ## NIH Preparedness and Response
# 
# - Contact tracing of confirmed cases being carried out by the RRTs.
# - Risk communication carried out among healthcare workers and affected communities during contact tracing.
# - NIH laboratory is providing support to provinces and regions for testing.

# ## Provinces Preparedness and Response
# ### 17th March 2020
# ### Balochistan
# - 5 PCR equipment made available.
# - Sampling of all quarantined persons at PCSIR village Quetta will be completed today.
# ### Sindh
# - 300 pilgrims have been shifted from Taftan to Sukkhur Quarantine Centre.
# - Arrangements being made for pilgrims to be shifted from Taftan to Sindh.
# ### Punjab
# - PDSRU is in 24/7 coordination with NIH, Islamabad.
# ### KP & TD
# - 15 confirmed cases quarantined at isolated facility in DI Khan.
# ### AJK
# - District RRTs have done screening of 02 workers from USA, 24 from UK, 10 from UAE, 21 from Saudi Arabia, 01 from Oman, France and Bahrain each so far.
# ### GB
# - Total 1000 PPE kits, 17 thermo-guns and 781 VTM, PCR Kits with testing capacity of 350 samples made available.

# In[ ]:


#!pip install urllib2


# In[ ]:


df = pd.read_csv('../input/corona-cases-in-pakistan/Corona Pakistan.csv',index_col='SNo')
df.shape


# In[ ]:


df.info()


# In[ ]:


df["Reporting Date"] =  pd.to_datetime(df['Reporting Date'])


# In[ ]:


df.sort_values("Reporting Date", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df


# ## Risk from Travellers Coming from Abroad
# 
# Travellers coming Iran are proving to be our worst nightmare. Travellers coming from other countries including Afghanistan, Italy, UK are also a great risk. Passengers coming from China are proving to be less risky due to great work being done by Chinese people and govt. As much as 60% of the travellers coming back from Iran are testing positive for Corona Virus.

# In[ ]:


fig, ax = plt.subplots(figsize=(12,7))
df[['Reporting Date','Cumulative Travellers Screened']].plot(x='Reporting Date',kind='line',ax=ax, title="Cumulative Travellers Screened")


# ### As shown above, there has been travel restrictions

# ### Returnees from Iran and China
# 
# Returnees from Iran and China are not rising, so this curve is now flat indicating we are entering into safe zone as far as foreign infections are concerned.

# In[ ]:


fig, ax = plt.subplots(figsize=(12,7))
df[['Reporting Date','Returnees from China']].plot(x='Reporting Date',kind='line',ax=ax, title="Cumulative Chinese Travellers Screened")


# In[ ]:


fig, ax = plt.subplots(figsize=(12,7))
df[['Reporting Date','Returnees from Iran']].plot(x='Reporting Date',kind='line',ax=ax, title="Cumulative Iranian Travellers Screened")


# ## Corona Virus Testing
# 
# Cheap and large scale testing against Corna Virus is important for quickly isolating patients and stopping spread of the disease. This has been successfully demonstrated by South Korea. In case of Pakistan, testing capacity has been very limited so it is possible that there lots of undetected cases and hence infection rates will increase. Pakistan's low number of positive cases can be largely attributed to poor testing capability and capacity. 

# In[ ]:


import matplotlib.pyplot as plt

ax.set_xlabel("Reporting Date")
ax.set_ylabel("Count of Confirmed Positive Cases, Tests Performed, and Admitted")
fig, ax = plt.subplots(figsize=(12,7))
df[['Reporting Date','Cumulative Tests Performed','Cumulative Test Positive Cases','Still Admitted']].plot(x='Reporting Date',kind='line', ax=ax, title="Cumulative Tests Performed Versus Positive Cases and Still Admitted")


# ### There is substantial increase in testing without corresponding increase in positive cases or deaths !

# ### More and more testing is being done as seen in graph. But still very limited testing being done can hide true number of local infections in Pakistan, and is a major concern at present. Good thing is with increase in testing, positive cases are not increasing at the same rate.

# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Pakistan | Comparing Count of Suspected, Positive, Admitted and Expired Cases")
df[['Reporting Date','Cumulative - Suspected Cases','Cumulative Test Positive Cases','Still Admitted','Expired']].plot(x='Reporting Date',kind='line',ax=ax, title="Pakistan | Positive Cases versus Still Admitted and Expired")


# In[ ]:


df['Positive Rate'] = (df['Cumulative Test Positive Cases']/df['Cumulative - Suspected Cases'])*100
df['Death Rate in Pakistan'] = (df['Expired']/df['Cumulative Test Positive Cases'])*100
#df[['Positive Rate','Admission Rate','Death Rate']]
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Positve Rate versus Death Rate")
df[['Reporting Date','Positive Rate','Death Rate in Pakistan']].plot(x='Reporting Date',kind='line', ax=ax, title="Pakistan's Corona Effort(%)")


# ### Positive rate continues to be around 10%, this will hopefully fall down with increased testing in place.

# From 15th March to 25th March, Pakistan detected huge number of foreign positive cases; but gradually the trend dived because flights were stopped around that period.

# In[ ]:



df['Death Rate in Italy']= (df['Total Deaths in Italy']/df['Total Cases in Italy'])*100


# While it is expected that positive might increase, and we need to make sure that we have enough capacity to take care of critically ill patients. So far so good, we have kept the death rate as flat as possible.

# In[ ]:


df['Death Rate in World']= (df['Global Deaths']/df['Global Cases'])*100
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Date Rate(%) in Pakistan versus World")
df[['Reporting Date','Death Rate in World','Death Rate in Pakistan','Death Rate in Italy']].plot(x='Reporting Date',kind='line',ax=ax, title="Corona Related Death Rate(%) in Pakistan versus World and Italy")
#df[['Reporting Date','Death Rate in Italy','Death Rate in Pakistan']].plot(x='Reporting Date',kind='line', ax=ax, title="Death Rate(%) in Pakistan versus Italy")


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Count of Positive Cases")
df[['Reporting Date','Cumulative Test Positive Cases','Global Cases']].plot(x='Reporting Date',kind='line',ax=ax, title="Global Positive Cases versus Pakistan's Positive Cases")


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Log of Positive Cases in Pakistan versus World")

df['log of Cumulative Pakistani Positive Cases']= np.log(df['Cumulative Test Positive Cases'])
df['log of Cumulative Global Positive Cases']= np.log(df['Global Cases'])
df[['Reporting Date','log of Cumulative Pakistani Positive Cases','log of Cumulative Global Positive Cases']].plot(x='Reporting Date',kind='line',ax=ax, title="Comparison on Lograthmic Scale | Global Positive Cases versus Pakistan's Positive Cases")


# Here lograthmic scale helps us with cases in two ways. First is, one or a few points are much larger than the bulk of the data. The second is to show percent change or multiplicative factors. Our rate of infections is still half as compared to world. However, we need to control it and bring it down to defeat Corona virus.

# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Log of Cumulative deaths in Pakistan versus World")

df['log of Cumulative Pakistani Deaths']= np.log(df['Expired'])
df['log of Cumulative Global Deaths']= np.log(df['Global Deaths'])
df[['Reporting Date','log of Cumulative Pakistani Deaths','log of Cumulative Global Deaths']].plot(x='Reporting Date',kind='line', ax=ax ,title="Comparison on Lograthmic Scale | Fatalities in Pakistan versus Rest of the World")


# ## Contact Tracing | Call Records
# 

# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Number of Calls")

df[['Reporting Date','New Calls','Cumulative Calls']].plot(x='Reporting Date',kind='line',ax=ax, title="Tracking Potential Sources of Infection")


# ### Aggressive contact tracing is in place, and would help a lot in combating spread of disease. Effectiveness of Contact tracing needs to be ensured to control Corona Virus in days ahead. 

# ## What is the Impact of Travelling Restrictions on Controlling Spread of Corona Virus ?
# 
# With travelling restrictions including flight ban taking effect on 21st March 2020, we might see number of foreign cases reducing with time. This may also reduce local cases. Major factor in local cases is of course social distance, restrictions on gatherings, weather etc.
# 
# ### Reference
# 
# https://www.theguardian.com/world/2020/mar/21/pakistan-suspends-international-flights-for-two-weeks-to-stem-covid-19-spread
# 

# In[ ]:



df['log of Travellers Count(Iran + China)']= np.log(df['Returnees from China'] + df['Returnees from Iran'] )

df['log of Cumulative Pakistani Positive Cases']= np.log(df['Cumulative Test Positive Cases'])

df['log of Cumulative Global Positive Cases']= np.log(df['Global Cases'])

fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Log of Cumulative Positive Cases in Pakistan versus World")

df[['Reporting Date','log of Cumulative Pakistani Positive Cases','log of Cumulative Global Positive Cases','log of Travellers Count(Iran + China)']].plot(x='Reporting Date',kind='line',ax=ax, title="Comparison on Lograthmic Scale | Global Positive Cases versus Pakistan's Positive Cases and Impact of Travelling Restrictions")


# 

# ## Why has Pakistan done better than Italy, Iran and China in prevention of corona virus Covid-19 ?
# 
# - Pakistan is culturally identical to Iran, Italy and China with large/close family units.
# - Pakistanis are not very disciplined people, and just want to carry on with life regardless of government instructions and current challenges.
# - Pakistan is much bigger than Iran and Italy
# - Pakistan has high proportion of young population than Italy
# - Pakistan is a poor country, with low quality housing, and health access to most of the population
# - Italy is approximately 301,340 sq km, while Pakistan is approximately 796,095 sq km. Meanwhile, the population of Italy is ~62.1 million people (142.8 million more people live in Pakistan). We have positioned the outline of Italy near your home location of Lahore, PB, Pakistan.
# - Pakistan is about 2 times smaller than Iran. Iran is approximately 1,648,195 sq km, while Pakistan is approximately 796,095 sq km. Meanwhile, the population of Iran is ~82.0 million people (122.9 million more people live in Pakistan). We have positioned the outline of Iran near your home location of Lahore, PB, Pakistan.
# - Pakistan was much near to epicenter of Corona virus i.e. China. 
# - Pakistan continued to receive thousands of visitors from Iran and China long after spread of Corona started.
# - Given Pakistan's close relationship with Iran and China means that Pakistan's day 1 is same as China when building any model to predict spread of Corona virus.
# - Pakistan continues to be high risk, and present status does not mean we can avoid bad situation in future due to failure of prevention/avoidance efforts !

# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlabel("Reporting Date")
ax.set_ylabel("Log of Counts")

df['log of Cumulative Positive Cases in Pakistan']= np.log(df['Cumulative Test Positive Cases'])
df['log of Cumulative Global Positive Cases']= np.log(df['Global Cases'])
df['log of Cumulative Deaths in Pakistan']= np.log(df['Expired'])
df['log of Cumulative Global Deaths']= np.log(df['Global Deaths'])
df['log of Cumulative Positive Cases in Italy ']= np.log(df['Total Cases in Italy'])
df['log of Cumulative Deaths in Italy']= np.log(df['Total Deaths in Italy'])
df[['Reporting Date','log of Cumulative Positive Cases in Italy ','log of Cumulative Global Positive Cases','log of Cumulative Positive Cases in Pakistan','log of Cumulative Deaths in Pakistan','log of Cumulative Global Deaths','log of Cumulative Deaths in Italy']].plot(x='Reporting Date',kind='line',ax=ax, title="Comparison on Lograthmic Scale | Fatalities in Pakistan versus Rest of the World and Italy")


# In[ ]:


## As Per Media Reports quoting Zafar Mirza

import matplotlib.pyplot as plt
classes = ['Misc - Mostly Secondary Cases','Umrah','Tableghis', 'Iranian Origin(Zaireen) Positive Cases','Positive Cases from Other Countries']
pop = [220,100,100,857,191]

plt.pie(pop,labels=classes,autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Breakdown of Pakistans Corona Virus Cases')

#Show
plt.show()


# ## Analysis of Data and Policy Suggestions

# ## Facts
# 
# 1). Most the cases i.e. over 71% have direct foreign connection.
# 2). 10% of those being tested due to symptoms are turning out to be Corona positive. 
# 3). Less than 1% of those who test positive are fatalities
# 4). There have been many cases of people escaping from quarantine
# 5). People don't understand quarantine, and self isolation at home
# 
# 
# ## Policy
# 
# 1). Given the facts that most of the positive cases have foreign origin, we need to keep borders closed and country open. We are doing completely opposite.
# 2). We need to increase testing bandwidth and make it easily available across the country to make sure that we do not miss out on potential outbreak.
# 3). Since less than 1% of positive cases are fatalities, therefore spend most of the effort in testing and prevention.
# 4). 
# ## Strategy
# 
# 1). Lockdown borders and not the whole country.  Avoid potentially dangerous spread through aggressive testing and quarantine.
# 2). Adapt the strategy, if the ground realities change. 
# 3). Use selective but effective lockdown at local levels e.g. village, mohallah when needed.
# 

# ## Summary of Conclusions
# 
# 1). With blockage of international flights, local travel restrictions and reduction in number of travellers from Iran/China risk from travellers should reduce gradually in coming days.
# 
# 2). Limited testing being done can hide true number of local infections in Pakistan, and is a major concern at present. However, right now with increase in testing we are not seeing large increase in positive cases. 
# 
# 3). Aggressive contact tracing is in place, and would help a lot in combating spread of disease.
# 
# 4). Pakistan is now at stage 2, and there is local spread which needs to be controlled through aggressive measures.
# 
# 5). Number of deaths, mortality is very small in Pakistan as compared to rest of the world.

# ## References
# 
# - https://www.nih.org.pk/novel-coranavirus-2019-ncov/
# - http://covid.gov.pk/
# 

# ## Please Upvote, Share and Make a Contribution to Help us Fight Corona virus if Possible. Thanks for Appreciation.
