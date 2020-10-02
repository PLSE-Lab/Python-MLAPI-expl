#!/usr/bin/env python
# coding: utf-8

# # Australian COVID-19 Data Analysis
# 
# All data sourced from https://github.com/CSSEGISandData/COVID-19
# 
# Australia is at a critical point in its fight against COVID-19. Analysing our known data against that of the world suggests that how we respond in the next week will significantly impact our ability to limit the effects of the virus. Social distancing to flatten the curve is well underway, the effects of which will be realised in the coming weeks.
# 
# ![](https://media.wired.com/photos/5e6aac7295ff060008467cf9/master/w_1600%2Cc_limit/Science_Covid19-Infographic.jpg)
# 
# Of particular interest to me here was Australia's rate of death and recovery against nations that have deployed successful strategies to combat COVID-19 - such as South Korea - and those who have been heavily impacted by the virus - Italy.
# 
# This notebook contains live metrics as made avaiable by Johns Hopkins University.
# 
# **Analysis around the charting is intentionally non-conclusive as this situation changes by the hour. The only thing that can be reliably concluded is that action in the short term is critical to prevent spread of the virus.**
# 
# > Feel free to copy and use notebook for your own analysis. Notebooks i've also found insightful and used in parts for code snippets include:
# > - https://www.kaggle.com/therealcyberlord/coronavirus-covid-19-visualization-prediction
# > - https://www.kaggle.com/imdevskp/covid-19-analysis-visualization-comparisons
# > - https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper
# > 
# > 
# > For more Australia data analysis (prettier than this):
# > - https://www.covid19data.com.au/
# > 
# > And please stay up to date with Australian health announcements at https://www.health.gov.au/. We're in this together!
# 
# 

# In[ ]:


#package import

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import random
import math
import time
import datetime
plt.style.use('ggplot')


# In[ ]:


confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')


# In[ ]:


#prep
cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]

#ausfilters
ausdeaths = deaths_df[deaths_df['Country/Region']=='Australia'].loc[:, cols[4]:cols[-1]]
ausrecoveries = recoveries_df[recoveries_df['Country/Region']=='Australia'].loc[:, cols[4]:cols[-1]]
ausconfirmed = confirmed_df[confirmed_df['Country/Region']=='Australia'].loc[:, cols[4]:cols[-1]]
ausall = confirmed_df[confirmed_df['Country/Region']=='Australia']

#southkoreafilters
southkoreadeaths = deaths_df[deaths_df['Country/Region']=='Korea, South'].loc[:, cols[4]:cols[-1]]
southkorearecoveries = recoveries_df[recoveries_df['Country/Region']=='Korea, South'].loc[:, cols[4]:cols[-1]]
southkoreaconfirmed = confirmed_df[confirmed_df['Country/Region']=='Korea, South'].loc[:, cols[4]:cols[-1]]

#italyfilters
italydeaths = deaths_df[deaths_df['Country/Region']=='Italy'].loc[:, cols[4]:cols[-1]]
italyrecoveries = recoveries_df[recoveries_df['Country/Region']=='Italy'].loc[:, cols[4]:cols[-1]]
italyconfirmed = confirmed_df[confirmed_df['Country/Region']=='Italy'].loc[:, cols[4]:cols[-1]]


#workingwithithedateconfines
dates = confirmed.keys()

#worldtotals
world_cases = []
total_deaths = [] 
total_recovered = [] 
total_active = [] 

#worldrates
mortality_rate = []
recovery_rate = [] 

#italyrates
italy_case_rate = []
italy_death_rate = []
italy_recovery_rate = []

#italytotals
italy_cases = [] 
italy_deaths = []
italy_recovery = []
italy_active = [] 

#ausrates
australia_case_rate = []
australia_death_rate = []
australia_recovery_rate = []

#austotals
australia_cases = [] 
australia_deaths = []
australia_recovery = []
australia_active = [] 

#southkoreatotals
southkorea_cases = [] 
southkorea_deaths = []
southkorea_recovery = []
southkorea_active = [] 

#southkorearates
southkorea_case_rate = []
southkorea_death_rate = []
southkorea_recovery_rate = []

for i in dates:
    
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    active_sum = confirmed_sum-death_sum-recovered_sum
    
    aus_confirmed_sum = ausconfirmed[i].sum()   
    aus_death_sum = ausdeaths[i].sum()
    aus_recovered_sum = ausrecoveries[i].sum()   
    aus_active_sum = aus_confirmed_sum-aus_death_sum-aus_recovered_sum
    
    italy_confirmed_sum = italyconfirmed[i].sum()   
    italy_death_sum = italydeaths[i].sum()
    italy_recovered_sum = italyrecoveries[i].sum()   
    italy_active_sum = italy_confirmed_sum-italy_death_sum-italy_recovered_sum
    
    southkorea_confirmed_sum = southkoreaconfirmed[i].sum()   
    southkorea_death_sum = southkoreadeaths[i].sum()
    southkorea_recovered_sum = southkorearecoveries[i].sum()   
    southkorea_active_sum = southkorea_confirmed_sum-southkorea_death_sum-southkorea_recovered_sum
    
    #metricstotals
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(active_sum)
    
    #metricsrates
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)

    #austotals
    australia_active.append(aus_active_sum)
    australia_cases.append(confirmed_df[confirmed_df['Country/Region']=='Australia'][i].sum())
    australia_deaths.append(deaths_df[deaths_df['Country/Region']=='Australia'][i].sum())
    australia_recovery.append(recoveries_df[recoveries_df['Country/Region']=='Australia'][i].sum())
    
    #ausrates
    australia_death_rate.append(aus_death_sum/aus_confirmed_sum)
    australia_recovery_rate.append(aus_recovered_sum/aus_confirmed_sum)
    
    #italytotals
    italy_active.append(italy_active_sum)
    italy_cases.append(confirmed_df[confirmed_df['Country/Region']=='Italy'][i].sum())
    italy_deaths.append(deaths_df[deaths_df['Country/Region']=='Italy'][i].sum())
    italy_recovery.append(recoveries_df[recoveries_df['Country/Region']=='Italy'][i].sum())
    
    #italyrates
    italy_death_rate.append(italy_death_sum/italy_confirmed_sum)
    italy_recovery_rate.append(italy_recovered_sum/italy_confirmed_sum)
    
    #southkoreatotals
    southkorea_active.append(southkorea_active_sum)
    southkorea_cases.append(confirmed_df[confirmed_df['Country/Region']=='Korea, South'][i].sum())
    southkorea_deaths.append(deaths_df[deaths_df['Country/Region']=='Korea, South'][i].sum())
    southkorea_recovery.append(recoveries_df[recoveries_df['Country/Region']=='Korea, South'][i].sum())
    
    #southkorearates
    southkorea_death_rate.append(southkorea_death_sum/southkorea_confirmed_sum)
    southkorea_recovery_rate.append(southkorea_recovered_sum/southkorea_confirmed_sum)
    
days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)

def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d 

def flatten_the_curve(data):
    e = [] 
    for i in range(len(data)):
        if i == 0:
            e.append(data[0])
        else:
            e.append((data[i].sum()/data[i-1].sum()-1))
    return e 

aus_daily_increase = daily_increase(australia_cases)
aus_curve_flatten = flatten_the_curve(australia_cases)
adjusted_dates = adjusted_dates.reshape(1, -1)[0]
total_recovered = np.array(total_recovered).reshape(-1, 1)


# # Global Overview
# 
# This section tracks some metrics around global stats of COVID-19. First, total cases over time. Then total cases and deaths and over time. Last, the rate of change for deaths over time.

# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, world_cases)
plt.title('Total number of Global COVID-19 Cases Over Time', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.legend(['Cases'], prop={'size': 20})
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, world_cases)
plt.plot(adjusted_dates, total_deaths)
#plt.plot(adjusted_dates, total_recovered)
plt.title('Total number of Global COVID-19 Cases and Deaths', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
#plt.legend(['Active','Deaths','Recoveries'], prop={'size': 20})
plt.legend(['Cases','Deaths'], prop={'size': 20})
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, mortality_rate)
#plt.plot(adjusted_dates, recovery_rate)
plt.title('Global COVID-19 Rate of Death Rate Over Time', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
#plt.legend(['Deaths','Recoveries'], prop={'size': 20})
plt.legend(['Deaths'], prop={'size': 20})
plt.ylabel('Rate', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# I find the rate of change to be the most indicative statistic of how things are tracking. Rates better express the effectiveness of measures as they are implemented. The inflection point of active cases at the 40-50 day mark is telling of the spread of COVID-19 from China to all parts of the globe, and the acceleration thereafter. Prior to this point, most of the data being relied upon was being sources from China, South Korea and Japan - each of which have implemented strict measures to deal with COVID-19.

# # Australian Overview
# 

# Now let's take a look at where Australia sits compared to the world. First, our total cases, also broken down by state. Then, our total cases and deaths. And finally our rates. Given our relatively small number of cases compared to other nations, I have marked the point in which the 100 case mark was hit.

# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, australia_cases)
plt.title('Total number of Australian COVID-19 Cases Over Time', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.vlines(x=47, ymin=0, ymax=100, color='black', linestyle='dashed')
plt.legend(['Cases','100 Cases'], prop={'size': 15})
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# Tracking this over time

# In[ ]:


plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, aus_daily_increase)
plt.title('Australia Daily Increases in Confirmed Cases', size=20)
plt.xlabel('Days Since 22 January 2020', size=20)
plt.ylabel('# of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# Are we flattening the curve?

# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, aus_curve_flatten)
plt.title('Rate of case increase', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.vlines(x=47, ymin=0, ymax=1, color='black', linestyle='dashed')
plt.legend(['Cases','100 Cases'], prop={'size': 15})
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# And those total cases broken down by states

# In[ ]:


ausstates = list(ausall['Province/State'].unique())
latest_confirmed = ausall[dates[-1]]

aus_confirmed_cases = []
for i in ausstates:
    cases = latest_confirmed[ausall['Province/State']==i].sum()
    if cases > 0:
        aus_confirmed_cases.append(cases)
        
plt.figure(figsize=(16, 9))
plt.barh(ausstates, aus_confirmed_cases)
plt.title('# of Covid-19 Confirmed Cases in Australian States', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# Charting this over time shows the greater impact that larger propulations have on the rate of growth.

# In[ ]:


wacases = []
viccases = []
tascases = []
sacases = []
qldcases = []
ntcases = []
nswcases = []
actcases = []


for i in dates:

    #states
    wacases.append(confirmed_df[confirmed_df['Province/State']=='Western Australia'][i].sum())
    viccases.append(confirmed_df[confirmed_df['Province/State']=='Victoria'][i].sum())
    tascases.append(confirmed_df[confirmed_df['Province/State']=='Tasmania'][i].sum())
    sacases.append(confirmed_df[confirmed_df['Province/State']=='South Australia'][i].sum())
    qldcases.append(confirmed_df[confirmed_df['Province/State']=='Queensland'][i].sum())
    ntcases.append(confirmed_df[confirmed_df['Province/State']=='Northern Territory'][i].sum())
    nswcases.append(confirmed_df[confirmed_df['Province/State']=='New South Wales'][i].sum())
    actcases.append(confirmed_df[confirmed_df['Province/State']=='Australian Capital Territory'][i].sum())

plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, wacases)
plt.plot(adjusted_dates, viccases)
plt.plot(adjusted_dates, tascases)
plt.plot(adjusted_dates, sacases)
plt.plot(adjusted_dates, qldcases)
plt.plot(adjusted_dates, ntcases)
plt.plot(adjusted_dates, nswcases)
plt.plot(adjusted_dates, actcases)
plt.title('Total number of Australian COVID-19 by State over Time', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.vlines(x=47, ymin=0, ymax=100, color='black', linestyle='dashed')
plt.legend(['WA','VIC','TAS','SA','QLD','NT','NSW','ACT','100 Cases'], prop={'size': 15})
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(15, 10))
#plt.plot(adjusted_dates, australia_active)
plt.plot(adjusted_dates, australia_cases)
plt.plot(adjusted_dates, australia_deaths)
plt.title('Total number of Australian COVID-19 Cases and Death Over Time', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.vlines(x=47, ymin=0, ymax=100, color='black', linestyle='dashed')
plt.legend(['Cases','Deaths','100 Cases'], prop={'size': 15})
#plt.legend(['Active Cases','Deaths','Recoveries','100 Cases'], prop={'size': 15})
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, australia_death_rate)
#plt.plot(adjusted_dates, australia_recovery_rate)
plt.title('Australian COVID-19 Rate of Deaths Over Time', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.vlines(x=47, ymin=0, ymax=0.7, color='black', linestyle='dashed')
#plt.legend(['Deaths','Recoveries','100 Cases'], prop={'size': 15})
plt.legend(['Deaths','100 Cases'], prop={'size': 15})
plt.ylabel('Rate', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# # South Korean Overview

# South Korea is an excellent example of how effective lockdown and testing can be used to significantly slow the transition of COVID-19. As per the previous two iterations of charting, we will first graph the total cases. Then, total cases and deaths. And finally the rates of increase of these data points.

# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, southkorea_cases)
plt.title('Total number of South Korean COVID-19 Cases Over Time', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.vlines(x=29, ymin=0, ymax=100, color='black', linestyle='dashed')
plt.legend(['Cases','100 Cases'], prop={'size': 15})
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# South Korea's total cases have significantly slowed in recent time. Attributed largely to their extensive testing and strict social distancing measures.

# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, southkorea_cases)
plt.plot(adjusted_dates, southkorea_deaths)
#plt.plot(adjusted_dates, southkorea_recovery)
plt.title('Total number of South Korean COVID-19 Cases and Deaths Over Time', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.vlines(x=29, ymin=0, ymax=100, color='black', linestyle='dashed')
plt.legend(['Cases','Deaths','100 Cases'], prop={'size': 15})
#plt.legend(['Active Cases','Deaths','Recoveries','100 Cases'], prop={'size': 15})
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# Allowing people time to recover reduces the ability of the virus to be trasmitted, thereby limiting its potentially exponential impact.

# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, southkorea_death_rate)
#plt.plot(adjusted_dates, southkorea_recovery_rate)
plt.title('South Korean COVID-19 Rate of Deaths Over Time', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.vlines(x=29, ymin=0, ymax=0.7, color='black', linestyle='dashed')
plt.legend(['Deaths','100 Cases'], prop={'size': 15})
plt.ylabel('Rate', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# As testing becomes more prevalent, the management of cases becomes more streamlined - ensuring that the death rate is sustainable and the burden on medical resources is less.

# # Italy Overivew

# The effects of COVID-19 on Italy are devastating and demonstrate the potential impact of the virus left unchecked at an early stage.

# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, italy_cases)
plt.title('Total number of Italian COVID-19 Cases Over Time', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.legend(['Cases'], prop={'size': 15})
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, italy_cases)
plt.plot(adjusted_dates, italy_deaths)
plt.title('Total number of Italian COVID-19 Cases and Deaths Over Time', size=20)
plt.legend(['Cases','Deaths'], prop={'size': 15})
plt.xlabel('Days since 22 January 2020', size=20)
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, italy_death_rate)
plt.title('Italian COVID-19 Rate of Death', size=20)
plt.xlabel('Days since 22 January 2020', size=20)
plt.vlines(x=30, ymin=0, ymax=0.7, color='black', linestyle='dashed')
plt.legend(['Deaths','100 Cases'], prop={'size': 15})
plt.ylabel('Rate', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# The death rate in Italy is high and has consistently grown over time.

# # Australia compared to South Korea and Italy
# 
# 

# It is still difficult to tell where Australia potentially sits when compared to responses to COVID-19 by other countries. One of our major barriers at the moment is the collection of information. We simply do not have enough data about COVID-19 to estimate effectively. Even when putting this dataset together, attemping to estimate or project cases still remains difficult. Instead, I have compared our figures with that of South Korea and Italy, nations whose fallout from COVID-19 has been significantly different.

# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, australia_cases)
plt.plot(adjusted_dates, southkorea_cases)
plt.plot(adjusted_dates, italy_cases)
plt.title('Total number of Australian, Italy and South Korea COVID-19 Cases Over Time', size=20)
plt.legend(['Australia','South Korea','Italy'], prop={'size': 15})
plt.xlabel('Days since 22 January 2020', size=20)
plt.ylabel('Total Number of Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(adjusted_dates, australia_death_rate)
plt.plot(adjusted_dates, southkorea_death_rate)
plt.plot(adjusted_dates, italy_death_rate)
plt.title('Rate of death from COVID-19 cases in Australia, Italy and South Korea Over Time', size=20)
plt.vlines(x=47, ymin=0, ymax=0.1, color='black', linestyle='dashed')
plt.legend(['Australia','South Korea','Italy','100 Aus Cases'], prop={'size': 15})
plt.xlabel('Days since 22 January 2020', size=20)
plt.ylabel('Rate of Death', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# Italy's death rate sits at just under 10%. They are further along their COVID-19 timeline, but when their death rate is tracked against that of South Korea it becomes evident that they share some similar characteristics. Initially, fluctuations, then an increasing plateau. Australia too, looks to be tracking along this path. Thus, how we act in the shorter term, will no doubt impact our ability to manage the rate of death arising from COVID-19. Testing will be a critical component of this as it informs action with concrete evidence.
