#!/usr/bin/env python
# coding: utf-8

# # Comparison of COVID-19 virus growth per country
# 
# ## Introduction
# The following notebook provides an analysis of the country specific growth rate of the COVID-19 virus. 
# 
# The data set used is the COVID-19 Complete Dataset (Updated every 24hrs) [[1](https://www.kaggle.com/imdevskp/corona-virus-report),[2](https://github.com/CSSEGISandData/COVID-19)]. 
# 
# The data includes the following columns 
# 
# |Column | Description | 
# |-------|-------------|
# |Country/Region|Country or region|
# |Province/State|Province within country or region|
# |Lat|Latitude of province|
# |Long|Longitude of province|
# |datetime|Day of measurement|
# |Confirmed|Cumulative count of confirmed positive test cases of COVID-19|
# |Deaths|Cumulative count of fatalities due to COVID-19|
# |Recovered|Cumulative count of people who have recovered from COVID-19 |

# In[ ]:


# Load libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # operating system
import matplotlib.pyplot as plt # plotting
import matplotlib.ticker
import matplotlib

# Add plot settings
matplotlib.rcParams.update({'font.size': 16})
#plt.style.use('dark_background')


# In[ ]:


df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
df = df.rename(columns={"Date": "datetime"})
df["datetime"] = pd.to_datetime(df["datetime"],format='%m/%d/%y')
df = df.sort_values("datetime")
df['Province/State'] = df['Province/State'].fillna('')
df.loc[df["Country/Region"]==df['Province/State'],'Province/State'] = ''
df["Country_Province"] = df["Country/Region"] + '-' + df['Province/State']
df["Country_Province"] = df["Country_Province"].map(lambda x: x.rstrip('-'))
df.tail()


# After combining the data per country (or region) a selection is made in the countries. Two criteria could be considered. The first is based on the confirmed cases, and the second is based on the number of deaths. As the COVID-19 testing criteria can be largely different per country, to make a best comparative evaluation of the virus growth, the choice is made to use the number of deaths to align the virus growth per country. 
# 
# Before continuing the countries which have more than 60 deaths are selected. This is done such that we make optimal use of the available data, but do not include data which is possibly not representative yet. The following countries were found: 

# In[ ]:


# Combine all data per country
table = pd.pivot_table(df, values=["Confirmed","Deaths"], index=["datetime"],
                    columns=["Country_Province"], aggfunc=sum)

total_deaths = pd.DataFrame(table['Deaths'].iloc[-1,:][table['Deaths'].iloc[-1,:]>60].sort_values(ascending=False))
total_deaths.head(20)


# Next the day at which 20, 30, 40, 50 and 60 deaths were found. These are reported in the table below, and show that China-Hubei is roughly 34 days ahead of Italy and Iran. 

# In[ ]:


# Select top 10 countries based on the number of deaths
country_list = [j for j in total_deaths.index][:10]

# Find time at which a certain number of deaths is found: 
time_at = {} # Time at number of deaths 
days = (table.index - table.index[0])
death_counts = [20,30,40,50,60]
for country in country_list:
    time_at[country] = {}
    for death_count in death_counts: 
        time_at[country][death_count] = np.interp(np.log(death_count),np.log(table['Deaths'][country][table['Deaths'][country]>0]),days.days[table['Deaths'][country]>0])
        #linear: time_at[country, death_count] = np.interp(death_count,table['Deaths'][country],days.days)
days_at_death = pd.DataFrame(time_at)
days_at_death.index.name = "Deaths"
days_at_death


# A comparison of the number of deaths is shown in the figure below. The left plot is referenced to the day that 30 deaths occured, and the right to 60 days. These show that a strong increase is found in Spain, whereas the South Korea shows the least increase in deaths. No direct spatial correlation appears apparent from the increase in the number of deaths per country. 

# In[ ]:


fig, axes = plt.subplots(2,1,figsize=(20,20))
axis_handles = axes.ravel()
for (j,death_count) in enumerate([30,60]):
    ax1 = axis_handles[j]
    for country in country_list: 
        ax1.semilogy(days.days - time_at[country][death_count], table['Deaths'][country],'.-')
    ax1.legend(loc='upper left')
    ax1.set_xlim([-10,50])
    ax1.set_ylim([1,100000])
    ax1.set_yticks([1,3,10,30,100,300,1000,3000,10000,30000,100000])
    ax1.set_xlabel(f'Days since {death_count} deaths')
    ax1.set_ylabel('Deaths')
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.grid(True)
plt.show()


# A comparison of the number of confirmed COVID-19 cases related to the day at which 30 or 60 deaths were seen is shown in the plots below. Assuming that the same number of people would die after being infected, this suggests that South Korea and Germany have much more tests than other countries. Iran appears to perform less tests on the population. The spread in the number of confirmed cases for the same number of deaths also indicates that it is likely that the testing criteria per country are different. 

# In[ ]:


fig, axes = plt.subplots(2,1,figsize=(20,20))
axis_handles = axes.ravel()
for (j,death_count) in enumerate([30,60]):
    ax1 = axis_handles[j]
    for country in country_list: 
        ax1.semilogy(days.days - time_at[country][death_count], table['Confirmed'][country],'.-')
    ax1.legend()
    ax1.set_xlim([-5,50])
    ax1.set_ylim([100,3000000])
    ax1.set_yticks([100,300,1000,3000,10000,30000,100000,300000,1000000,3000000])
    ax1.set_xlabel(f'Days since {death_count} deaths')
    ax1.set_ylabel('Confirmed')
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.grid(True)
plt.show()


# An outlook of the number of deaths in China suggest that after 60 deaths in the country it takes roughly 50 days for the death count to stabilise. It is important to realise that different countries have performed different measures for controling the spread of the disease, which means this may take a while longer to stabilise in other countries. 

# In[ ]:


fig, ax1 = plt.subplots(1,1,figsize=(20,10))
death_count = 60 #days
for country in country_list: 
    ax1.semilogy(days.days - time_at[country][death_count], table['Deaths'][country],'.-')
ax1.legend()
ax1.set_xlim([-5,60])
ax1.set_ylim([10,100000])
ax1.set_yticks([10,30,100,300,1000,3000,10000,30000,100000])
ax1.set_xlabel(f'Days since {death_count} deaths')
ax1.set_ylabel('Deaths')
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.grid(True)
plt.show()


# ## Summarising comments
# 
# Based on an analysis relating countries by number of fatalities at a certain day the following was found: 
# 
# * No direct spatial geographic correlation was found between the 10 different selected countries
# * Iran and Italy appear to lag 34 days behind China in the spread of the disease
# * Strong differences in testing are found, Germany and South Korea have tested a lot at an early stage whereas Iran did not test as much 
# * It appears from the day where 60 people died, it takes roughly 50 days to stabilise

# ## References 
# 
# [1] https://www.kaggle.com/imdevskp/corona-virus-report
# 
# [2] https://github.com/CSSEGISandData/COVID-19
