#!/usr/bin/env python
# coding: utf-8

# Import libraries and assign dataframe

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')


# In[ ]:


df


# Need to take values of each case that are more than average

# In[ ]:


df_country = df.groupby(['Country/Region'], as_index = False)


mean_recovered = df_country.aggregate({'Recovered': 'max'})['Recovered'].mean()
mean_deaths = df_country.aggregate({'Deaths': 'max'})['Deaths'].mean()
mean_confirmed = df_country.aggregate({'Confirmed': 'max'})['Confirmed'].mean()


values_recovered = df_country.aggregate({'Recovered': 'max'})
values_confirmed = df_country.aggregate({'Confirmed': 'max'})
values_deaths = df_country.aggregate({'Deaths': 'max'})


df_recovered_mta = values_recovered.query('Recovered > @mean_recovered') # mta - more than average
df_deaths_mta = values_deaths.query('Deaths > @mean_deaths')
df_confirmed_mta = values_confirmed.query('Confirmed > @mean_confirmed')


# Use Pie Diagram and with "explode" define top 3 for each case

# In[ ]:


fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi= 80)

data = df_recovered_mta['Recovered']
categories = df_recovered_mta['Country/Region']
explode = list(np.zeros(len(data)))
listed = list(sorted(data))[-3:]
explode[list(data).index(listed[0])] = 0.1
explode[list(data).index(listed[1])] = 0.05
explode[list(data).index(listed[2])] = 0.1

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}% ({:d})".format(pct, absolute)

wedges, texts, autotexts = ax.pie(data, 
                                  autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"), 
                                  colors=plt.cm.tab20b.colors,
                                 startangle=140,
                                 explode = explode)

# Decoration
ax.legend(wedges, categories, title="Countries/Regions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=10, weight=700)
ax.set_title("Pie Diagram of Recoveries by CoronaVirus")
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi= 80)

data = df_deaths_mta['Deaths']
categories = df_deaths_mta['Country/Region']
explode = list(np.zeros(len(data)))
listed = list(sorted(data))[-3:]
explode[list(data).index(listed[0])] = 0.1
explode[list(data).index(listed[1])] = 0.05
explode[list(data).index(listed[2])] = 0.1


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}% ({:d})".format(pct, absolute)

wedges, texts, autotexts = ax.pie(data, 
                                  autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"), 
                                  colors=plt.cm.Dark2.colors,
                                 startangle=140,
                                 explode = explode)

# Decoration
ax.legend(wedges, categories, title="Countries/Regions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=10, weight=700)
ax.set_title("Pie Diagram of Death by CoronaVirus")
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi= 80)

data = df_confirmed_mta['Confirmed']
categories = df_confirmed_mta['Country/Region']
explode = list(np.zeros(len(data)))
listed = list(sorted(data))[-3:]
explode[list(data).index(listed[0])] = 0.2
explode[list(data).index(listed[1])] = 0.1
explode[list(data).index(listed[2])] = 0.1

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}% ({:d})".format(pct, absolute)

wedges, texts, autotexts = ax.pie(data, 
                                  autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"), 
                                  colors=plt.cm.tab20b.colors,
                                 startangle=210,
                                 explode = explode)

# Decoration
ax.legend(wedges, categories, title="Countries/Regions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=10, weight=700)
ax.set_title("Pie Diagram of Confirmed Disease by CoronaVirus")
plt.show()


# Top 3 Recoveries:
# 
# 1) US 2) Germany 3) Spain
# 
# Top 3 Deaths:
# 
# 1) US 2) United Kingdom 3) Italy
# 
# Top 3 Confirmed:
# 
# 1) US 2) Spain 3) Italy

# Make a function for defining raise. List the countries and cases

# In[ ]:


def defining_raise(array):
    some_list = []
    for i in range(len(array)):
        if i == 0:
            some_list.append(array[0])
        else:
            some_list.append(array[i] - array[i-1])
    return some_list


list_of_countries = ['China','Italy','US', 'Germany', 'Spain', 'United Kingdom']
list_of_cases = ['Recovered', 'Deaths', 'Confirmed']


# Use generator to assign each case for each country

# In[ ]:


def defining_values(list_of_countries):
    for i in list_of_countries:
        df_of_country = df[df['Country/Region'] == '{}'.format(i)]
        for i in list_of_cases:
            array_values = df_of_country.groupby(['Country/Region','Date'], as_index = False)['{0}'.format(i)]            .sum()['{0}'.format(i)].sort_values().values
            yield array_values
        
gen = defining_values(list_of_countries)
china_recovered, china_deaths, china_confirmed = next(gen), next(gen), next(gen)
italy_recovered, italy_deaths, italy_confirmed = next(gen), next(gen), next(gen)
us_recovered, us_deaths, us_confirmed = next(gen), next(gen), next(gen)
germany_recovered, germany_deaths, germany_confirmed = next(gen), next(gen), next(gen)
spain_recovered, spain_deaths, spain_confirmed = next(gen), next(gen), next(gen)
uk_recovered, uk_deaths, uk_confirmed = next(gen), next(gen), next(gen)


china_recovered_raise, china_deaths_raise, china_confirmed_raise = defining_raise(china_recovered), defining_raise(china_deaths), defining_raise(china_confirmed)
italy_recovered_raise, italy_deaths_raise, italy_confirmed_raise = defining_raise(italy_recovered), defining_raise(italy_deaths), defining_raise(italy_confirmed)
us_recovered_raise, us_deaths_raise, us_confirmed_raise = defining_raise(us_recovered), defining_raise(us_deaths), defining_raise(us_confirmed)
germany_recovered_raise, germany_deaths_raise, germany_confirmed_raise = defining_raise(germany_recovered), defining_raise(germany_deaths), defining_raise(germany_confirmed)
spain_recovered_raise, spain_deaths_raise, spain_confirmed_raise = defining_raise(spain_recovered), defining_raise(spain_deaths), defining_raise(spain_confirmed)
uk_recovered_raise, uk_deaths_raise, uk_confirmed_raise = defining_raise(uk_recovered), defining_raise(uk_deaths), defining_raise(uk_confirmed)


# In[ ]:




########################################################## CHINA ##########################################################


plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(china_recovered_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Recovered #', fontsize = 25)
plt.title('China Recovery Cases', fontsize = 30)
plt.show()

#### #### #### Recovery #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(china_deaths_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Deaths #', fontsize = 25)
plt.title('China Death Cases', fontsize = 30)
plt.show()

#### #### #### Deaths #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(china_confirmed_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Confirmed #', fontsize = 25)
plt.title('China Confirmed Cases', fontsize = 30)
plt.show()

#### #### #### Confirmed #### #### ####


# In[ ]:




########################################################## Italy ##########################################################


plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(italy_recovered_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Recovered #', fontsize = 25)
plt.title('Italy Recovery Raise Cases', fontsize = 30)
plt.show()

#### #### #### Recovery #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(italy_deaths_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Deaths #', fontsize = 25)
plt.title('Italy Death Raise Cases', fontsize = 30)
plt.show()

#### #### #### Deaths #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(italy_confirmed_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Confirmed #', fontsize = 25)
plt.title('Italy Confirmed Raise Cases', fontsize = 30)
plt.show()

#### #### #### Confirmed #### #### ####


# In[ ]:




########################################################## US ##########################################################


plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(us_recovered_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Recovered #', fontsize = 25)
plt.title('United States Recovery Raise Cases', fontsize = 30)
plt.show()

#### #### #### Recovery #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(us_deaths_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Deaths #', fontsize = 25)
plt.title('United States Death Raise Cases', fontsize = 30)
plt.show()

#### #### #### Deaths #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(us_confirmed_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Confirmed #', fontsize = 25)
plt.title('United States Confirmed Raise Cases', fontsize = 30)
plt.show()

#### #### #### Confirmed #### #### ####


# In[ ]:




######################################################## GERMANY #########################################################


plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(germany_recovered_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Recovered #', fontsize = 25)
plt.title('Germany Recovery Raise Cases', fontsize = 30)
plt.show()

#### #### #### Recovery #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(germany_deaths_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Deaths #', fontsize = 25)
plt.title('Germany Death Raise Cases', fontsize = 30)
plt.show()

#### #### #### Deaths #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(germany_confirmed_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Confirmed #', fontsize = 25)
plt.title('Germany Confirmed Raise Cases', fontsize = 30)
plt.show()

#### #### #### Confirmed #### #### ####


# In[ ]:




######################################################## SPAIN #########################################################


plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(spain_recovered_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Recovered #', fontsize = 25)
plt.title('Spain Recovery Raise Cases', fontsize = 30)
plt.show()

#### #### #### Recovery #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(spain_deaths_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Deaths #', fontsize = 25)
plt.title('Spain Death Raise Cases', fontsize = 30)
plt.show()

#### #### #### Deaths #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(spain_confirmed_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Confirmed #', fontsize = 25)
plt.title('Spain Confirmed Raise Cases', fontsize = 30)
plt.show()

#### #### #### Confirmed #### #### ####


# In[ ]:




######################################################## UK #########################################################


plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(uk_recovered_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Recovered #', fontsize = 25)
plt.title('United Kingdom Recovery Raise Cases', fontsize = 30)
plt.show()

#### #### #### Recovery #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(uk_deaths_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Deaths #', fontsize = 25)
plt.title('United Kingdom Death Raise Cases', fontsize = 30)
plt.show()

#### #### #### Deaths #### #### ####

plt.style.use('bmh')
plt.figure(figsize = (16,8))
plt.plot(uk_confirmed_raise, 'c', marker = 'd', linestyle = ':')
plt.xlabel('Days passed from 22 January', fontsize =25)
plt.ylabel('Confirmed #', fontsize = 25)
plt.title('United Kingdom Confirmed Raise Cases', fontsize = 30)
plt.show()

#### #### #### Confirmed #### #### ####


# In[ ]:




