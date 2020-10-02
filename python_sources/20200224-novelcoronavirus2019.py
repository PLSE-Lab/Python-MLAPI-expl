#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Loading the Datasets

# In[ ]:


#df1 = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')  #older Data......
df2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df_conf = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df_deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
df_recov = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')


# # **Column Description**
# 
# covid19data.csv
# 
#     Sno - Serial number
#     ObservationDate - Date of the observation in MM/DD/YYYY
#     Province/State - Province or state of the observation (Could be empty when missing)
#     Country/Region - Country of observation
#     Last Update - Time in UTC (**-1h for Germany**)at which the row is updated for the given province or country. (Not standardised and so please clean before using it)
#     Confirmed - Cumulative number of confirmed cases till that date
#     Deaths - Cumulative number of of deaths till that date
#     Recovered - Cumulative number of recovered cases till that date
# 
# 2019ncovdata.csv
# 
# This is old dataset and not being updated now

# # Aim of the investigation and disclaimer
# 
# This data analysis is for training purposes only. This notebook is not meant to give any medical advice, predict the further spread of the Coronavirus or evaluate any measures taken by the local governements. It is a work in progress (**WIP**).
# 
# Background Knowledge:
# 
# The current outbreak from 2019/2020 is caused by the Severe acute respiratory syndrome coronavirus 2 (**SARS-CoV-2**). [Link to Wikipedia](https://en.wikipedia.org/wiki/Severe_acute_respiratory_syndrome_coronavirus_2)
# 
# The virus is a positive-sense single-stranded RNA virus from the family of Coronaviridae. The name comes from the ,,envelope" structure formed by the SEM proteins looking under a microscope like a crown (corona). 
# In November 2019 supposedly the virus migrated from an animal and infected a human in Wuhan, China where the animal was sold on a food marked. Up to now it is not clear what kind of animal was the carrier of the virus. During the following month the virus spread from China to more than 80 countries around the globe. 
# Few facts are known about the transmission, the symptoms and the sevierity of the infection. But the american CDC states that it is transmitted by water droplets spread by the infected patients while sneezing an coughing and therefore is not airborne. In human it infects mostly the upper resperatory systems but can as well lead do severe cases up to the death of the patient. 
# 

# # Investigation of the Datasets

# First and foremost we investigae the datasets individually, using the head, describe and info method of pandas

# In[ ]:


df2.head()


# In[ ]:


df_conf.head()


# In[ ]:


df_deaths.head()


# In[ ]:


df_recov.head()


# In[ ]:


df2.describe()


# In[ ]:


df_conf.describe()


# In[ ]:


#df_deaths.describe()


# In[ ]:


#df_recov.describe()


# In[ ]:


df2.info()


# In[ ]:


#df_conf.info()


# In[ ]:


#df_deaths.info()


# In[ ]:


#df_recov.info()


# **NOTES**
# 
# in some cases the 'Province/State' column has no entry when the region inside of a country is not known. 

# # Exploratory Data Analysis

# With this dataset constantly being updated. The result of the analysis might vary after some time because of new incoming data.
# First I would like to give an overview about the infection distribution around the world.

# In[ ]:


df_conf2 = df_conf.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)
df_deaths2 = df_deaths.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)
df_recov2 = df_recov.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)


# Some data preparations for plotting the total numbers of cases.

# # Calculation of total increase (worldwide) in infected people from Janury 22nd on

# In[ ]:


patnum = [] 
for columns in df_conf2.columns:           #calculation of the total cases
    patnum.append(sum(df_conf2[columns])) 

plt.figure(figsize=(25,10))
ax = sns.barplot(x=df_conf2.columns, y = patnum)
ax.set(xlabel='date', ylabel='number of infected people')
ax.set_title('Number of infections from 22nd of January 2020 on worldwide')
plt.xticks(rotation=80)
plt.show()
#-------------------------------------------------------------------------------------------------------------------
deathnum = [] 
for columns in df_deaths2.columns: 
    deathnum.append(sum(df_deaths2[columns]))

plt.figure(figsize=(25,10))
ax = sns.barplot(x=df_deaths2.columns, y = deathnum)
ax.set(xlabel='date', ylabel='number of patients died')
ax.set_title('Number of deaths from 22nd of January 2020 on worldwide')
plt.xticks(rotation=80)
plt.show()
#---------------------------------------------------------------------------------------------------------------------
recovnum = [] 
for columns in df_recov2.columns: 
    recovnum.append(sum(df_recov2[columns]))

plt.figure(figsize=(25,10))
ax = sns.barplot(x=df_recov2.columns, y = recovnum)
ax.set(xlabel='date', ylabel='number of patients recovered')
ax.set_title('Number of recoveries during 22nd of January 2020 on worldwide from Cov19-2 infection')
plt.xticks(rotation=80)
plt.show()


# We can see an exponential growth of infection cases in the first graph. With infections in Mainland China accounting for the majority of reported cases until the middle of March. By the 31st of January the first confirmed infection in Italy was reported. By this time (9th of April 2020) we count more than 1,5 million SARS-CoV-2 infections resulting in ~94.000 deaths related to the COVID19 desease.
# 
# How is are the infections distributed by country?

# In[ ]:


df3 = df2.groupby(['Country/Region']).sum()
df3 = df3.iloc[:, 1:4]
df3 = df3.reset_index()
#---------------------------------------------------
fig, axs = plt.subplots(ncols=3, sharey = True, figsize=(30,40))
ax1 = sns.barplot(x = 'Confirmed', y ='Country/Region', data=df3 , ci = None, ax = axs[0])
ax1.set(xlabel='Infections', ylabel='Country/Region')
ax1.set_xscale("log")
ax1.set_title('Total Infections per Country')
#-------------------------------------------------
ax2 = sns.barplot(x = 'Deaths', y ='Country/Region', data=df3 , ci = None, ax = axs[1])
ax2.set(xlabel='Deaths', ylabel='Country/Region')
ax2.set_title('Total Deaths per Country')
ax2.set_xscale("log")
#-------------------------------------------------------
ax3 = sns.barplot(x = 'Recovered', y ='Country/Region', data=df3 , ci = None, ax = axs[2])
ax3.set(xlabel='Recoveries', ylabel='Country/Region')
ax3.set_title('Total Recoveries per Country')
ax3.set_xscale("log")
plt.show()


# In[ ]:


x = df2.groupby("Country/Region").sum().reset_index()
print("The SARS-CoV-2 Virus hast infected " + str(x["Country/Region"].count())+ " countries up to now.")


# The figures above show the virus has spread throughout the world infecting more than 200 countries and including small islands like St.Martin and Saint Lucia. This shows a high infection potential of the virus. Resulting in drastig measures of called ,,Social Distancing" installed by some governements trying to to spread out the infections over a longer period of time. The governemental recomendations include instructions regarding personal hygiene (hand washing, sneezing & coughing nettiquete, handshaking) and the instrunction to keep at least a 2m distance to other people at all times.  
# 
# But let's take a look at China, where the virus first appeared:

# In[ ]:


df_china = df2.loc[df2['Country/Region']== 'Mainland China']
df_china2 = df_china.groupby("ObservationDate").sum()


# In[ ]:


df_china2 = df_china2.reset_index()


# In[ ]:


fig, axs = plt.subplots(nrows=3,sharex=True ,figsize=(30,15))
ax1 =sns.barplot(x = 'ObservationDate', y='Confirmed',  data = df_china2, ax = axs[0], ci = None)
ax1.set_title('Number of Cov19-2 infections in China')
#------------------------------------------------------------------------------------
ax2 = sns.barplot(x = 'ObservationDate', y='Deaths',  data = df_china2, ax = axs[1],ci = None)
ax2.set_title('Number of Cov19-2 related deaths in China')
#----------------------------------------------------------------------------------------
ax3 = sns.barplot(x = 'ObservationDate', y='Recovered',  data = df_china2, ax = axs[2],ci = None)
ax3.set_title('Number of Cov19-2 recoveries in China')
plt.xticks(rotation=80)

plt.show()


# In the figure above we see total number of infections, deaths and recoveries **NOT** a daily increase in cases. In total up to now 80.000 cases of COVID19 infections were reported in Mainland China. China, as many other countries later on implemented strinct rules for the peoples daily movement by the end of January 2020. Still showing a steady increase in reported cases up to the 23rd of February when the numbers of new infections stopped increasing in huge numbers daily.
# This is most likely due to the possibility that although a CoV-2 infection can lead to fatal complications, people can be unawear of a CoV-2 infection, spreading the virus unknowingly. Making patient tracing difficult. Furthermore is the time of patients being infections is reported with between 14 and 21 days. Leaving a long time period with many possible transfections on the way.  

# In[ ]:


df_china = df2.loc[df2['Country/Region']== 'Mainland China'] 

plt.figure(figsize=(30,15))
ax = sns.barplot(x = 'Province/State', y='Confirmed',  data = df_china, hue = 'ObservationDate')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Number of Cov19-2 infections per Chinese Province and date') 
plt.show()


# The majority of infections where reported from the province of Hubei in which the city of Wuhan is located. Wuhan is reported to be the starting point of the SARS-CoV-2 pandemic.

# # SARS-CoV-2 infection numbers for Germany

# As a native german citizen I'm naturally interested in the numbers of SARS-Cov-2 infections in my Country. So in the following chapter I willl have a special focus on Germany.
# 
# Naturally we start with some data crunching, because we need the numbers of infections, deaths and revoveries **ONLY** for Germany.

# In[ ]:


df_ger = df2.loc[df2['Country/Region']== 'Germany'] 

fig, axs = plt.subplots(nrows=3,sharex=True ,figsize=(30,15))
ax1 =sns.barplot(x = 'ObservationDate', y='Confirmed',  data = df_ger, ax = axs[0])
ax1.set_title('Number of Cov19-2 infections in Germany')
#------------------------------------------------------------------------------------
ax2 = sns.barplot(x = 'ObservationDate', y='Deaths',  data = df_ger, ax = axs[1])
ax2.set_title('Number of Cov19-2 related deaths in Germany')
#----------------------------------------------------------------------------------------
ax3 = sns.barplot(x = 'ObservationDate', y='Recovered',  data = df_ger, ax = axs[2])
ax3.set_title('Number of Cov19-2 recoveries in Germany')
plt.xticks(rotation=80)

plt.show()


# The first reported case of a SARS-CoV-2 infection was reported on the 27th of January 2020 in Munich, Bavaria. But the numbers of infections gained momentum by the middle of March 2020. Resulting in more than 115.000 reported cases up to today (09.04.2020). The first confirmed death by COVID19 was reported on the 9th of March 2020. 

# In[ ]:


df_ger2 = df_ger
df_ger2 = df_ger2.drop(["SNo","Province/State","Last Update","Country/Region"], axis = 1)
#df_ger2.head()


# In[ ]:


ax1 = sns.pairplot(df_ger2,  kind="reg")
ax1.fig.suptitle('Pairplot for COVID19 Cases in Germany over time', y=1.08)
plt.show()


# # Fatality Ratio calculation for Germany

# To have an estimate how deadly the virus is the fatality ration, a ration between reported death and reported infections is calculated. 

# In[ ]:


df_ger2["Ratio"] = (df_ger2["Deaths"]/df_ger2["Confirmed"]*100)
df_ger2.head()


# In[ ]:


plt.figure(figsize=(15,8))
sns.stripplot(x = 'ObservationDate', y='Ratio',  data = df_ger2)
plt.xticks(rotation=80)
plt.title('Development of the death to confirmed infection ratio over time in percent')
plt.show()


# Over time the fatality ration for Germany is increasing. This is expected, because more infections are reported und the fatalities after a COVID19 infection increase simultaniously. This graph above can **NOT** really tell how deadly the SARS-CoV-2 virus is. The total numbers can be calculated after the first wave of the outbreak or a total immunisation of the human population. In general experts talked about a fatality rate between 1 and 3% initially. But even experts say every day, that they learn something new about this Virus and the illness coming with it. 

# In[ ]:


x = df_ger2.iloc[-1,1] 
y = df_ger2.iloc[-1,2]
z = round((y/x)*100, 3)
print("Percentage of fatal cases per confirmed COVID19 infection in Germany:" + " " + str(z) + "%")


# Above I calculated the fatality rate in percent. The number of 2% says that out of 100 infected patiens 2 died on average. This number is not a forcast on and can vary over the time of the pandemic. One reason of the unreliability of the fatality ratio at the moment is that the number of tested people is increasing daily. Leading to a high varianz of the fatality ratio in countries with not sufficiant amount of testing.  

# In[ ]:




