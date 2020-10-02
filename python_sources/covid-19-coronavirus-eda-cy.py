#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Coronaviruses (CoV) are a large family of viruses that cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV).
# 
# Coronavirus disease (COVID-19) is a new strain that was discovered in 2019 and has not been previously identified in humans.
# 
# Coronaviruses are zoonotic, meaning they are transmitted between animals and people.  Detailed investigations found that SARS-CoV was transmitted from civet cats to humans and MERS-CoV from dromedary camels to humans. Several known coronaviruses are circulating in animals that have not yet infected humans. 
# 
# Common signs of infection include respiratory symptoms, fever, cough, shortness of breath and breathing difficulties. In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death. 
# 
# Standard recommendations to prevent infection spread include regular hand washing, covering mouth and nose when coughing and sneezing, thoroughly cooking meat and eggs. Avoid close contact with anyone showing symptoms of respiratory illness such as coughing and sneezing.
# 
# <font color= 'blue'>
# Content:
# 1. [Load and Check Data](#1)
# 2. [Variable Description](#2)
# 3. [Basic Data Analysis](#3)
# 4. [Visualization](#4)

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns
from collections import Counter
import warnings 
warnings.filterwarnings("ignore")


# for kaggle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id = "1"></a><br>
# ## Load and Check Data

# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])
df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)
df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)
df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# In[ ]:


df2 = df.groupby(["Date", "Country", "Province/State"])[['SNo', 'Date', 'Province/State', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()


# In[ ]:


df2


# In[ ]:


df.query('Country=="Mainland China"').groupby("Last Update")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()


# In[ ]:


df.groupby("Country")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()


# <a id = "2"></a><br>
# ## Variable Description
# 
# 1. Sno - Serial number
# 1. ObservationDate - Date of the observation in MM/DD/YYYY
# 1. Province/State - Province or state of the observation (Could be empty when missing)
# 1. Country/Region - Country of observation
# 1. Last Update - Time in UTC at which the row is updated for the given province or country. (Not standardised and so please clean before using it)
# 1. Confirmed - Cumulative number of confirmed cases till that date
# 1. Deaths - Cumulative number of of deaths till that date
# 1. Recovered - Cumulative number of recovered cases till that date

# <a id = "3"></a><br>
# # Basic Data Analysis
# 
# * Country/Region - Confirmed - Deaths
# * Country/Region - Confirmed - Recovered

# In[ ]:


df.groupby('Date').sum()


# In[ ]:


confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
deaths = df.groupby('Date').sum()['Deaths'].reset_index()
recovered = df.groupby('Date').sum()['Recovered'].reset_index()


# <a id = "4"></a><br>
# # Visualization
# 

# In[ ]:


f,ax = plt.subplots(figsize = (20,15))
sns.barplot(x=confirmed['Confirmed'],y=confirmed['Date'],color='yellow',alpha = 0.5,label='confirmed' )
sns.barplot(x=deaths['Deaths'],y=deaths['Date'],color='purple',alpha = 0.7,label='deaths')
sns.barplot(x=recovered['Recovered'],y=recovered['Date'],color='green',alpha = 0.6,label='recovered')


ax.legend(loc='lower right',frameon = True)     
ax.set(xlabel='confirmed, deaths and recovered', ylabel='Date',title = "confirmed, deaths and recovered rate by dates")
plt.show()


# In[ ]:


dfForVisualization = pd.DataFrame({'Date': confirmed.Date,'Confirmed': confirmed.Confirmed, 'Deaths': deaths.Deaths, 'Recovered': recovered.Recovered})


# In[ ]:


f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='Date',y='Confirmed',data=dfForVisualization,color='orange',alpha=0.8)
sns.pointplot(x='Date',y='Deaths',data=dfForVisualization,color='red',alpha=0.8)
sns.pointplot(x='Date',y='Recovered',data=dfForVisualization,color='lime',alpha=0.8)

plt.text(20,130000,'number of Confirmed',color='orange',fontsize = 17,style = 'italic')
plt.text(20,120000,'number of Deaths',color='red',fontsize = 18,style = 'italic')
plt.text(20,110000,'number of Recovered',color='lime',fontsize = 18,style = 'italic')



plt.xlabel('Dates',fontsize = 15,color='blue')
plt.xticks(rotation=45,size=10)
plt.ylabel('Numbers',fontsize = 15,color='blue')
plt.title('number of Confirmed - Deaths - Recovered',fontsize = 20,color='blue')
plt.grid()

