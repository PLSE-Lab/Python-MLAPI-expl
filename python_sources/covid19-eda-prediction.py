#!/usr/bin/env python
# coding: utf-8

# ## Last updated on 2020-05-15
# This notebook tracks the spread of the novel coronavirus, also known as the 2019-nCoV. It is a contagious respiratory virus that first started in Wuhan in December 2019. As of 02/11/2020, the virus is officially named COVID-19 by the World Health Organization (WHO). The WHO declared the coronavirus outbreak as a pandemic on 03/11/2020.
# 
# ### Acknowledgements
# [https://github.com/CSSEGISandData/COVID-19](http://github.com/CSSEGISandData/COVID-19)

# In[ ]:


#Importing required libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#Prophet for prediction
from fbprophet import Prophet


# In[ ]:


# Loading the dataset
#'Last Update' is parsed as datetime format
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['Last Update'])
df_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv", index_col='Country/Region')
df_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv",index_col='Country/Region')

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)


# In[ ]:


df.head(3)


# In[ ]:


#Selecting only the required columns for analysis and droping others
df = df.drop(['Province/State'], axis = 'columns')
df.head(3)


# In[ ]:


df_confirmed = df_confirmed.drop(['Province/State', 'Lat', 'Long'], axis = 1)
df_confirmed.head(3)


# In[ ]:


df_deaths = df_deaths.drop(['Province/State', 'Lat', 'Long'], axis = 1)
df_deaths.head(3)


# In[ ]:


# Group df dataset by 'Date' with sum parameter and analyse the 'Confirmed','Deaths' values.
cases = df.groupby('Date').sum()[['Confirmed', 'Recovered', 'Deaths', ]].reset_index()
cases.plot(kind='line', figsize = (15,7) , marker='o',linewidth=2)
plt.bar(cases.index, cases['Confirmed'],alpha=0.3,color='c')
plt.xlabel('Days', fontsize=15)
plt.ylabel('Number of cases', fontsize=15)
plt.title('Worldwide Covid-19 cases - Confirmed & Deaths',fontsize=20)
plt.grid(True)
plt.style.use('ggplot')
plt.legend()


# ### Evaluating the most affected country - China 

# In[ ]:


# Evaluate China's cases
df_china = df[df['Country']=='US']
df_china_daily = df_china.groupby('Date')[['Confirmed','Recovered', 'Deaths']].sum().reset_index()
df_china_daily.plot(kind='line', figsize = (12,6))
plt.xlabel('Days')
plt.ylabel('Number of cases')
plt.title('Confirmed cases in US')
plt.grid(True)
plt.show()


# ## Most affected countries

# In[ ]:


m_count = df.groupby(['Date', 'Country']).sum()[['Confirmed','Recovered','Deaths']].reset_index()
#The latest date reported in the dataset
latest_date = m_count['Date'].max()
m_count = m_count[(m_count['Date']==latest_date)][['Country', 'Confirmed', 'Recovered','Deaths']]
top_5=m_count.nlargest(5,['Confirmed']).reset_index()
top_5


# In[ ]:


plt.figure(figsize=(10,3))
plt.title('Top 5 Countries with confirmed Covid-19 cases',fontsize=15)
plt.barh(top_5['Country'],top_5['Confirmed'],color='blue')
plt.yticks(fontsize=12)
plt.xlabel('Confirmed', fontsize=12)
plt.grid(True)
plt.show()


# In[ ]:


top_5_d=m_count.nlargest(5,['Deaths'])
plt.figure(figsize=(10,3))
plt.title('Top 5 Countries with deaths due to Covid-19',fontsize=15)
plt.barh(top_5_d['Country'],top_5_d['Deaths'],color='red')
plt.yticks(fontsize=12)
plt.xlabel('Death Count', fontsize=12)
plt.grid(True)
plt.show()


# In[ ]:


top_5_d=m_count.nlargest(5,['Recovered'])
plt.figure(figsize=(10,3))
plt.title('Top 5 Countries with recovered Covid-19',fontsize=15)
plt.barh(top_5_d['Country'],top_5_d['Recovered'],color='green')
plt.yticks(fontsize=12)
plt.xlabel('Recovered', fontsize=12)
plt.grid(True)
plt.show()


# In[ ]:


# Evaluate India's cases
df_india = df[df['Country']=='India']
df_india_daily = df_india.groupby('Date')[['Confirmed','Recovered', 'Deaths']].sum().reset_index()
df_india_daily.plot(kind='line', figsize = (12,6), marker='o',linewidth=2)
plt.xlabel('Days')
plt.ylabel('Number of cases')
plt.title('Confirmed cases in India')
plt.grid(True)
plt.show()


# In[ ]:


#Evaluate affected countries
confirmed = df.groupby(['Date', 'Country']).sum()[['Confirmed']].reset_index()
deaths = df.groupby(['Date', 'Country']).sum()[['Deaths']].reset_index()


# In[ ]:


#The latest date reported in the dataset
latest_date = confirmed['Date'].max()
latest_date


# In[ ]:


confirmed = confirmed[(confirmed['Date']==latest_date)][['Country', 'Confirmed']]
deaths = deaths[(deaths['Date']==latest_date)][['Country', 'Deaths']]


# In[ ]:


# All the affected countries by Covid-19
all_countries = confirmed['Country'].unique()
print("Number of countries/regions with cases: " + str(len(all_countries)))
print("Countries/Regions with cases: ")
for i in all_countries:
    print("*    " + str(i))


# ## Predicting future Covid-19 cases
# We are using Prophet forecasting method to predict the count of cases and deaths.

# In[ ]:


confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
deaths = df.groupby('Date').sum()['Deaths'].reset_index()


# In[ ]:


confirmed.columns = ['ds','y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.head()


# In[ ]:


m = Prophet(interval_width=0.95)
m.fit(confirmed)
future = m.make_future_dataframe(periods=15)
future_confirmed = future.copy() # for non-baseline predictions later on
future.tail()


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast, figsize = (12,6))
plt.xlabel('Days')
plt.ylabel('Number of cases')
plt.title('Predicted growth in the no. of Covid19 cases')
plt.grid(True)
plt.style.use('ggplot')
plt.show()


# In[ ]:


deaths.columns = ['ds','y']
deaths['ds'] = pd.to_datetime(deaths['ds'])
m = Prophet(interval_width=0.95)
m.fit(deaths)
future = m.make_future_dataframe(periods=15)
future_deaths = future.copy() # for non-baseline predictions later on
future.tail()


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
deaths_forecast_plot = m.plot(forecast, figsize = (12,6))
plt.xlabel('Days')
plt.ylabel('Death Toll')
plt.title('Predicted growth in the no. of deaths due to Covid19')
plt.grid(True)
plt.style.use('ggplot')
plt.show()


# # Thanks!
# ## Will keep updating the notebook
