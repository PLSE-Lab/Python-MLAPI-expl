#!/usr/bin/env python
# coding: utf-8

# ## COVID-19 Spread : exploratory analysis
# 
# <font size="3">**Coronavirus disease (COVID-19)** is an infectious disease caused by a new virus.
# The disease causes respiratory illness (like the flu) with symptoms such as a cough, fever, and in more severe cases, difficulty breathing.
# 
# 
# In this notebook exploratory analysis of spread of covid-19 is studied and which will help to determine which factors impact the transmission behavior of COVID-19. 
# Right now this notebook examines covid-19 spread based on the following datasets through visualisations. Though it does not consider some key factors such as 
# 1. number of tests performed, 
# 2. asymptomatic spread, 
# 3. population data for a region/country
# 
# I will update this notebook to consider the above information and models to forecast spread. (some kernels have implemented some of them)
# <br/><br/>
# 
# There are two datasets used here : </font>
# 1. 2019-nCoV Data Repository by Johns Hopkins CSSE : https://github.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning
# 2. Novel Corona Virus 2019 Dataset : https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset#COVID19_open_line_list.csv
# 
# 
# STAY SAFE.

# <font size="5">**DATA EXPLORATION**</font>
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# We will use - 'corona-virus-report', 'novel-corona-virus-2019-dataset' 


# In[ ]:


# import os
# print(os.listdir("../input/"))
# for item in os.listdir("../input/"):
#     print(os.listdir(f'../input/{item}'))


# <font size="4">**Reading and understanding the data**</font>

# In[ ]:


table1 = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
table1 = table1.dropna(axis=0, how='all')
print(table1.shape)
table1.head(2)


# In[ ]:


table2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

# Removing columns that might be not necessary
table2 = table2.drop(['Unnamed: 33','Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37','Unnamed: 38', 
                      'Unnamed: 39','Unnamed: 40', 'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44',
                      'location', 'admin3', 'admin2', 'admin1','country_new', 'admin_id', 'data_moderator_initials'], axis=1)
table2 = table2.dropna(axis=0, how='all')
print(table2.shape)
table2.head(2)


# <font size="4">**Preparing the data** </font>

# ** -> TABLE1 DATA **

# In[ ]:


#PREPRARING TABLE1 DATA

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Active Case: Active Case = confirmed - deaths - recovered
table1['Active'] = table1['Confirmed'] - table1['Deaths'] - table1['Recovered']

# Renaming Mainland china as China in the data table
table1['Country/Region'] = table1['Country/Region'].replace('Mainland China', 'China')

# filling missing values 
table1[['Province/State']] = table1[['Province/State']].fillna('')
table1[cases] = table1[cases].fillna(0)

# latest
table1_latest = table1[table1['Date'] == max(table1['Date'])].reset_index()
# latest_reduced
table1_latest_gp = table1_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
table1.head(2)


# ** -> TABLE2 DATA **

# In[ ]:


# PREPARING TABLE2 DATA

# Filling null values with a past date (to detect it easily) and removing irregular datetime values
table2_time = table2[['country','date_onset_symptoms', 'date_admission_hospital', 'date_confirmation']].fillna('01.01.2000')
table2_time = table2_time.drop(table2_time[table2_time['date_onset_symptoms'].str.len() != 10].index)
table2_time = table2_time.drop(table2_time[table2_time['date_admission_hospital'].str.len() != 10].index)
table2_time = table2_time.drop(table2_time[table2_time['date_confirmation'].str.len() != 10].index)

# converting to datetime
table2_time['date_onset_symptoms'] = pd.to_datetime(table2_time['date_onset_symptoms'].str.strip(), format='%d.%m.%Y', errors='coerce')
table2_time['date_admission_hospital'] = pd.to_datetime(table2_time['date_admission_hospital'].str.strip(), format='%d.%m.%Y', errors='coerce')
table2_time['date_confirmation'] = pd.to_datetime(table2_time['date_confirmation'].str.strip(), format='%d.%m.%Y', errors='coerce')

# print(table2.columns)
table2_time.head(2)


# In[ ]:


# Looking for which country has how much data (important to understand and interpret results)
print(table2['country'].value_counts()[:5])


# **1. Maximum number of cases from country**
# 
# 

# In[ ]:


# temp = table1.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max().reset_index()
# temp[temp['Confirmed'] == max(temp['Confirmed'])]

# OR
temp = table1_latest_gp[table1_latest_gp['Confirmed'] == max(table1_latest_gp['Confirmed'])]
print(temp['Country/Region'])
temp


# ** 2. Total Global cases**

# In[ ]:


temp = table1.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
(temp.style.background_gradient(cmap='Pastel1'))


# In[ ]:


temp_f = table1_latest_gp.sort_values(by='Confirmed', ascending=False)
temp_f = temp_f.reset_index(drop=True)
temp_f.style.background_gradient(cmap='Reds')


# <font size="5">**Data Visualisation**</font>

# **Visual 1 :** A Time-series graph of the confirmed and recovered cases of COVID-19. This shows that with increasing number of positive cases the number of recovered cases as a whole also increasing telling us that the situation is improving.
# 
# 1. The rate of increase of confirmed cases significantly drops from March 1st week and almost flattens.
# 2. We expect the bar of **recovered cases** to increase in coming weeks to compensate the total number of positive cases and neutralise the situation.

# In[ ]:


import plotly as py
import plotly.graph_objects as go
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
                x=table1.Date,
                y=table1['Confirmed'],
                name="Confirmed",
                line_color='blue'))

fig.add_trace(go.Scatter(
                x=table1.Date,
                y=table1['Recovered'],
                name="Recovered",
                line_color='red'))
fig.update_layout(title_text='Rate of cases over time (Time Series with Rangeslider)',
                  xaxis_rangeslider_visible=True)
py.offline.iplot(fig)


# **Visual 2 :** Confirmed cases in China - plot. (Apart from Hubei)
# 
# **It is clearly visible that the region with second most number cases in china is just only 2.1% of number of cases in Hubei, which is also origin of the covid-19.** This shows that China has contained the spread very sporadically.

# In[ ]:


# Top 3 regions in china where confirmed cases are recorded
table1[table1['Country/Region'] == 'China'].groupby('Province/State').max()['Confirmed'].sort_values(ascending=False)[:3]


# In[ ]:


import plotly.express as px
# import plotly.graph_objects as go
temp = table1[table1['Country/Region'] == 'China'].groupby('Province/State').max().reset_index()
hubei = temp[temp['Province/State'] == 'Hubei']
hubei['color'] = 'red'
temp = temp[temp['Province/State'] != 'Hubei']
fig = px.scatter_geo(lat='Lat', lon='Long', data_frame=temp, size='Confirmed')
fig.add_trace(px.scatter_geo(lat='Lat', lon='Long', data_frame=hubei, size='Confirmed', color_discrete_sequence=['red']).data[0])
# fig = px.scatter_geo(lat='Lat', lon='Long', data_frame=hubei, size='Confirmed')
fig.update_layout(
        title = 'Spread in China - Hubei(red) ',
        geo = dict(
            lonaxis =dict(range=[72,135]),
            lataxis =dict(range=[17,50]),
            showcountries=True,
            countrycolor="black",
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            countrywidth = 0.8,
        ),
    )
fig.show()


# **Visual 3 :** Total confirmed, recovered, active cases.

# In[ ]:


plt.figure(figsize=(12, 7))
ax = plt.plot(table1.groupby('Date').sum().reset_index()[cases], linewidth=3)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('total Cases', size=20)
plt.legend(cases, prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# **Visual 4 :** Trend for top 5 most spread countries

# In[ ]:


top_5 = list(temp_f['Country/Region'][:5])
plt.figure(figsize=(12,8))
plt.plot(table1[table1['Country/Region'] == top_5[0]].groupby('Date').sum().reset_index()['Confirmed'], linewidth=3)
plt.plot(table1[table1['Country/Region'] == top_5[1]].groupby('Date').sum().reset_index()['Confirmed'], linewidth=3)
plt.plot(table1[table1['Country/Region'] == top_5[2]].groupby('Date').sum().reset_index()['Confirmed'], linewidth=3)
plt.plot(table1[table1['Country/Region'] == top_5[3]].groupby('Date').sum().reset_index()['Confirmed'], linewidth=3)
plt.plot(table1[table1['Country/Region'] == top_5[4]].groupby('Date').sum().reset_index()['Confirmed'], linewidth=3)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('total Cases', size=20)
plt.legend(top_5, prop={'size': 15})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# **Visual 5 :** Analysis for average time from onset of symtoms, admission to hospital and actual testing positive for the covid-19.
# 
# Here, i am considering cases available for **CHINA** only as we have maximum number of data available for it, and 
# **Then we will compare China to South Korea and try to notice any visible difference in their approach**. 

# In[ ]:


table2_time_china = table2_time[table2_time['country'] == 'China']
time_to_hospital = pd.DataFrame((table2_time_china['date_admission_hospital'] - table2_time_china['date_onset_symptoms']).astype(int))
time_to_confirm = pd.DataFrame((table2_time_china['date_confirmation'] - table2_time_china['date_onset_symptoms']).astype('timedelta64[D]'))

print('Average number of days for china = '+ str(np.mean(np.array(time_to_confirm[time_to_confirm[0].between(0,500)]))))
plt.figure(figsize=(16,10))

sns.countplot(y=0, data = time_to_confirm[time_to_confirm[0].between(1,500) ], orient='v')
plt.xlabel('count', size=20)
plt.ylabel('# of days from first symptom to tested positive(CHINA)', size=20)
plt.show()


# In[ ]:


table2_time_korea = table2_time[table2_time['country'] == 'South Korea']
time_to_hospital = pd.DataFrame((table2_time_korea['date_admission_hospital'] - table2_time_korea['date_onset_symptoms']).astype(int))
time_to_confirm = pd.DataFrame((table2_time_korea['date_confirmation'] - table2_time_korea['date_onset_symptoms']).astype('timedelta64[D]'))

print('Average number of days for china = '+ str(np.mean(np.array(time_to_confirm[time_to_confirm[0].between(0,500)]))))
plt.figure(figsize=(16,10))

sns.countplot(y=0, data = time_to_confirm[time_to_confirm[0].between(1,500) ], orient='v')
plt.xlabel('count', size=20)
plt.ylabel('# of days from first symptom to tested positive(S. KOREA)', size=20)
plt.show()


# In[ ]:


time_to_hospital = pd.DataFrame((table2_time['date_admission_hospital'] - table2_time['date_onset_symptoms']).astype(int))
time_to_confirm = pd.DataFrame((table2_time['date_confirmation'] - table2_time['date_onset_symptoms']).astype('timedelta64[D]'))

print('Average number of days for china = '+ str(np.mean(np.array(time_to_confirm[time_to_confirm[0].between(0,500)]))))


# The average number of days for both China and South Korea is almost equal and much similar to overall average of 5.8 days, i.e. It took 5.8 days for an average country to test its people since the onset of first symtom.
# 
# **The idea to plot is to observe any lack efficiency in response of a country towards testing its people. But surely due to lack of testing data it is still an open question.**

# **Visual 6 :** Plot for mortality and recovery rates.
# 
# For the last week, there is a steep decline in recovery rate and a gradual increase in mortality rate. The possible reasons can be increase in number of cases in US.

# In[ ]:


temp = table1.groupby('Date').max().reset_index()
dates = temp.keys()
mortality_rate = []
recovery_rate = [] 

for i,row in temp.iterrows():
    confirmed_sum = temp.iloc[i]['Confirmed']
    death_sum = temp.iloc[i]['Deaths']
    recovered_sum = temp.iloc[i]['Recovered']

    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)


plt.figure(figsize=(12, 7))
plt.plot(mortality_rate, linewidth=3)
plt.plot(recovery_rate, linewidth=3)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('rate (0-1)', size=20)
plt.legend(['mortality rate', 'recovery rate'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# **Visual 7 :** Plot for number of deaths and recoveries.
# 
# From below plot, it is visible that Italy, Spain has a sudden increase in number of deaths. US and France also following the same trend.

# In[ ]:


# TOP 10 COUNTRIES WITH MOST NUMBER OF DEATHS

temp_f = table1.groupby('Country/Region').max().reset_index()[['Country/Region','Deaths','Recovered']].sort_values('Deaths', ascending=False)[:10].reset_index(drop=True)
temp_f.style.background_gradient(cmap='Reds')


# In[ ]:


top_5 = temp_f['Country/Region'][:6]
temp = table1[table1['Country/Region'] == 'US'].groupby('Date').max().reset_index()

plt.figure(figsize=(12,8))
plt.plot(table1[table1['Country/Region'] == top_5[0]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)
plt.plot(table1[table1['Country/Region'] == top_5[1]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)
plt.plot(table1[table1['Country/Region'] == top_5[2]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)
plt.plot(table1[table1['Country/Region'] == top_5[3]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)
plt.plot(table1[table1['Country/Region'] == top_5[4]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)
plt.plot(table1[table1['Country/Region'] == top_5[5]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('total deaths', size=20)
plt.legend(top_5, prop={'size': 15})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# **Visual 8 :** Analysis for travel history associated countrywise.

# In[ ]:


# import re
# tempr = table2.drop(['latitude', 'longitude', 'wuhan(0)_not_wuhan(1)','date_onset_symptoms','date_admission_hospital','date_confirmation', 'lives_in_Wuhan',
#                      'sequence_available','notes_for_discussion', 'chronic_disease', 'outcome','date_death_or_discharge', 'chronic_disease_binary', 
#                      'reported_market_exposure'], axis=1)

# italy = tempr[tempr['country'] == 'Italy'][['country','travel_history_location']]
# for index,row in tempr.iterrows():
#     if(row['travel_history_location']) != (row['travel_history_location']):
#         row['travel_history_location'] = np.nan
#     else:
#         if re.findall(r"Wuhan", row['travel_history_location']) or re.findall(r"wuhan", row['travel_history_location']):
#             print(row['travel_history_location'])
#             tempr.loc[index, 'travel_history_location'] = 'Wuhan'
#             print(row['travel_history_location'])
#             print('\n')

# D = tempr['travel_history_location'].value_counts().to_dict()
# plt.bar(range(len(D)), list(D.values()))
# plt.xticks(range(len(D)), list(D.keys()), rotation=90)
# plt.show()


# Note: This kernel is highly inspired from few other kaggle kernels and other data science resources. Any traces of replications, which may appear, is purely for the purpose to use them for further developement. Due respect & credit to all my fellow kagglers. Thanks
