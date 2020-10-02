#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import express as px


# Any results you write to the current directory are saved as output.


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

base = '../input/covid19-in-india/'
age_group_data = pd.read_csv(base + 'AgeGroupDetails.csv')
covid_19_data = pd.read_csv(base + 'covid_19_india.csv')


# In[ ]:


print(age_group_data.info())
age_group_data['Percentage'] = age_group_data['Percentage'].apply(lambda row: float(row.strip('%')))
age_group_data['start_age'] = age_group_data['AgeGroup'].apply(lambda row: int(row.split('-')[0].strip('>=')) if row != 'Missing' else np.nan)
age_group_data.head(4)


# In[ ]:


covid_19_data['Date'] = covid_19_data['Date'].apply(lambda row: pd.to_datetime(row, format = '%d/%m/%y'))
print(covid_19_data.info())
covid_19_data.head(4)


# - Age data speaks about the fraction of people in each decade of age, affected by the Corona virus.
# - The dataset on covid_19 cases in India enumerates a detailed outline of the cases that have been confirmed in India (both Indian & Foreign nationals), along with the number of cured people and the fatalities.

# # Age-wise distribution of confirmed cases

# Let's look at the pie-chart of age-wise distribution people affected by the COVID-19.

# In[ ]:


labels = age_group_data['AgeGroup']
values = age_group_data['Percentage']

explode = [0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0]

fig, ax = plt.subplots()

ax.pie(values, labels = labels, explode = explode, autopct='%1.0f%%', 
       shadow=False, startangle=0, pctdistance = 1.15, labeldistance = 1.5)
ax.axis('equal')
ax.set_title('Age-Wise distribution of affected people')
ax.legend(frameon=True, bbox_to_anchor=(1.5,0.8))


print('Percentage of people above the age of 40, that are in the list of Confirmed Cases: {}%'.format(np.sum(age_group_data.loc[age_group_data['start_age'] >= 40, 'Percentage'])))


# It can be seen that most of the affected people lie in the age group of 20 to 69. This is very understandable considering that the disease is contagiously spread, and this is the age group that is most found in the day-to-day crowd.

# # Cases in India & comparison with other countries

# ## Growth of confirmed cases, recoveries & deaths in India

# In[ ]:


country_inception = pd.to_datetime('30/01/2020')
date_wise_cum_national_cases = covid_19_data.groupby(['Date'])['ConfirmedIndianNational'].sum()
date_wise_cum_fatalities = covid_19_data.groupby(['Date'])['Deaths'].sum()
date_wise_cum_recoveries = covid_19_data.groupby(['Date'])['Cured'].sum()

date_wise_cum_cases = pd.DataFrame(date_wise_cum_national_cases.values, index = date_wise_cum_national_cases.index, columns = ['ConfirmedNationalCases'])

for date in date_wise_cum_fatalities.index:
    date_wise_cum_cases.loc[date, 'Fatalities'] = date_wise_cum_fatalities[date]
    
for date in date_wise_cum_recoveries.index:
    date_wise_cum_cases.loc[date, 'Recoveries'] = date_wise_cum_recoveries[date]

date_wise_cum_foreign_cases = covid_19_data.groupby(['Date'])['ConfirmedForeignNational'].sum()

for date in date_wise_cum_foreign_cases.index:
    date_wise_cum_cases.loc[date, 'ConfirmedForeignNational'] = date_wise_cum_foreign_cases[date]

date_wise_cum_cases['TotalConfirmedCases'] = date_wise_cum_cases.apply(lambda row: row['ConfirmedNationalCases'] + row['ConfirmedForeignNational'], axis = 1)

for date in date_wise_cum_cases.index:
    date_wise_cum_cases.loc[date, 'days'] = (date - country_inception).days

india_lockdown_date = pd.to_datetime('25/03/2020')
india_days_marker = (india_lockdown_date - country_inception).days


# In[ ]:


plt.figure(figsize = (20,15))
plt.plot(date_wise_cum_cases['days'], date_wise_cum_cases['ConfirmedNationalCases'], label = 'National Cases')
plt.plot(date_wise_cum_cases['days'], date_wise_cum_cases['ConfirmedForeignNational'], label = 'Foreign Cases')
plt.plot(date_wise_cum_cases['days'], date_wise_cum_cases['TotalConfirmedCases'], label = 'Total Cases')
plt.plot(date_wise_cum_cases['days'], date_wise_cum_cases['Fatalities'], label = 'Fatalities')
plt.plot(date_wise_cum_cases['days'], date_wise_cum_cases['Recoveries'], label = 'Recoveries')
plt.title('Cases in India')
plt.legend(frameon = False, prop= {'size': 15})
plt.ylabel('Confirmed Cases')
plt.xlabel('Days after 30th Jan')
None


# In[ ]:


plt.figure(figsize = (20,15))
date_wise_cum_cases['TotalConfirmedCases'] = date_wise_cum_cases.apply(lambda row: row['ConfirmedNationalCases'] + row['ConfirmedForeignNational'], axis = 1)
plt.plot(date_wise_cum_cases['days'], np.log(date_wise_cum_cases['TotalConfirmedCases'] - date_wise_cum_cases['Recoveries']))
plt.title('Difference between total confirmed cases and recoveries: Log Scale')
plt.ylabel('Count')
plt.xlabel('Days')
plt.show()


# #### Inferences
# - The number of cases is increasing exponentially (as expected from observations in other countries). However, the number of foreign national cases is not as high, both in magnitude and growth, probably due to the scrutinized flights and lockdown of foreign flights.
# - Lately, the number of recoveries has also started increasing rapidly. On a log scale, the difference between total confirmed cases and total recoveries is linear in the recent trend. This implies that, in contrast to the exponential growth in total confirmed cases, there isn't a matching growth in the recoveries, sadly.

# ## Comparison with Spain, Italy, USA & China

# In[ ]:


world_data = pd.read_csv('../input/covid19-forecasting-week-two-launch-data/train.csv')
world_data['Date'] = world_data['Date'].apply(lambda date: pd.to_datetime(date, format = '%Y-%m-%d'))

china_data = world_data.loc[world_data['Country_Region'] == 'China']
china_date_of_inception = pd.to_datetime('17/11/2019')
china_date_wise_cases = china_data.groupby(['Date'])['ConfirmedCases'].sum()
china_date_wise_fatalities = china_data.groupby(['Date'])['Fatalities'].sum()
china_date_wise_cum_cases = pd.DataFrame(china_date_wise_cases.values, index = china_date_wise_cases.index, columns = ['TotalConfirmedCases'])
for date in china_date_wise_fatalities.index:
    china_date_wise_cum_cases.loc[date, 'Fatalities'] = china_date_wise_fatalities[date]

for date in china_date_wise_cum_cases.index:
    china_date_wise_cum_cases.loc[date, 'days'] = (date - china_date_of_inception).days
    
china_lockdown_date = pd.to_datetime('24/01/2020')
china_days_marker = (china_lockdown_date - china_date_of_inception).days
    
italy_data = world_data.loc[world_data['Country_Region'] == 'Italy']
italy_date_of_inception = pd.to_datetime('31/01/2020')
italy_date_wise_cases = italy_data.groupby(['Date'])['ConfirmedCases'].sum()
italy_date_wise_fatalities = italy_data.groupby(['Date'])['Fatalities'].sum()
italy_date_wise_cum_cases = pd.DataFrame(italy_date_wise_cases.values, index = italy_date_wise_cases.index, columns = ['TotalConfirmedCases'])
for date in italy_date_wise_fatalities.index:
    italy_date_wise_cum_cases.loc[date, 'Fatalities'] = italy_date_wise_fatalities[date]

for date in italy_date_wise_cum_cases.index:
    italy_date_wise_cum_cases.loc[date, 'days'] = (date - italy_date_of_inception).days
    
italy_lockdown_date = pd.to_datetime('09/03/2020', format = '%d/%m/%Y')
italy_days_marker = (italy_lockdown_date - italy_date_of_inception).days
    
spain_data = world_data.loc[world_data['Country_Region'] == 'Spain']
spain_date_of_inception = pd.to_datetime('31/01/2020')
spain_date_wise_cases = spain_data.groupby(['Date'])['ConfirmedCases'].sum()
spain_date_wise_fatalities = spain_data.groupby(['Date'])['Fatalities'].sum()
spain_date_wise_cum_cases = pd.DataFrame(spain_date_wise_cases.values, index = spain_date_wise_cases.index, columns = ['TotalConfirmedCases'])
for date in spain_date_wise_fatalities.index:
    spain_date_wise_cum_cases.loc[date, 'Fatalities'] = spain_date_wise_fatalities[date]

for date in spain_date_wise_cum_cases.index:
    spain_date_wise_cum_cases.loc[date, 'days'] = (date - spain_date_of_inception).days
    
spain_lockdown_date = pd.to_datetime('15/03/2020')
spain_days_marker = (spain_lockdown_date - spain_date_of_inception).days
    
us_data = world_data.loc[world_data['Country_Region'] == 'US']
us_date_of_inception = pd.to_datetime('19/01/2020')
us_date_wise_cases = us_data.groupby(['Date'])['ConfirmedCases'].sum()
us_date_wise_fatalities = us_data.groupby(['Date'])['Fatalities'].sum()
us_date_wise_cum_cases = pd.DataFrame(us_date_wise_cases.values, index = us_date_wise_cases.index, columns = ['TotalConfirmedCases'])
for date in us_date_wise_fatalities.index:
    us_date_wise_cum_cases.loc[date, 'Fatalities'] = us_date_wise_fatalities[date]

for date in us_date_wise_cum_cases.index:
    us_date_wise_cum_cases.loc[date, 'days'] = (date - us_date_of_inception).days
    


# In[ ]:


plt.figure(figsize = (15,10))

plt.plot(date_wise_cum_cases['days'], date_wise_cum_cases['TotalConfirmedCases'], label = 'India')
plt.plot(china_date_wise_cum_cases['days'], china_date_wise_cum_cases['TotalConfirmedCases'], label = 'China')
plt.plot(spain_date_wise_cum_cases['days'], spain_date_wise_cum_cases['TotalConfirmedCases'], label = 'Spain')
plt.plot(italy_date_wise_cum_cases['days'], italy_date_wise_cum_cases['TotalConfirmedCases'], label = 'Italy')
plt.plot(us_date_wise_cum_cases['days'], us_date_wise_cum_cases['TotalConfirmedCases'], label = 'USA')

plt.axvline(india_days_marker, 0, 100000, color = 'blue', linestyle = '--', label='India Lockdown')
plt.axvline(china_days_marker, 0, 100000, color = 'orange', linestyle = '--', label='China Lockdown')
plt.axvline(italy_days_marker, 0, 100000, color = 'red', linestyle = '--', label='Italy Lockdown')
plt.axvline(spain_days_marker, 0, 100000, color = 'green', linestyle = '--', label='Spain Lockdown')

plt.legend(frameon = False, prop={'size': 12})
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
plt.show()


# In[ ]:


plt.figure(figsize = (15,10))

plt.plot(date_wise_cum_cases['days'], date_wise_cum_cases['Fatalities'], label = 'India')
plt.plot(china_date_wise_cum_cases['days'], china_date_wise_cum_cases['Fatalities'], label = 'China')
plt.plot(spain_date_wise_cum_cases['days'], spain_date_wise_cum_cases['Fatalities'], label = 'Spain')
plt.plot(italy_date_wise_cum_cases['days'], italy_date_wise_cum_cases['Fatalities'], label = 'Italy')
plt.plot(us_date_wise_cum_cases['days'], us_date_wise_cum_cases['Fatalities'], label = 'USA')

plt.axvline(india_days_marker, 0, 100000, color = 'blue', linestyle = '--', label='India Lockdown')
plt.axvline(china_days_marker, 0, 100000, color = 'orange', linestyle = '--', label='China Lockdown')
plt.axvline(italy_days_marker, 0, 100000, color = 'red', linestyle = '--', label='Italy Lockdown')
plt.axvline(spain_days_marker, 0, 100000, color = 'green', linestyle = '--', label='Spain Lockdown')

plt.legend(frameon = False, prop={'size': 12})
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
plt.show()


# Dates of first cases by country - 
# 1. China: 17th November, 2019
# 2. Italy: 31st January, 2020
# 3. Spain: 31st January, 2020
# 4. US: 19th January, 2020
# 5. India: 30th January, 2020

# #### Inferences from the above plots
# - India seems to be growing very slowly as compared to the other nations. This seems to be a positive news for us all, but we have to look at the rate tests are conducted in India as compared to these big nations.
# - Strangely, given the first case in China was reported on 17th November, 2019, their curve doesn't start exploding until **_3 months_** later - in February.
# - In general, containment ranking among Spain, Italy and USA can be given as -
#         USA > Spain > Italy
#     From the plots of both confirmed cases and the fatalities, it seems that India is outperforming even US. Hope this continues and we get over this pandemic quickly.
# - India has been quicker than the other countries - USA, Italy and Spain, in enforcing a lockdown. However, given the population density of India and the incubation period of the virus, it is too early to boldly state the effectivity of the lockdown

# # State wise data visualization and analysis

# ## Spread of the Virus across India - Cluster Formations

# In[ ]:


latitudes = {'Kerala': 8.900372741, 'Telengana': 17.39998313, 'Delhi': 28.6699929, 'Rajasthan': 26.44999921, 'Uttar Pradesh': 27.59998069, 'Haryana': 28.45000633, 'Ladakh': 34.152588, 'Tamil Nadu': 12.92038576, 'Karnataka': 12.57038129, 'Maharashtra': 19.25023195, 'Punjab': 31.51997398, 'Jammu and Kashmir': 34.29995933, 'Andhra Pradesh': 14.7504291, 'Uttarakhand': 30.32040895, 'Odisha': 19.82042971, 'Pondicherry': 11.93499371, 'West Bengal': 22.58039044, 'Chattisgarh': 22.09042035, 'Chandigarh': 30.71999697, 'Gujarat': 22.2587, 'Himachal Pradesh': 31.10002545, 'Madhya Pradesh': 21.30039105, 'Bihar': 25.78541445, 'Manipur': 24.79997072, 'Mizoram': 23.71039899, 'Andaman and Nicobar Islands': 11.66702557, 'Goa': 15.491997}
longitudes = {'Kerala': 76.56999263, 'Telengana': 78.47995357, 'Delhi': 77.23000403,'Rajasthan': 74.63998124,'Uttar Pradesh': 78.05000565,'Haryana': 77.01999101,'Ladakh': 77.577049,'Tamil Nadu': 79.15004187,'Karnataka': 76.91999711,'Maharashtra': 73.16017493,'Punjab': 75.98000281,'Jammu and Kashmir': 74.46665849,'Andhra Pradesh': 78.57002559,'Uttarakhand': 78.05000565,'Odisha': 85.90001746,'Pondicherry': 79.83000037,'West Bengal': 88.32994665,'Chattisgarh': 82.15998734,'Chandigarh': 76.78000565,'Gujarat': 71.1924,'Himachal Pradesh': 77.16659704,'Madhya Pradesh': 76.13001949,'Bihar': 87.4799727,'Manipur': 93.95001705,'Mizoram': 92.72001461,'Andaman and Nicobar Islands': 92.73598262, 'Goa': 73.81800065}

covid_19_data.loc[covid_19_data['State/UnionTerritory'] == 'Chhattisgarh', 'State/UnionTerritory'] = 'Chattisgarh'
covid_19_data.loc[covid_19_data['State/UnionTerritory'] == 'Puducherry', 'State/UnionTerritory'] = 'Pondicherry'

covid_19_data['Latitudes'] = covid_19_data.apply(lambda row: latitudes[row['State/UnionTerritory']], axis = 1)
covid_19_data['Longitudes'] = covid_19_data.apply(lambda row: longitudes[row['State/UnionTerritory']], axis = 1)
covid_19_data['TotalConfirmedCases'] = covid_19_data.apply(lambda row: row['ConfirmedIndianNational'] + row['ConfirmedForeignNational'], axis = 1)
covid_19_data['Date_Value'] = covid_19_data['Date'].apply(str)


# In[ ]:


fig = px.scatter_geo(covid_19_data, 
                    lat = 'Latitudes',
                    lon = 'Longitudes',
                    color="TotalConfirmedCases", # which column to use to set the color of markers
                    hover_data=['TotalConfirmedCases', 'Deaths', 'State/UnionTerritory'], # column added to hover information
                    size="TotalConfirmedCases", # size of markers
                    range_color= [0, covid_19_data['TotalConfirmedCases'].max()],
                    projection="orthographic",
                    animation_frame="Date_Value", 
                    title='Spread of confirmed cases over India',
                    color_continuous_scale="portland")
fig.update_layout(height=700)
fig.update_geos(
    lataxis_range=[0,40], lonaxis_range=[60, 100]
)
fig.show()


# #### Inferences from the above visualization
# 
# - This visualization shows the spread of the virus in India, and the cluster formations. Two clusters, which are quite profound, are **Kerala** and **Maharashtra** - with 186 and 182 confirmed cases respectively, and 6 and 1 deaths respectively.
# - Kerala was the first affected place in India.
# - There are small clusters forming in North India. Given the population density of India, this can prove harmful, and lead to a massive single cluster, if people abstain from good practices, such as self-quarantine, sanitization etc.
# - The clusters in rest of India are significantly large, mostly, but are sparsely situated. Hence, proper caution can gradually lead to the death of these clusters.
