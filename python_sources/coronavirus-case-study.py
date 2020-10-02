#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap


# In[ ]:


print('Last updated: ',datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))


# In[ ]:


# Read the datasets
wc = pd.read_csv("../input/world-coordinates/world_coordinates.csv")
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df


# In[ ]:


# Drop duplicate entries, if any
df.drop_duplicates(inplace=True)
df


# In[ ]:


# Remove columns not required for study
df.drop(['SNo','Last Update'], axis=1, inplace=True)
# Rename certain values
df.rename(columns={'ObservationDate':'Date'}, inplace=True)
df['Country/Region'].replace({'Mainland China':'China'},inplace=True)
df.head()


# In[ ]:


# List of affected provinces/states
aff_ps = df['Province/State'].unique()
print(aff_ps)
print("Total:", len(aff_ps))


# In[ ]:


# Number of cases in each Province/State
case_ps = df.groupby('Province/State', as_index=False)[['Confirmed','Deaths','Recovered']].max()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): # Prevent truncation
    display(case_ps) # Maintain rich formatting by using display() instead of print()


# In[ ]:


# List of affected countries/regions
aff_c = df['Country/Region'].unique()
print(aff_c)
print("Total:", len(aff_c))


# In[ ]:


# Number of cases in each Country/Region
case_c = df.groupby(['Country/Region', 'Date']).sum().reset_index()
case_c = case_c.sort_values('Date', ascending=False)
case_c = case_c.drop_duplicates(subset = ['Country/Region'])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  display(case_c.sort_values('Country/Region')[['Country/Region','Confirmed','Deaths','Recovered']].reset_index(drop=True))


# In[ ]:


# Total number of cases
print("Total Confirmed:",case_c['Confirmed'].sum())
print("Total Deaths:",case_c['Deaths'].sum())
print("Total Recovered:",case_c['Recovered'].sum())


# In[ ]:


# Plot top 10 countries with confirmed cases
plt.rcParams['figure.figsize']=(16,8)
sns.barplot(x='Country/Region', y='Confirmed', data=case_c.nlargest(10,'Confirmed'))
plt.xticks(rotation=90)
plt.xlabel('10 most affected countries',fontsize=15)
plt.ylabel('Number of cases',fontsize=15)


# In[ ]:


# Plot top 10 countries with death cases
plt.rcParams['figure.figsize']=(16,8)
sns.barplot(x='Country/Region', y='Deaths', data=case_c.nlargest(10,'Deaths'))
plt.xticks(rotation=90)
plt.xlabel('10 countries with most deaths',fontsize=15)
plt.ylabel('Number of cases',fontsize=15)


# In[ ]:


# Time-series analysis
df_date = df.groupby('Date', as_index=False)[['Confirmed','Deaths','Recovered']].sum()

# If Timestamp is required, run the following code
# df['Timestamp'] = pd.to_datetime(df['Date']).astype(int)/10**10

# If data-time is given, convert to date format
# df_date['Date'] = pd.to_datetime(df_date['Date']).dt.date

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(df_date)


# In[ ]:


# Plot the confirmed cases
plt.plot('Date', 'Confirmed', data=df_date.groupby(['Date']).sum().reset_index(), color='blue')
plt.xticks(rotation=80)
plt.xlabel('Dates',fontsize=12)
plt.ylabel('Number of cases',fontsize=12)
plt.legend()
plt.rcParams['figure.figsize']=(16,8)
plt.show()


# In[ ]:


# Plot the deaths & recoveries
plt.plot('Date', 'Deaths', data=df_date.groupby(['Date']).sum().reset_index(), color='red')
plt.plot('Date', 'Recovered', data=df_date.groupby(['Date']).sum().reset_index(), color='green')
plt.xticks(rotation=80)
plt.xlabel('Dates',fontsize=12)
plt.ylabel('Number of cases',fontsize=12)
plt.legend()
plt.rcParams['figure.figsize']=(16,8)
plt.show()


# In[ ]:


# Plot mortality rate
df_date['Mortality']=df_date.apply(lambda x: (x['Deaths']/x['Confirmed'])*100, axis=1)
plt.plot('Date', 'Mortality', data=df_date, color='red')
plt.xticks(rotation=80)
plt.xlabel('Dates',fontsize=12)
plt.ylabel('Death percentage',fontsize=12)
plt.legend()
plt.rcParams['figure.figsize']=(16,8)
plt.show()


# In[ ]:


# Merge world coordinates with covid_19 dataframe
case_c.rename(columns={'Country/Region':'Country'}, inplace=True)
wc_df = pd.merge(wc,case_c,on='Country') # Might not be an exact match due to unequal data
wc_df.drop(['Code','Date','Deaths','Recovered'], axis=1, inplace=True)


# In[ ]:


# Heatmap using Folium
heatmap = folium.Map(location=[38.963745, 35.243322], zoom_start=2)

heat_data = [[row['latitude'],row['longitude'],row['Confirmed']] for index, row in wc_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(heatmap)

# Display the map
heatmap

