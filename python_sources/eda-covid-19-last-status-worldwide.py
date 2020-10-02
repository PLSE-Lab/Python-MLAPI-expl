#!/usr/bin/env python
# coding: utf-8

# # EDA COVID-19 for last status
# 
# This notebook explore only the last status of covid-19 for each country or worldwide.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

sns.set()


# # Feature Engineering

# In[ ]:


df_covid = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df_coord = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')


# ## Merge datasets

# In[ ]:


# Create relation latitude and longitude per country/province
df_coord = df_coord.groupby(['Province/State', 'Country/Region']).agg({'Lat' : 'first', 'Long' : 'first'})
df_coord.reset_index(inplace=True)


# In[ ]:


df_covid['Country/Region'] = df_covid['Country/Region'].replace('Mainland China', 'China')
df_covid = df_covid.merge(df_coord, how='left')


# ## Force columns to be snake case 

# In[ ]:


cols = df_covid.columns
df_covid.columns = [col.lower() for col in cols]


# In[ ]:


df_covid.rename(columns={
    'observationdate' : 'observation_date',
    'country/region' : 'country',
    'province/state' : 'province_state', 
    'last update' : 'last_update',
}, inplace=True)


# ## Drop unwanted columns
# 
# Even thought is just one columns, there's no need to keep it

# In[ ]:


df_covid.drop('sno', axis=1, inplace=True)


# ## Fixing missing values

# In[ ]:


df_covid.isnull().any()


# In[ ]:


df_covid['province_state'].fillna('', inplace=True)
df_covid['lat'].fillna('', inplace=True)
df_covid['long'].fillna('', inplace=True)


# ## Transform date column to date format
# 
# This will be usefull to sort our data, even if our dataset looks sorted it's always a good idea to do it again.

# In[ ]:


df_covid['observation_date'] = pd.to_datetime(df_covid['observation_date'])
df_covid['last_update'] = pd.to_datetime(df_covid['last_update'])

df_covid.sort_values('observation_date', inplace=True)


# ## Feature extraction

# In[ ]:


df_covid['diseased'] = df_covid['confirmed'] - df_covid['recovered'] - df_covid['deaths']
df_covid['observation_month'] = df_covid['observation_date'].astype(str).str.extract(r'(\d{4}-\d{2})')
df_covid['last_update_month'] = df_covid['last_update'].astype(str).str.extract(r'(\d{4}-\d{2})')


# ## Get updated information of every region
# 
# In some analysis we're only interested in the last update for each region

# In[ ]:


df_covid_last = df_covid.groupby(['country', 'province_state']).last()
df_covid_last.reset_index(inplace=True)


# ## Group updated informations by country

# In[ ]:


df_grouped = df_covid_last[['country', 'confirmed', 'deaths', 'recovered', 'diseased']]    .groupby('country')    .sum()    .sort_values('recovered', ascending=False)

# Removed countries without any confirmed case of covid-19
df_grouped = df_grouped[df_grouped['confirmed'] > 0]

# Get percentages of recovered, deaths and diseased
df_grouped['pct_recovered'] = round(df_grouped['recovered'] / df_grouped['confirmed'], 4)
df_grouped['pct_deaths'] = round(df_grouped['deaths'] / df_grouped['confirmed'], 4)
df_grouped['pct_diseased'] = round(df_grouped['diseased'] / df_grouped['confirmed'], 4)


# # EDA

# ## Last update worldwide

# In[ ]:


df_covid_last[['confirmed', 'recovered', 'deaths', 'diseased']].sum()


# In[ ]:


df_covid_last[['recovered', 'deaths', 'diseased']]    .sum()    .plot(kind='pie', subplots=True, figsize=(8, 8), autopct='%1.1f%%')

fig = plt.gcf()

fig.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))


# ## Density plot
# 
# We're missing a lot of informaton for lack of lat and long, i'll keep this plot but i won't pay much attention to it

# In[ ]:


df_density_mapbox = df_covid_last[df_covid_last['lat'] != ''][['country', 'province_state', 'lat', 'long', 'confirmed','deaths','recovered', 'diseased']]

fig = px.density_mapbox(
    df_density_mapbox,
    lat = 'lat',
    lon = 'long',
    color_continuous_scale='Inferno',
    hover_name = 'province_state',
    hover_data = ['confirmed','deaths','recovered', 'diseased'],
    radius = 10, zoom = 1, height = 600
)

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)
fig.update_layout()
fig.show()


# ## Distribution of confirmed cases worldwide
# 
# Let's see in one plot how bad is the situation for confirmed cases.

# In[ ]:


plt.figure(figsize=(14, 6))

sns.distplot(np.log10(df_grouped['confirmed']))


# ## Last status per countries

# In[ ]:


df_grouped['confirmed_log'] = np.log10(df_grouped['confirmed'])
df_grouped['deaths_log'] = np.log10(df_grouped['deaths'])


# ### Confirmed cases

# In[ ]:


plt.figure(figsize=(13, 11))

x = df_grouped[df_grouped['confirmed_log'] > 0].sort_values('confirmed_log', ascending=False)

sns.barplot(x=x['confirmed_log'], y=x.index);
plt.show()


# #### Top 10

# In[ ]:


plt.figure(figsize=(13, 11))

x = df_grouped[df_grouped['confirmed_log'] > 0].sort_values('confirmed_log', ascending=False)[:10]

sns.barplot(x=x['confirmed_log'], y=x.index);
plt.show()


# ### Deaths

# In[ ]:


plt.figure(figsize=(13, 11))

x = df_grouped[df_grouped['deaths_log'] > 0].sort_values('deaths_log', ascending=False)

sns.barplot(x=x['deaths_log'], y=x.index);
plt.show()


# #### Top 10

# In[ ]:


plt.figure(figsize=(13, 11))

x = df_grouped[df_grouped['deaths_log'] > 0].sort_values('deaths_log', ascending=False)[:10]

sns.barplot(x=x['deaths_log'], y=x.index);
plt.show()


# ## Countries with high death rate
# 
# I'm currently using **5% as my high death threshold** and **0.1% as my low death threshold** but **this is very subjective** and can differ from differents perspectives. Also, **i do not want to count countries that have less than 30 confimed cases of covid-19**

# In[ ]:


threshold_high_death_rate = 0.05
threshold_low_death_rate  = 0.001
threshold_confimed_cases  = 30


# In[ ]:


f"{df_grouped[(df_grouped['pct_deaths'] >= threshold_high_death_rate) & (df_grouped['confirmed'] >= threshold_confimed_cases)].shape[0]} countries with high death rate"


# In[ ]:


df_grouped[(df_grouped['pct_deaths'] >= threshold_high_death_rate) & (df_grouped['confirmed'] >= threshold_confimed_cases)]    .sort_values('confirmed', ascending=False)


# ## Countries with low death rate

# In[ ]:


f"{df_grouped[(df_grouped['pct_deaths'] <= threshold_low_death_rate) & (df_grouped['confirmed'] >= threshold_confimed_cases)].shape[0]} countries with high death rate"


# In[ ]:


df_grouped[(df_grouped['pct_deaths'] <= threshold_low_death_rate) & (df_grouped['confirmed'] >= threshold_confimed_cases)]    .sort_values('confirmed', ascending=False)


# ## How many months been past since first occurancy for each country

# In[ ]:


df_first_occurency = df_covid.groupby('country').agg({
    'observation_month' : 'nunique'
}) 

df_first_occurency.rename(columns={'observation_month' : 'qnt_months_since_first_occurency'}, inplace=True)


# In[ ]:


df_first_occurency['qnt_months_since_first_occurency'].value_counts()


# The majority of the countires are still in their first month of covid-19

# ### Relation of deaths per months passed since first occurence of covid-19

# In[ ]:


df_covid_last = df_covid_last.set_index('country')    .join(df_first_occurency['qnt_months_since_first_occurency'])

df_grouped_first_occurency = df_covid_last.groupby('qnt_months_since_first_occurency').sum()


# In[ ]:


# Percentage change between the current and a prior element
df_grouped_first_occurency.pct_change()


# There's still small amount of data to infer anything, i'll keep this here tho

# ## Countries that recovered every diseased

# In[ ]:


df_grouped[df_grouped['pct_recovered'] == 1]


# In[ ]:


df_tmp = df_covid[df_covid['country'].isin(df_grouped[df_grouped['pct_recovered'] == 1].index)]    .groupby('country')    .agg({'observation_month' : ['first', 'last']})


# In[ ]:


df_grouped[
    (df_grouped['pct_recovered'] == 1) & 
    (df_grouped.index.isin(df_tmp[df_tmp['observation_month']['first'] != df_tmp['observation_month']['last']].index))
]


# This countries have more than one month since first occurency and recovery all diseased

# # Last comments
# 
# I do find this subject really interesting and i'm not done with this notebook yet, fell free to contribute with toughts and suggestions :D
