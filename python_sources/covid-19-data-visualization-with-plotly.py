#!/usr/bin/env python
# coding: utf-8

# ## Visualizations using PlotLy
# 
# ### Please upvote if you like this notebook.
# 
# #### I would also appreciate any suggestions you might have.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly.express as px

train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv", index_col = 'Id')
test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv", index_col = 'ForecastId')
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


train_df.rename(columns={"Country_Region": "country", "Province_State": "province"}, inplace=True, errors="raise")
df = train_df.fillna('NA').groupby(['country','province','Date'])['ConfirmedCases'].sum()                           .groupby(['country','province']).max().sort_values()                           .groupby(['country']).sum().sort_values(ascending = False)

top10 = pd.DataFrame(df).head(10)
fig = px.bar(top10, x=top10.index, y='ConfirmedCases', labels={'x':'Country'},
             color="ConfirmedCases", color_continuous_scale=px.colors.sequential.Brwnyl)
fig.update_layout(title_text='Confirmed COVID-19 cases by country')
fig.show()


# In[ ]:


df_by_date = pd.DataFrame(train_df.fillna('NA').groupby(['country','Date'])['ConfirmedCases'].sum().sort_values().reset_index())

fig = px.bar(df_by_date.loc[(df_by_date['country'] == 'Canada') &(df_by_date.Date >= '2020-03-01')].sort_values('ConfirmedCases',ascending = False), 
             x='Date', y='ConfirmedCases', color="ConfirmedCases", color_continuous_scale=px.colors.sequential.BuGn)
fig.update_layout(title_text='Confirmed COVID-19 cases per day in Canada')
fig.show()


# In[ ]:


fig = px.bar(df_by_date.loc[(df_by_date.country == 'Russia') &(df_by_date.Date >= '2020-03-01')].sort_values(['Date','ConfirmedCases'],ascending = False),
             x='Date', y='ConfirmedCases', color="ConfirmedCases", color_continuous_scale=px.colors.sequential.Blues)
fig.update_layout(title_text='Confirmed COVID-19 cases per day in Russia')
fig.show()


# In[ ]:


fig = px.bar(df_by_date.loc[(df_by_date['country'] == 'US') &(df_by_date.Date >= '2020-03-02')].sort_values('ConfirmedCases',ascending = False), x='Date', 
             y='ConfirmedCases', color="ConfirmedCases", color_continuous_scale=px.colors.sequential.Brwnyl)
fig.update_layout(title_text='Confirmed COVID-19 cases per day in US')
fig.show()


# In[ ]:


fig = px.bar(df_by_date.loc[(df_by_date['country'] == 'Italy' ) &(df_by_date.Date >= '2020-03-02')].sort_values('ConfirmedCases',ascending = False), x='Date', y='ConfirmedCases',
            color="ConfirmedCases", color_continuous_scale=px.colors.sequential.Mint)
fig.update_layout(title_text='Confirmed COVID-19 cases per day in Italy')
fig.show()


# In[ ]:


fig = px.bar(df_by_date.loc[df_by_date['country'] == 'China'].sort_values('ConfirmedCases',ascending = False), x='Date', y='ConfirmedCases',
            color="ConfirmedCases", color_continuous_scale=px.colors.sequential.Purples)
fig.update_layout(title_text='Confirmed COVID-19 cases per day in China')
fig.show()


# Confirmed Cases vs Deaths by Country

# In[ ]:


df = train_df.fillna('NA').groupby(['country','province','Date'])['ConfirmedCases','Fatalities'].sum()                           .groupby(['country','province']).max().sort_values(by='ConfirmedCases')                           .groupby(['country']).sum().sort_values(by='ConfirmedCases',ascending = False)

df = pd.DataFrame(df).reset_index()


df = pd.DataFrame(df)

df_new_cases = pd.DataFrame(train_df.fillna('NA').groupby(['country','Date'])['ConfirmedCases'].sum()                             .reset_index()).sort_values(['country','Date'])
df_new_cases.ConfirmedCases = df_new_cases.ConfirmedCases.diff().fillna(0)
df_new_cases = df_new_cases.loc[df_new_cases['Date'] == max(df_new_cases['Date']),['country','ConfirmedCases']]
df_new_cases.rename(columns={"ConfirmedCases": "NewCases"}, inplace=True, errors="raise")

df_new_deaths = pd.DataFrame(train_df.fillna('NA').groupby(['country','Date'])['Fatalities'].sum()                             .reset_index()).sort_values(['country','Date'])

df_new_deaths.Fatalities = df_new_deaths.Fatalities.diff().fillna(0)
df_new_deaths = df_new_deaths.loc[df_new_deaths['Date'] == max(df_new_deaths['Date']),['country','Fatalities']]

df_new_deaths.rename(columns={"Fatalities": "NewFatalities"}, inplace=True, errors="raise")

merged = df.merge(df_new_cases, left_on='country', right_on='country')            .merge(df_new_deaths, left_on='country', right_on='country')


merged.style.background_gradient(cmap="Blues", subset=['ConfirmedCases'])            .background_gradient(cmap="Reds", subset=['Fatalities'])            .background_gradient(cmap="Blues", subset=['NewCases'])            .background_gradient(cmap="Reds", subset=['NewFatalities'])


# In[ ]:


df_by_date.ConfirmedCases = df_by_date.ConfirmedCases.diff().fillna(0)
df_by_date.index = pd.to_datetime(df_by_date.Date)


# In[ ]:


# install calmap
get_ipython().system(' pip install calmap')


# In[ ]:


import matplotlib.pyplot as plt
import calmap


# In[ ]:


fig = plt.figure(figsize=(20, 3));
ax = fig.add_subplot(111)
cax = calmap.yearplot(df_by_date.ConfirmedCases, fillcolor='white', cmap='Blues', linewidth=0.5)
fig.colorbar(cax.get_children()[1], ax=cax, orientation='horizontal')
plt.title('Number of new confirmed cases per day worldwide');


# In[ ]:


ft_by_date = train_df.fillna('NA').groupby(['country','Date'])['Fatalities'].sum().sort_values().reset_index()                           .groupby(['Date'])['Fatalities'].sum().sort_values()

ft_by_date.index = pd.to_datetime(ft_by_date.index)
ft_by_date = ft_by_date.diff().fillna(0)


# In[ ]:


fig = plt.figure(figsize=(20,3))

ax = fig.add_subplot(111)
cax = calmap.yearplot(ft_by_date, fillcolor='white', cmap='Reds', linewidth=0.5)
fig.colorbar(cax.get_children()[1], ax=cax, orientation='horizontal')
plt.title('Number of deaths per day worldwide');


# In[ ]:


# Remove columns we do not need
cols = ['Fatalities']
times_series_cntr = train_df.drop(cols, axis=1).fillna('N/A')

# Aggregate cases by date and country
times_series_cntr = times_series_cntr.groupby(['Date','province','country'])['ConfirmedCases'].max()                    .groupby(['Date','country']).sum()                    .reset_index()

# Indexing with Time Series Data
times_series_cntr = times_series_cntr.set_index('Date')


# In[ ]:


import seaborn as sns
from matplotlib import rcParams, pyplot as plt, style as style

style.use('ggplot')
rcParams['figure.figsize'] = 15,10
country_province = train_df.fillna('N/A').groupby(['country','province'])['ConfirmedCases', 'Fatalities'].max().sort_values(by='ConfirmedCases', ascending=False)

countries = country_province.groupby('country')['ConfirmedCases','Fatalities'].sum().sort_values(by= 'ConfirmedCases',ascending=False)

countries['country'] = countries.index

# Unpivot the dataframe from wide to long format
df_long = pd.melt(countries, id_vars=['country'] , value_vars=['ConfirmedCases','Fatalities'])

#Top countries by confirmed cases
top_countries = countries.index[:10]

top_countries_tm = times_series_cntr[times_series_cntr['country'].isin(top_countries)]
plt.xticks(rotation=45)


ax = sns.lineplot(x=top_countries_tm.index, y="ConfirmedCases", hue="country", data=top_countries_tm).set_title('Cumulative line')
plt.legend(loc=2, prop={'size': 12});

