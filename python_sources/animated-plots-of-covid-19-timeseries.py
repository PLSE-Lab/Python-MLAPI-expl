#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

from datetime import datetime
import plotly.express as px
import math


# In[ ]:


dataset = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['ObservationDate', 'Last Update'])

dfConfirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
dfRecovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
dfDeaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
dfConfirmed.fillna(0, inplace=True)
dfRecovered.fillna(0, inplace=True)
dfDeaths.fillna(0, inplace=True)


# 1. **Animated Scatter Plot**

# In[ ]:


dataset['ObservationDate'] = dataset['ObservationDate'].apply(lambda x: datetime.combine(x.date(), datetime.min.time()))
dataset['Country/Region'] = dataset['Country/Region'].apply(lambda x: x.strip().replace("('", '').replace("',)", ''))
tmp = dataset[['ObservationDate', 'Country/Region', 'Confirmed', 'Deaths']].groupby(['ObservationDate', 'Country/Region']).sum()
tmp = tmp.unstack(1).fillna(0)


# In[ ]:


tmp.sort_index(inplace=True)
cumsum = tmp.copy(deep=True)
cumsum = tmp.cumsum(axis=0)
dataset = cumsum.T.unstack().T.reset_index()
for i, d in enumerate(sorted(dataset['ObservationDate'].unique())):
    dataset.loc[dataset['ObservationDate']==d,'days'] = i


# In[ ]:


#increase 0 values to apply log scale
dataset_log = dataset.copy(deep=True)
dataset_log.loc[dataset_log.Confirmed == 0, 'Confirmed'] = dataset_log.loc[dataset_log.Confirmed == 0, 'Confirmed'] + 0.1
dataset_log.loc[dataset_log.Deaths == 0, 'Deaths'] = dataset_log.loc[dataset_log.Deaths == 0, 'Deaths'] + 0.1


# In[ ]:


fig = px.scatter(dataset_log
           , y="Deaths"
           , x="Confirmed"
           , animation_frame="days"
           , color="Country/Region"
           , hover_name="Country/Region"
           , range_y=[0.1,dataset_log.Deaths.max()+5000]
           , range_x = [0.1,dataset_log.Confirmed.max()+5000]
           , text="Country/Region"
           , size=len(dataset_log)*[20]
           , log_x=True
           , log_y=True
           , title='Time-lapse of COVID-19 by Country'
          )
fig.show()


# 2. **Animated World Map**

# In[ ]:


def reshape(df, value):
    return df.melt(id_vars=['Lat','Long','Province/State','Country/Region'], 
                var_name="Date", 
                value_name=value)

def prepare_data_country(df):
    df.Date = pd.to_datetime(df.Date)
    df.loc[:,'Date'] = df.loc[:,'Date'].dt.date
    #df.Date = df.Date.dt.to_pydatetime()

    df.loc[df['Province/State'] == 0, 'Country/State'] = df.loc[df['Province/State'] == 0, 'Country/Region']
    df.loc[df['Province/State'] != 0, 'Country/State'] = df.loc[df['Province/State'] != 0, 'Country/Region']                                                         + '-'                                                         + df.loc[df['Province/State'] != 0, 'Province/State']    
    df.drop(['Country/Region', 'Province/State'], axis=1, inplace=True)
    df.set_index(['Lat','Long','Date','Country/State'], inplace=True)
    return df


# In[ ]:


dfConfirmed = reshape(dfConfirmed, 'Confirmed')
dfRecovered = reshape(dfRecovered, 'Recovered')
dfDeaths = reshape(dfDeaths, 'Deaths')

dfConfirmed = prepare_data_country(dfConfirmed)
dfRecovered = prepare_data_country(dfRecovered)
dfDeaths = prepare_data_country(dfDeaths)


# In[ ]:


dfMap = pd.concat([dfConfirmed, dfRecovered, dfDeaths], axis=1
         , join='inner')
dfMap.reset_index(inplace=True)
dfMap.sort_values('Date', inplace=True)
dfMap['Date'] =dfMap['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))


# In[ ]:


dfMap.loc[dfMap['Recovered']>0, 'Recovered_log'] = dfMap.loc[dfMap['Recovered']>0, 'Recovered'].apply(lambda x: math.log(x))
dfMap.loc[dfMap['Confirmed']>0, 'Confirmed_log'] = dfMap.loc[dfMap['Confirmed']>0, 'Confirmed'].apply(lambda x: math.log(x))
dfMap.fillna(0, inplace=True)


# In[ ]:


fig = px.scatter_mapbox(dfMap[dfMap.Confirmed > 0], 
                        lat="Lat", 
                        lon="Long", 
                        hover_name="Country/State", 
                        hover_data=["Confirmed","Deaths","Recovered"], 
                        animation_frame="Date",
                        color='Recovered_log',
                        size='Confirmed_log',
                        size_max=15,
                        zoom=0.6,height=600
                       )

fig.update_layout(title='Time-lapse of COVID-19: confirmed and recovered cases and deaths',
                  font=dict(size=16)
                 )
fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)
fig.update_layout(margin={"r":10,"t":35,"l":0,"b":0})


fig.show()


# In[ ]:




