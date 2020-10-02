#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_df=pd.read_csv("../input/covid-predictive-analysis/train.csv")
train_df = train_df.drop('Province_State', axis=1)
train_df.head()


# In[ ]:


df=train_df.groupby(['Country_Region','Date'])['Confirmed'].sum()             .groupby(['Country_Region']).max().sort_values()             .groupby(['Country_Region']).sum().sort_values(ascending = False)

#df.head(10)

top10 = pd.DataFrame(df).head(10)
top10


# In[ ]:


import plotly.express as px
figure = px.bar(top10, x=top10.index, y='Confirmed', labels={'x':'Country'},
             color="Confirmed", color_continuous_scale=px.colors.sequential.Brwnyl)
figure.update_layout(title_text='Confirmed COVID-19 cases by country')
figure.show()


# In[ ]:


df_by_date = pd.DataFrame(train_df.groupby(['Country_Region','Date'])['Confirmed'].sum().sort_values().reset_index())
df_by_date.head(10)


# In[ ]:


figure = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'US') &(df_by_date.Date >= '2020-01-22')].sort_values('Confirmed',ascending = False), 
             x='Date', y='Confirmed', color="Confirmed", color_continuous_scale=px.colors.sequential.BuGn)
figure.update_layout(title_text='Confirmed COVID-19 cases per day in US')
figure.show()


# In[ ]:


figure = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'US') &(df_by_date.Date >= '2020-03-15')].sort_values('Confirmed',ascending = False), 
             x='Date', y='Confirmed', color="Confirmed", color_continuous_scale=px.colors.sequential.BuGn)
figure.update_layout(title_text='Confirmed COVID-19 cases per day in US')
figure.show()


# In[ ]:


figure = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'Turkey') &(df_by_date.Date >= '2020-01-22')].sort_values('Confirmed',ascending = False), 
             x='Date', y='Confirmed', color="Confirmed", color_continuous_scale=px.colors.sequential.BuGn)
figure.update_layout(title_text='Confirmed COVID-19 cases per day in Turkey')
figure.show()


# In[ ]:


figure = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == 'Turkey') &(df_by_date.Date >= '2020-03-15')].sort_values('Confirmed',ascending = False), 
             x='Date', y='Confirmed', color="Confirmed", color_continuous_scale=px.colors.sequential.BuGn)
figure.update_layout(title_text='Confirmed COVID-19 cases per day in Turkey')
figure.show()


# In[ ]:


df=train_df.groupby(['Date','Country_Region']).agg('sum').reset_index()
df.tail(10)


# In[ ]:


train_df['Date'] = pd.to_datetime(train_df['Date'])


# In[ ]:


import matplotlib.pyplot as plt

case='Confirmed'
def timeCompare(time,*argv):
    Coun1=argv[0]
    Coun2=argv[1]
    f,ax=plt.subplots(figsize=(16,5))
    labels=argv  
    country=df.loc[(df['Country_Region']==Coun1)]
    plt.plot(country['Date'],country[case],linewidth=2)
    plt.xticks([])
    plt.legend(labels)
    ax.set(title=' Evolution of actual cases',ylabel='Number of cases' )

    country2=df.loc[df['Country_Region']==Coun2]
    #country2['Date']=country2['Date']-datetime.timedelta(days=time)
    plt.plot(country2['Date'],country2[case],linewidth=2)
    #plt.xticks([])
    plt.legend(labels)
    ax.set(title=' Evolution of cases in %d days difference '%time ,ylabel='Number of %s cases'%case )


# In[ ]:


timeCompare(7,'Italy','Turkey')
timeCompare(7,'Italy','US')


# In[ ]:


timeCompare(7,'Turkey','US')
timeCompare(7,'Turkey','France')


# In[ ]:


case='Deaths'
def timeCompare_f(time,*argv):
    Coun1=argv[0]
    Coun2=argv[1]
    f,ax=plt.subplots(figsize=(16,5))
    labels=argv  
    country=df.loc[(df['Country_Region']==Coun1)]
    plt.plot(country['Date'],country[case],linewidth=2)
    plt.xticks([])
    plt.legend(labels)
    ax.set(title=' Evolution of actual cases',ylabel='Number of cases' )

    country2=df.loc[df['Country_Region']==Coun2]
    #country2['Date']=country2['Date']-datetime.timedelta(days=time)
    plt.plot(country2['Date'],country2[case],linewidth=2)
    #plt.xticks([])
    plt.legend(labels)
    ax.set(title=' Evolution of Deaths in %d days difference '%time ,ylabel='Number of %s cases'%case )


# In[ ]:


timeCompare_f(7,'Italy','Turkey')
timeCompare_f(7,'Italy','US')


# In[ ]:


timeCompare_f(7,'Turkey','US')
timeCompare_f(7,'Turkey','France')


# In[ ]:


train_df.info()


# In[ ]:


from fbprophet import Prophet
df = train_df.loc[:, ["Date","Deaths"]]
df['Date'] = pd.DatetimeIndex(df['Date'])
df.dtypes


# In[ ]:


df = df.rename(columns={'Date': 'ds',
                        'Deaths': 'y'})

ax = df.set_index('ds').plot(figsize=(20, 12))
ax.set_ylabel('Monthly Deaths')
ax.set_xlabel('Date')

plt.show()


# In[ ]:


def countryCompare(df):
    df = df.rename(columns={'Date': 'ds',
                            'Deaths': 'y'})

    my_model = Prophet()
    my_model.fit(df)

 
    future_dates = my_model.make_future_dataframe(periods=100)
    forecast =my_model.predict(future_dates)

    #fig2 = my_model.plot_components(forecast)


    forecastnew = forecast['ds']
    forecastnew2 = forecast['yhat']

    forecastnew = pd.concat([forecastnew,forecastnew2], axis=1)
    mask = (forecastnew['ds'] > "5-1-2020") & (forecastnew['ds'] <= "5-10-2020")
    forecastedvalues = forecastnew.loc[mask]

    mask = (forecastnew['ds'] > "1-22-2020") & (forecastnew['ds'] <= "4-30-2020")
    forecastnew = forecastnew.loc[mask]
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(forecastnew.set_index('ds'), color='b')
    ax1.plot(forecastedvalues.set_index('ds'), color='r')
    ax1.set_ylabel('Deaths')
    ax1.set_xlabel('Date')
    print("Red = Predicted Values, Blue = Base Values")


# In[ ]:


dfTurkey=train_df[train_df["Country_Region"]=="Turkey"]

countryCompare(dfTurkey)


# In[ ]:


dfItaly=train_df[train_df["Country_Region"]=="Italy"]

countryCompare(dfItaly)


# In[ ]:


df_by_date2 = pd.DataFrame(train_df.groupby(['Country_Region','Date'])['Deaths'].sum().reset_index())

dfUS=df_by_date2[df_by_date2["Country_Region"]=="US"]
countryCompare(dfUS)


# In[ ]:


def countryDeaths(df):
    df = df.rename(columns={'Date': 'ds',
                            'Deaths': 'y'})

    my_model = Prophet()
    my_model.fit(df)

    
    future_dates = my_model.make_future_dataframe(periods=100)
    forecast =my_model.predict(future_dates)

    #fig2 = my_model.plot_components(forecast)


    forecastnew = forecast['ds']
    forecastnew2 = forecast['yhat']

    forecastnew = pd.concat([forecastnew,forecastnew2], axis=1)
    mask = (forecastnew['ds'] > "5-1-2020") & (forecastnew['ds'] <= "5-10-2020")
    forecastedvalues = forecastnew.loc[mask]
    

    mask = (forecastnew['ds'] > "1-22-2020") & (forecastnew['ds'] <= "4-30-2020")
    forecastnew = forecastnew.loc[mask]
    
    print(forecastedvalues.set_index('ds'))


# In[ ]:


def countryConfirmed(df):
    df = df.rename(columns={'Date': 'ds',
                            'Confirmed': 'y'})

    my_model = Prophet()
    my_model.fit(df)

   
    future_dates = my_model.make_future_dataframe(periods=100)
    forecast =my_model.predict(future_dates)

    #fig2 = my_model.plot_components(forecast)


    forecastnew = forecast['ds']
    forecastnew2 = forecast['yhat']

    forecastnew = pd.concat([forecastnew,forecastnew2], axis=1)
    mask = (forecastnew['ds'] > "5-1-2020") & (forecastnew['ds'] <= "5-10-2020")
    forecastedvalues = forecastnew.loc[mask]
    

    mask = (forecastnew['ds'] > "1-22-2020") & (forecastnew['ds'] <= "4-30-2020")
    forecastnew = forecastnew.loc[mask]
    
    print(forecastedvalues.set_index('ds'))


# In[ ]:


arr=train_df["Country_Region"].unique()
df_by_date2 = pd.DataFrame(train_df.groupby(['Country_Region','Date'])['Deaths'].sum().reset_index())
df_by_date3 = pd.DataFrame(train_df.groupby(['Country_Region','Date'])['Confirmed'].sum().reset_index())
for i in arr:
    dfTemp=df_by_date2[df_by_date2["Country_Region"]==i]
    dfTemp1=df_by_date3[df_by_date3["Country_Region"]==i]
    print(i)
    countryDeaths(dfTemp)
    countryConfirmed(dfTemp1)

