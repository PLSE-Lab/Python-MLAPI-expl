#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Coronavirus disease 2019 (COVID-19) is an infectious disease caused by SARS-CoV-2, a virus closely related to the SARS virus. 
# 
# Cases were initially identified in Wuhan, capital of Hubei province in China in December 2019. Subsequently, infections have been reported around the world. Cases reported outside China have predominantly been in people who have recently travelled to Mainland China, however a few cases of local transmission have also occurred.

# # Data 

# Data used for this notebook comes from the repo https://github.com/CSSEGISandData/COVID-19 provided by John Hopkins University, at the moment the data is updated on everyday basis. This notebook reads the data directly from the repo, so it updates on every rerun. 

# Online dashboard provided by John Hopkins University: 
# 
# https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6

# ## Get Data and Reshape

# In[ ]:


import os
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[ ]:


path = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
confirmed_path = "time_series_19-covid-Confirmed.csv"
deaths_path = "time_series_19-covid-Deaths.csv"
recovered_path = "time_series_19-covid-Recovered.csv"


# In[ ]:


confirmed = pd.read_csv(os.path.join(path, confirmed_path) )
confirmed = confirmed.melt(id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='date',
                          value_name = 'confirmed')
confirmed['date_dt'] = pd.to_datetime(confirmed.date, format="%m/%d/%y")
confirmed.date = confirmed.date_dt.dt.date
confirmed.rename(columns={'Country/Region': 'country', 'Province/State': 'province'}, inplace=True)


# In[ ]:


recovered = pd.read_csv(os.path.join(path, recovered_path) )
recovered = recovered.melt(id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='date',
                          value_name = 'recovered')
recovered['date_dt'] = pd.to_datetime(recovered.date, format="%m/%d/%y")
recovered.date = recovered.date_dt.dt.date
recovered.rename(columns={'Country/Region': 'country', 'Province/State': 'province'}, inplace=True)


# In[ ]:


deaths = pd.read_csv(os.path.join(path, deaths_path) )
deaths = deaths.melt(id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='date',
                          value_name = 'deaths')
deaths['date_dt'] = pd.to_datetime(deaths.date, format="%m/%d/%y")
deaths.date = deaths.date_dt.dt.date

deaths.rename(columns={'Country/Region': 'country', 'Province/State': 'province'}, inplace=True)


# In[ ]:


print('confirmed', confirmed.shape)
print('deaths', deaths.shape)
print('recovered', recovered.shape)


# In[ ]:


deaths.tail()


# In[ ]:


confirmed.query('country == "Others"').province.value_counts()


# In[ ]:


confirmed.isnull().sum().where(lambda x : x!=0).dropna()


# In[ ]:


recovered.isnull().sum().where(lambda x : x!=0).dropna()


# In[ ]:


deaths.isnull().sum().where(lambda x : x!=0).dropna()


# In all three datasets there are missing values for province 

# In[ ]:


confirmed[confirmed.province.isnull()].country.unique()


# In[ ]:


def check_null_province(data):
    no_province_countries = data[data.province.isnull()].country.unique()
    print(f"{len(no_province_countries)} Countries for which province is not specified")
    print(no_province_countries)
    keep = data.country.isin(no_province_countries)
    print('-'*60)
    print("Nb not null province for these countries:", data[keep].province.notnull().sum())
    print("Max nb of records for country, date:", data[keep].groupby(['country', 'date']).Lat.count().max())


# In[ ]:


check_null_province(confirmed)


# In[ ]:


check_null_province(deaths)


# In[ ]:


check_null_province(recovered)


# Fill in missing province with country name

# In[ ]:


confirmed.province = confirmed.province.where(confirmed.province.notnull(), other=confirmed.country)


# In[ ]:


deaths.province = deaths.province.where(deaths.province.notnull(), other=deaths.country)
recovered.province = recovered.province.where(recovered.province.notnull(), other=recovered.country)


# In[ ]:


merge_on = ['province', 'country', 'date']
all_stat = confirmed.merge(deaths[merge_on + ['deaths']], how='left', on=merge_on).                         merge(recovered[merge_on + ['recovered']], how='left', on=merge_on)


# In[ ]:


all_stat['sick'] = all_stat['confirmed'] - all_stat['recovered'] - all_stat['deaths']


# In[ ]:


all_stat.head()


# In[ ]:





# ## Data Cleaning

# In[ ]:


all_stat = all_stat.sort_values(['country', 'province', 'date'])
all_stat['confirmed_day'] = all_stat.groupby(['country', 'province']).confirmed.diff()
all_stat['next_confirmed_day'] = all_stat.groupby(['country', 'province']).confirmed_day.shift(-1)
all_stat['prev_confirmed'] = all_stat.groupby(['country', 'province']).confirmed.shift()


# There are a few cases of inconsistent data when accumulated number of cases decrease in time, we will fix it manually here

# In[ ]:


all_stat.query('next_confirmed_day<0')


# In[ ]:


all_stat.loc[all_stat.eval('country=="Japan" and date_dt=="2020-02-06"'), 'confirmed'] = 22
all_stat.loc[all_stat.eval('country=="Japan" and date_dt=="2020-01-22"'), 'confirmed'] = 1
all_stat.loc[all_stat.eval('country=="Australia" and province=="Queensland" and date_dt=="2020-02-01"'), 'confirmed'] = 1
all_stat.loc[all_stat.eval('country=="Australia" and province=="Queensland" and date_dt=="2020-01-30"'), 'confirmed'] = 2


# In[ ]:


all_stat['confirmed_day'] = all_stat.groupby(['country', 'province']).confirmed.diff()
all_stat['deaths_day'] = all_stat.groupby(['country', 'province']).deaths.diff()
all_stat['recovered_day'] = all_stat.groupby(['country', 'province']).recovered.diff()


# ## Reshaped Dataset

# In[ ]:


all_stat.columns


# In[ ]:


all_stat.head()


# In[ ]:


all_stat.date.max()


# # Total Confirmed: Inpatient, Recovered and Deaths

# In[ ]:


daily_stat = all_stat.groupby(['date'])[['confirmed', 'deaths', 'recovered', 'sick',
                                        'confirmed_day', 'deaths_day', 'recovered_day']].sum().reset_index()
daily_stat['deaths_to_confirmed'] = daily_stat['deaths'] / daily_stat['confirmed']
daily_stat['recovered_to_confirmed'] = daily_stat['recovered'] / daily_stat['confirmed']
daily_stat['sick_to_confirmed'] = daily_stat['sick'] / daily_stat['confirmed']


# In[ ]:


px.bar(all_stat.melt(id_vars=['province', 'country', 'date'], 
                     value_vars=['sick', 'recovered', 'deaths'], var_name='type', value_name='nb_cases'),
       
       x='date', y='nb_cases', color='type', barmode='stack', 
            category_orders={'type':['sick', 'deaths', 'recovered']} )


# Last five days daily statisctics

# In[ ]:


daily_stat.set_index('date').tail().transpose()


# In[ ]:


daily_stat['sick_pct'] = daily_stat['sick'].pct_change()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(x=daily_stat.date, y=100*daily_stat.sick_pct, opacity=0.7))
fig.update_layout( title="Daily Increase of Sick Patients, %",
                 xaxis_title="Date", )
fig.show()


# As on 19 Feb 2020 there are 75 641 confirmed cases in total, around 75.9% are still inpatient (57 398 cases) and 21.3% totally recovered. Because of the rapid spread of the virus the number of patients who are sick and possibly requires medical treatment has been growing, however in the recent days the growth slowed down and even decreased by ~2% in the last day     

# # Number of Confirmed Cases vs Distance to Outbreak

# In[ ]:


import geopy.distance

hubei_coords = (30.97564, 112.2707)
dist_to_hubei = lambda lat, long: geopy.distance.distance((lat, long), hubei_coords).km


# In[ ]:


all_stat['dist_to_hubei'] = all_stat.apply(lambda r: dist_to_hubei(r.Lat, r.Long), axis=1)


# In[ ]:


def get_geo(country, province): 
    if province == 'Hubei':
        return 'China/Hubei'
    elif country == 'Mainland China':
        return 'China/Others'
    else:
        return 'Others'
    
all_stat['geo'] = all_stat.apply(lambda r: get_geo(r.country, r.province), axis=1)


# In[ ]:


all_stat['month'] = all_stat.date_dt.dt.month_name()


# In[ ]:


d = all_stat.groupby(['country', 'province', 'dist_to_hubei', 'geo', 'month']).confirmed.max().reset_index()
fig = px.scatter(d, x='dist_to_hubei', y='confirmed', color='geo', facet_col='month',
          hover_data=['country', 'province', 'confirmed'], log_y=True,
          category_orders={'month': ['January', 'February']}, 
                 labels={'dist_to_hubei':'Distance, km', 
                         'confirmed':'Number of Confirmed Cases (Log scale)'})
fig.update_layout(title="Number of Confirmed depending on the distance to the outbreak (Hubei province)")
fig.show()


# Along with China provinces some other Asian countries located relatively close to the outbreak observe increase of confirmed cases in February

# # Daily Incremet of Confirmed Cases

# In[ ]:


all_stat['location'] = 'Countries'
all_stat['location'] = all_stat.location.where(all_stat.country!="Others", 'Cruise ships')


# In[ ]:


d = all_stat.groupby(['geo', 'date', 'location']).confirmed_day.sum().reset_index()
px.bar(d, x='date', y='confirmed_day', 
       facet_col='geo', log_y=True, color='location',
      color_continuous_scale=px.colors.diverging.Tealrose)


# In[ ]:


print('Average daily increament in the last 5 days')
print('China/Hubei',  d.query('geo=="China/Hubei"')[-5:].confirmed_day.mean())
print('China/Others',  d.query('geo=="China/Others"')[-5:].confirmed_day.mean())
print('Other Countries',  d.query('geo=="Others" and location=="Countries"')[-5:].confirmed_day.mean())
print('Cruise Ship',  d.query('geo=="Others" and location=="Cruise ships"')[-5:].confirmed_day.mean())


# Descrease in the number of new cases in other Chinese provinces except Hubei

# In[ ]:


d.query('geo=="China/Others"')[-5:]


# There is still a constant increase of confirmed cases in Hubei province with average of 1525 for the last 5 days, however there is an evident descrease in the number of new cases in other parts of mainland China. In other Chinese provinces for the last 5 days daily number of new cases decreased from 212 to 58. There is a constant increase of new cases with the average 20/day for the last 5 days

# ## Country Comparison

# In[ ]:


all_stat['cur_nb_cases'] = all_stat.groupby(['country', 'province']).confirmed.transform('max')
cns = all_stat.sort_values('cur_nb_cases', ascending = False)['country'].unique()


# In[ ]:


fig = px.area(all_stat.query('geo=="Others" and location=="Countries" ' ),
              x="date", y="confirmed", color="country",
              category_orders={"country": list(cns)}
             )
fig.update_layout(title="Total Number of Confirmed Cases for Countries except China and cruise ship")
fig.show()


# In[ ]:


last_date = all_stat.date.max()


# Top China regions with most confirmed cases

# In[ ]:


all_stat.query('date == @last_date and country=="Mainland China" ' ).groupby(['country', 'province']).confirmed.sum().                reset_index().sort_values('confirmed', ascending=False).head(10)


# Top countries with most confirmed cases

# In[ ]:


d = all_stat.query('date == @last_date and country!="Mainland China" ' ).groupby(['country', 'province']).confirmed.sum().                reset_index().sort_values('confirmed', ascending=False)
d.head(10)


# In[ ]:


print('Total outside China: ', (d[d.country!="Mainland China"]).confirmed.sum()) 
print('Total in Japan, Singapore, Hong Kong: ', (d[d.country.isin(['Japan', 'Singapore', 'Hong Kong'])]).confirmed.sum())
print('Total in countries: ', (d[~d.country.isin(['Others'])]).confirmed.sum())
print('Share of Japan, Singapore, Hong Kong: ', d[d.country.isin(['Japan', 'Singapore', 'Hong Kong'])].confirmed.sum()/d[~d.country.isin(['Others'])].confirmed.sum() )


# Total number of cases ouside China is 1095 with highest number related to Diamond Princess cruise ship with 621 cases. Japan, Singapore and Hong Kong are countries with most confirmed cases for now with 231 in total, it is 50% of all countries related cases (other countries except cruise ship numbers)   

# In[ ]:


def add_plot(fig, country, row, col, showlegend=False):
    data = all_stat.query('country==@country')
    fig.add_trace(go.Scatter(x=data.date, y=data.confirmed,
                    mode='lines',
                    name='Total Confirmed', line={'width':1, 'color':'blue'},
                        opacity=1, showlegend=showlegend),
             secondary_y=False, row=row, col=col)
    fig.add_trace(go.Bar(x=data.date, y=data.confirmed_day,
                   marker_color='rgb(55, 83, 109)',
                    name='Confirmed (Day Increment)',
                        opacity=0.5 , showlegend=showlegend),
             secondary_y=False, row=row, col=col)

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Singapore", "Japan", "Hong Kong"),
    
    specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
    shared_yaxes=True,
    horizontal_spacing=0.02)

add_plot(fig, 'Singapore', row=1, col=1, showlegend=True)
add_plot(fig, 'Japan', row=1, col=2)
add_plot(fig, 'Hong Kong', row=1, col=3)


fig.update_layout( title="Total Confirmed and Day Increment" )
# fig.update_yaxes(title_text="Number of Confirmed Cases", secondary_y=False)
fig.update_yaxes(title_text="Number of Confirmed Cases", row=1, col=1)
fig.update_layout(legend_orientation="h")

fig.show()


# In[ ]:


fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Thailand', 'Malaysia', 'Taiwan'),
    
    specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
    shared_yaxes=True,
    horizontal_spacing=0.02)

add_plot(fig, 'Thailand', row=1, col=1, showlegend=True)
add_plot(fig, 'Malaysia', row=1, col=2)
add_plot(fig, 'Taiwan', row=1, col=3)


fig.update_layout( title="Total Confirmed and Day Increment" )
# fig.update_yaxes(title_text="Number of Confirmed Cases", secondary_y=False)
fig.update_yaxes(title_text="Number of Confirmed Cases", row=1, col=1)
fig.update_layout(legend_orientation="h")

fig.show()


# # Geo Dynamics of Spread

# In[ ]:


stat_map = all_stat.groupby(['date_dt', 'province'])['confirmed', 'deaths', 
                                                 'recovered', 'Lat', 'Long'].max().reset_index()


# In[ ]:


stat_map['size'] = stat_map.confirmed.pow(0.5)
stat_map['date_dt'] = stat_map['date_dt'].dt.strftime('%Y-%m-%d')


# In[ ]:



fig = px.scatter_geo(stat_map, lat='Lat', lon='Long', scope='asia',
                     color="size", size='size', hover_name='province', 
                     hover_data=['confirmed', 'deaths', 'recovered'],
                     projection="natural earth", animation_frame="date_dt", title='Spread in Asia Region over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()


# # Deaths and Recoveries

# In[ ]:


data = daily_stat.loc[1:,:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.date, y=data.confirmed_day,
                    mode='lines+markers',
                    name='confirmed', line={'width':0.5},
                        opacity=0.7))
fig.add_trace(go.Scatter(x=data.date, y=data.deaths_day,
                    mode='lines+markers',
                    name=f'deaths', line={'width':0.5},
                        opacity=0.7))
fig.add_trace(go.Scatter(x=data.date, y=data.recovered_day,
                    mode='lines+markers', 
                    name=f'recovered', line={'width':0.5},
                        opacity=0.7))

fig.update_layout(yaxis_type="log", title="Daily New Number of Cases (Confirmed, Deaths, Recovered)",
                 yaxis_title="Number of Cases (Log Scale)", 
                 xaxis_title="Date", )
fig.show()


# In[ ]:


fig = px.bar(all_stat.query('country=="Mainland China"').groupby(['geo', 'date']).deaths_day.sum().reset_index(), x='date', y='deaths_day', 
       facet_col='geo', log_y=True,
      color_continuous_scale=px.colors.diverging.Tealrose)
fig.update_layout(title="Number of Deaths per Day")
fig.show()


#  ## Comapre with Number of Cases Confirmed Earlier

# In[ ]:


daily_stat['confirmed_14d'] = daily_stat.confirmed.shift(14)
daily_stat['confirmed_7d'] = daily_stat.confirmed.shift(7)


# In[ ]:



def compare_death_recovery (daily_stat, shift_days=0, log_y=False):

    y = daily_stat.confirmed.shift(shift_days)
    data = daily_stat[y.notnull()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.date, y=data.deaths,
                        mode='lines',
                        name='deaths'))
    fig.add_trace(go.Scatter(x=data.date, y=y[y.notnull()],
                        mode='lines',
                        name=f'confirmed {shift_days} days ago'))
    fig.add_trace(go.Scatter(x=data.date, y=data.recovered,
                        mode='lines',
                        name='recovered'))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.update_layout(title=f"Compare Deaths and Recovered with Confirmed {shift_days} days ago")
    fig.show()
#     return fig, y


# In[ ]:


compare_death_recovery(daily_stat, 14, log_y=True)


# In[ ]:


compare_death_recovery(daily_stat, 7, log_y=True)


# In[ ]:


daily_stat['death_rate_14d'] = daily_stat.deaths / daily_stat.confirmed_14d
daily_stat['recovery_rate_14d'] = daily_stat.recovered / daily_stat.confirmed_14d


# In[ ]:


daily_stat[['death_rate_14d', 'recovery_rate_14d']].tail()


# # Compare with SARS

# * Historical Data from SARS outbreak
# 
# SARS image https://commons.wikimedia.org/wiki/File:2003_Probable_cases_of_SARS_-_Worldwide.svg
# <img src="https://upload.wikimedia.org/wikipedia/commons/e/e0/2003_Probable_cases_of_SARS_-_Worldwide.svg" width="700">

# In[ ]:


data = daily_stat

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=data.date, y=data.confirmed,
                    mode='lines',
                    name='Total Confirmed', line={'width':1},
                        opacity=1),
             secondary_y=False)
fig.add_trace(go.Scatter(x=data.date, y=data.deaths,
                    mode='lines',
                    name=f'Total Deaths', line={'width':1},
                        opacity=1),
             secondary_y=True)
fig.add_trace(go.Bar(x=data.date, y=data.confirmed_day,
                   
                    name='Confirmed (Day Increment)',
                        opacity=0.5 ),
             secondary_y=True)

fig.update_layout( title="Total Number of Confirmed and Death Cases",
                 
                 xaxis_title="Date", )
fig.update_yaxes(title_text="Number of Confirmed Cases", secondary_y=False)
fig.update_yaxes(title_text="Number of Death Cases", secondary_y=True)

fig.show()


# # Summary

# **The summary was prepared for the data as on 19 Feb 2020**
# 
# * As on 19 Feb 2020 there are 75 641 confirmed cases in total, around 75.9% are still inpatient (57 398 cases) and 21.3% totally recovered. Because of the rapid spread of the virus the number of patients who are sick and possibly requires medical treatment has been growing, however in the recent days the growth slowed down and even decreased by ~2% in the last day 
# 
# * Along with China provinces some other Asian countries located relatively close to the outbreak observe increase of confirmed cases in February
# 
# * There is still a constant increase of confirmed cases in Hubei province with average of 1525 for the last 5 days, however there is an evident descrease in the number of new cases in other parts of mainland China. In other Chinese provinces for the last 5 days daily number of new cases decreased from 212 to 58. There is a constant increase of new cases with the average 20/day for the last 5 days
# 
# * Total number of cases ouside China is 1095 with highest number related to Diamond Princess cruise ship with 621 cases. Japan, Singapore and Hong Kong are countries with most confirmed cases for now with 231 in total, it is 50% of all countries related cases (other countries except cruise ship numbers) 
# 
# * While new cases are still comming every day it is not clear what is the real death rate. It is possible to estimate from tracking each case from confirmation to recovery or death, but not from total numbers. Comparing with SARS historical data we can see stabilized curves of confirmed cases and deaths from what we can estimate the death rate to be around 10%. Total numbers of confirmed cases and deaths significantly higher than for SARS. Comparing current death number with the number of confirmed cases 2 weeks ago we can see numbers around 7-8%, however the 2 week period is quite arbitrary here and assumed to be related to the expected time between confirmation and recovery or death.
# 
# 

# In[ ]:




