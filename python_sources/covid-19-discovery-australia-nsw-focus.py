#!/usr/bin/env python
# coding: utf-8

# ![](https://health.mil/-/media/Images/MHS/Photos/COVID-19.ashx?h=407&la=en&mw=720&w=720&hash=332EAD54CFF5B415044711D1F966B399F05E032CA7CFA38CC61CB42EA88753AE)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use("seaborn-dark-palette")
plt.rcParams.update({'font.size': 22})
default_fig_size = (25,10)
sub_fig_size = (25,25)

#importing plotly and cufflinks in offline mode
import cufflinks as cf
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# # Setup Dataset

# In[ ]:


import os
import re
from datetime import datetime

source_dir = "/kaggle/input/jhucovid19/csse_covid_19_data/csse_covid_19_daily_reports"
get_date_regex = re.compile(r"(?P<month>\d{0,2})-(?P<day>\d{1,2})-(?P<year>\d{4})\.csv$", re.IGNORECASE)

col_renames ={
    "Country_Region": "Country",
    "Country/Region": "Country",
    "Province_State": "State",
    "Province/State": "State",
    "Last Update": "Last_Update",
    "Latitude":"Lat",
    "Longitude":"Long",
    "Long_":"Long",
}

new_covid_df = pd.DataFrame()
for file_name in os.listdir(source_dir):
    result = get_date_regex.search(file_name)
    if result:
        file_date = datetime(int(result['year']), int(result['month']), int(result['day']))
        new_csv_df = pd.read_csv(f'{source_dir}/{file_name}')
        new_csv_df['Date'] = file_date
        new_csv_df.rename(columns=col_renames, inplace=True)
        new_covid_df = new_covid_df.append(new_csv_df, ignore_index=True, sort=True)
        
new_covid_df.sort_values('Date', ascending=True, inplace=True)
new_covid_df.Country.replace('China', 'Mainland China', inplace=True)
new_covid_df.Country.replace('UK', 'United Kindgdom', inplace=True)
new_covid_df = new_covid_df[((new_covid_df.Country == "Australia") & (~new_covid_df.State.isin(['From Diamond Princess', 'External territories', 'Jervis Bay Territory']))) | (new_covid_df.Country != 'Australia')]


# In[ ]:


#Select Datasource
covid_df = new_covid_df.sort_values('Date')


# In[ ]:


#Create Daily new figures from totals
covid_df['Daily'] = covid_df.groupby(['State', 'Country'])['Confirmed'].diff().fillna(0)

#Create Fatality to Case Ratio from totals
covid_df['FatalRatio'] = covid_df.Deaths / covid_df.Confirmed
covid_df['FatalRatio'].fillna(0, inplace=True)


# In[ ]:


#Get NSW Data
nsw_data_path = '/kaggle/input/nsw-health-data-covid19'


# In[ ]:


nsw_local_df = pd.read_csv(f'https://data.nsw.gov.au/data/dataset/aefcde60-3b0c-4bc0-9af1-6fe652944ec2/resource/21304414-1ff1-4243-a5d2-f52778048b29/download/covid-19-cases-by-notification-date-and-postcode-local-health-district-and-local-government-area.csv')

nsw_age_df = pd.read_csv(f'https://data.nsw.gov.au/data/dataset/3dc5dc39-40b4-4ee9-8ec6-2d862a916dcf/resource/24b34cb5-8b01-4008-9d93-d14cf5518aec/download/covid-19-cases-by-notification-date-and-age-range.csv')

nsw_source_df = pd.read_csv(f'https://data.nsw.gov.au/data/dataset/c647a815-5eb7-4df6-8c88-f9c537a4f21e/resource/2f1ba0f3-8c21-4a86-acaf-444be4401a6d/download/covid-19-cases-by-notification-date-and-likely-source-of-infection.csv')

ausmapinfo = pd.read_csv('/kaggle/input/australian-postcodes/australian_postcodes.csv')
ausmapinfo = ausmapinfo.groupby('postcode').agg({
                             'locality': ' | '.join, 
                             'long':'first',
                             'lat':'first'}).reset_index()
nsw_local_df = pd.merge(nsw_local_df, ausmapinfo, on='postcode', how='left')


# ---

# # NSW Trends

# In[ ]:


aus_state = "New South Wales"


# In[ ]:


base_layout = go.Layout(
    legend=dict(orientation="h"),
    margin={"r":0,"t":40,"l":0,"b":0},
    title={
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        }
    )


# # Sydney & NSW Confirmed Case Maps

# In[ ]:


title=f'Sydney COVID-19 Confirmed Cases Map'
layout = base_layout.update(title_text=title)


# In[ ]:


#Prepare Sydney Metro Cases
nswmastermap = nsw_local_df[nsw_local_df.long != 0.000000]
nswmastermap = nswmastermap[nswmastermap['postcode'].between(2000, 2234)]
nswmastermap['short_local'] = nswmastermap['locality'].str.split('|').str.get(0)
nswmastermap = nswmastermap.groupby('locality').agg({'locality':'count', 'short_local':'first','long':'first','lat':'first'}).rename(columns={'locality':'Count'}).reset_index()


# In[ ]:


#Map Sydney Metro Cases
fig = px.scatter_mapbox(nswmastermap, lat="lat", lon="long", hover_name="short_local", hover_data=["Count"],
                        color_discrete_sequence=["red"], zoom=9, height=500, size="Count",size_max=25)

fig.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
mtype = "stamen-terrain"
fig.update_layout(mapbox_style=mtype)
fig.update_layout(layout)
fig.show()


# In[ ]:


title=f'{aus_state} COVID-19 Confirmed Cases Map'
layout = base_layout.update(title_text=title)


# In[ ]:


#Prepare NSW Cases
nswmastermap = nsw_local_df[nsw_local_df.long != 0.000000]
nswmastermap = nswmastermap.groupby('lga_name19').agg({'lga_name19':'count','long':'first','lat':'first'}).rename(columns={'lga_name19':'Count'}).reset_index()


# In[ ]:


#Map NSW Cases
fig = px.scatter_mapbox(nswmastermap, lat="lat", lon="long", hover_name="lga_name19", hover_data=["Count"],
                        color_discrete_sequence=["red"], zoom=5, height=500, size="Count",size_max=15)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
mtype = "stamen-terrain"
fig.update_layout(mapbox_style=mtype)
fig.update_layout(layout)
fig.show()


# # NSW Timeline Trends

# In[ ]:


title=f'{aus_state} COVID-19 Total Confirmed Cases'
layout = base_layout.update(title_text=title)


# In[ ]:


covid_df[(covid_df.Country == "Australia") & (covid_df.State == aus_state)].groupby(['Date', 'State']).Confirmed.max().unstack().iplot(kind='bar', layout=layout)


# Looking ok, curve flattening from about Mar 28 perhaps

# In[ ]:


aus_state = "New South Wales"

title=f'{aus_state} COVID-19 Daily Confirmed Cases'
layout = base_layout.update(title_text=title)


# In[ ]:


covid_df[(covid_df.Country == "Australia") & (covid_df.State == aus_state)].groupby(['Date', 'State']).Daily.max().unstack().iplot(kind='bar', layout=layout)


# Definatly looking like a downward trend in new case numbers when looking at the daily total after Mar 25.
# 
# Assume this would be related to the restrictions inplace perhaps from 6 days prior due to the incubation lag.
# 
# This needs further investigation.

# In[ ]:


title=f'{aus_state} COVID-19 Total Recovered Cases'
layout = base_layout.update(title_text=title)


# In[ ]:


covid_df[(covid_df.Country == "Australia") & (covid_df.State == aus_state)].groupby(['Date', 'State']).Recovered.max().unstack().iplot(kind='bar', layout=layout)


# This looks like recoveries are probably not reported or tracked correctly?

# # NSW Demographics & Infection Source

# In[ ]:


title=f'{aus_state} COVID-19 Confirmed Cases By Age'
layout = base_layout.update(title_text=title)


# In[ ]:


nsw_age_df['age_group'] = pd.Categorical(nsw_age_df['age_group'], categories=['AgeGroup_0-4', 'AgeGroup_5-9', 'AgeGroup_10-14', 'AgeGroup_15-19', 'AgeGroup_20-24', 'AgeGroup_25-29', 'AgeGroup_30-34', 'AgeGroup_35-39', 'AgeGroup_40-44', 'AgeGroup_45-49', 'AgeGroup_50-54', 'AgeGroup_55-59',
       'AgeGroup_60-64', 'AgeGroup_65-69', 'AgeGroup_70+'])


# In[ ]:


fig = px.histogram(nsw_age_df.sort_values(by='age_group'), x="age_group", labels={'age_group':'Age Groups'})
fig.update_layout(layout)
fig.show()


# Peaks are in the 70+ and late 20's

# In[ ]:


title=f'{aus_state} COVID-19 Confirmed Cases By Likley Source of Infection'
layout = base_layout.update(title_text=title)


# In[ ]:


fig = px.histogram(nsw_source_df, x='likely_source_of_infection', labels={'likely_source_of_infection':'Likely Source of Infection'})
fig.update_layout(layout)
fig.show()


# # NSW Fataility Rates

# In[ ]:


title=f'{aus_state} COVID-19 Cumulative Fatality to Case Ratio'
layout = base_layout.update(title_text=title)
layout = layout.update(legend=dict(orientation="h"))


# In[ ]:


covid_df[(covid_df.Country == "Australia") & (covid_df.State == aus_state)].groupby(['Date', 'State']).FatalRatio.max().unstack().iplot(kind='line', layout=layout)


# I suspect numbers are too low to draw accurate conclusions from this, but the downward trend could indicate health care preparedness getting better over time?

# ---

# # Australia Trends

# In[ ]:


title=f'Australia COVID-19 Total Confirmed Cases'
layout = base_layout.update(title_text=title)


# In[ ]:


covid_df[(covid_df.Country == "Australia")].groupby(['Date', 'State']).Confirmed.max().unstack().iplot(kind='bar', layout=layout)


# Australia as awhole tracking similar to NSW. This makes sense due to NSW making the bulk of confirmed cases.

# In[ ]:


title=f'Australia COVID-19 Daily Confirmed Cases'

layout = base_layout.update(title_text=title, barmode='stack')


# In[ ]:


covid_df[(covid_df.Country == "Australia")].groupby(['Date', 'State']).Daily.max().unstack().iplot(kind='bar', layout=layout)


# Smaller states with low case counts are more erratic and harder ro draw conclusions from but as above the trend roughtly matches NSW for Australia as a whole.

# In[ ]:


title=f'Australia COVID-19 Total Confirmed Cases by State'
ftitle={
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
}


# In[ ]:


covid_df[(covid_df.Country == "Australia")].groupby(['Date', 'State']).max().Confirmed.unstack().iplot(kind='bar', subplots=True, subplot_titles=True, showlegend=False, title=ftitle)


# In[ ]:


title=f'Australia COVID-19 Daily Confirmed Cases by State'
ftitle={
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
}


# In[ ]:


covid_df[(covid_df.Country == "Australia")].groupby(['Date', 'State']).max().Daily.unstack().iplot(kind='bar', subplots=True, subplot_titles=True, showlegend=False, title=ftitle)


# All states roughly tracking the same.

# In[ ]:


title=f'Australia COVID-19 Cumulative Fatality to Case Ratio'
layout = base_layout.update(title_text=title)


# In[ ]:


country_sums = covid_df[covid_df.Country.isin(['Australia'])].groupby(['Date', 'Country'])['Deaths', 'Confirmed'].sum()
country_rations = country_sums.Deaths / country_sums.Confirmed
country_rations.unstack().iplot(kind='line', layout=layout)


# Higher and longer spike than NSW although a similar trend is visable.

# In[ ]:


title=f'Australia COVID-19 Cumulative Fatality to Case Ratio by State'
layout = base_layout.update(title_text=title)


# In[ ]:


aus_sums = covid_df[(covid_df.Country == "Australia")].groupby(['Date', 'State'])['Deaths', 'Confirmed'].sum()
aus_rations = aus_sums.Deaths / aus_sums.Confirmed
aus_rations.unstack().iplot(kind='line', layout=layout)


# Looks like the initial very high death rate in WA skewed the australis totals up, WA and NSW seem to be the biggest influence on the Australia overal rate trend.
# WA and NSW also had first deaths recorded earlier than other states.

# In[ ]:


DFactor=1/0.034


# Using the above fatality rate as reported [here](https://www.msn.com/en-us/health/health-news/what-the-official-covid-19-mortality-rate-actually-means/ar-BB10O0Dx) to estimate what the expected number of cases should be given how many deaths have occured.
# 
# This does not attempt to explain any discrepency as there are many factors such as
# * Testing rate
# * Population Age
# * Urbanisation
# * Health care Standards
# * Weather/Temprature
# * Economy
# * Many more
# 
# I will try to incorperare and explan these over time

# In[ ]:


title = f'Australia Spread between: Expected Cases (Death Rate) vs Reported Confirmed Cases'

ftitle={
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
}


# In[ ]:


Ecovid_df = covid_df.copy()
Ecovid_df['Estimated Cases from Deaths'] =  Ecovid_df['Deaths'] * DFactor
Ecovid_df[Ecovid_df.Country.isin(['Australia'])].groupby(['Date']).agg({'Confirmed':'sum', 'Estimated Cases from Deaths':'sum'}).iplot(kind='spread', title=ftitle)


# This shows Australia has many more reported cases than expected from the current amount of deaths. (Essentially the death rate is lower than expected)
# This could indicate many things including: Good health care standars, high testing rates etc.
# 
# In future I will use the testing rates as a factor for this.

# # Basic Predictions

# **Based on existing data point using FB Prophet to predict the future cases for Australia**

# In[ ]:


import numpy as np
import pandas as pd
from fbprophet import Prophet
from math import e

def prophet_log_predict(in_df, days_ahead, date_col='ds', y_col='y', log_base=10, changepoint_prior_scale=0.1, mcmc_samples=150):
  in_df = in_df.copy()
  #Convert to log
  in_df[y_col] = (np.log(in_df[y_col]) / np.log(log_base)).replace([np.inf, -np.inf], np.nan)

  #Prep col names
  in_df.rename(columns={date_col:'ds', y_col:'y'}, inplace=True)

  #Create Model
  model = Prophet(changepoint_prior_scale=changepoint_prior_scale, mcmc_samples=mcmc_samples)
  model.fit(in_df)

  #Make Predictions
  future = model.make_future_dataframe(periods=days_ahead)
  forecast = model.predict(future)

  #Create output dataframe
  out_df = pd.DataFrame()
  out_df[date_col] = forecast['ds']

  #Convert back from log and attach to output df
  out_df[y_col] = log_base**forecast['yhat']
  
  return out_df


# In[ ]:


pdf = covid_df[(covid_df.Country == "Australia")][['Date', 'Confirmed']].groupby(['Date']).Confirmed.sum().to_frame().reset_index()
pdf.rename(columns={'Date':'ds', 'Confirmed':'y'}, inplace=True)


# In[ ]:


pre_res = prophet_log_predict(pdf, 10)


# In[ ]:


title=f'Australia COVID-19 Case Prediction (Prophet)'
layout = base_layout.update(title_text=title)


# In[ ]:


pre_res.iplot(x='ds', layout=layout)


# In[ ]:


from datetime import datetime
restrict_date = datetime(2020, 4, 9)
pastdf = pdf[pdf.ds < restrict_date]

d_pred = (datetime.today() - restrict_date).days


# In[ ]:


no_res = prophet_log_predict(pastdf, d_pred)


# In[ ]:


title=f'Australia COVID-19 Case Prediction - Before Restrictions (Prophet)'
layout = base_layout.update(title_text=title)


# In[ ]:


no_res.iplot(x='ds', layout=layout)


# In[ ]:


comb = pd.DataFrame()
comb['ds'] = no_res['ds']
comb['Restricted (Actuals)'] = pre_res.y
comb['Unrestricted (Prediction)'] = no_res.y


# In[ ]:


title=f'Australia COVID-19 Case Prediction - No Restrictions (Prophet) and Actuals'
layout = base_layout.update(title_text=title)


# In[ ]:


comb.iplot(x='ds', layout=layout)


# ---

# # Global Trends

# In[ ]:


scope_countries = ['US', 'Mainland China', 'Italy', 'Spain', 'United Kingdom', 'Germany']


# In[ ]:


title = f'COVID-19 Confirmed Cases in {", ".join(scope_countries)}'

layout = base_layout.update(title_text=title, barmode='group')


# In[ ]:


covid_df[covid_df.Country.isin(scope_countries)].groupby(['Date', 'Country']).Confirmed.sum().unstack().iplot(kind='bar', layout=layout)


# China showing basically no new cases for a while.
# 
# US looking like its continuing up at the typical 3-5 day doubling rate... needs further investigation.
# 
# While Italy had a head start on Spain, Spain is heading up at a steeper slope and looks to have overtaken Italy on Apr 4. Possibly harsher restrictions in Itialy can explan this?

# In[ ]:


title = f'COVID-19 Fatal Cases in {", ".join(scope_countries)}'

layout = base_layout.update(title_text=title, barmode='group')


# In[ ]:


covid_df[covid_df.Country.isin(scope_countries)].groupby(['Date', 'Country']).Deaths.sum().unstack().iplot(kind='bar', layout=layout)


# Itialy still has a higher death rate by far than Spain.
# 
# Testing rates might need to be looked at to determine if there are different rates of fataility or different rates of untracked infection?

# In[ ]:


title = f'COVID-19 Cumulative Fatality to Case Ratio in {", ".join(scope_countries)}'

layout = base_layout.update(title_text=title, barmode='group')


# In[ ]:


country_sums = covid_df[covid_df.Country.isin(scope_countries)].groupby(['Date', 'Country'])['Deaths', 'Confirmed'].sum()
country_rations = country_sums.Deaths / country_sums.Confirmed
country_rations.unstack().iplot(kind='line', layout=layout)


# To be discussed

# In[ ]:


import plotly.graph_objects as go

wrld_df = covid_df.copy()
wrld_df['CLoc'] = wrld_df['State'].str.cat(wrld_df['Country'], join='outer', sep=', ').fillna(wrld_df['Country'])

wrld_df = wrld_df[(wrld_df.Date == wrld_df['Date'].max())].groupby('CLoc').agg({'Confirmed':'sum', 'Lat':'mean', 'Long':'mean'}).reset_index()


scale = 100
fig = go.Figure(data=go.Scattergeo(
        lon = wrld_df['Long'],
        lat = wrld_df['Lat'],
        text = wrld_df[['CLoc','Confirmed']],
        hoverinfo = 'text',
        mode = 'markers',
        marker_sizemode = 'area',
        marker_color = 'crimson',
        marker_size = wrld_df['Confirmed']/scale,
        ))

title=f'World Map of Current Confirmed Cases'
ftitle={
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
}

fig.update_layout(
        margin={"r":0,"t":20,"l":0,"b":0},
        title = ftitle,
        geo_scope='world',
        geo_landcolor = "rgb(229, 229, 229)",
        geo_countrycolor = "white" ,
        geo_coastlinecolor = "white",
        geo_showcoastlines = True,
        geo_showcountries = True,
    )
fig.show()


# Pretty much Obligitory for this kind of data.
# Nothing to groundbreaking to see but looks pretty nice.

# ---

# Similar to the Australia Graph, Using the fatality rate as reported [here](https://www.msn.com/en-us/health/health-news/what-the-official-covid-19-mortality-rate-actually-means/ar-BB10O0Dx) to estimate what the expected number of cases should be given how many deaths have occured.
# 
# This does not attempt to explain any discrepency as there are many factors such as
# * Testing rate
# * Population Age
# * Urbanisation
# * Health care Standards
# * Weather/Temprature
# * Economy
# * Many more
# 
# I will try to incorperare and explan these over time

# In[ ]:


title = f'Global Spread between: Expected Cases (Death Rate) vs Reported Confirmed Cases'

ftitle={
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
}


# In[ ]:


DFactor=1/0.034
Ecovid_df = covid_df.copy()
Ecovid_df['Estimated Cases from Deaths'] =  Ecovid_df['Deaths'] * DFactor
Ecovid_df.groupby(['Date']).agg({'Confirmed':'sum', 'Estimated Cases from Deaths':'sum'}).iplot(kind='spread', title=ftitle)


# Globally this shows that the death rate is rising as time goes on. This could be an indicator of strain on the medical system of heavily affected countires.

# # Spread Graphs: Expected Cases (Death Rate) vs Reported Confirmed Cases

# In[ ]:


for country in scope_countries:
    title = f'{country} Expected Cases vs Reported Confirmed Cases'

    ftitle={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
    }
    Ecovid_df = covid_df.copy()
    Ecovid_df['Estimated Cases from Deaths'] =  Ecovid_df['Deaths'] * DFactor
    Ecovid_df[Ecovid_df.Country.isin([country])].groupby(['Date']).agg({'Confirmed':'sum', 'Estimated Cases from Deaths':'sum'}).iplot(kind='spread', title=ftitle)


# The US actually shows slightly more cases than expected (a sign of lower death rate).
# This will need further analasys as the assumption is that the testing rates are low in the US and this seems unusual.
# **Update**: US data has recently swapped to a more expected case.

# # TODO

# * More details on tracking turning point to restrictions and/or other factors
# * More comparision with other countries
# * LSTM or VAR to make some predictions
# * Try to make whatif predictions, e.g. what if restriction X was applied
# * Estimate untested cases
