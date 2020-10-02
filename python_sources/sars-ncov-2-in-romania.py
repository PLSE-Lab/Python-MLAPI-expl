#!/usr/bin/env python
# coding: utf-8

# <h1>SARS-nCoV-2 in Romania</h1>
# 
# 
# For this analysis, I switched to use @orianao daily updated dataset with the SARS-nCoV-2 collected information about Romania.
# 
# The dataset contains, besides Confirmed, Recovered, Deaths data as well the number of Tests results communicated daily.
# 
# **Note**: as a fallback measure, if the above mentioned dataset is not maintained constantly, I also load John Hopkins that have the same information (with one day lag) and data is used from the dataset with newest information.
# 
# I also added my dataset on daily county-level confirmed cases for Romania.  Here I also collect (from scrapped web content on public announcements site of Internal Affairs Ministry of Romania) number of quarantined, isolated people as well as ICU daily patients.
# 
# For the geographical distribution of the data, I am also importing GeoJSON data with the Romanian counties polygons.
# 
# This Kernel is updated frequently, to include the latest (daily) updates from the two datasets.
# 
# **Disclaimer** The analysis includes exponential and logistic curve fit for the confirmed cases data. These should not be interpreted as predictions, we are just trying to understand dynamic of the evolution based on current available data points. As well, the mortality rates presented, calculated based on the available data, does not represent actual values but (under/and over) estimated values (for which we mention as well the approximations we did).
# 
# 

# ## Load packages
# 
# We will use mostly Plotly and Folium for visualization.

# In[ ]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime as dt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from shapely.geometry import shape, Point, Polygon
import folium
from folium.plugins import HeatMap, HeatMapWithTime
init_notebook_mode(connected=True)


# # Load and process the data
# 
# There is only one file in the dataset, updated daily.
# 
# 
# ## Cumulative data
# 
# We glimpse the data, looking to shape of the data and some samples from the head and tail of the dataset.

# In[ ]:


data_df = pd.read_csv("/kaggle/input/covid19-coronavirus-romania/covid-19RO.csv")
county_data_df = pd.read_csv("/kaggle/input/covid19-romania-county-level/ro_covid_19_time_series.csv")
country_data_df = pd.read_csv("/kaggle/input/covid19-romania-county-level/ro_covid_19_country_data_time_series.csv")
ro_geo_data = "/kaggle/input/elementary-school-admission-romania-2014/romania.geojson"
ro_large_geo_data = "/kaggle/input/elementary-school-admission-romania-2014/ro_judete_poligon.geojson"


# In[ ]:


data_df.shape


# In[ ]:


data_df.head()


# In[ ]:


data_df.tail()


# In[ ]:


data_df.head()


# In[ ]:


county_data_df.shape


# In[ ]:


county_data_df.head()


# In[ ]:


country_data_df.shape


# In[ ]:


country_data_df.head()


# In[ ]:


country_data_df.tail()


# Let's fix the issue with *2020-06-04*. The data is missing. We will just fill in the average of the days before and after.

# In[ ]:


for feature in ['ati', 'quarantine', 'isolation', 'tests', 'confirmed', 'recovered', 'deaths']:
    country_data_df.loc[country_data_df.date=="2020-06-04",feature] =     int((country_data_df.loc[country_data_df.date=="2020-06-05", feature].values[0] +         country_data_df.loc[country_data_df.date=="2020-06-03", feature].values[0])/2)


# Let's fix the issue with 2020-06-26. The data is missing. We will just fill in the average of the days before and after.

# In[ ]:


for feature in ['ati', 'quarantine', 'isolation', 'tests', 'confirmed', 'recovered', 'deaths']:
    country_data_df.loc[country_data_df.date=="2020-06-26",feature] =     int((country_data_df.loc[country_data_df.date=="2020-06-25", feature].values[0] +         country_data_df.loc[country_data_df.date=="2020-06-27", feature].values[0])/2)


# Let's fix the issue for 2020-06-29.

# In[ ]:


for feature in ['ati', 'quarantine', 'isolation', 'tests', 'confirmed', 'recovered', 'deaths']:
    country_data_df.loc[country_data_df.date=="2020-06-29",feature] =     int((country_data_df.loc[country_data_df.date=="2020-06-28", feature].values[0] +         country_data_df.loc[country_data_df.date=="2020-06-30", feature].values[0])/2)


# In[ ]:


country_data_df.tail()


# Convert the string storing the date to an actual date.

# In[ ]:


data_df['date'] = data_df['date'].apply(lambda x: dt.datetime.strptime(x, "%d/%m/%Y"))
country_data_df['date'] = country_data_df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))


# In[ ]:


fallback_data_two_df = country_data_df.copy()
fallback_data_two_df = fallback_data_two_df[['date', 'confirmed', 'recovered', 'deaths', 'tests']].copy()
fallback_data_two_df.columns = ['date', 'cases', 'recovered', 'deaths', 'tests']
#fallback_data_df['date'] = fallback_data_df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
print(f"Base RO data: {min(data_df['date'])}, {max(data_df['date'])}")
print(f"Fallback RO data: {min(fallback_data_two_df['date'])}, {max(fallback_data_two_df['date'])}")
fallback_data_two_df.tail()


# In[ ]:


#data in date_df and not in country_data_df
date_list_primary = data_df.date.unique()
date_list_fallback_two = fallback_data_two_df.date.unique()
delta_date_list = list(set(date_list_primary) - set(date_list_fallback_two))

data_1_df = data_df.loc[data_df.date.isin(delta_date_list)]
print(data_1_df.shape)
data_2_df = fallback_data_two_df

data_df = pd.concat([data_1_df, fallback_data_two_df], axis=0)
print(data_1_df.shape, data_2_df.shape, data_df.shape)


# In[ ]:


data_df.tail()


# Let's calculate the active cases as well.  
# 
# The number of current active cases is very important, because this is the number that tests the capacity of the health system to respond to the crisis. This crisis is not only a medical crisis, it is also a resources crisis: supply and logistic resources, human resources, managmement resources. Limiting the number of current active cases or finding effective measures to distribute the effort, so that the capacity of health system will not be overhealmed, is of first priority.
# 

# In[ ]:


data_df['active'] = data_df['cases'] - data_df['recovered'] - data_df['deaths']


# ## Daily data
# 
# Let's use pandas diff function to calculate the daily data from the cumulative data.

# In[ ]:


data_daily_df = data_df.diff()
data_daily_df.dropna(inplace=True)
data_daily_df['date'] = data_df['date'][1:]


# We also glimpse the daily data, looking to the shape and some samples of the data.

# In[ ]:


data_daily_df.shape


# In[ ]:


data_daily_df.head()


# In[ ]:


data_daily_df.tail()


# From the Confirmed cases and tests numbers, let's calculate a percent of positive tests results every day. This is calculated from the number of tests and confirmed cases in each day. This is not exact number, because typically a test result will be received 1-2 days later. That means this number is underestimated.

# In[ ]:


data_daily_df['cases_per_tests'] = np.round(data_daily_df['cases'] / data_daily_df['tests'] * 100, 2)
data_daily_df = data_daily_df.replace([np.inf, -np.inf], np.nan)
data_daily_df = data_daily_df.fillna(0)
data_daily_df = data_daily_df.reset_index()


# 
# # Daily data

# In[ ]:


def plot_bars_time_variation(d_df, feature, title, color='Red'):
    
    hover_text = []
    for index, row in d_df.iterrows():
        hover_text.append(('Date: {}<br>'+
                          'Confirmed cases: {}<br>'+
                          'Recovered cases: {}<br>'+
                          'Deaths: {}<br>'+
                          'Tests: {}').format(row['date'],row['cases'], 
                                                   row['recovered'], row['deaths'], row['tests']))
    d_df['hover_text'] = hover_text

    d_df['text'] = hover_text
    trace = go.Bar(
        x = d_df['date'],y = d_df[feature],
        name=feature,
        marker=dict(color=color),
        text = hover_text
    )

    data = [trace]
    layout = dict(title = title,
              xaxis = dict(title = 'Date', showticklabels=True), 
              yaxis = dict(title = title),
              hovermode = 'closest'
             )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='cases-covid19')


# In[ ]:


plot_bars_time_variation(data_daily_df, 'cases', 'Number of daily Confirmed cases', 'Magenta')


# In[ ]:


plot_bars_time_variation(data_daily_df, 'recovered', 'Number of daily Recovered cases', 'Green')


# In[ ]:


plot_bars_time_variation(data_daily_df, 'deaths', 'Number of daily Deaths', 'Red')


# In[ ]:


d_df = data_daily_df.copy()

hover_text = []
for index, row in d_df.iterrows():
    hover_text.append(('Date: {}<br>'+
                      'Confirmed cases: {}<br>'+
                      'Recovered cases: {}<br>'+
                      'Deaths: {}<br>'+
                      'Tests: {}').format(row['date'],row['cases'], 
                                               row['recovered'], row['deaths'], row['tests']))
d_df['hover_text'] = hover_text

d_df['text'] = hover_text
traceR = go.Bar(
    x = d_df['date'],y = d_df['recovered'],
    name='Recovered',
    marker=dict(color='Green'),
    text = hover_text
)
traceD = go.Bar(
    x = d_df['date'],y = d_df['deaths'],
    name='Deaths',
    marker=dict(color='Red'),
    text = hover_text
)

data = [traceD, traceR]
layout = dict(title = 'Recovered & Deaths (New Daily)',
          xaxis = dict(title = 'Date', showticklabels=True), 
          yaxis = dict(title = 'Recovered & Deaths (Daily)'),
          hovermode = 'closest',
          barmode='stack'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='cases-covid19')


# In[ ]:


d_df = data_daily_df.copy()

hover_text = []
for index, row in d_df.iterrows():
    hover_text.append(('Date: {}<br>'+
                      'Confirmed cases: {}<br>'+
                      'Recovered cases: {}<br>'+
                      'Deaths: {}<br>'+
                      'Tests: {}').format(row['date'],row['cases'], 
                                               row['recovered'], row['deaths'], row['tests']))
d_df['hover_text'] = hover_text

d_df['text'] = hover_text
traceR = go.Bar(
    x = d_df['date'],y = d_df['recovered'],
    name='Recovered',
    marker=dict(color='Green'),
    text = hover_text
)
traceD = go.Bar(
    x = d_df['date'],y = d_df['cases'],
    name='Confirmed',
    marker=dict(color='Magenta'),
    text = hover_text
)

data = [traceD, traceR]
layout = dict(title = 'Recovered & Confirmed (Daily New)',
          xaxis = dict(title = 'Date', showticklabels=True), 
          yaxis = dict(title = 'Recovered & Confirmed (Daily)'),
          hovermode = 'closest',
          barmode='stack'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='cases-covid19-recovered-confirmed')


# In[ ]:


plot_bars_time_variation(data_daily_df, 'tests', 'Daily Tests', 'Black')


# The testing capacity increased in the last days.

# In[ ]:


plot_bars_time_variation(data_daily_df, 'cases_per_tests', 'Daily positives per Tests [%]', 'Magenta')


# Let's represent a regression curve associated with the number of tests / day and as well .

# In[ ]:


from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor

d_df = data_daily_df.copy()
d_df = d_df.reset_index()

x = d_df[['index']]
y = d_df['cases_per_tests']

# Linear Regression
model_LR = LinearRegression()
model_LR.fit(x,y)
y1_fit = model_LR.predict(x)

# Bayesian Ridge Regression
model_BR = BayesianRidge()
model_BR.set_params(alpha_init=1.0, lambda_init=0.01)
model_BR.fit(x, y)
y2_fit = model_BR.predict(x)

# Random Forest Regression
model_RF = RandomForestRegressor(max_depth = 5, n_estimators=10)
model_RF.fit(x,y)
y3_fit = model_RF.predict(x)

traceCPTR = go.Scatter(
    x = d_df['date'],y = d_df['cases_per_tests'],
    name='Positives tests %',
    marker=dict(color='Magenta'),
    mode = "markers",
    text = d_df['cases_per_tests']
)

traceLReg = go.Scatter(
    x = d_df['date'],y = y1_fit,
    name='Linear Regression',
    marker=dict(color='Red'),
    mode = "lines",
    text = d_df['cases_per_tests']
)


traceBRReg = go.Scatter(
    x = d_df['date'],y = y2_fit,
    name='Bayesian Ridge Regression',
    marker=dict(color='Blue'),
    mode = "lines",
    text = d_df['cases_per_tests']
)

traceRFReg = go.Scatter(
    x = d_df['date'],y = y3_fit,
    name='RandomForest Regression',
    marker=dict(color='Green'),
    mode = "lines",
    text = d_df['cases_per_tests']
)

data = [traceCPTR, traceLReg, traceBRReg, traceRFReg]
layout = dict(title = 'Percent of positive tests / day (values and regression lines)',
          xaxis = dict(title = 'Date', showticklabels=True), 
          yaxis = dict(title = 'Percent of positive tests / day'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='cases-covid19')


# # Cumulative data
# 
# 
# Let's visualize as well the original (cumulative) data.

# Let's see the graphs of all (cumulative) values: Confirmed cases, Recovered cases, Deaths and Active cases.

# In[ ]:


traceC = go.Scatter(
    x = data_df['date'],y = data_df['cases'],
    name="Confirmed cases",
    marker=dict(color="Magenta"),
    mode = "markers+lines",
    text=data_df['cases'],
)
traceR = go.Scatter(
    x = data_df['date'],y = data_df['recovered'],
    name="Recovered cases",
    marker=dict(color="Green"),
    mode = "markers+lines",
    text=data_df['recovered'],
)
traceD = go.Scatter(
    x = data_df['date'],y = data_df['deaths'],
    name="Deaths",
    marker=dict(color="Red"),
    mode = "markers+lines",
    text=data_df['deaths'],
)
traceA = go.Bar(
    x = data_df['date'],y = data_df['active'],
    name="Active cases",
    marker=dict(color="Blue"),
    #mode = "markers+lines",
    text=data_df['active'],
)


data = [traceC, traceR, traceD, traceA]

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Confirmed cases', 'Recovered cases', 'Deaths', 'Active cases'))
fig.append_trace(traceC, 1, 1)
fig.append_trace(traceR, 1, 2)
fig.append_trace(traceD, 2, 1)
fig.append_trace(traceA, 2, 2)

fig['layout'].update(width=700)
fig['layout'].update(height=600)
fig['layout'].update(title='Cumulative values for Romania')

iplot(fig, filename='covid19-cumulative')


# Let's see cumulative Confirmed cases vs. Active cases.

# In[ ]:


traceC = go.Scatter(
    x = data_df['date'],y = data_df['cases'],
    name="Confirmed cases",
    marker=dict(color="Magenta"),
    mode = "markers+lines",
    text=data_df['cases'],
)

traceA = go.Bar(
    x = data_df['date'],y = data_df['active'],
    name="Active cases (Daily)",
    marker=dict(color="Blue"),
    text=data_df['active'],
)


data = [traceC, traceA]

layout = dict(title = 'Cumulative confirmed and active cases',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Number of cases'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_1')


# Let's see recovered vs. active cases.

# In[ ]:


traceR = go.Scatter(
    x = data_df['date'],y = data_df['recovered'],
    name="Recovered",
    marker=dict(color="Green"),
    mode = "markers+lines",
    text=data_df['recovered'],
)

traceA = go.Bar(
    x = data_df['date'],y = data_df['active'],
    name="Active cases (Daily)",
    marker=dict(color="LightBlue"),
    text=data_df['active'],
)


data = [traceR, traceA]

layout = dict(title = 'Cumulative recovered and active cases',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Number of cases'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_1')


# **01-05-2020** We observe a slow trend toward reducing the growth of daily active patients, with few daily decrease events.
# 
# Let's see cumulative Recovered cases vs. Deaths.

# In[ ]:


traceC = go.Scatter(
    x = data_df['date'],y = data_df['cases'],
    name="Confirmed cases",
    marker=dict(color="Magenta"),
    mode = "markers+lines",
    text=data_df['cases'],
)

traceR = go.Scatter(
    x = data_df['date'],y = data_df['recovered'],
    name="Recovered",
    marker=dict(color="Green"),
    mode = "markers+lines",
    text=data_df['recovered'],
)

traceD = go.Scatter(
    x = data_df['date'],y = data_df['deaths'],
    name="Deaths",
    marker=dict(color="Red"),
    mode = "markers+lines",
    text=data_df['active'],
)

traceA = go.Bar(
    x = data_df['date'],y = data_df['active'],
    name="Active cases (Daily)",
    marker=dict(color="LightBlue"),
    text=data_df['active'],
)


data = [traceC, traceR, traceD, traceA]

layout = dict(title = 'Cumulative Confirmed, Recovered, Deaths  and Daily Active cases',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Number of cases'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_1111')


# In[ ]:


layout = dict(title = 'Cumulative Confirmed, Recovered, Deaths  and Daily Active cases - log scale',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Number of cases', type="log"),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_1111')


# In[ ]:


traceR = go.Scatter(
    x = data_df['date'],y = data_df['recovered'],
    name="Recovered cases",
    marker=dict(color="Green"),
    mode = "markers+lines",
    text=data_df['recovered'],
)
traceD = go.Scatter(
    x = data_df['date'],y = data_df['deaths'],
    name="Deaths",
    marker=dict(color="Red"),
    mode = "markers+lines",
    text=data_df['deaths'],
)
data = [traceR, traceD]

layout = dict(title = 'Cumulative recovered cases and deaths',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Number of cases'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_2')


# Let's see cumulative Death rate (calculated as a ratio of Deaths to Confirmed) and Recovery rate (calculated as a ration of Recovery cases to Confirmed).

# In[ ]:


traceR = go.Scatter(
    x = data_df['date'],y = data_df['recovered'] / data_df['cases'] * 100,
    name="Recovered ratio",
    marker=dict(color="Green"),
    mode = "markers+lines",
    text=data_df['recovered'],
)
traceD = go.Scatter(
    x = data_df['date'],y = data_df['deaths'] / data_df['cases'] * 100,
    name="Deaths ratio",
    marker=dict(color="Red"),
    mode = "markers+lines",
    text=data_df['deaths'],
)
data = [traceR, traceD]

layout = dict(title = 'Cumulative recovered ratio and deaths ratio [%]',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Recovered ratio and deaths ratio [%]'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_3')


# Let's evaluate now mortality (or Death rate) in 2 ways:
# 
# * Mortality defined as ratio of Deaths to Confirmed; this is an underestimate of the morbidity, since the deaths result after a number of days after infection;
# * Mortality defined as ratio of Deaths to Recovered; this is an overestimate, since the duration to recover is typically larger that to die;  
# 
# 

# In[ ]:


d_df = data_df.copy()
d_df = d_df.loc[d_df['deaths']>0]

traceDC = go.Scatter(
    x = d_df['date'],y = d_df['deaths'] / d_df['cases'] * 100,
    name="Deaths / Confirmed ratio",
    marker=dict(color="Magenta"),
    mode = "markers+lines",
    text=np.round(d_df['deaths'] / d_df['cases'] * 100,1),
)
traceDR = go.Scatter(
    x = d_df['date'],y = d_df['deaths'] / d_df['recovered'] * 100,
    name="Deaths / Recovered ratio",
    marker=dict(color="Red"),
    mode = "markers+lines",
    text=np.round(d_df['deaths'] / d_df['recovered'] * 100,1),
)
traceDA = go.Scatter(
    x = d_df['date'],y = d_df['deaths'] / d_df['active'] * 100,
    name="Deaths / Active ratio",
    marker=dict(color="Orange"),
    mode = "markers+lines",
    text=np.round(d_df['deaths'] / d_df['active'] * 100,1),
)

data = [traceDC, traceDR]

layout = dict(title = 'Cumulative Mortality (Deaths ratio to Confirmed and Recovered)  [%]',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Mortality [%]'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_4')


# **01-05-2020** Mortality ratios starts to converge. The Deaths/Recovered ration (which is an overestimate) is decreasing steadily, whilst the Deaths/Confirmed ratio is slowly growing over the last 2 weeks in the range 5-5.7% (this is an underestimate). The Deaths/Active ratio is now at 9.2% (this is difficult to judge if it is an underestimate or an overestimate, at this point). The real mortality ratio might be approximated by the imaginary convergence point of these 3 curves.
# 
# **27-05-2020** Removed Death/Active.
# 

# In[ ]:


layout = dict(title = 'Cumulative Mortality (Deaths ratio to Confirmed and Recovered) - log scale [%]',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Mortality [%]', type="log"),
          hovermode = 'closest',
         )
fig = dict(data=data, layout=layout)

iplot(fig, filename='covid-cases_5')


# We see that mortality calculated as Deaths/Recovered ratio is on a steadly rise (with Death / Confirmed & Deaths / Active as well).  
# 
# From the analysis of similar graphs for countries with a couple of weeks or even a month before Romania in the epidemics development, we conclude that we are still on the early stages of this development: the Deaths / Recovered will peak and then slowly decrease; eventually these two curves will converge to the "real" cumulative mortality.   
# 
# The dynamics of this process really depends on the effectiveness of the imposed measures for social distancing (to reduce the number of confirmed cases), on one side, and on the capacity of the medical system to deal with the difficult cases, to reduce the mortality of the severe confirmed cases.  
# 
# **06-04-2020** update: we see that Mortality % calculated as Deaths / Recovered started to decrease - this usually happens (if we look to the cases of Italy, Spain, France in the first days, after a peak when mostly Deaths are registered (typically death will occur in shorter time than full recovery).
# 
# **09-04-2020** update: we see that Mortality % calculated as Deaths / Recovered continue the descending trend.
# 
# **27-05-2020** update: removed Deaths/Active since no longer relevant due to reduced number of Active.
# 
# 

# # Aditional country-level data
# 
# I am using the second dataset, with some country-level data collected, most notably ICU (in Romanian ATI) patients, quarantined and isolated people.
# 
# Note: we will fix an obvious error, where one data was mistakenly published.

# In[ ]:


country_data_df.loc[country_data_df.ati == 274, 'ati'] = 174
country_data_df.loc[country_data_df.quarantine==0, 'quarantine'] = None


# In[ ]:


traceICU = go.Bar(
    x = country_data_df['date'],y = country_data_df['ati'],
    name="ICU patients (Daily)",
    marker=dict(color="Magenta"),
    #mode = "markers+lines",
    text=country_data_df['ati'],
)

dd_df = data_daily_df.loc[data_daily_df.date >= min(country_data_df.date)].copy()

traceDD = go.Bar(
    x = dd_df['date'],y = dd_df['deaths'],
    name="Deaths (Daily)",
    marker=dict(color="Red"),
    text=dd_df['deaths'],
)

traceDR = go.Bar(
    x = dd_df['date'],y = dd_df['recovered'],
    name="Recovered (Daily)",
    marker=dict(color="Green"),
    text=dd_df['recovered'],
)

d_df = data_df.loc[data_df.date >= min(country_data_df.date)].copy()

traceR = go.Scatter(
    x = d_df['date'],y = d_df['recovered'],
    name="Recovered (Cumulative)",
    marker=dict(color="Green"),
    mode = "markers+lines",
    text=d_df['recovered'],
)


traceD = go.Scatter(
    x = d_df['date'],y = d_df['deaths'],
    name="Deaths (Cumulative)",
    marker=dict(color="Red"),
    mode = "markers+lines",
    text=d_df['deaths'],
)

data = [traceICU, traceDD, traceDR, traceR, traceD]

layout = dict(title = 'Current number of patients in ICU vs. Recovered & Deaths',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Number of patients'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_icu')


# In[ ]:


layout = dict(title = 'Current number of patients in ICU vs. Recovered & Deaths - log scale',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Number of patients', type="log"),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_icu')


# In[ ]:


traceICU = go.Bar(
    x = country_data_df['date'],y = country_data_df['ati'],
    name="ICU patients (Daily)",
    marker=dict(color="Magenta"),
    #mode = "markers+lines",
    text=country_data_df['ati'],
)

dd_df = data_daily_df.loc[data_daily_df.date >= min(country_data_df.date)].copy()

d_df = data_df.loc[data_df.date >= min(country_data_df.date)].copy()

traceA = go.Bar(
    x = dd_df['date'],y = d_df['active'],
    name="Active (Daily)",
    marker=dict(color="Blue"),
    text=d_df['active'],
)

data = [traceA,traceICU]

layout = dict(title = 'Current number of patients - Active vs. in ICU',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Number of patients'),
          hovermode = 'closest',
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_icu')


# **25-04-2020** Yesterday was the first day when total number of active cases droped.
# 
# **02-05-2020** Over the last week, the number of active cases continued to drop several times.

# In[ ]:


layout = dict(title = 'Current number of patients - Active vs. in ICU - log scale',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Number of patients', type="log"),
          hovermode = 'closest',
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_icu')


# In[ ]:


traceQ = go.Bar(
    x = country_data_df['date'],y = country_data_df['quarantine'],
    name="Quarantined",
    marker=dict(color="Orange"),
    text=country_data_df['quarantine'],
)

traceI = go.Bar(
    x = country_data_df['date'],y = country_data_df['isolation'],
    name="Isolated",
    marker=dict(color="Blue"),
    text=country_data_df['isolation'],
)

d_df = data_df.loc[data_df.date >= min(country_data_df.date)].copy()
traceC = go.Scatter(
    x = d_df['date'],y = d_df['cases'],
    name="Confirmed",
    marker=dict(color="Red"),
    mode = "markers+lines",
    text=d_df['cases'],
)

data = [traceQ, traceI, traceC]

layout = dict(title = 'Current number of people in quarantine and isolation + Confirmed cases',
          xaxis = dict(title = 'date', showticklabels=True), 
          yaxis = dict(title = 'Number of people'),
          hovermode = 'closest',
          #barmode='stack'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_qi')


# Update **20-05-2020** - number of isolated citizens is in sharp increase, due to return in Romania, after opening the borders, of Romanian citizens from <font color="red">red</font> zones.
# 
# Update **27-05-2020** - number of isolated citizens continue the increase, reached over 80K as of today.
# 
# Update **01-06-2020** - number of isolated citizens decreased.

# # Evolution projection 
# 
# **Disclaimer**: this should not be interpreted as a prediction, since this data has a very dynamic characteristic and there are a lot of unknowns governing this dynamic.   
# 
# We are making here an extremely simplified assumption, i.e. that the evolution follows a logistic curve, and we are fitting a **logistic curve**.
# 
# 
# ## Fit a logistic curve
# 
# 
# Let's try to fit a Logistic curve for predicting future behavior of the cumulative number of confirmed cases.
# 
# I took the formulae from @oriano Kernel: https://www.kaggle.com/orianao/covid-19-logistic-curve-prediction
# 
# * L (the maximum number of confirmed cases) = 250000 taken from the US example (this is from long time obsolete now)
# * k (growth rate) = 0.25 approximated value from most of the countries
# * x0 (the day of the inflexion) = 80 approximated
# 
# The curve being:
# 
# $$y = \frac{L}{1 + e^{-k (x-x_0)}} + 1$$
# 

# In[ ]:


def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0))) + 1


# In[ ]:


import datetime
import scipy

p0 = (0,0,0)
def plot_logistic_fit_data(d_df, title, p0=p0):
    d_df = d_df.sort_values(by=['date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1
    d_df['y'] = d_df['cases']

    x = d_df['x']
    y = d_df['y']

    c2 = scipy.optimize.curve_fit(logistic,  x,  y,  p0=p0 )
    #y = logistic(x, L, k, x0)
    popt, pcov = c2

    x = range(1,d_df.shape[0] + int(popt[2]))
    y_fit = logistic(x, *popt)
    
    p_df = pd.DataFrame()
    p_df['x'] = x
    p_df['y'] = y_fit.astype(int)
    
    print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
    print("Predicted k (growth rate): " + str(float(popt[1])))
    print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")

    x0 = int(popt[2])
    
    traceC = go.Scatter(
        x=d_df['x'], y=d_df['y'],
        name="Confirmed",
        marker=dict(color="Red"),
        mode = "markers+lines",
        text=d_df['cases'],
    )

    traceP = go.Scatter(
        x=p_df['x'], y=p_df['y'],
        name="Projected",
        marker=dict(color="blue"),
        mode = "lines",
        text=p_df['y'],
    )
    
    trace_x0 = go.Scatter(
        x = [x0, x0], y = [0, p_df.loc[p_df['x']==x0,'y'].values[0]],
        name = "X0 - Inflexion point",
        marker=dict(color="black"),
        mode = "lines",
        text = "X0 - Inflexion point"
    )

    data = [traceC, traceP, trace_x0]

    layout = dict(title = 'Cumulative Confirmed cases and logistic curve projection',
          xaxis = dict(title = 'Day since first case', showticklabels=True), 
          yaxis = dict(title = 'Number of cases'),
          hovermode = 'closest'
         )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='covid-logistic-forecast')


# In[ ]:


d_df = data_df.copy()
L = 250000
k = 0.25
x0 = 100
p0 = (L, k, x0)
plot_logistic_fit_data(d_df, 'Romania')


# **29-03-2020**: Current projection using logistic curve shows for Romania that inflexion will be reached on day 37 (in one week from today), with the peak of the epidemics in another 5 weeks (May).
# 
# **31-03-2020**: Current projection shows  that inflexion will be reached on day 36, with the peak of the epidemics in another 5 weeks (May).
# 
# **4-04-2020**: Current prediction shows  that inflexion was already reached and plateau will be in day 75, with 6.7K cases.
# 
# **14-04-2020**: Current projection shows  that inflexion moved to 41st day, was already reached and plateau will be in day 88, with 8.5K cases. This evolution shows obviously that fitting a logistic curve is not accurately representing the data and needs daily adjustment due to significant dynamic of the data.
# 
# **17-04-2020**: Current projection  shows  that inflexion moved to 42nd day, was already reached and plateau will be in day 92, with 9.2K cases. This evolution shows obviously that fitting a logistic curve is not accurately representing the data and needs daily adjustment due to significant dynamic of the data. The real curve started to move away from the logistic curve fitted.
# 
# **20-04-2020**: Current projection  shows  that inflexion moved to 43nd day, and will plateau at 10.1K, day 97. The evolution will have to be continously monitored over the next 5-7 days, without being able to confirm or infirm a saturation effect.
# 
# 
# **25-04-2020**: Current projection  shows that logistic curve evolution for cumulative cases is no longer possible.
# 
# **01-05-2020**: Current projection  shows that logistic curve evolution for cumulative cases is no longer possible. The inflexion point is moved now to day 48.
# 
# **15-05-2020**: Current projection shows that we are approaching the plateau zone.
# 
# 
# We will continue to monitor the evolution over the next days.
# 
# **Disclaimer note**: this analysis is not intended as a prediction or forecast, is just studying how the evolution will be if we try to fit the data with a logistic curve. The assumption is most probably not accurate.
# 
# 
# ## Fitting an exponential curve
# 
# 
# **Disclaimer**: this should not be interpreted as a prediction, since this data has a very dynamic characteristic and there are a lot of unknowns governing this dynamic.
# 
# We are making here an extremely **simplified assumption**, i.e. that the evolution follows an exponential curve, and we are fitting this **exponential curve**.
# 
# The parameters for the curve are:
# 
# * A - the constant multiplier for the exponential
# * B - the multiplier for the exponent
# 
# The curve is thus:
# 
# $$y = Ae^{Bx}$$
# 
# 

# In[ ]:


import datetime
import scipy
p0 = (0,0)
def plot_exponential_fit_data(d_df, title, delta, p0):
    d_df = d_df.sort_values(by=['date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1
    d_df['y'] = d_df['cases']

    x = d_df['x'][:-delta]
    y = d_df['y'][:-delta]

    c2 = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y,  p0=p0)

    A, B = c2[0]
    print(f'(y = Ae^(Bx)) A: {A}, B: {B}')
    x = range(1,d_df.shape[0] + 1)
    y_fit = A * np.exp(B * x)
    
    traceC = go.Scatter(
        x=d_df['x'][:-delta], y=d_df['y'][:-delta],
        name="Confirmed (included for fit)",
        marker=dict(color="Red"),
        mode = "markers+lines",
        text=d_df['cases'],
    )

    traceV = go.Scatter(
        x=d_df['x'][-delta-1:], y=d_df['y'][-delta-1:],
        name="Confirmed (validation)",
        marker=dict(color="blue"),
        mode = "markers+lines",
        text=d_df['cases'],
    )
    
    traceP = go.Scatter(
        x=np.array(x), y=y_fit,
        name="Projected values (fit curve)",
        marker=dict(color="green"),
        mode = "lines",
        text=y_fit,
    )

    data = [traceC, traceV, traceP]

    layout = dict(title = 'Cumulative Conformed cases and exponential curve projection',
          xaxis = dict(title = 'Day since first case', showticklabels=True), 
          yaxis = dict(title = 'Number of cases'),
          hovermode = 'closest'
         )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='covid-exponential-forecast')


# In[ ]:


d_df = data_df.copy()
p0 = (40, 0.2)
plot_exponential_fit_data(d_df, 'Romania', 7, p0)


# **Disclaimer note**: this analysis is not intended as a prediction or forecast, is just studying how the evolution will be if we try to fit the data with an exponential curve. The assumption is most probably not accurate, as it is shown from the comparison of the projected values and real values, used in the validation (for the last week). 
# 
# **06-04-2020** update - the trend of the real curve is to distance from the exponential fited.
# 
# **08-04-2020** update - the trend of the real curve continue to depart from the exponential fit.
# 
# **14-04-2020** update - the trend of the real curve continue to take distance from the exponential fit.
# 
# **17-04-2020** update - the trend of the real curve continue to move away from the exponential fit.
# 
# **20-04-2020** update - we can confirm that the trend of the real data is not possible to be modeled using an exponential curve.

# ## Fitting Active cases with polynomials
# 
# 
# We fit the active cases with polynomials (with p=3,4,5). We only show the fitted curve for p=3.   
# 
# We then calculate the evolution (based on the fitted curve) for more 7 days.

# In[ ]:


def plot_polinomial_fit_data(d_df):
    d_df = d_df.sort_values(by=['date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1
    d_df['y'] = d_df['active']

    x = d_df['x']
    y = d_df['y']

    p3 =  np.poly1d(np.polyfit(x, y, 3))
    p4 =  np.poly1d(np.polyfit(x, y, 4))
    p5 =  np.poly1d(np.polyfit(x, y, 5))

    
    xp = range(20,d_df.shape[0] + 7)
    yp3 = p3(xp)
    yp4 = p4(xp)
    yp5 = p5(xp)
    
    p_df = pd.DataFrame()
    p_df['x'] = xp
    p_df['y3'] = np.round(yp3,0)
    p_df['y4'] = np.round(yp4,0)
    p_df['y5'] = np.round(yp5,0)


    traceA = go.Scatter(
        x=d_df['x'], y=d_df['y'],
        name="Active",
        marker=dict(color="Red"),
        mode = "markers+lines",
        text=d_df['active'],
    )

    traceP3 = go.Scatter(
        x=p_df['x'], y=p_df['y3'],
        name="p = 3",
        marker=dict(color="blue"),
        mode = "lines",
        text=p_df['y3'],
    )
    traceP4 = go.Scatter(
        x=p_df['x'], y=p_df['y4'],
        name="p = 4",
        marker=dict(color="lightblue"),
        mode = "lines",
        text=p_df['y4'],
    )
    traceP5 = go.Scatter(
        x=p_df['x'], y=p_df['y5'],
        name="p = 5",
        marker=dict(color="darkblue"),
        mode = "lines",
        text=p_df['y5'],
    )

    
    data = [traceA, traceP3, traceP4, traceP5]

    layout = dict(title = 'Active cases and polynomial (p=3, p=4) curve projection (for +2 weeks)',
          xaxis = dict(title = 'Day since first case', showticklabels=True), 
          yaxis = dict(title = 'Number of active cases'),
          hovermode = 'closest'
         )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='covid-polinomial-projection')


# In[ ]:


d_df = data_df.copy()
plot_polinomial_fit_data(d_df)


# Update **20-05-2020** the decrease of the number of active cases slowed down over last 2 days.
# 
# Update **27-05-2020** the decrease of the number of active cases continues, but with a slow pace.
# 
# 

# # County-level Confirmed cases
# 
# 
# Note: on May 7th, the official site from where we scrap these informations discontinued to publish county-level data.

# In[ ]:


import json
import difflib 

# retrieve the county names from geoJson
with open(ro_geo_data) as json_file:
    json_data = json.load(json_file)
county_lat_long_df = pd.DataFrame()
for item in json_data['features']:
    polygons = list(shape(item['geometry']))
    county = item['properties']['name']
    county_lat_long_df = county_lat_long_df.append(pd.DataFrame({'county': county, 'Lat':polygons[0].centroid.y, 'Long': polygons[0].centroid.x}, index=[0]))
# merge county data    
county_join = pd.DataFrame(list(county_data_df.County.unique()))
county_join.columns = ['County']
county_join = county_join.loc[~(county_join.County=='Not identified')]
county_join.head()
# match the county names
difflib.get_close_matches
county_lat_long_df['County'] = county_lat_long_df.county.map(lambda x: difflib.get_close_matches(x, county_join.County)[0])
print(f"Validation [polygons]: {county_lat_long_df.County.nunique()},{county_lat_long_df.county.nunique()}")

with open(ro_large_geo_data) as json_file:
    json_data = json.load(json_file)
county_population_df = pd.DataFrame()
for item in json_data['features']:
    county = item['properties']['name']
    population = item['properties']['pop2011']
    county_population_df = county_population_df.append(pd.DataFrame({'county': county, 'population': population}, index=[0]))
difflib.get_close_matches
county_population_df['County'] = county_population_df.county.map(lambda x: difflib.get_close_matches(x, county_join.County)[0])
print(f"Validation [population]: {county_population_df.County.nunique()},{county_population_df.county.nunique()}")


county_data_df = county_data_df.merge(county_lat_long_df, on='County', how='inner')
county_data_df = county_data_df.merge(county_population_df[['County', 'population']], on='County', how='inner')

county_data_df['percent_confirmed'] =  county_data_df['Confirmed'] / county_data_df['population'] * 100

last_data_df = county_data_df.loc[county_data_df.Date==max(county_data_df.Date)].reset_index()
print(f"Validation [last date]: {last_data_df.county.nunique()}")
county_data_df.head()


# In[ ]:


ro_map = folium.Map(location=[45.9, 24.9], zoom_start=6)

folium.Choropleth(
    geo_data=ro_geo_data,
    name='Counties countour plots',
    data=last_data_df,
    columns=['county', 'Confirmed'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    bins=[0, 250, 500, 750, 1000, 2000, 3000, 4000, 5000],
    fill_opacity=0.6,
    line_opacity=0.5,
    legend_name='Confirmed cases / county'
).add_to(ro_map)



radius_min = 3
radius_max = 20
weight = 1
fill_opacity = 0.5

_color_conf = 'Magenta'
group0 = folium.FeatureGroup(name='<span style=\\"color: #EFEFE8FF;\\">Popups</span>')
for i in range(len(last_data_df)):
    lat = last_data_df.loc[i, 'Lat']
    lon = last_data_df.loc[i, 'Long']
    county = last_data_df.loc[i, 'county']

    _radius_conf = 0.5 * np.sqrt(last_data_df.loc[i, 'Confirmed'])
    if _radius_conf < radius_min:
        _radius_conf = radius_min

    if _radius_conf > radius_max:
        _radius_conf = radius_max

    _popup_conf = str(county) + '\nConfirmed: '+str(last_data_df.loc[i, 'Confirmed'])
                                                    
    folium.CircleMarker(location = [lat,lon], 
                        radius = _radius_conf, 
                        popup = _popup_conf, 
                        tooltip = _popup_conf,
                        color = _color_conf, 
                        fill_opacity = fill_opacity,
                        weight = weight, 
                        fill = True, 
                        fillColor = _color_conf).add_to(group0)

group0.add_to(ro_map)


folium.LayerControl().add_to(ro_map)

ro_map


# Let's plot the population / County (as of the last census, in 2011).

# In[ ]:


ro_map = folium.Map(location=[45.9, 24.9], zoom_start=6)

folium.Choropleth(
    geo_data=ro_geo_data,
    name='Population / County',
    data=last_data_df,
    columns=['county', 'population'],
    key_on='feature.properties.name',
    fill_color='Blues',
    fill_opacity=0.8,
    line_opacity=0.5,
    legend_name='Population / county'
).add_to(ro_map)

radius_min = 3
radius_max = 15
weight = 1
fill_opacity = 0.5

_color_conf = 'Blue'
group0 = folium.FeatureGroup(name='<span style=\\"color: #EFEFE8FF;\\">Popups</span>')
for i in range(len(last_data_df)):
    lat = last_data_df.loc[i, 'Lat']
    lon = last_data_df.loc[i, 'Long']
    county = last_data_df.loc[i, 'county']

    _radius_conf = 0.01 * np.sqrt(last_data_df.loc[i, 'population'])
    if _radius_conf < radius_min:
        _radius_conf = radius_min

    if _radius_conf > radius_max:
        _radius_conf = radius_max

    _popup_conf = str(county) + '\nPopulation: '+str(last_data_df.loc[i, 'population'])
                                                    
    folium.CircleMarker(location = [lat,lon], 
                        radius = _radius_conf, 
                        popup = _popup_conf, 
                        color = _color_conf, 
                        fill_opacity = fill_opacity,
                        weight = weight, 
                        fill = True, 
                        fillColor = _color_conf).add_to(group0)

group0.add_to(ro_map)
folium.LayerControl().add_to(ro_map)
ro_map


# And let's calculate and plot the percent of population / county with Confirmed cases.

# In[ ]:


ro_map = folium.Map(location=[45.9, 24.9], zoom_start=6)


folium.Choropleth(
    geo_data=ro_geo_data,
    name='Percent confirmed from County population',
    data=last_data_df,
    columns=['county', 'percent_confirmed'],
    key_on='feature.properties.name',
    fill_color='Reds',
    bins = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.4, 0.7],
    fill_opacity=0.8,
    line_opacity=0.5,
    legend_name='Confirmed from Population [%] / county'
).add_to(ro_map)

_color_conf = 'Magenta'
radius_min = 3
radius_max = 15
weight = 1
fill_opacity = 0.5

group0 = folium.FeatureGroup(name='<span style=\\"color: #EFEFE8FF;\\">Popups</span>')
for i in range(len(last_data_df)):
    lat = last_data_df.loc[i, 'Lat']
    lon = last_data_df.loc[i, 'Long']
    county = last_data_df.loc[i, 'county']

    _radius_conf = 25 * np.sqrt(last_data_df.loc[i, 'percent_confirmed'])
    if _radius_conf < radius_min:
        _radius_conf = radius_min

    if _radius_conf > radius_max:
        _radius_conf = radius_max

    _tooltip_conf_percent = str(county) + '\n - Confirmed/Population: '+str(np.round(last_data_df.loc[i, 'percent_confirmed'],3))+'%'
                                                    
    folium.CircleMarker(location = [lat,lon], 
                        radius = _radius_conf,  
                        tooltip = _tooltip_conf_percent,
                        color = _color_conf, 
                        fill_opacity = fill_opacity,
                        weight = weight, 
                        fill = True, 
                        fillColor = _color_conf).add_to(group0)

group0.add_to(ro_map)


folium.LayerControl().add_to(ro_map)

ro_map


# # County-level Confirmed data - Animation

# In[ ]:


mindate = min(county_data_df['Date'])
maxdate = max(county_data_df['Date'])
print(f"Date min/max: {mindate}, {maxdate}")
data_ro_df = county_data_df.copy()
data_ro_df['Date'] = data_ro_df['Date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
data_ro_df['Days'] = data_ro_df['Date'].apply(lambda x: (x - dt.datetime.strptime('2020-04-02', "%Y-%m-%d")).days)
print(f"Date min/max: {min(data_ro_df['Days'])}, {max(data_ro_df['Days'])}, Days: {data_ro_df['Days'].nunique()} ")


# In[ ]:


import plotly.express as px
data_ro_df.loc[data_ro_df.Confirmed.isna(), 'Confirmed'] = 0
data_ro_df['Size'] = np.round(5 * (data_ro_df['Confirmed']),0)
max_confirmed = max(data_ro_df['Confirmed'])
min_confirmed = min(data_ro_df['Confirmed'])
hover_text = []
for index, row in data_ro_df.iterrows():
    hover_text.append(('Date: {}<br>'+
                       'County: {}<br>'+
                       'Population: {}<br>'+
                      'Confirmed: {}<br>'+
                      'Confirmed/Population: {}%').format(row['Date'], 
                                            row['County'],
                                            row['population'],
                                            row['Confirmed'],
                                            round(row['percent_confirmed'],3)))
data_ro_df['hover_text'] = hover_text
fig = px.scatter_geo(data_ro_df, scope = 'europe',
                     width=700, height=525, size_max=50,
                     lon='Long', lat='Lat', color="Confirmed",
                     hover_name="hover_text", size="Size",
                     animation_frame="Days",
                     projection="natural earth", 
                     range_color =[min_confirmed,max_confirmed])
fig.update_geos(
    center=dict(lat=45.9, lon=24.9),
    projection_rotation=dict(lon=24.9, lat=45.9, roll=0),
    lataxis_range=[43.5,48.5], lonaxis_range=[20,30],
   
    showcoastlines=True, coastlinecolor="DarkBlue",
    showland=True, landcolor="LightGrey",
    showocean=True, oceancolor="LightBlue",
    showlakes=True, lakecolor="Blue",
    showrivers=True, rivercolor="Blue",
    showcountries=True, countrycolor="DarkBlue"
)

fig.show()


# We can also compare the count / countries (for the last/current day) using a bar plot.

# In[ ]:


d_df = last_data_df.copy()
d_df = d_df.sort_values(by=['Confirmed'], ascending = False)

hover_text = []
for index, row in d_df.iterrows():
    hover_text.append(('County: {}<br>'+
                      'Confirmed cases: {}').format(row['County'], row['Confirmed']))
d_df['hover_text'] = hover_text

    
trace = go.Bar(
    x = d_df['County'],y = d_df['Confirmed'],
    name='Confirmed',
    marker=dict(color='Red'),
    text = hover_text,
)

data = [trace]
layout = dict(title = 'Confirmed cases per County - last day',
          xaxis = dict(title = 'County', showticklabels=True),
          yaxis = dict(title = 'Confirmed cases'),
          hovermode = 'closest',
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='cases-covid19-county-all')


# In[ ]:


d_df = last_data_df.copy()
d_df = d_df.sort_values(by=['percent_confirmed'], ascending = False)

hover_text = []
for index, row in d_df.iterrows():
    hover_text.append(('County: {}<br>'+
                      'Confirmed cases: {}<br>'+
                      'Percent from population: {}').format(row['County'], row['Confirmed'], np.round(row['percent_confirmed'],4)))
d_df['hover_text'] = hover_text

    
trace = go.Bar(
    x = d_df['County'],y = d_df['percent_confirmed'],
    name='Confirmed',
    marker=dict(color='Magenta'),
    text = hover_text,
)

data = [trace]
layout = dict(title = 'Percent from population of Confirmed cases per County - last day',
          xaxis = dict(title = 'County', showticklabels=True),
          yaxis = dict(title = 'Percent from population of Confirmed [%]'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='cases-covid19-county-all-percent')


# ## County-level Data - Time Evolution

# In[ ]:


d_df = county_data_df.copy()
d_df = d_df.loc[d_df['Confirmed']>0]
counties = list(d_df.County.unique())

data = []
for county in counties:
    dc_df = d_df.loc[d_df.County==county]
    traceC = go.Scatter(
        x = dc_df['Date'],y = dc_df['Confirmed'],
        name=county,
        mode = "markers+lines",
        text=dc_df['Confirmed']
    )
    data.append(traceC)

layout = dict(title = 'Confirmed cases per County (log scale)',
          xaxis = dict(title = 'Date', showticklabels=True), 
          yaxis = dict(title = 'Confirmed cases (log scale)'),
          yaxis_type="log",
          hovermode = 'y',
          height=1000
         )

fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_7')


# In[ ]:


import plotly.express as px
fig = px.bar(d_df, x="Confirmed", y="County", animation_frame='Date', orientation='h',range_color =[min_confirmed,max_confirmed],
             width=600, height=800, range_x = [min_confirmed,max_confirmed])
fig.update_layout(font=dict(family="Courier New, monospace",size=10,color="#7f7f7f"))
fig.show()


# We can also look to the percent of population that represents the Confirmed cases, in each County.

# In[ ]:


d_df = county_data_df.copy()
d_df = d_df.loc[d_df['Confirmed']>0]
counties = list(d_df.County.unique())

data = []
for county in counties:
    dc_df = d_df.loc[d_df.County==county]
    traceC = go.Scatter(
        x = dc_df['Date'],y = dc_df['percent_confirmed'],
        name=county,
        mode = "markers+lines",
        text=dc_df['percent_confirmed']
    )
    data.append(traceC)

layout = dict(title = 'Percent of Confirmed cases (from the population) per County (log scale)',
          xaxis = dict(title = 'Date', showticklabels=True), 
          yaxis = dict(title = 'Percent of population representing Confirmed cases (log scale)'),
          yaxis_type="log",
          hovermode = 'y',
          height=1000
         )

fig = dict(data=data, layout=layout)
iplot(fig, filename='covid-cases_9')


# In[ ]:


import plotly.express as px
min_percent_confirmed = min(d_df['percent_confirmed'])
max_percent_confirmed = max(d_df['percent_confirmed'])
fig = px.bar(d_df, x="percent_confirmed", y="County", animation_frame='Date', orientation='h',
             range_x =[min_percent_confirmed,max_percent_confirmed],
             width=700, height=800)
fig.update_layout(font=dict(family="Courier New, monospace",size=10,color="#7f7f7f"))
fig.show()

