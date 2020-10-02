#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# #### The predict value is not accurate unless the confirm/deaths/recorverd has a huge increasement duration > 2-3 weeks, so please not be scared and keep patient. Figures are for reference only.
# 
# 
# 
# 
# 
# 

# # Exploratory Data Analysis and  prediction models of Covid-19 in Canada
# 
# This project charts the progress of Covid-19 in Canada and presently implements one model for prediction,  Please see the References section for reference data sources and notebooks. This Notebook is an evolving work. The goal is to implement multiple data sources, charts and models to present a clear portrait of the pandemic in Canada.

# # To Do:
# 
# 
# * Explore currrent Ontario data and find cause of model bug. -- In progress 04042020
# * Explore other data sets and models , implement.
# * Explore NLP ideas for media coverage.
# 
# 

# # Basic Data Analysis

# 

# In[ ]:


# Project: Novel Corona Virus 2019 Dataset by Kaggle
# Program: COVID-19 in Canada with model
# Author:  Michel LeBlond
# Date:   April 3, 2020

# TODO explore ontario data and other models.



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots

import os

# Input data files are available in the "../input/" directory.


# In[ ]:


df_covid = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])
df_covid.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country', 'Province/State':'Province' }, inplace=True)


# Check the latest observation date

# In[ ]:


ondate=max(df_covid['Date'])
print ("The last observation date is " + ondate)


# # Functions for plotting

# In[ ]:


def plot_bar_chart(confirmed, deaths, recoverd, country, fig=None):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Bar(x=confirmed['Date'],
                y=confirmed['Confirmed'],
                name='Confirmed'
                ))
    fig.add_trace(go.Bar(x=deaths['Date'],
                y=deaths['Deaths'],
                name='Deaths'
                ))
    fig.add_trace(go.Bar(x=recovered['Date'],
                y=recovered['Recovered'],
                name='Recovered'
                ))

    fig.update_layout(
        title= 'Cumulative Daily Cases of COVID-19 (Confirmed, Deaths, Recovered) - ' + country + ' as of ' + ondate ,
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Number of Cases',
            titlefont_size=14,
            tickfont_size=12,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, 
        bargroupgap=0.1 
    )
    return fig


# In[ ]:


def plot_line_chart(confirmed, deaths, recoverd, country, fig=None):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=confirmed['Date'], 
                         y=confirmed['Confirmed'],
                         mode='lines+markers',
                         name='Confirmed'
                         ))
    fig.add_trace(go.Scatter(x=deaths['Date'], 
                         y=deaths['Deaths'],
                         mode='lines+markers',
                         name='Deaths'
                         ))
    fig.add_trace(go.Scatter(x=recovered['Date'], 
                         y=recovered['Recovered'],
                         mode='lines+markers',
                         name='Recovered'
                        ))
    fig.update_layout(
        title= 'Number of COVID-19 Cases Over Time - ' + country + ' as of ' + ondate ,
        xaxis_tickfont_size=12,
        yaxis=dict(
           title='Number of Cases',
           titlefont_size=14,
           tickfont_size=12,
        ),
        legend=dict(
           x=0,
           y=1.0,
           bgcolor='rgba(255, 255, 255, 0)',
           bordercolor='rgba(255, 255, 255, 0)'
        )
     )
    return fig


# # What is happening Worldwide

# In[ ]:


confirmed = df_covid.groupby('Date').sum()['Confirmed'].reset_index() 
deaths = df_covid.groupby('Date').sum()['Deaths'].reset_index() 
recovered = df_covid.groupby('Date').sum()['Recovered'].reset_index()


# In[ ]:


plot_bar_chart(confirmed, deaths, recovered,'Worldwide').show()


# In[ ]:


plot_line_chart(confirmed, deaths, recovered,'Worldwide').show()


# 

# # What is going on in China the World and Canada

# 

# In[ ]:


full_table = df_covid.copy() # we need this to build the model later
# Active Case = confirmed - deaths - recovered
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



full_table['Date'] = pd.to_datetime(full_table['Date'])
#full_table.tail()

# Clean Data for Canada 
full_table = full_table.replace(to_replace =["Calgary, Alberta", "Edmonton, Alberta"],value ="Alberta")
full_table =full_table.replace(to_replace =[" Montreal, QC"],value ="Quebec") 
# This generated duplicate dates that broke when running predictions through our model
# Changing to other labels for now.  Merge data on duplicate dates in next pass at code.
#full_table = full_table.replace(to_replace =["Toronto, ON", "London, ON"],value ="Ontario") 
#full_table =full_table.replace(to_replace =["Diamond Princess cruise ship"],value ="Ontario") 
full_table = full_table.replace(to_replace =["Toronto, ON", "London, ON"],value ="OntarioOther") 
full_table =full_table.replace(to_replace =["Diamond Princess cruise ship"],value ="OntarioOther") 

China_df = full_table[full_table['Country'] == 'Mainland China'].copy()
#China_df.tail() 
#full_table.Active.describe()

China_df.tail() 
#df_covid['Country'].unique()


# # ****Build Model

# In[ ]:


def get_time_series(country):
    # for some countries, data is spread over several Provinces
    if full_table[full_table['Country'] == country]['Province'].nunique() > 1:
        country_table = full_table[full_table['Country'] == country]
        country_df = pd.DataFrame(pd.pivot_table(country_table, values = ['Confirmed', 'Deaths', 'Recovered', 'Active'],
                              index='Date', aggfunc=sum).to_records())
        return country_df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]
    df = full_table[(full_table['Country'] == country) 
                & (full_table['Province'].isin(['', country]))]
    return df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]


def get_time_series_province(province):
    # for some countries, data is spread over several Provinces
    df = full_table[(full_table['Province'] == province)]
    return df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]


# I will use a model from a marketing paper by Emmanuelle Le Nagard and Alexandre Steyer, that attempts to reflect the social structure of a diffusion process. Their application was the diffusion of innovations, not epidemics. However, there are commonalities in both domains, as the number of contacts each infected person / innovation adopter has seems relevant. It also has the added benefit to allow fitting parameters to the beginning of a time series.
# 
# paper is available (in French) here
# 
# The model is also sensitive to when we define the origin of time for the epidemic process. Here, I just took the first point of the time series available, but adding a lag parameter could be attempted.
# 

# In[ ]:


country = 'Mainland China'
df = get_time_series(country)
if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:
    df.drop(df.tail(1).index,inplace=True)
df.tail(10)


# In[ ]:


import math
def model_with_lag(N, a, alpha, lag, t):
    # we enforce N, a and alpha to be positive numbers using min and max functions
    lag = min(max(lag, -100), 100) # lag must be less than +/- 100 days 
    return max(N, 0) * (1 - math.e ** (min(-a, 0) * (t - lag))) ** max(alpha, 0)

def model(N, a, alpha, t):
    return max(N, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)


# In[ ]:


model_index = 0

def model_loss(params):
#     N, a, alpha, lag = params
    N, a, alpha = params
    model_x = []
    r = 0
    for t in range(len(df)):
        r += (model(N, a, alpha, t) - df.iloc[t, model_index]) ** 2
#         r += (math.log(1 + model(N, a, alpha, t)) - math.log(1 + df.iloc[t, 0])) ** 2 
#         r += (model_with_lag(N, a, alpha, lag, t) - df.iloc[t, 0]) ** 2
#         print(model(N, a, alpha, t), df.iloc[t, 0])
    return math.sqrt(r) 


# We need to explore the 3d parameter space to find a minimum, using gradient descent. There are a number of algorithms to do that in scipy.optimize, I stopped at the first one that seemed to work. Generalized Reduced Gradient as in Excel solver also works.

# In[ ]:


import numpy as np
from scipy.optimize import minimize
use_lag_model = False
if use_lag_model:
    opt = minimize(model_loss, x0=np.array([200000, 0.05, 15, 0]), method='Nelder-Mead', tol=1e-5).x
else:
    model_index = 0
    opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
    model_index = 1
    opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
    model_index = 2
    opt_recovered = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

model_x = []
for t in range(len(df)):
    model_x.append([df.index[t], model(*opt_confirmed, t), model(*opt_deaths, t), model(*opt_recovered, t)])
model_sim = pd.DataFrame(model_x, dtype=int)
model_sim.set_index(0, inplace=True)
model_sim.columns = ['Model-Confirmed', 'Model-Deaths', 'Model-Recovered']

model_sim['Model-Active'] = model_sim['Model-Confirmed'] - model_sim['Model-Deaths'] - model_sim['Model-Recovered']
model_sim.loc[model_sim['Model-Active']<0,'Model-Active'] = 0
plot_color = ['#99990077', '#FF000055', '#0000FF55', '#00FF0055', '#999900FF', '#FF0000FF', '#0000FFFF', '#00FF00FF']

pd.concat([model_sim, df], axis=1).plot(color = plot_color)
plt.show()


# Curve look perfect, let's extend the prediction curve

# In[ ]:


import datetime
start_date = df.index[0]
n_days = len(df) + 30
extended_model_x = []
last_row = []

isValid = True
last_death_rate = 0

for t in range(n_days):
    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t), model(*opt_deaths, t), model(*opt_recovered, t)])
   
    #if deaths + recovered > confirmed or deaths rate > 5%, maybe not valid
    if (t > len(df)):
        last_row = extended_model_x[-1]
        if (last_row[2] + last_row[3] > last_row[1]) or (last_row[2] > last_row[1]*0.12):
            if (isValid):
                last_row2 = extended_model_x[-2]
                last_death_rate = last_row2[2]/last_row2[1]
                isValid = False

        if (last_row[2] > last_row[1]*0.05):
            last_row[2] = last_row[1]*last_death_rate
            
        if (last_row[2] + last_row[3] > last_row[1]):
            last_row[2] = last_row[1]*last_death_rate
            last_row[3] = last_row[1]*(1-last_death_rate)

extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)
extended_model_sim.set_index(0, inplace=True)

extended_model_sim.columns = ['Model-Confirmed', 'Model-Deaths', 'Model-Recovered']
extended_model_sim['Model-Active'] = extended_model_sim['Model-Confirmed'] - extended_model_sim['Model-Deaths'] - extended_model_sim['Model-Recovered']
extended_model_sim.loc[extended_model_sim['Model-Active']<0,'Model-Active'] = 0

plot_color = ['#99990077', '#FF000055', '#0000FF55', '#00FF0055', '#999900FF', '#FF0000FF', '#0000FFFF', '#00FF00FF']

pd.concat([extended_model_sim, df], axis=1).plot(color = plot_color)
print('China COVID-19 Prediction')
plt.show()


# let's display predictions for future weeks

# In[ ]:


df.tail()


# In[ ]:


pd.options.display.float_format = '{:20,.0f}'.format
concat_df = pd.concat([df, extended_model_sim], axis=1)
concat_df[concat_df.index.day % 3 == 0]


# Looks like no problem, Let's build the model.
# 

# In[ ]:


def display_fit(df, opt_confirmed, opt_deaths, opt_recovered, ax):
    model_x = []
    
    isValid = True
    last_death_rate = 0
    
    for t in range(len(df)):
        model_x.append([df.index[t], model(*opt_confirmed, t), model(*opt_deaths, t), model(*opt_recovered, t)])
        
        #if deaths + recovered > confirmed or deaths rate > 5%, maybe not valid
        if (t > len(df)):
            last_row = model_x[-1]
            if (last_row[2] + last_row[3] > last_row[1]) or (last_row[2] > last_row[1]*0.05):
                if (isValid):
                    last_row2 = model_x[-2]
                    last_death_rate = last_row2[2]/last_row2[1]
                    isValid = False
                    
            if (last_row[2] > last_row[1]*0.05):
                last_row[2] = last_row[1]*last_death_rate
                
            if (last_row[2] + last_row[3] > last_row[1]):
                last_row[2] = last_row[1]*last_death_rate
                last_row[3] = last_row[1]*(1-last_death_rate)
                
                
    model_sim = pd.DataFrame(model_x, dtype=int)
    model_sim.set_index(0, inplace=True)
    model_sim.columns = ['Model-Confirmed', 'Model-Deaths', 'Model-Recovered']

    model_sim['Model-Active'] = model_sim['Model-Confirmed'] - model_sim['Model-Deaths'] - model_sim['Model-Recovered']
    model_sim.loc[model_sim['Model-Active']<0,'Model-Active'] = 0
    plot_color = ['#99990077', '#FF000055', '#0000FF55', '#00FF0055', '#999900FF', '#FF0000FF', '#0000FFFF', '#00FF00FF']

    return pd.concat([model_sim, df], axis=1).plot(ax=ax, figsize=(14, 10), color = plot_color)

def display_extended_curve(df, opt_confirmed, opt_deaths, opt_recovered, ax):
    start_date = df.index[0]
    n_days = len(df) + 40
    extended_model_x = []
    
    isValid = True
    last_death_rate = 0
    
    for t in range(n_days):
        extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t), model(*opt_deaths, t), model(*opt_recovered, t)])
        
        #if deaths + recovered > confirmed or deaths rate > 5%, maybe not valid
        if (t > len(df)):
            last_row = extended_model_x[-1]
            if (last_row[2] + last_row[3] > last_row[1]) or (last_row[2] > last_row[1]*0.05):
                if (isValid):
                    last_row2 = extended_model_x[-2]
                    last_death_rate = last_row2[2]/last_row2[1]
                    isValid = False
            
            if (last_row[2] > last_row[1]*0.05):
                last_row[2] = last_row[1]*last_death_rate
                    
            if (last_row[2] + last_row[3] > last_row[1]):
                last_row[2] = last_row[1]*last_death_rate
                last_row[3] = last_row[1]*(1-last_death_rate)
                
                
    extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)
    extended_model_sim.set_index(0, inplace=True)
    extended_model_sim.columns = ['Model-Confirmed', 'Model-Deaths', 'Model-Recovered']

    extended_model_sim['Model-Active'] = extended_model_sim['Model-Confirmed'] - extended_model_sim['Model-Deaths'] - extended_model_sim['Model-Recovered']
    
    extended_model_sim.loc[extended_model_sim['Model-Active']<0,'Model-Active'] = 0
    plot_color = ['#99990077', '#FF000055', '#0000FF55', '#00FF0055', '#999900FF', '#FF0000FF', '#0000FFFF', '#00FF00FF']

    return pd.concat([extended_model_sim, df], axis=1).plot(ax=ax, figsize=(14, 10), color = plot_color)


def opt_display_model(df, stats):
    # if the last data point repeats the previous one, or is lower, drop it
    if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:
        df.drop(df.tail(1).index,inplace=True)
    global model_index
    model_index = 0
    opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
    model_index = 1
    opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
    model_index = 2
    opt_recovered = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
    if min(opt_confirmed) > 0:
        stats.append([country, *opt_confirmed, *opt_deaths, *opt_recovered])
        n_plot = len(stats)
        plt.figure(1)
        ax1 = plt.subplot(221)
        display_fit(df, opt_confirmed, opt_deaths, opt_recovered, ax1)
        ax2 = plt.subplot(222)
        display_extended_curve(df, opt_confirmed, opt_deaths, opt_recovered, ax2)
        plt.show()


# **World COVID-19 Prediction**

# 1. * Predict World (With China Data) 

# In[ ]:


stats = []

df = full_table[['Province','Country', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']].groupby('Date').sum()
print('World COVID-19 Prediction (With China Data)')
opt_display_model(df, stats)


# * Predict World (Without China Data) 
# 

# Because china nearly cleared COVID-19, and data is ahead of the world, so maybe exclude china data sounds resonable, the tend looks more worse!
# 

# In[ ]:


stats = []

df = full_table[full_table['Country'] != 'Mainland China'][['Province','Country', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']].groupby('Date').sum()
print('World COVID-19 Prediction(Without China Data)')
opt_display_model(df, stats)


# # Predict by Specific Province

# 

# In[ ]:


stats = []

# Province Specify
for Province in ['Hong Kong', 'Hubei']:
    df = get_time_series_province(Province)
    print('{} COVID-19 Prediction'.format(Province))
    opt_display_model(df, stats)


# # What is going on in Canada

# # Predict Canada

# In[ ]:


stats = []

df = full_table[full_table['Country'] == 'Canada'][['Province','Country', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']].groupby('Date').sum()
print('World COVID-19 Prediction(Canada)')
opt_display_model(df, stats)


# # Predict Canadian Province

# We can see that we probably need more data and more feature engineering for ontario.
# Taking out the cruiseship and London data generates a chart but it is off.
# The predictive model does not line up with the reported data(04/04/2020).
# 

# In[ ]:


stats = []

# Province Specify
for Province in ['British Columbia','Ontario','Quebec']:
    df = get_time_series_province(Province)
    print('{} COVID-19 Prediction'.format(Province))
    opt_display_model(df, stats)


# In[ ]:





# First Cases

# In[ ]:


Canada_df = full_table[full_table['Country'] == 'Canada'].copy()
Ontario_df = Canada_df[Canada_df['Province'] == 'Ontario'].copy()


# tests to see why ontario is  not running in model.
# Some Active showed - 1
#Canada_df.head() 
#Canada_df['Province'].unique()
#Ontario_df.describe()
#Ontario_df[Ontario_df.Active < 0] = 0  move this up to the complete data_frame

#Ontario_df.describe()
#Export the ontario data to check integrity.
Ontario_df.to_csv('Ontario.csv', index=False)



# The Latest Cases (based on available data)

# In[ ]:


Canada_df.tail()


# In[ ]:


confirmed = Canada_df.groupby(['Date', 'Province'])['Confirmed'].sum().reset_index()
provinces = Canada_df['Province'].unique()
provinces


# In[ ]:


# Clean Data
Canada_df = Canada_df.replace(to_replace =["Toronto, ON", "London, ON"],  
                            value ="Ontario") 
Canada_df = Canada_df.replace(to_replace =["Calgary, Alberta", "Edmonton, Alberta"],  
                            value ="Alberta") 
Canada_df =Canada_df.replace(to_replace =[" Montreal, QC"],  
                            value ="Quebec") 
# Here recovered is assumed to be from BC, and Cruise ship data from Ontario
#Canada_df =Canada_df.replace(to_replace =["Recovered"],  
#                            value ="British Columbia") 
Canada_df =Canada_df.replace(to_replace =["Diamond Princess cruise ship"],  
                            value ="Ontario") 


# # Visualization

# In[ ]:


confirmed = Canada_df.groupby('Date').sum()['Confirmed'].reset_index()
deaths = Canada_df.groupby('Date').sum()['Deaths'].reset_index()
recovered = Canada_df.groupby('Date').sum()['Recovered'].reset_index()


# In[ ]:


plot_bar_chart(confirmed, deaths, recovered,'Canada').show()


# In[ ]:


plot_line_chart(confirmed, deaths, recovered,'Canada').show()


# # Across Canada

# In[ ]:


provinces = Canada_df['Province'].unique()
#df = Canada_df
provinces


# In[ ]:


confirmed = Canada_df.groupby(['Date', 'Province'])['Confirmed'].sum().reset_index()


# In[ ]:


fig = go.Figure()
for province in provinces:
 
    fig.add_trace(go.Scatter(
        x=confirmed[confirmed['Province']==province]['Date'],
        y=confirmed[confirmed['Province']==province]['Confirmed'],
        name = province, # Style name/legend entry with html tags
        connectgaps=True # override default to connect the gaps
    ))
fig.update_layout(title="Number of Confirmed COVID-19 Cases Over Time - Canada - By Province" + ' as of ' + ondate)       
fig.show()


# In[ ]:


grouped_country = Canada_df.groupby(["Province"] ,as_index=False)["Confirmed","Recovered","Deaths"].last().sort_values(by="Confirmed",ascending=False)
grouped_country


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Bar(
    
    y=grouped_country['Province'],
    x=grouped_country['Confirmed'],
    orientation='h',
    text=grouped_country['Confirmed']
    ))
fig.update_traces(textposition='outside')
fig.update_layout(title="Cumulative Number of COVID-19 Confirmed Cases - By Province" + ' as of ' + ondate)    
fig.show()


# In[ ]:


fig = go.Figure()

trace1 = go.Bar(
    x=grouped_country['Confirmed'],
    y=grouped_country['Province'],
    orientation='h',
    name='Confirmed'
)
trace2 = go.Bar(
    x=grouped_country['Deaths'],
    y=grouped_country['Province'],
    orientation='h',
    name='Deaths'
)
trace3 = go.Bar(
    x=grouped_country['Recovered'],
    y=grouped_country['Province'],
    orientation='h',
    name='Recovered'
)

data = [trace1, trace2, trace3]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
fig.update_layout(title="Stacked Number of COVID-19 Cases (Confirmed, Deaths, Recoveries) - Canada by Province" + ' as of ' + ondate)    
fig.show()


# # Advanced Data Analysis
# 
# <b>Time Series Analysis</b>
# 
# 

# In[ ]:


ts_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
ts_confirmed.rename(columns={'Country/Region':'Country', 'Province/State':'Province' }, inplace=True)
Canada_C_ts = ts_confirmed[ts_confirmed['Country'] == 'Canada'].copy()
Canada_C_ts


# In[ ]:


ts_diff =Canada_C_ts[Canada_C_ts.columns[4:Canada_C_ts.shape[1]]]
new = ts_diff.diff(axis = 1, periods = 1) 
ynew=list(new.sum(axis=0))


# <b>Epidemic Curve</b>
# 
# An epidemic curve shows the frequency of new cases over time based on the date of onset of disease. This curve is an important plot in epidemiology. The shape of the curve in relation to the incubation period for a particular disease can give clues about the spread and duration of the epidemy.

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    y=ynew,
    x=ts_diff.columns,
    text=list(new.sum(axis=0)),
    ))
fig.update_traces(textposition='outside')
fig.update_layout(title="Epidemic Curve - Daily Number of COVID-19 Confirmed Cases in Canada " + ' as of  ' + ondate,
                 yaxis=dict(title='Number of Cases'))    
fig.show()


# To Do:
# 
# * Clustering of Worldwide Cases
# * Forecasting ....
# * More granular analysis - Age...

# # References
# * The model and modified code snippets and comments are  inherited from these notebooks :
# 
# *  https://www.kaggle.com/yuanquan/covid-19-prediction-by-country-and-province
# *  https://www.kaggle.com/iris2007/covid-19-in-canada
# *  https://www.kaggle.com/alixmartin/covid-19-predictions
# * 
# 
# 
# **Data Sources**
# 
# 1. [Novel Corona Virus 2019 Dataset on Kaggle](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)
# 2. [Novel Coronavirus (COVID-19) Cases, provided by JHU CSSE](https://github.com/CSSEGISandData/COVID-19)
