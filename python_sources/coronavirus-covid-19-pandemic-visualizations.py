#!/usr/bin/env python
# coding: utf-8

# Here are some visualizations for the COVID-19 pandemic, tracking the growth rate for confirmed cases and deaths for various regions.

# In[ ]:


import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go

from scipy.optimize import curve_fit
from sympy.solvers import solve
from sympy import Symbol

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)

from plotly.offline import iplot, init_notebook_mode # Using plotly + cufflinks in offline mode
import cufflinks as cf
cf.go_offline(connected=True)
init_notebook_mode(connected=True)

# cf.getThemes()
# ['ggplot', 'pearl', 'solar', 'space', 'white', 'polar', 'henanigans']
cf.set_config_file(theme='space')


# ## Data Import and Preparation

# In[ ]:


# Data taken from Johns Hopkins CSSE - https://github.com/CSSEGISandData/COVID-19

df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')


# In[ ]:


df_confirmed.head(n=10)


# In[ ]:


df = df_confirmed.copy()
df.rename(columns={'Country/Region':'Country'}, inplace=True)
df = df.groupby(['Country']).sum().reset_index()
df = df.drop(['Lat', 'Long'], axis=1)
df_confirmed = df.copy()

df = df_deaths.copy()
df.rename(columns={'Country/Region':'Country'}, inplace=True)
df = df.groupby(['Country']).sum().reset_index()
df = df.drop(['Lat', 'Long'], axis=1)
df_deaths = df.copy()

df = df_recovered.copy()
df.rename(columns={'Country/Region':'Country'}, inplace=True)
df = df.groupby(['Country']).sum().reset_index()
df = df.drop(['Lat', 'Long'], axis=1)
df_recovered = df.copy()


# In[ ]:


df_confirmed = df_confirmed.melt(id_vars=['Country'],
                                 var_name='Date',
                                 value_name='Confirmed')

df_deaths = df_deaths.melt(id_vars=['Country'],
                           var_name='Date',
                           value_name='Deaths')

df_recovered = df_recovered.melt(id_vars=['Country'],
                                 var_name='Date',
                                 value_name='Recovered')

df_all = df_confirmed.copy()
df_all['Deaths'] = df_deaths['Deaths']
df_all['Recovered'] = df_recovered['Recovered']
df_all['Date'] = pd.to_datetime(df_all['Date'])
# df_all['Country'] = df_all['Country'].replace('Mainland China', 'China')
df_all['Country'] = df_all['Country'].replace('Holy See', 'Vatican City')
df_all['Country'] = df_all['Country'].replace('Korea, South', 'South Korea')

df_all = df_all.reset_index(drop=True)


# In[ ]:


df_all['Country'].unique()


# In[ ]:


df_diamondprincess = df_all[df_all['Country'] == 'Diamond Princess']

df_usa = df_all[(df_all['Country'] == 'US')]

df_china = df_all[df_all['Country'] == 'China']
df_nochina = df_all[df_all['Country'] != 'China']
df_italy = df_all[df_all['Country'] == 'Italy']
df_spain = df_all[df_all['Country'] == 'Spain']
df_uk = df_all[df_all['Country'] == 'United Kingdom']
df_southkorea = df_all[df_all['Country'] == 'South Korea']
df_india = df_all[df_all['Country'] == 'India']
df_iran = df_all[df_all['Country'] == 'Iran']
df_europe = df_all[(df_all['Country'] == 'France') |
                   (df_all['Country'] == 'Germany') |
                   (df_all['Country'] == 'Finland') |
                   (df_all['Country'] == 'Italy') |
                   (df_all['Country'] == 'United Kingdom') |
                   (df_all['Country'] == 'Sweden') |
                   (df_all['Country'] == 'Spain') |
                   (df_all['Country'] == 'Belgium') |
                   (df_all['Country'] == 'Croatia') |
                   (df_all['Country'] == 'Switzerland') |
                   (df_all['Country'] == 'Austria') |
                   (df_all['Country'] == 'Greece') |
                   (df_all['Country'] == 'North Macedonia') |
                   (df_all['Country'] == 'Norway') |
                   (df_all['Country'] == 'Denmark') |
                   (df_all['Country'] == 'Estonia') |
                   (df_all['Country'] == 'Netherlands') |
                   (df_all['Country'] == 'San Marino') |
                   (df_all['Country'] == 'Belarus') |
                   (df_all['Country'] == 'Lithuania') |
                   (df_all['Country'] == 'Ireland') |
                   (df_all['Country'] == 'Luxembourg') |
                   (df_all['Country'] == 'Monaco') |
                   (df_all['Country'] == 'Czech Republic') |
                   (df_all['Country'] == 'Portugal') |
                   (df_all['Country'] == 'Andorra') |
                   (df_all['Country'] == 'Latvia') |
                   (df_all['Country'] == 'Ukraine') |
                   (df_all['Country'] == 'Hungary') |
                   (df_all['Country'] == 'Gibraltar') |
                   (df_all['Country'] == 'Liechtenstein') |
                   (df_all['Country'] == 'Poland') |
                   (df_all['Country'] == 'Bosnia and Herzegovina') |
                   (df_all['Country'] == 'Slovenia') |
                   (df_all['Country'] == 'Serbia') |
                   (df_all['Country'] == 'Slovakia') |
                   (df_all['Country'] == 'Bulgaria') |
                   (df_all['Country'] == 'Malta') |
                   (df_all['Country'] == 'Ireland') |
                   (df_all['Country'] == 'Moldova') |
                   (df_all['Country'] == 'Vatican City')]


# # Table: Case Fatality Rates

# Obviously these aren't meant to represent the true case fatality rate (CFR) of the virus. South Korea's 0.6% (now over 1%) seems like a pretty solid lower bound for now, though, given the amount of testing they've done and that very few of their cases are resolved. Likewise, the CFR of those infected on the Diamond Princess is now above 1%, and while that population skews older, they also received very good medical care.
# 
# cCFR = Deaths / Confirmed

# In[ ]:


print(f'Data current as of {df_all.Date.dt.date.max()}.')


# ## Global - Case Fatality Rate for Confirmed Cases

# In[ ]:


df = df_all.copy()
df = df[df['Date'] == max(df['Date'])].reset_index(drop=True)
df['Region'] = 'World'
df = df.groupby(['Region']).sum()

df2 = df_nochina.copy()
df2 = df2[df2['Date'] == max(df2['Date'])].reset_index(drop=True)
df2['Region'] = 'World (-China)'
df2 = df2.groupby(['Region']).sum()
df = df.append(df2, sort=False)

df2 = df_europe.copy()
df2 = df2[df2['Date'] == max(df2['Date'])].reset_index(drop=True)
df2['Region'] = 'Europe'
df2 = df2.groupby(['Region']).sum()
df = df.append(df2, sort=False)

df2 = df_usa.copy()
df2 = df2[df2['Date'] == max(df2['Date'])].reset_index(drop=True)
df2['Region'] = 'USA'
df2 = df2.groupby(['Region']).sum()
df = df.append(df2, sort=False)

df2 = df_diamondprincess.copy()
df2 = df2[df2['Date'] == max(df2['Date'])].reset_index(drop=True)
df2['Region'] = 'Diamond Princess'
df2 = df2.groupby(['Region']).sum()
df = df.append(df2, sort=False)


df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

df['cCFR [%]'] = df['Deaths'] / df['Confirmed'] * 100
df['cCFR [%]'] = round(df['cCFR [%]'], 2)


df = df.reset_index()
df = df.sort_values('Confirmed', axis=0, ascending=False)

df.style.background_gradient(subset=['Confirmed'], cmap='Blues', axis=None)        .background_gradient(subset=['Deaths'], cmap='Reds', axis=None)        .background_gradient(subset=['Recovered'], cmap='Greens', axis=None)        .background_gradient(subset=['Active'], cmap='Greys', axis=None)        .background_gradient(subset=['cCFR [%]'], cmap='Purples', axis=None)


# ## By Country - Case Fatality Rate for Confirmed Cases

# In[ ]:


df = df_all.copy()
df = df[df['Date'] == max(df['Date'])].reset_index(drop=True)
#df = df.groupby(['Country'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
#df['Country'].replace(to_replace='Others', value='Diamond Princess', inplace=True)
df = df[df.Country != 'Others'].reset_index(drop=True)


df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

df['cCFR [%]'] = df['Deaths'] / df['Confirmed'] * 100
df['cCFR [%]'] = round(df['cCFR [%]'], 2)

df = df.drop('Date', axis=1)

df = df.sort_values('Confirmed', axis=0, ascending=False).reset_index(drop=True)

df.style.background_gradient(subset=['Confirmed'], cmap='Blues', axis=None)        .background_gradient(subset=['Deaths'], cmap='Reds', axis=None)        .background_gradient(subset=['Recovered'], cmap='Greens', axis=None)        .background_gradient(subset=['Active'], cmap='Greys', axis=None)        .background_gradient(subset=['cCFR [%]'], cmap='Purples', axis=None)


# # World (-China)

# In[ ]:


df = df_all.copy()
df = df[df['Date'] == max(df['Date'])].reset_index()
df = df.groupby(['Country'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
fig = px.choropleth(df,
                    locations='Country',
                    locationmode='country names',
                    color='Confirmed',
                    hover_name='Country',
                    hover_data=['Confirmed', 'Deaths', 'Recovered'],
                    color_continuous_scale='Burg')
                    #color_continuous_scale=px.colors.sequential.thermal_r)
fig.show()

print('You can double-click on any country in the legend to see just its plot.')

df = df_nochina.copy()
df = df.groupby(['Date', 'Country'])['Confirmed'].sum().reset_index()
df = df.pivot(index='Date', columns='Country', values='Confirmed').reset_index()
df.iplot(kind='bar', barmode='stack', x='Date', xTitle='Date', yTitle='Confirmed', title='World (-China) - Total Confirmed Cases', colorscale='blues', width=20)


df = df_nochina.copy()
df = df.groupby(['Date', 'Country'])['Deaths'].sum().reset_index()
df = df.pivot(index='Date', columns='Country', values='Deaths').reset_index()
df.iplot(kind='bar', barmode='stack', x='Date', xTitle='Date', yTitle='Deaths', title='World (-China) - Total Deaths', colorscale='reds')


df = df_nochina.copy()
df = df.groupby(['Date', 'Country'])['Recovered'].sum().reset_index()
df = df.pivot(index='Date', columns='Country', values='Recovered').reset_index()
df.iplot(kind='bar', barmode='stack', x='Date', xTitle='Date', yTitle='Recovered', title='World (-China) - Total Recovered', colorscale='greens')


df = df_nochina.copy()
df = df.groupby(['Date', 'Country'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date', 'Country'])['Active'].sum().reset_index()
df = df.pivot(index='Date', columns='Country', values='Active').reset_index()
df.iplot(kind='bar', barmode='stack', x='Date', xTitle='Date', yTitle='Active', title='World (-China) - Number of Active Cases', colorscale='purples')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_nochina.copy()
df = df.groupby(['Date'])['Confirmed'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Confirmed', 'yTitle':'Confirmed', 'title':'World (-China) - Model Fitting, Confirmed Cases', 'kind':'scatter', 'mode':'markers', 'color':'polarbluelight', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_nochina.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_nochina.copy()
df = df.groupby(['Date'])['Deaths'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Deaths'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Deaths', 'yTitle':'Deaths', 'title':'World (-China) - Model Fitting, Deaths', 'kind':'scatter', 'mode':'markers', 'color':'polarred', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_nochina.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Deaths.diff()[-5:] / df.Deaths[-5:])
print(f'Average daily growth rate of deaths over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Deaths double every: {t_d[0]:0.3} days.')


# # USA

# In[ ]:


df = df_usa.groupby(['Date'])['Confirmed'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Confirmed', title='USA - Total Confirmed Cases', color='polarbluelight')

df = df_usa.groupby(['Date'])['Deaths'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Deaths', title='USA - Total Deaths', color='polarred')


df = df_usa.groupby(['Date'])['Recovered'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Recovered', title='USA - Total Recovered', color='polargreen')


df = df_usa.copy()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date'])['Active'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Active', title='USA - Number of Active Cases', color='polarpurple')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_usa.copy()
df = df.groupby(['Date'])['Confirmed'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Confirmed', 'yTitle':'Confirmed', 'title':'USA - Model Fitting, Confirmed Cases', 'kind':'scatter', 'mode':'markers', 'color':'polarbluelight', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_usa.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_usa.copy()
df = df.groupby(['Date'])['Deaths'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Deaths'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Deaths', 'yTitle':'Deaths', 'title':'USA - Model Fitting, Deaths', 'kind':'scatter', 'mode':'markers', 'color':'polarred', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_usa.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Deaths.diff()[-5:] / df.Deaths[-5:])
print(f'Average daily growth rate of deaths over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Deaths double every: {t_d[0]:0.3} days.')


# # Europe

# In[ ]:


print('You can double-click on any country in the legend to see just its plot.')

df = df_europe.copy()
df = df.groupby(['Date', 'Country'])['Confirmed'].sum().reset_index()
df = df.pivot(index='Date', columns='Country', values='Confirmed').reset_index()
df.iplot(kind='bar', barmode='stack', x='Date', xTitle='Date', yTitle='Confirmed', title='Europe - Total Confirmed Cases', colorscale='blues')


df = df_europe.copy()
df = df.groupby(['Date', 'Country'])['Deaths'].sum().reset_index()
df = df.pivot(index='Date', columns='Country', values='Deaths').reset_index()
df.iplot(kind='bar', barmode='stack', x='Date', xTitle='Date', yTitle='Deaths', title='Europe - Total Deaths', colorscale='reds')


df = df_europe.copy()
df = df.groupby(['Date', 'Country'])['Recovered'].sum().reset_index()
df = df.pivot(index='Date', columns='Country', values='Recovered').reset_index()
df.iplot(kind='bar', barmode='stack', x='Date', xTitle='Date', yTitle='Recovered', title='Europe - Total Recovered', colorscale='greens')


df = df_europe.copy()
df = df.groupby(['Date', 'Country'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date', 'Country'])['Active'].sum().reset_index()
df = df.pivot(index='Date', columns='Country', values='Active').reset_index()
df.iplot(kind='bar', barmode='stack', x='Date', xTitle='Date', yTitle='Active', title='Europe - Number of Active Cases', colorscale='purples')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_europe.copy()
df = df.groupby(['Date'])['Confirmed'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Confirmed', 'yTitle':'Confirmed', 'title':'Europe - Model Fitting, Confirmed Cases', 'kind':'scatter', 'mode':'markers', 'color':'polarbluelight', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_europe.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_europe.copy()
df = df.groupby(['Date'])['Deaths'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Deaths'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Deaths', 'yTitle':'Deaths', 'title':'Europe - Model Fitting, Deaths', 'kind':'scatter', 'mode':'markers', 'color':'polarred', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_europe.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Deaths.diff()[-5:] / df.Deaths[-5:])
print(f'Average daily growth rate of deaths over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Deaths double every: {t_d[0]:0.3} days.')


# # Italy

# In[ ]:


df = df_italy.groupby(['Date'])['Confirmed'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Confirmed', title='Italy - Total Confirmed Cases', color='polarbluelight')

df = df_italy.groupby(['Date'])['Deaths'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Deaths', title='Italy - Total Deaths', color='polarred')


df = df_italy.groupby(['Date'])['Recovered'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Recovered', title='Italy - Total Recovered', color='polargreen')


df = df_italy.copy()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date'])['Active'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Active', title='Italy - Number of Active Cases', color='polarpurple')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_italy.copy()
df = df.groupby(['Date'])['Confirmed'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index[:53], df['Confirmed'].values[:53], maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)
df.Model[df['Model'] > 150000] = False

fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Confirmed', 'yTitle':'Confirmed', 'title':'Italy - Model Fitting, Confirmed Cases', 'kind':'scatter', 'mode':'markers', 'color':'polarbluelight', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_italy.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
#print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_italy.copy()
df = df.groupby(['Date'])['Deaths'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Deaths'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Deaths', 'yTitle':'Deaths', 'title':'Italy - Model Fitting, Deaths', 'kind':'scatter', 'mode':'markers', 'color':'polarred', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_usa.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Deaths.diff()[-5:] / df.Deaths[-5:])
print(f'Average daily growth rate of deaths over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Deaths double every: {t_d[0]:0.3} days.')


# # Spain

# In[ ]:


df = df_spain.groupby(['Date'])['Confirmed'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Confirmed', title='Spain - Total Confirmed Cases', color='polarbluelight')

df = df_spain.groupby(['Date'])['Deaths'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Deaths', title='Spain - Total Deaths', color='polarred')


df = df_spain.groupby(['Date'])['Recovered'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Recovered', title='Spain - Total Recovered', color='polargreen')


df = df_spain.copy()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date'])['Active'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Active', title='Spain - Number of Active Cases', color='polarpurple')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_spain.copy()
df = df.groupby(['Date'])['Confirmed'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Confirmed', 'yTitle':'Confirmed', 'title':'Spain - Model Fitting, Confirmed Cases', 'kind':'scatter', 'mode':'markers', 'color':'polarbluelight', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_spain.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_spain.copy()
df = df.groupby(['Date'])['Deaths'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Deaths'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Deaths', 'yTitle':'Deaths', 'title':'Spain - Model Fitting, Deaths', 'kind':'scatter', 'mode':'markers', 'color':'polarred', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_spain.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Deaths.diff()[-5:] / df.Deaths[-5:])
print(f'Average daily growth rate of deaths over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Deaths double every: {t_d[0]:0.3} days.')


# # United Kingdom

# In[ ]:


df = df_uk.groupby(['Date'])['Confirmed'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Confirmed', title='UK - Total Confirmed Cases', color='polarbluelight')

df = df_uk.groupby(['Date'])['Deaths'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Deaths', title='UK - Total Deaths', color='polarred')


df = df_uk.groupby(['Date'])['Recovered'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Recovered', title='UK - Total Recovered', color='polargreen')


df = df_uk.copy()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date'])['Active'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Active', title='UK - Number of Active Cases', color='polarpurple')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_uk.copy()
df = df.groupby(['Date'])['Confirmed'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Confirmed', 'yTitle':'Confirmed', 'title':'UK - Model Fitting, Confirmed Cases', 'kind':'scatter', 'mode':'markers', 'color':'polarbluelight', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_uk.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_uk.copy()
df = df.groupby(['Date'])['Deaths'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Deaths'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Deaths', 'yTitle':'Deaths', 'title':'UK - Model Fitting, Deaths', 'kind':'scatter', 'mode':'markers', 'color':'polarred', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_uk.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Deaths.diff()[-5:] / df.Deaths[-5:])
print(f'Average daily growth rate of deaths over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Deaths double every: {t_d[0]:0.3} days.')


# # Iran

# In[ ]:


df = df_iran.groupby(['Date'])['Confirmed'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Confirmed', title='Iran - Total Confirmed Cases', color='polarbluelight')


df = df_iran.groupby(['Date'])['Deaths'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Deaths', title='Iran - Total Deaths', color='polarred')


df = df_iran.groupby(['Date'])['Recovered'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Recovered', title='Iran - Total Recovered', color='polargreen')


df = df_iran.copy()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date'])['Active'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Active', title='Iran - Number of Active Cases', color='polarpurple')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_iran.copy()
df = df.groupby(['Date'])['Confirmed'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index[:49], df['Confirmed'].values[:49], maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)
df.Model[df['Model'] > 50000] = False

fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Confirmed', 'yTitle':'Confirmed', 'title':'Iran - Model Fitting, Confirmed Cases', 'kind':'scatter', 'mode':'markers', 'color':'polarbluelight', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_iran.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
#t_d
#print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# # South Korea

# In[ ]:


df = df_southkorea.groupby(['Date'])['Confirmed'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Confirmed', title='South Korea - Total Confirmed Cases', color='polarbluelight')


df = df_southkorea.groupby(['Date'])['Deaths'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Deaths', title='South Korea - Total Deaths', color='polarred')


df = df_southkorea.groupby(['Date'])['Recovered'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Recovered', title='South Korea - Total Recovered', color='polargreen')


df = df_southkorea.copy()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date'])['Active'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Active', title='South Korea - Number of Active Cases', color='polarpurple')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_southkorea.copy()
df = df.groupby(['Date'])['Confirmed'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index[:43], df['Confirmed'].values[:43], maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)
df.Model[df['Model'] > 25000] = False

fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Confirmed', 'yTitle':'Confirmed', 'title':'South Korea - Model Fitting, Confirmed Cases', 'kind':'scatter', 'mode':'markers', 'color':'polarbluelight', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_southkorea.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

# x = Symbol('x')
# t_d = solve((1 + growth)**x - 2 , x)
# print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# # India

# In[ ]:


df = df_india.groupby(['Date'])['Confirmed'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Confirmed', title='India - Total Confirmed Cases', color='polarbluelight')


df = df_india.groupby(['Date'])['Deaths'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Deaths', title='India - Total Deaths', color='polarred')


df = df_india.groupby(['Date'])['Recovered'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Recovered', title='India - Total Recovered', color='polargreen')


df = df_india.copy()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date'])['Active'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Active', title='India - Number of Active Cases', color='polarpurple')


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_india.copy()
df = df.groupby(['Date'])['Confirmed'].sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)


fig=cf.tools.figures(df,[{'x':'Date', 'y':'Model', 'kind':'lines', 'color':'ghostwhite', 'dash':'dash'},
                         {'x':'Date', 'y':'Confirmed', 'yTitle':'Confirmed', 'title':'India - Model Fitting, Confirmed Cases', 'kind':'scatter', 'mode':'markers', 'color':'polarbluelight', 'size':7}])
cf.iplot(fig)

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_india.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# # China

# In[ ]:


df = df_china.groupby(['Date'])['Confirmed'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Confirmed', title='China - Total Confirmed Cases', color='polarbluelight')


df = df_china.groupby(['Date'])['Deaths'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Deaths', title='China - Total Deaths', color='polarred')


df = df_china.groupby(['Date'])['Recovered'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Recovered', title='China - Total Recovered', color='polargreen')


df = df_china.copy()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date'])['Active'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Active', title='China - Number of Active Cases', color='polarpurple')


# In[ ]:


df = df_china.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

# x = Symbol('x')
# t_d = solve((1 + growth)**x - 2 , x)
# print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# # World

# In[ ]:


df = df_all.groupby(['Date'])['Confirmed'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Confirmed', title='World - Total Confirmed Cases', color='polarbluelight')


df = df_all.groupby(['Date'])['Deaths'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Deaths', title='World - Total Deaths', color='polarred')


df = df_all.groupby(['Date'])['Recovered'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Recovered', title='World - Total Recovered', color='polargreen')


df = df_all.copy()
df['Active'] = df.Confirmed - df.Deaths - df.Recovered
df = df.groupby(['Date'])['Active'].sum().reset_index()
df.iplot(kind='bar', x='Date', xTitle='Date', yTitle='Active', title='World - Number of Active Cases', color='polarpurple')


# In[ ]:


df = df_all.copy()

df = df[df['Date'] == max(df['Date'])].reset_index()
df = df.groupby(['Country'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

fig = px.treemap(df,
                 path=['Country'],
                 values=df['Active'],
                 title='World - Active Cases per Country',
                 hover_data=['Confirmed', 'Deaths', 'Recovered', 'Active'],
                 color_discrete_sequence=px.colors.qualitative.Prism)
fig.show()


# In[ ]:


df = df_all.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

# x = Symbol('x')
# t_d = solve((1 + growth)**x - 2 , x)
# print(f'Confirmed cases double every: {t_d[0]:0.3} days.')

