#!/usr/bin/env python
# coding: utf-8

# I figured I'd put together some visualizations for the COVID-19 pandemic, specifically for the things I wanted to track (especially growth rate and active cases per region).

# In[ ]:


import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go

from scipy.optimize import curve_fit
from sympy.solvers import solve
from sympy import Symbol


pd.set_option('display.max_rows', None)


# ## Data Import and Preparation

# In[ ]:


# Data taken from Johns Hopkins CSSE - https://github.com/CSSEGISandData/COVID-19
#

df_confirmed = pd.read_csv('/kaggle/input/covid19-csea/time_series_covid_19_confirmed.csv')
df_deaths = pd.read_csv('/kaggle/input/covid19-csea/time_series_covid_19_deaths.csv')
df_recovered = pd.read_csv('/kaggle/input/covid19-csea/time_series_covid_19_recovered.csv')


# In[ ]:


df_confirmed.head(n=10)


# In[ ]:


df_confirmed[df_confirmed['Country/Region'] == 'Italy']


# In[ ]:


# Johns Hopkins changed how they're sorting the data sooo gotta do all this stuff to try to fix it
# No longer necessary as of 3/12/2020. But I'll leave it since it won't affect anything, and in case they change the data structure again


def fix_USA(df_orig):
    us_state_abbrev = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'District of Columbia': 'D.C.',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Northern Mariana Islands':'MP',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Palau': 'PW',
        'Pennsylvania': 'PA',
        'Puerto Rico': 'PR',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virgin Islands': 'VI',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY',
    }


    df_noUS = df_orig[df_orig['Country/Region'] != 'US']

    df = df_orig[df_orig['Country/Region'] == 'US']
    df = df[(df['Country/Region'] == 'US') & (df['Province/State'].str.contains(","))]
    df = df.fillna(0)
    df['3/10/20'] = 0
    df['3/11/20'] = 0

    df2 = df_orig[df_orig['Country/Region'] == 'US']
    df2 = df2[(df2['Country/Region'] == 'US') & (~df2['Province/State'].str.contains(","))]
    df2 = df2[(df2['Province/State'] != 'Diamond Princess') & (df2['Province/State'] != 'Grand Princess')]



    df2['Province/State'] = df2['Province/State'].map(us_state_abbrev)
    df2 = df2.groupby(['Province/State', 'Country/Region']).sum().reset_index()

    df['Province/State'] = df['Province/State'].str[-2:]
    df.replace(to_replace='R ', value='OR', inplace=True) # One of the counties is formatted incorrectly
    df.replace(to_replace='C.', value='D.C.', inplace=True)
    df = df.groupby(['Province/State', 'Country/Region']).sum().reset_index()

    df3 = df.append(df2)
    df3 = df3.groupby(['Province/State', 'Country/Region']).sum().reset_index()


    df_orig = pd.concat([df_noUS, df3], sort=False)
    df_orig = df_orig.fillna(method='ffill', axis=1)
    #df_orig = df_orig.fillna(0)

    return df_orig



#df_confirmed[df_confirmed['Country/Region'] == 'Republic of Korea'].fillna(0, inplace=True)
df_confirmed.loc[df_confirmed['Country/Region'] == 'Republic of Korea', '3/11/20'] = 0
df_confirmed.loc[df_confirmed['Country/Region'] == 'Iran (Islamic Republic of)', '3/11/20'] = 0
df_confirmed.loc[df_confirmed['Country/Region'] == 'Mainland China', '3/11/20'] = 0
df_confirmed.loc[(df_confirmed['Country/Region'] == 'France') & (df_confirmed['Province/State'].isnull()), '3/11/20'] = 0



df_deaths.loc[df_deaths['Country/Region'] == 'Republic of Korea', '3/11/20'] = 0
df_deaths.loc[df_deaths['Country/Region'] == 'Iran (Islamic Republic of)', '3/11/20'] = 0
df_deaths.loc[df_deaths['Country/Region'] == 'Mainland China', '3/11/20'] = 0
df_deaths.loc[(df_deaths['Country/Region'] == 'France') & (df_deaths['Province/State'].isnull()), '3/11/20'] = 0


df_recovered.loc[df_recovered['Country/Region'] == 'Republic of Korea', '3/11/20'] = 0
df_recovered.loc[df_recovered['Country/Region'] == 'Iran (Islamic Republic of)', '3/11/20'] = 0
df_recovered.loc[df_recovered['Country/Region'] == 'Mainland China', '3/11/20'] = 0
df_recovered.loc[(df_recovered['Country/Region'] == 'France') & (df_recovered['Province/State'].isnull()), '3/11/20'] = 0


df_confirmed.loc[df_confirmed['Country/Region'] == 'Italy', '3/12/20'] = 15113
df_recovered.loc[df_recovered['Country/Region'] == 'Italy', '3/12/20'] = 1258
df_deaths.loc[df_deaths['Country/Region'] == 'Italy', '3/12/20'] = 1016
df_confirmed.loc[df_confirmed['Province/State'] == 'United Kingdom', '3/12/20'] = 590
df_deaths.loc[df_deaths['Province/State'] == 'United Kingdom', '3/12/20'] = 10
df_recovered.loc[df_recovered['Province/State'] == 'United Kingdom', '3/12/20'] = 18



df_confirmed = fix_USA(df_confirmed)
df_deaths = fix_USA(df_deaths)
df_recovered = fix_USA(df_recovered)


# In[ ]:


df_confirmed = df_confirmed.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                                 var_name='Date',
                                 value_name='Confirmed')

df_deaths = df_deaths.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                           var_name='Date',
                           value_name='Deaths')

df_recovered = df_recovered.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                                 var_name='Date',
                                 value_name='Recovered')

df_all = df_confirmed.copy()
df_all['Deaths'] = df_deaths['Deaths']
df_all['Recovered'] = df_recovered['Recovered']
df_all.rename(columns={'Country/Region':'Country'}, inplace=True)
df_all['Date'] = pd.to_datetime(df_all['Date'])
df_all['Country'] = df_all['Country'].replace('Mainland China', 'China')
df_all['Country'] = df_all['Country'].replace('Iran (Islamic Republic of)', 'Iran')
df_all['Country'] = df_all['Country'].replace('Republic of Korea', 'South Korea')
df_all['Country'] = df_all['Country'].replace('Republic of Moldova', 'Moldova')
df_all['Country'] = df_all['Country'].replace('Holy See', 'Vatican City')

df_all = df_all[df_all['Country'] != 'occupied Palestinian territory']

df_all['Country'] = df_all['Country'].replace('Korea, South', 'South Korea')
#df_all.groupby('South Korea').sum()


# In[ ]:


df_all.head(n=10)


# In[ ]:


print(f'Data current as of {df_all.Date.dt.date.max()}.')


# ### Let's separate the data into different regions

# In[ ]:


df_all['Country'].unique()


# In[ ]:


df_all['Province/State'].unique()


# In[ ]:


df_all[df_all['Country'] == 'US']['Province/State'].unique()


# In[ ]:


df_diamondprincess = df_all[df_all['Province/State'] == 'Diamond Princess']

df_all = df_all[(df_all['Province/State'] != 'Omaha, NE (From Diamond Princess)') &
                (df_all['Province/State'] != 'Grand Princess') &
                (df_all['Province/State'] != 'Diamond Princess') &
                (df_all['Province/State'] != 'From Diamond Princess') &
                (df_all['Province/State'] != 'Travis, CA (From Diamond Princess)') &
                (df_all['Province/State'] != 'Lackland, TX (From Diamond Princess)') &
                (df_all['Province/State'] != 'Grand Princess Cruise Ship') &       
                (df_all['Province/State'] != 'Unassigned Location (From Diamond Princess)')]
#df_all = df_all[(df_all['Country'] != 'Others')]

# df_usa2 = (df_all[(df_all['Country'] == 'US') &
#             (df_all['Province/State'].str.contains(",") == True)].index, axis=0, inplace=True)
# df_all.drop(df_all[(df_all['Country'] == 'US') &
#                    (df_all['Province/State'].str.contains(",") == False)].index, axis=0, inplace=True)

#df_usa = df_all[(df_all['Country'] == 'US')]
# df_usa = df_usa[~df_usa['Province/State'].str.contains(",")]
# df_usa = df_usa[df_usa['Province/State'].str.contains(",")]
#df_all = df_all[(df_all['Country'] != 'Others')]


df_usa = df_all[(df_all['Country'] == 'US')]
# df_usa = df_usa[~df_usa['Province/State'].str.contains(",")]



df_china = df_all[df_all['Country'] == 'China']
df_nochina = df_all[df_all['Country'] != 'China']
df_italy = df_all[df_all['Country'] == 'Italy']
df_southkorea = df_all[df_all['Country'] == 'South Korea']
df_india = df_all[df_all['Country'] == 'India']
df_iran = df_all[df_all['Country'] == 'Iran']
df_europe = df_all[(df_all['Country'] == 'France') |
                   (df_all['Country'] == 'Germany') |
                   (df_all['Country'] == 'Finland') |
                   (df_all['Country'] == 'Italy') |
                   (df_all['Country'] == 'UK') |
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
                   (df_all['Country'] == 'Republic of Ireland') |
                   (df_all['Country'] == 'Moldova') |
                   (df_all['Country'] == 'Vatican City')]


# # Table: Case Fatality Rates

# ## Case Fatality Rate for Confirmed Cases

# Obviously these aren't meant to represent the true CFR. South Korea's 0.6% seems like a pretty solid lower bound for now, though, given the amount of testing they've done and that very few of their cases are resolved.
# 
# cCFR = Deaths / Confirmed

# In[ ]:


df = df_all.copy()
df = df[df['Date'] == max(df['Date'])].reset_index()
df = df.groupby(['Country'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
df = df.sort_values('Confirmed', axis=0, ascending=False).reset_index(drop=True)
#df['Country'].replace(to_replace='Others', value='Diamond Princess', inplace=True)
df = df[df.Country != 'Others']

df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

df['cCFR [%]'] = df['Deaths'] / df['Confirmed'] * 100
df['cCFR [%]'] = round(df['cCFR [%]'], 2)

# df['cCRR [%]'] = df['Recovered'] / df['Confirmed'] * 100
# df['cCRR [%]'] = round(df['cCRR [%]'], 2)

# print(f'cCFR = Deaths / Confirmed')
#print(f'cCRR = Recovered / Confirmed')

#df.head(n=20)
df.style.background_gradient(subset=['Confirmed'], cmap='Blues', axis=None)        .background_gradient(subset=['Deaths'], cmap='Reds', axis=None)        .background_gradient(subset=['Recovered'], cmap='Greens', axis=None)        .background_gradient(subset=['Active'], cmap='Greys', axis=None)        .background_gradient(subset=['cCFR [%]'], cmap='PuRd', axis=None)


# ## Global - Case Fatality Rate for Confirmed Cases

# In[ ]:


df = df_all.copy()
df = df[df['Date'] == max(df['Date'])].reset_index()
df = df.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered'].sum()
df['cCFR [%]'] = df['Deaths'] / df['Confirmed'] * 100
df['cCFR [%]'] = round(df['cCFR [%]'], 2)
df.style.background_gradient(subset=['Confirmed'], cmap='Blues', axis=None)        .background_gradient(subset=['Deaths'], cmap='Reds', axis=None)        .background_gradient(subset=['Recovered'], cmap='Greens', axis=None)        .background_gradient(subset=['cCFR [%]'], cmap='PuRd', axis=None)


# ## Diamond Princess Cruise Ship - Case Fatality Rate for Confirmed Cases

# In[ ]:


df = df_diamondprincess.copy()
df = df[df['Date'] == max(df['Date'])].reset_index()
df = df.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered'].sum()
df['cCFR [%]'] = df['Deaths'] / df['Confirmed'] * 100
df['cCFR [%]'] = round(df['cCFR [%]'], 2)
df.style.background_gradient(subset=['Confirmed'], cmap='Blues', axis=None)        .background_gradient(subset=['Deaths'], cmap='Reds', axis=None)        .background_gradient(subset=['Recovered'], cmap='Greens', axis=None)        .background_gradient(subset=['cCFR [%]'], cmap='PuRd', axis=None)


# # World (-China)

# In[ ]:


# fig = px.bar(df_nochina.groupby(['Date']).sum().reset_index(), x='Date', y='Confirmed', title='World (-China) - Total Confirmed Cases')
# fig.show()

df = df_nochina.copy()
df = df.groupby(['Date', 'Country']).sum().reset_index()
df = df.replace(1603, 892) # accounting for how John's Hopkins doubled up the counts for 2020-03-10 when they switched to State reporting for the US
fig = px.bar(df, x='Date', y='Confirmed', title='World (-China) - Total Confirmed Cases', color='Country', color_discrete_sequence=px.colors.sequential.Plasma_r)
fig.show()
print('Note that you can double-click on one of the countries in the legend to only see it on the chart.')

fig = px.bar(df_nochina.groupby(['Date']).sum().reset_index(), x='Date', y='Deaths', title='World (-China) - Total Deaths', color_discrete_sequence=px.colors.diverging.Picnic_r)
fig.update_layout(showlegend=False)
fig.show()

# df = df_nochina.copy()
# df = df.groupby(['Date', 'Country']).sum().reset_index()
# fig = px.bar(df, x='Date', y='Deaths', title='World (-China) - Total Deaths', color='Country', color_discrete_sequence=px.colors.sequential.RdPu_r)
# fig.show()

fig = px.bar(df_nochina.groupby(['Date']).sum().reset_index(), x='Date', y='Recovered', title='World (-China) - Total Recovered', color_discrete_sequence=px.colors.diverging.Tropic)
fig.show()

df = df_nochina.groupby(['Date']).sum().reset_index()
fig = px.bar(df, x='Date', y=(df.Confirmed - df.Deaths - df.Recovered), title='World (-China) - Number of Active Cases', color_discrete_sequence=px.colors.sequential.thermal)
fig.update_layout(yaxis_title='Active Cases')
fig.show()


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_nochina.copy()
df = df.groupby(['Date']).sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)

trace1 = go.Scatter(x=df['Date'], y=df['Confirmed'], mode='markers', name='Confirmed')
trace2 = go.Scatter(x=df['Date'], y=df['Model'], mode='lines', line={'dash':'dash', 'color':'grey'}, name='Fit')
data = [trace1, trace2]

fig = go.Figure(data=data)
fig.update_layout(title="World (-China) - Model Fitting", yaxis_title="Confirmed")
fig.show()


# residuals = df['Confirmed'] - func(df['Date'].index, *popt)
# ss_res = np.sum(residuals**2)
# ss_tot = np.sum((df['Confirmed']-np.mean(df['Confirmed']))**2)
# r_squared = 1 - (ss_res / ss_tot)
# print(f'R^2 = {r_squared:.3}')

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_nochina.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# # USA

# In[ ]:


df = df_usa.copy()
# df['Province/State'] = df['Province/State'].str[-2:]
# df.replace(to_replace='R ', value='OR', inplace=True) # One of the counties is formatted incorrectly
# df.replace(to_replace='C.', value='D.C.', inplace=True)
df = df[df['Date'] == max(df['Date'])].reset_index()
df = df.groupby(['Province/State'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

fig = px.choropleth(df,
                    locations='Province/State',
                    locationmode='USA-states',
                    color='Confirmed',
                    hover_name='Province/State',
                    hover_data=['Confirmed', 'Deaths', 'Recovered'],
                    scope='usa',
                    title='USA - Total Confirmed Cases',
                    color_continuous_scale='Burg')
fig.show()

df = df_usa.copy()
# df['Province/State'] = df['Province/State'].str[-2:]
# df.replace(to_replace='R ', value='OR', inplace=True) # One of the counties is formatted incorrectly
# df.replace(to_replace='C.', value='D.C.', inplace=True)
df = df.groupby(['Date', 'Province/State']).sum().reset_index()
fig = px.bar(df, x='Date', y='Confirmed', title='USA - Total Confirmed Cases', color='Province/State', color_discrete_sequence=px.colors.sequential.Plasma)
fig.show()

fig = px.bar(df, x='Date', y='Deaths', title='USA - Total Deaths', color='Province/State', color_discrete_sequence=px.colors.sequential.Reds_r)
#fig.update_layout(showlegend=False)
fig.show()

fig = px.bar(df.groupby(['Date']).sum().reset_index(), x='Date', y='Recovered', title='USA - Total Recovered', color_discrete_sequence=px.colors.diverging.Tropic)
fig.show()

df = df.groupby(['Date']).sum().reset_index()
fig = px.bar(df, x='Date', y=(df.Confirmed - df.Deaths - df.Recovered), title='USA - Number of Active Cases', color_discrete_sequence=px.colors.sequential.thermal)
fig.update_layout(yaxis_title='Active Cases')
fig.show()

df = df_usa.copy()
df = df[df['Date'] == max(df['Date'])].reset_index()
df = df.groupby(['Province/State'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
fig = px.treemap(df,
                 path=['Province/State'],
                 values=df['Active'],
                 title='USA - Active Cases per State',
                 hover_data=['Confirmed', 'Deaths', 'Recovered', 'Active'],
                 color_discrete_sequence=px.colors.qualitative.Prism)
fig.show()


# Note that you can double-click on one of the states in the legend to only see it on the chart.

# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_usa.copy()
df = df.groupby(['Date']).sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)

trace1 = go.Scatter(x=df['Date'], y=df['Confirmed'], mode='markers', name='Confirmed')
trace2 = go.Scatter(x=df['Date'], y=df['Model'], mode='lines', line={'dash':'dash', 'color':'grey'}, name='Fit')
data = [trace1, trace2]

fig = go.Figure(data=data)
fig.update_layout(title="USA - Model Fitting", yaxis_title="Confirmed")
fig.show()

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


df = df_usa.copy()
# df['Province/State'] = df['Province/State'].str[-2:]
# df.replace(to_replace='R ', value='OR', inplace=True) # One of the counties is formatted incorrectly
# df.replace(to_replace='C.', value='D.C.', inplace=True)
df = df[df['Date'] == max(df['Date'])].reset_index()
df = df.groupby(['Province/State'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
df = df.sort_values('Confirmed', axis=0, ascending=False).reset_index(drop=True)
#df['Country'].replace(to_replace='Others', value='Diamond Princess', inplace=True)
#df = df[df.Country != 'Others']

df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

df['cCFR [%]'] = df['Deaths'] / df['Confirmed'] * 100
df['cCFR [%]'] = round(df['cCFR [%]'], 2)

#print(f'cCFR = Deaths / Confirmed')

#df.head(n=20)
df.style.background_gradient(subset=['Confirmed'], cmap='Blues', axis=None)        .background_gradient(subset=['Deaths'], cmap='Reds', axis=None)        .background_gradient(subset=['Recovered'], cmap='Greens', axis=None)        .background_gradient(subset=['Active'], cmap='Greys', axis=None)        .background_gradient(subset=['cCFR [%]'], cmap='PuRd', axis=None)


# # Italy

# In[ ]:


fig = px.bar(df_italy.groupby(['Date']).sum().reset_index(), x='Date', y='Confirmed', title='Italy - Total Confirmed Cases')
fig.show()

fig = px.bar(df_italy.groupby(['Date']).sum().reset_index(), x='Date', y='Deaths', title='Italy - Total Deaths', color_discrete_sequence=px.colors.diverging.Picnic_r)
fig.show()

fig = px.bar(df_italy.groupby(['Date']).sum().reset_index(), x='Date', y='Recovered', title='Italy - Total Recovered', color_discrete_sequence=px.colors.diverging.Tropic)
fig.show()

fig = px.bar(df_italy, x='Date', y=(df_italy.Confirmed - df_italy.Deaths - df_italy.Recovered), title='Italy - Number of Active Cases', color_discrete_sequence=px.colors.sequential.thermal)
fig.update_layout(yaxis_title='Active Cases')
fig.show()


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_italy.copy()
df = df.groupby(['Date']).sum().reset_index()
# df = df.append({'Date':datetime.datetime(2020, 3, 11), 'Confirmed':np.int64(12462)}, ignore_index=True).reset_index()


popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)

trace1 = go.Scatter(x=df['Date'], y=df['Confirmed'], mode='markers', name='Confirmed')
trace2 = go.Scatter(x=df['Date'], y=df['Model'], mode='lines', line={'dash':'dash', 'color':'grey'}, name='Fit')
data = [trace1, trace2]

fig = go.Figure(data=data)
fig.update_layout(title="Italy - Model Fitting", yaxis_title="Confirmed")
fig.show()

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_italy.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# # Iran

# In[ ]:


fig = px.bar(df_iran.groupby(['Date']).sum().reset_index(), x='Date', y='Confirmed', title='Iran - Total Confirmed Cases')
fig.show()

fig = px.bar(df_iran.groupby(['Date']).sum().reset_index(), x='Date', y='Deaths', title='Iran - Total Deaths', color_discrete_sequence=px.colors.diverging.Picnic_r)
fig.show()

fig = px.bar(df_iran.groupby(['Date']).sum().reset_index(), x='Date', y='Recovered', title='Iran - Total Recovered', color_discrete_sequence=px.colors.diverging.Tropic)
fig.show()

fig = px.bar(df_iran, x='Date', y=(df_iran.Confirmed - df_iran.Deaths - df_iran.Recovered), title='Iran - Number of Active Cases', color_discrete_sequence=px.colors.sequential.thermal)
fig.update_layout(yaxis_title='Active Cases')
fig.show()


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]

df = df_iran.copy()
df = df.groupby(['Date']).sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)

trace1 = go.Scatter(x=df['Date'], y=df['Confirmed'], mode='markers', name='Confirmed')
trace2 = go.Scatter(x=df['Date'], y=df['Model'], mode='lines', line={'dash':'dash', 'color':'grey'}, name='Fit')
data = [trace1, trace2]

fig = go.Figure(data=data)
fig.update_layout(title="Iran - Model Fitting", yaxis_title="Confirmed")
fig.show()

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_iran.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
#t_d
print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# # South Korea

# In[ ]:


fig = px.bar(df_southkorea.groupby(['Date']).sum().reset_index(), x='Date', y='Confirmed', title='South Korea - Total Confirmed Cases')
fig.show()

fig = px.bar(df_southkorea.groupby(['Date']).sum().reset_index(), x='Date', y='Deaths', title='South Korea - Total Deaths', color_discrete_sequence=px.colors.diverging.Picnic_r)
fig.show()

fig = px.bar(df_southkorea.groupby(['Date']).sum().reset_index(), x='Date', y='Recovered', title='South Korea - Total Recovered', color_discrete_sequence=px.colors.diverging.Tropic)
fig.show()

fig = px.bar(df_southkorea, x='Date', y=(df_southkorea.Confirmed - df_southkorea.Deaths - df_southkorea.Recovered), title='South Korea - Number of Active Cases', color_discrete_sequence=px.colors.sequential.thermal)
fig.update_layout(yaxis_title='Active Cases')
fig.show()


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_southkorea.copy()
df = df.groupby(['Date']).sum().reset_index()
# df = df.append({'Date':datetime.datetime(2020, 3, 11), 'Confirmed':np.int64(7755)}, ignore_index=True).reset_index()

popt, pconv = curve_fit(func, df['Date'].index[:43], df['Confirmed'].values[:43], maxfev=1000)
# print(popt)

df['Model'] = func(df['Date'].index, *popt)

trace1 = go.Scatter(x=df['Date'], y=df['Confirmed'], mode='markers', name='Confirmed')
trace2 = go.Scatter(x=df['Date'], y=df['Model'], mode='lines', line={'dash':'dash', 'color':'grey'}, name='Fit')
data = [trace1, trace2]

fig = go.Figure(data=data)
fig.update_layout(title="South Korea - Model Fitting", yaxis_title="Confirmed")
fig.show()

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')
print(f'Shows how much South Korea has decreased their rate of infection growth.')


# In[ ]:


df = df_southkorea.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

# x = Symbol('x')
# t_d = solve((1 + growth)**x - 2 , x)
# print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# # Europe

# In[ ]:


fig = px.bar(df_europe.groupby(['Date', 'Country']).sum().reset_index(), x='Date', y='Confirmed', title='Europe - Total Confirmed Cases', color='Country', color_discrete_sequence=px.colors.sequential.Plasma_r)
fig.show()


fig = px.bar(df_europe.groupby(['Date', 'Country']).sum().reset_index(), x='Date', y='Deaths', title='Europe - Total Deaths', color='Country', color_discrete_sequence=px.colors.sequential.Reds)
fig.show()

fig = px.bar(df_europe.groupby(['Date', 'Country']).sum().reset_index(), x='Date', y='Recovered', title='Europe - Total Recovered', color='Country', color_discrete_sequence=px.colors.sequential.Viridis)
fig.show()

fig = px.bar(df_europe, x='Date', y=(df_europe.Confirmed - df_europe.Deaths - df_europe.Recovered), title='Europe - Number of Active Cases', color='Country', color_discrete_sequence=px.colors.sequential.thermal)
fig.update_layout(yaxis_title='Active Cases')
fig.show()


# Note that you can double-click on one of the countries in the legend to only see it on the chart.

# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_europe.copy()
df = df.groupby(['Date']).sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)

trace1 = go.Scatter(x=df['Date'], y=df['Confirmed'], mode='markers', name='Confirmed')
trace2 = go.Scatter(x=df['Date'], y=df['Model'], mode='lines', line={'dash':'dash', 'color':'grey'}, name='Fit')
data = [trace1, trace2]

fig = go.Figure(data=data)
fig.update_layout(title="Europe - Model Fitting", yaxis_title="Confirmed")
fig.show()

print(f'Daily growth rate (model): +{(popt[1] - 1) * 100 : 0.3}%')


# In[ ]:


df = df_europe.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

x = Symbol('x')
t_d = solve((1 + growth)**x - 2 , x)
print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# # India

# In[ ]:


fig = px.bar(df_india.groupby(['Date']).sum().reset_index(), x='Date', y='Confirmed', title='India - Total Confirmed Cases')
fig.show()

fig = px.bar(df_india.groupby(['Date']).sum().reset_index(), x='Date', y='Deaths', title='India - Total Deaths', color_discrete_sequence=px.colors.diverging.Picnic_r)
fig.show()

fig = px.bar(df_india.groupby(['Date']).sum().reset_index(), x='Date', y='Recovered', title='India - Total Recovered', color_discrete_sequence=px.colors.diverging.Tropic)
fig.show()

fig = px.bar(df_india, x='Date', y=(df_india.Confirmed - df_india.Deaths - df_india.Recovered), title='India - Number of Active Cases', color_discrete_sequence=px.colors.sequential.thermal)
fig.update_layout(yaxis_title='Active Cases')
fig.show()


# In[ ]:


def func(t, a, b):
    return a*b**(t)
guess = [1, 20]


df = df_india.copy()
df = df.groupby(['Date']).sum().reset_index()

popt, pconv = curve_fit(func, df['Date'].index, df['Confirmed'].values, maxfev=1000)
#print(popt)

df['Model'] = func(df['Date'].index, *popt)

trace1 = go.Scatter(x=df['Date'], y=df['Confirmed'], mode='markers', name='Confirmed')
trace2 = go.Scatter(x=df['Date'], y=df['Model'], mode='lines', line={'dash':'dash', 'color':'grey'}, name='Fit')
data = [trace1, trace2]

fig = go.Figure(data=data)
fig.update_layout(title="India - Model Fitting", yaxis_title="Confirmed")
fig.show()

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


fig = px.bar(df_china.groupby(['Date']).sum().reset_index(), x='Date', y='Confirmed', title='China - Total Confirmed Cases')
fig.show()

fig = px.bar(df_china.groupby(['Date']).sum().reset_index(), x='Date', y='Deaths', title='China - Total Deaths', color_discrete_sequence=px.colors.diverging.Picnic_r)
fig.update_layout(showlegend=False)
fig.show()

fig = px.bar(df_china.groupby(['Date']).sum().reset_index(), x='Date', y='Recovered', title='China - Total Recovered', color_discrete_sequence=px.colors.diverging.Tropic)
fig.show()

df = df_china.groupby(['Date']).sum().reset_index()
fig = px.bar(df, x='Date', y=(df.Confirmed - df.Deaths - df.Recovered), title='China - Number of Active Cases', color_discrete_sequence=px.colors.sequential.thermal)
fig.update_layout(yaxis_title='Active Cases')
fig.show()


# #### Numbers declining in China... for now!

# In[ ]:


df = df_china.copy()
df = df.groupby(['Date']).sum().reset_index()
growth = np.mean(df.Confirmed.diff()[-5:] / df.Confirmed[-5:])
print(f'Average daily growth rate of confirmed cases over the last 5 days: +{growth * 100 : 0.3}%')

# x = Symbol('x')
# t_d = solve((1 + growth)**x - 2 , x)
# print(f'Confirmed cases double every: {t_d[0]:0.3} days.')


# In[ ]:


df = df_china.copy()
df = df[df['Date'] == max(df['Date'])].reset_index()
df = df.groupby(['Province/State'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
df = df.sort_values('Confirmed', axis=0, ascending=False).reset_index(drop=True)
#df['Country'].replace(to_replace='Others', value='Diamond Princess', inplace=True)
#df = df[df.Country != 'Others']

df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

df['cCFR [%]'] = df['Deaths'] / df['Confirmed'] * 100
df['cCFR [%]'] = round(df['cCFR [%]'], 2)

#print(f'cCFR = Deaths / Confirmed')

#df.head(n=20)
df.style.background_gradient(subset=['Confirmed'], cmap='Blues', axis=None)        .background_gradient(subset=['Deaths'], cmap='Reds', axis=None)        .background_gradient(subset=['Recovered'], cmap='Greens', axis=None)        .background_gradient(subset=['Active'], cmap='Greys', axis=None)        .background_gradient(subset=['cCFR [%]'], cmap='PuRd', axis=None)


# # World

# In[ ]:


df = df_all.copy()
# df[(df['Country'] == 'US') & (df['Province/State'].str.contains(","))].fillna(0)
# df = df.groupby(['Date', 'Country'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
# df = df.groupby(['Country'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index()
df = df[df['Date'] == max(df['Date'])].reset_index()
df = df.groupby(['Country'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

fig = px.choropleth(df,
                    locations='Country',
                    locationmode='country names',
                    color='Confirmed',
                    hover_name='Country',
                    hover_data=['Confirmed', 'Deaths', 'Recovered'],
                    #color_continuous_scale='Burg')
                    color_continuous_scale=px.colors.sequential.thermal_r)
fig.show()

df = df_all.copy()

fig = px.bar(df.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index(), x='Date', y='Confirmed', title='World - Total Confirmed Cases')
fig.show()

fig = px.bar(df.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index(), x='Date', y='Deaths', title='World - Total Deaths', color_discrete_sequence=px.colors.diverging.Picnic_r)
fig.update_layout(showlegend=False)
fig.show()

fig = px.bar(df.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index(), x='Date', y='Recovered', title='World - Total Recovered', color_discrete_sequence=px.colors.diverging.Tropic)
fig.show()

df = df.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
fig = px.bar(df, x='Date', y=(df.Confirmed - df.Deaths - df.Recovered), title='World - Number of Active Cases', color_discrete_sequence=px.colors.sequential.thermal)
fig.update_layout(yaxis_title='Active Cases')
fig.show()


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

