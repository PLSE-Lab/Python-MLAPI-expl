#!/usr/bin/env python
# coding: utf-8

# # COVID-19 W2: A few charts and a simple baseline
# 
# 
# # Summary
# 
# **Disclaimer** We still have limited data to predict or understand what will happen in the next few weeks (months).
# 
# At this point I see more value in collecting data and monitoring the outbreak than trying to predict the future.
# 
# [Please don't kill yourself because I published a notebook](https://www.reddit.com/r/datascience/comments/fsfdn2/the_best_thing_you_can_do_to_fight_covid19_is/)
# 
# 
# 
# ### Challenges
#  * The outbreak patterns vary a lot among countries
#  * Most countries have only 2 weeks data
#  * Only a handful countries managed to succesfuly slow down the outbreak
#  * Almost every country had several serious regulations in recent weeks
#  * Increasing testing capacity could have serious impact on confirmed cases
# 
# 
# 
#  ### Assumptions
#   * As we are still in the early period, we will see exponential growth in the next few weeks
#   * Thanks to the panic/awareness/regulations/social distancing the exponential increase will slow down
# 
# As the process is not stationary at all I decided to use a simple heuristic approach. Maybe I will import sklearn next week.
#   
#   ### TIL
#  * Namibia's country code is NA. Now I remember I heard it in joke before, but I had to investigate a bug learn it again :)
#  * I haven't used plotly recently, I quite enjoyed the "new" Plotly Express interface
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_rows', 99)
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import datetime as dt


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))

import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


COMP = '../input/covid19-global-forecasting-week-2'
DATEFORMAT = '%Y-%m-%d'


def get_comp_data(COMP):
    train = pd.read_csv(f'{COMP}/train.csv')
    test = pd.read_csv(f'{COMP}/test.csv')
    submission = pd.read_csv(f'{COMP}/submission.csv')
    print(train.shape, test.shape, submission.shape)
    train['Country_Region'] = train['Country_Region'].str.replace(',', '')
    test['Country_Region'] = test['Country_Region'].str.replace(',', '')

    train['Location'] = train['Country_Region'] + '-' + train['Province_State'].fillna('')

    test['Location'] = test['Country_Region'] + '-' + test['Province_State'].fillna('')

    train['LogConfirmed'] = to_log(train.ConfirmedCases)
    train['LogFatalities'] = to_log(train.Fatalities)
    train = train.drop(columns=['Province_State'])
    test = test.drop(columns=['Province_State'])

    country_codes = pd.read_csv('../input/covid19-metadata/country_codes.csv', keep_default_na=False)
    train = train.merge(country_codes, on='Country_Region', how='left')
    test = test.merge(country_codes, on='Country_Region', how='left')

    train['DateTime'] = pd.to_datetime(train['Date'])
    test['DateTime'] = pd.to_datetime(test['Date'])
    
    return train, test, submission


def process_each_location(df):
    dfs = []
    for loc, df in tqdm(df.groupby('Location')):
        df = df.sort_values(by='Date')
        df['Fatalities'] = df['Fatalities'].cummax()
        df['ConfirmedCases'] = df['ConfirmedCases'].cummax()
        df['LogFatalities'] = df['LogFatalities'].cummax()
        df['LogConfirmed'] = df['LogConfirmed'].cummax()
        df['LogConfirmedNextDay'] = df['LogConfirmed'].shift(-1)
        df['ConfirmedNextDay'] = df['ConfirmedCases'].shift(-1)
        df['DateNextDay'] = df['Date'].shift(-1)
        df['LogFatalitiesNextDay'] = df['LogFatalities'].shift(-1)
        df['FatalitiesNextDay'] = df['Fatalities'].shift(-1)
        df['LogConfirmedDelta'] = df['LogConfirmedNextDay'] - df['LogConfirmed']
        df['ConfirmedDelta'] = df['ConfirmedNextDay'] - df['ConfirmedCases']
        df['LogFatalitiesDelta'] = df['LogFatalitiesNextDay'] - df['LogFatalities']
        df['FatalitiesDelta'] = df['FatalitiesNextDay'] - df['Fatalities']
        dfs.append(df)
    return pd.concat(dfs)


def add_days(d, k):
    return dt.datetime.strptime(d, DATEFORMAT) + dt.timedelta(days=k)


def to_log(x):
    return np.log(x + 1)


def to_exp(x):
    return np.exp(x) - 1


# In[ ]:


start = dt.datetime.now()
train, test, submission = get_comp_data(COMP)
train.shape, test.shape, submission.shape
train.head(2)
test.head(2)


# In[ ]:


train.describe()
train.nunique()
train.dtypes
train.count()

TRAIN_START = train.Date.min()
TEST_START = test.Date.min()
TRAIN_END = train.Date.max()
TEST_END = test.Date.max()
TRAIN_START, TRAIN_END, TEST_START, TEST_END


# # Worldwide

# In[ ]:


train = train.sort_values(by='Date')
countries_latest_state = train[train['Date'] == TRAIN_END].groupby([
    'Country_Region', 'continent', 'geo_region', 'country_iso_code_3']).sum()[[
    'ConfirmedCases', 'Fatalities']].reset_index()
countries_latest_state['Log10Confirmed'] = np.log10(countries_latest_state.ConfirmedCases + 1)
countries_latest_state['Log10Fatalities'] = np.log10(countries_latest_state.Fatalities + 1)
countries_latest_state = countries_latest_state.sort_values(by='Fatalities', ascending=False)
countries_latest_state.to_csv('countries_latest_state.csv', index=False)

countries_latest_state.shape
countries_latest_state.head()


# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations = countries_latest_state['country_iso_code_3'],
    z = countries_latest_state['Log10Confirmed'],
    text = countries_latest_state['Country_Region'],
    colorscale = 'viridis_r',
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix = '10^',
    colorbar_title = 'Confirmed cases <br>(log10 scale)',
))

_ = fig.update_layout(
    title_text=f'COVID-19 Global Cases [Updated: {TRAIN_END}]',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

fig.show()


# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations = countries_latest_state['country_iso_code_3'],
    z = countries_latest_state['Log10Fatalities'],
    text = countries_latest_state['Country_Region'],
    colorscale = 'viridis_r',
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix = '10^',
    colorbar_title = 'Deaths <br>(log10 scale)',
))

_ = fig.update_layout(
    title_text=f'COVID-19 Global Deaths [Updated: {TRAIN_END}]',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

fig.show()


# In[ ]:


countries_latest_state['DeathConfirmedRatio'] = (countries_latest_state.Fatalities + 1) / (countries_latest_state.ConfirmedCases + 1)
countries_latest_state['DeathConfirmedRatio'] = countries_latest_state['DeathConfirmedRatio'].clip(0, 0.1) 
fig = px.scatter(countries_latest_state,
                 x='ConfirmedCases',
                 y='Fatalities',
                 color='DeathConfirmedRatio',
                 size='Log10Fatalities',
                 size_max=20,
                 hover_name='Country_Region',
                 color_continuous_scale='viridis_r'
)
_ = fig.update_layout(
    title_text=f'COVID-19 Deaths vs Confirmed Cases by Country [Updated: {TRAIN_END}]',
    xaxis_type="log",
    yaxis_type="log"
)
fig.show()


# In[ ]:


# The source dataset is not necessary cumulative we will force it
latest_loc = train[train['Date'] == TRAIN_END][['Location', 'ConfirmedCases', 'Fatalities']]
max_loc = train.groupby(['Location'])[['ConfirmedCases', 'Fatalities']].max().reset_index()
check = pd.merge(latest_loc, max_loc, on='Location')
np.mean(check.ConfirmedCases_x == check.ConfirmedCases_y)
np.mean(check.Fatalities_x == check.Fatalities_y)
check[check.Fatalities_x != check.Fatalities_y]
check[check.ConfirmedCases_x != check.ConfirmedCases_y]


# In[ ]:


train_clean = process_each_location(train)

train_clean.shape
train_clean.tail()


# # Continents

# In[ ]:


regional_progress = train_clean.groupby(['DateTime', 'continent']).sum()[['ConfirmedCases', 'Fatalities']].reset_index()
regional_progress['Log10Confirmed'] = np.log10(regional_progress.ConfirmedCases + 1)
regional_progress['Log10Fatalities'] = np.log10(regional_progress.Fatalities + 1)
regional_progress = regional_progress[regional_progress.continent != '#N/A']


# In[ ]:


fig = px.area(regional_progress, x="DateTime", y="ConfirmedCases", color="continent")
_ = fig.update_layout(
    title_text=f'COVID-19 Cumulative Confirmed Cases by Continent [Updated: {TRAIN_END}]'
)
fig.show()
fig2 = px.line(regional_progress, x='DateTime', y='ConfirmedCases', color='continent')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Confirmed Cases by Continent [Updated: {TRAIN_END}]'
)
fig2.show()


# In[ ]:


fig = px.area(regional_progress, x="DateTime", y="Fatalities", color="continent")
_ = fig.update_layout(
    title_text=f'COVID-19 Cumulative Confirmed Deaths by Continent [Updated: {TRAIN_END}]'
)
fig.show()
fig2 = px.line(regional_progress, x='DateTime', y='Fatalities', color='continent')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Confirmed Deaths by Continent [Updated: {TRAIN_END}]'
)
fig2.show()


# In[ ]:


china = train_clean[train_clean.Location.str.startswith('China')]
top10_locations = china.groupby('Location')[['ConfirmedCases']].max().sort_values(
    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]
fig2 = px.line(china[china.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Confirmed Cases in China [Updated: {TRAIN_END}]'
)
fig2.show()


# In[ ]:


europe = train_clean[train_clean.continent == 'Europe']
top10_locations = europe.groupby('Location')[['ConfirmedCases']].max().sort_values(
    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]
fig2 = px.line(europe[europe.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Confirmed Cases in Europe [Updated: {TRAIN_END}]'
)
fig2.show()


# In[ ]:


us = train_clean[train_clean.Country_Region == 'US']
top10_locations = us.groupby('Location')[['ConfirmedCases']].max().sort_values(
    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]
fig2 = px.line(us[us.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Confirmed Cases in the USA [Updated: {TRAIN_END}]'
)
fig2.show()


# In[ ]:


africa = train_clean[train_clean.continent == 'Africa']
top10_locations = africa.groupby('Location')[['ConfirmedCases']].max().sort_values(
    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]
fig2 = px.line(africa[africa.Location.isin(top10_locations)], x='DateTime', y='ConfirmedCases', color='Location')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Confirmed Cases in Africa [Updated: {TRAIN_END}]'
)
fig2.show()


# # Countries

# In[ ]:


country_progress = train_clean.groupby(['Date', 'DateTime', 'Country_Region']).sum()[[
    'ConfirmedCases', 'Fatalities', 'ConfirmedDelta', 'FatalitiesDelta']].reset_index()
top10_countries = country_progress.groupby('Country_Region')[['Fatalities']].max().sort_values(
    by='Fatalities', ascending=False).reset_index().Country_Region.values[:10]

fig2 = px.line(country_progress[country_progress.Country_Region.isin(top10_countries)],
               x='DateTime', y='ConfirmedCases', color='Country_Region')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Confirmed Cases by Country [Updated: {TRAIN_END}]'
)
fig2.show()
fig3 = px.line(country_progress[country_progress.Country_Region.isin(top10_countries)],
               x='DateTime', y='Fatalities', color='Country_Region')
_ = fig3.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Deaths by Country [Updated: {TRAIN_END}]'
)
fig3.show()


# # Outbreak during March

# In[ ]:


countries_0301 = country_progress[country_progress.Date == '2020-03-01'][[
    'Country_Region', 'ConfirmedCases', 'Fatalities']]
countries_0331 = country_progress[country_progress.Date == '2020-03-31'][[
    'Country_Region', 'ConfirmedCases', 'Fatalities']]
countries_in_march = pd.merge(countries_0301, countries_0331, on='Country_Region', suffixes=['_0301', '_0331'])
countries_in_march['IncreaseInMarch'] = countries_in_march.ConfirmedCases_0331 / (countries_in_march.ConfirmedCases_0301 + 1)
countries_in_march = countries_in_march[countries_in_march.ConfirmedCases_0331 > 200].sort_values(
    by='IncreaseInMarch', ascending=False)
countries_in_march.tail(15)


# In[ ]:


selected_countries = [
    'Italy', 'Vietnam', 'Bahrain', 'Singapore', 'Taiwan*', 'Japan', 'Kuwait', 'Korea, South', 'China']
fig2 = px.line(country_progress[country_progress.Country_Region.isin(selected_countries)],
               x='DateTime', y='ConfirmedCases', color='Country_Region')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Confirmed Cases by Country [Updated: {TRAIN_END}]'
)
fig2.show()
fig3 = px.line(country_progress[country_progress.Country_Region.isin(selected_countries)],
               x='DateTime', y='Fatalities', color='Country_Region')
_ = fig3.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Deaths by Country [Updated: {TRAIN_END}]'
)
fig3.show()


# In[ ]:


train_clean['Geo#Country#Contintent'] = train_clean.Location + '#' + train_clean.Country_Region + '#' + train_clean.continent
latest = train_clean[train_clean.Date == '2020-03-31'][[
    'Geo#Country#Contintent', 'ConfirmedCases', 'Fatalities', 'LogConfirmed', 'LogFatalities']]
daily_confirmed_deltas = train_clean[train_clean.Date >= '2020-03-17'].pivot(
    'Geo#Country#Contintent', 'Date', 'LogConfirmedDelta').round(3).reset_index()
daily_confirmed_deltas = latest.merge(daily_confirmed_deltas, on='Geo#Country#Contintent')
daily_confirmed_deltas.shape
daily_confirmed_deltas.head()
daily_confirmed_deltas.to_csv('daily_confirmed_deltas.csv', index=False)


# In[ ]:


deltas = train_clean[np.logical_and(
        train_clean.LogConfirmed > 2,
        ~train_clean.Location.str.startswith('China')
)].dropna().sort_values(by='LogConfirmedDelta', ascending=False)

deltas['start'] = deltas['LogConfirmed'].round(0)
confirmed_deltas = pd.concat([
    deltas.groupby('start')[['LogConfirmedDelta']].mean(),
    deltas.groupby('start')[['LogConfirmedDelta']].std(),
    deltas.groupby('start')[['LogConfirmedDelta']].count()
], axis=1)

deltas.mean()

confirmed_deltas.columns = ['avg', 'std', 'cnt']
confirmed_deltas
confirmed_deltas.to_csv('confirmed_deltas.csv')


# In[ ]:


fig = px.box(deltas,  x="start", y="LogConfirmedDelta", range_y=[0, 0.35])
fig.show()


# In[ ]:


fig = px.box(deltas[deltas.Date >= '2020-03-01'],  x="DateTime", y="LogConfirmedDelta", range_y=[0, 0.6])
fig.show()


# In[ ]:


deltas = train_clean[np.logical_and(
        train_clean.LogConfirmed > 0,
        ~train_clean.Location.str.startswith('China')
)].dropna().sort_values(by='LogConfirmedDelta', ascending=False)
deltas = deltas[deltas['Date'] >= '2020-03-12']

confirmed_deltas = pd.concat([
    deltas.groupby('Location')[['LogConfirmedDelta']].mean(),
    deltas.groupby('Location')[['LogConfirmedDelta']].std(),
    deltas.groupby('Location')[['LogConfirmedDelta']].count(),
    deltas.groupby('Location')[['LogConfirmed']].max()
], axis=1)
confirmed_deltas.columns = ['avg', 'std', 'cnt', 'max']

confirmed_deltas.sort_values(by='avg').head(10)
confirmed_deltas.sort_values(by='avg').tail(10)
confirmed_deltas.to_csv('confirmed_deltas.csv')


# # Create prediction

# In[ ]:


DECAY = 0.93
DECAY ** 7, DECAY ** 14, DECAY ** 21, DECAY ** 28

confirmed_deltas = train.groupby(['Location', 'Country_Region', 'continent'])[[
    'Id']].count().reset_index()

GLOBAL_DELTA = 0.11
confirmed_deltas['DELTA'] = GLOBAL_DELTA

confirmed_deltas.loc[confirmed_deltas.continent=='Africa', 'DELTA'] = 0.14
confirmed_deltas.loc[confirmed_deltas.continent=='Oceania', 'DELTA'] = 0.06
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Korea South', 'DELTA'] = 0.011
confirmed_deltas.loc[confirmed_deltas.Country_Region=='US', 'DELTA'] = 0.15
confirmed_deltas.loc[confirmed_deltas.Country_Region=='China', 'DELTA'] = 0.01
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Japan', 'DELTA'] = 0.05
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Singapore', 'DELTA'] = 0.05
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Taiwan*', 'DELTA'] = 0.05
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Switzerland', 'DELTA'] = 0.05
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Norway', 'DELTA'] = 0.05
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Iceland', 'DELTA'] = 0.05
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Austria', 'DELTA'] = 0.06
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Italy', 'DELTA'] = 0.04
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Spain', 'DELTA'] = 0.08
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Portugal', 'DELTA'] = 0.12
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Israel', 'DELTA'] = 0.12
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Iran', 'DELTA'] = 0.08
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Germany', 'DELTA'] = 0.07
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Malaysia', 'DELTA'] = 0.06
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Russia', 'DELTA'] = 0.18
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Ukraine', 'DELTA'] = 0.18
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Brazil', 'DELTA'] = 0.12
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Turkey', 'DELTA'] = 0.18
confirmed_deltas.loc[confirmed_deltas.Country_Region=='Philippines', 'DELTA'] = 0.18
confirmed_deltas.loc[confirmed_deltas.Location=='France-', 'DELTA'] = 0.1
confirmed_deltas.loc[confirmed_deltas.Location=='United Kingdom-', 'DELTA'] = 0.12
confirmed_deltas.loc[confirmed_deltas.Location=='Diamond Princess-', 'DELTA'] = 0.00
confirmed_deltas.loc[confirmed_deltas.Location=='China-Hong Kong', 'DELTA'] = 0.08
confirmed_deltas.loc[confirmed_deltas.Location=='San Marino-', 'DELTA'] = 0.03


confirmed_deltas.shape, confirmed_deltas.DELTA.mean()

confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA].shape, confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA].DELTA.mean()
confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA]
confirmed_deltas.describe()


# In[ ]:


daily_log_confirmed = train_clean.pivot('Location', 'Date', 'LogConfirmed').reset_index()
daily_log_confirmed = daily_log_confirmed.sort_values(TRAIN_END, ascending=False)
daily_log_confirmed.to_csv('daily_log_confirmed.csv', index=False)

for i, d in tqdm(enumerate(pd.date_range(add_days(TRAIN_END, 1), add_days(TEST_END, 1)))):
    new_day = str(d).split(' ')[0]
    last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)
    last_day = last_day.strftime(DATEFORMAT)
    for loc in confirmed_deltas.Location.values:
        confirmed_delta = confirmed_deltas.loc[confirmed_deltas.Location == loc, 'DELTA'].values[0]
        daily_log_confirmed.loc[daily_log_confirmed.Location == loc, new_day] = daily_log_confirmed.loc[daily_log_confirmed.Location == loc, last_day] +             confirmed_delta * DECAY ** i


# In[ ]:


daily_log_confirmed.head()


# In[ ]:


confirmed_prediciton = pd.melt(daily_log_confirmed[:25], id_vars='Location')
confirmed_prediciton['ConfirmedCases'] = to_exp(confirmed_prediciton['value'])
fig2 = px.line(confirmed_prediciton,
               x='Date', y='ConfirmedCases', color='Location')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Confirmed Cases Prediction [Updated: {TRAIN_END}]'
)
fig2.show()


# # Fatalities

# In[ ]:


train_clean['Geo#Country#Contintent'] = train_clean.Location + '#' + train_clean.Country_Region + '#' + train_clean.continent
latest = train_clean[train_clean.Date == TRAIN_END][[
    'Geo#Country#Contintent', 'ConfirmedCases', 'Fatalities', 'LogConfirmed', 'LogFatalities']]
daily_death_deltas = train_clean[train_clean.Date >= '2020-03-17'].pivot(
    'Geo#Country#Contintent', 'Date', 'LogFatalitiesDelta').round(3).reset_index()
daily_death_deltas = latest.merge(daily_death_deltas, on='Geo#Country#Contintent')
daily_death_deltas.shape
daily_death_deltas.head()
daily_death_deltas.to_csv('daily_death_deltas.csv', index=False)


# In[ ]:


death_deltas = train.groupby(['Location', 'Country_Region', 'continent'])[[
    'Id']].count().reset_index()

GLOBAL_DELTA = 0.11
death_deltas['DELTA'] = GLOBAL_DELTA

death_deltas.loc[death_deltas.Country_Region=='China', 'DELTA'] = 0.005
death_deltas.loc[death_deltas.continent=='Oceania', 'DELTA'] = 0.08
death_deltas.loc[death_deltas.Country_Region=='Korea South', 'DELTA'] = 0.04
death_deltas.loc[death_deltas.Country_Region=='Japan', 'DELTA'] = 0.04
death_deltas.loc[death_deltas.Country_Region=='Singapore', 'DELTA'] = 0.05
death_deltas.loc[death_deltas.Country_Region=='Taiwan*', 'DELTA'] = 0.06



death_deltas.loc[death_deltas.Country_Region=='US', 'DELTA'] = 0.17

death_deltas.loc[death_deltas.Country_Region=='Switzerland', 'DELTA'] = 0.15
death_deltas.loc[death_deltas.Country_Region=='Norway', 'DELTA'] = 0.15
death_deltas.loc[death_deltas.Country_Region=='Iceland', 'DELTA'] = 0.01
death_deltas.loc[death_deltas.Country_Region=='Austria', 'DELTA'] = 0.14
death_deltas.loc[death_deltas.Country_Region=='Italy', 'DELTA'] = 0.07
death_deltas.loc[death_deltas.Country_Region=='Spain', 'DELTA'] = 0.1
death_deltas.loc[death_deltas.Country_Region=='Portugal', 'DELTA'] = 0.13
death_deltas.loc[death_deltas.Country_Region=='Israel', 'DELTA'] = 0.16
death_deltas.loc[death_deltas.Country_Region=='Iran', 'DELTA'] = 0.06
death_deltas.loc[death_deltas.Country_Region=='Germany', 'DELTA'] = 0.14
death_deltas.loc[death_deltas.Country_Region=='Malaysia', 'DELTA'] = 0.14
death_deltas.loc[death_deltas.Country_Region=='Russia', 'DELTA'] = 0.2
death_deltas.loc[death_deltas.Country_Region=='Ukraine', 'DELTA'] = 0.2
death_deltas.loc[death_deltas.Country_Region=='Brazil', 'DELTA'] = 0.2
death_deltas.loc[death_deltas.Country_Region=='Turkey', 'DELTA'] = 0.22
death_deltas.loc[death_deltas.Country_Region=='Philippines', 'DELTA'] = 0.12
death_deltas.loc[death_deltas.Location=='France-', 'DELTA'] = 0.14
death_deltas.loc[death_deltas.Location=='United Kingdom-', 'DELTA'] = 0.14
death_deltas.loc[death_deltas.Location=='Diamond Princess-', 'DELTA'] = 0.00

death_deltas.loc[death_deltas.Location=='China-Hong Kong', 'DELTA'] = 0.01
death_deltas.loc[death_deltas.Location=='San Marino-', 'DELTA'] = 0.05


death_deltas.shape
death_deltas.DELTA.mean()

death_deltas[death_deltas.DELTA != GLOBAL_DELTA].shape
death_deltas[death_deltas.DELTA != GLOBAL_DELTA].DELTA.mean()
death_deltas[death_deltas.DELTA != GLOBAL_DELTA]
death_deltas.describe()


# In[ ]:


daily_log_deaths = train_clean.pivot('Location', 'Date', 'LogFatalities').reset_index()
daily_log_deaths = daily_log_deaths.sort_values(TRAIN_END, ascending=False)
daily_log_deaths.to_csv('daily_log_deaths.csv', index=False)

for i, d in tqdm(enumerate(pd.date_range(add_days(TRAIN_END, 1), add_days(TEST_END, 1)))):
    new_day = str(d).split(' ')[0]
    last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)
    last_day = last_day.strftime(DATEFORMAT)
    for loc in death_deltas.Location:
        death_delta = death_deltas.loc[death_deltas.Location == loc, 'DELTA'].values[0]
        daily_log_deaths.loc[daily_log_deaths.Location == loc, new_day] = daily_log_deaths.loc[daily_log_deaths.Location == loc, last_day] +             death_delta * DECAY ** i


# In[ ]:


confirmed_prediciton = pd.melt(daily_log_deaths[:25], id_vars='Location')
confirmed_prediciton['Fatalities'] = to_exp(confirmed_prediciton['value'])
fig2 = px.line(confirmed_prediciton,
               x='Date', y='Fatalities', color='Location')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Fatalities Prediction [Updated: {TRAIN_END}]'
)
fig2.show()


# # Create submission file

# In[ ]:


submission.head(2)


# In[ ]:


confirmed = []
fatalities = []
for id, d, loc in tqdm(test[['ForecastId', 'Date', 'Location']].values):
    c = to_exp(daily_log_confirmed.loc[daily_log_confirmed.Location == loc, d].values[0])
    f = to_exp(daily_log_deaths.loc[daily_log_deaths.Location == loc, d].values[0])
    confirmed.append(c)
    fatalities.append(f)


# In[ ]:


my_submission = test.copy()
my_submission['ConfirmedCases'] = confirmed
my_submission['Fatalities'] = fatalities
my_submission.shape
my_submission.head()



# In[ ]:


my_submission.groupby('Date').sum().tail()


# # Sanity check

# In[ ]:


total = my_submission.groupby('Date')[['ConfirmedCases', 'Fatalities']].sum().reset_index()

fig2 = px.line(pd.melt(total, id_vars=['Date']), x='Date', y='value', color='variable')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Prediction Total [Updated: {TRAIN_END}]'
)
fig2.show()


# In[ ]:


my_submission[[
    'ForecastId', 'ConfirmedCases', 'Fatalities'
]].to_csv('submission.csv', index=False)
print(DECAY)
my_submission.head()
my_submission.tail()
my_submission.shape


# In[ ]:


end = dt.datetime.now()
print('Finished', end, (end - start).seconds, 's')

