#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


DATEFORMAT = '%Y-%m-%d'


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


COMP = 'covid19-global-forecasting-week-1'


# In[ ]:


train = pd.read_csv(f'../input/{COMP}/train.csv')
test = pd.read_csv(f'../input/{COMP}/test.csv')
submission = pd.read_csv(f'../input/{COMP}/submission.csv')
train.shape, test.shape, submission.shape


# In[ ]:


train.head()
submission.head()
test.head()


# In[ ]:


set(test['Country/Region']) - set(train['Country/Region'])
set(test['Country/Region']) - set(train['Country/Region'])


# In[ ]:


def to_log(x):
    return np.log(x + 1)

def to_exp(x):
    return np.exp(x) - 1


# In[ ]:


np.arange(10)
to_exp(to_log(np.arange(10)))


# In[ ]:


train['Location'] = train['Country/Region'] + '-' + train['Province/State'].fillna('')
train['Location'] = train['Location'].str.replace(',', '')

test['Location'] = test['Country/Region'] + '-' + test['Province/State'].fillna('')
test['Location'] = test['Location'].str.replace(',', '')

train['LogConfirmed'] = to_log(train.ConfirmedCases)
train['LogFatalities'] = to_log(train.Fatalities)
train = train.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])
test = test.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])


# In[ ]:


train.tail()


# In[ ]:


set(test['Location']) - set(train['Location'])


# In[ ]:


dfs = []
for loc, df in tqdm(train.groupby('Location')):
    df = df.sort_values(by='Date')
    df['LogFatalities'] = df['LogFatalities'].cummax()
    df['LogConfirmed'] = df['LogConfirmed'].cummax()
    df['LogConfirmedNextDay'] = df['LogConfirmed'].shift(-1)
    df['DateNextDay'] = df['Date'].shift(-1)
    df['LogFatalitiesNextDay'] = df['LogFatalities'].shift(-1)
    df['LogConfirmedDelta'] = df['LogConfirmedNextDay'] - df['LogConfirmed']
    df['LogFatalitiesDelta'] = df['LogFatalitiesNextDay'] - df['LogFatalities']
    dfs.append(df)
dfs = pd.concat(dfs)


# In[ ]:


dfs.shape
dfs.tail()


# In[ ]:


deltas = dfs[np.logical_and(
        dfs.LogConfirmed > 0,
        ~dfs.Location.str.startswith('China')
)].dropna().sort_values(by='LogConfirmedDelta', ascending=False)

deltas['start'] = deltas['LogConfirmed'].round(1)
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


plt.scatter(confirmed_deltas.index, confirmed_deltas.avg, s=confirmed_deltas.cnt)
plt.grid()


# In[ ]:


daily_confirmed_deltas = pd.concat([
    deltas.groupby('Date')[['LogConfirmedDelta']].mean(),
    deltas.groupby('Date')[['LogConfirmedDelta']].std(),
    deltas.groupby('Date')[['LogConfirmedDelta']].count()
], axis=1)


daily_confirmed_deltas.columns = ['avg', 'std', 'cnt']

plt.scatter(daily_confirmed_deltas.index, daily_confirmed_deltas.avg, s=daily_confirmed_deltas.cnt)
plt.xticks(rotation=90)
plt.grid()
plt.show();


# In[ ]:


deltas = dfs[np.logical_and(
        dfs.LogConfirmed > 0,
        ~dfs.Location.str.startswith('China')
)].dropna().sort_values(by='LogConfirmedDelta', ascending=False)
deltas = deltas[deltas['Date'] >= '2020-03-01']

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


# In[ ]:





# # Confirmed Cases

# In[ ]:


DECAY = 0.93
DECAY ** 14, DECAY ** 27


# In[ ]:


confirmed_deltas = train.groupby('Location')[['Id']].count()

confirmed_deltas['DELTA'] = 0.18
confirmed_deltas.loc[confirmed_deltas.index.str.startswith('China'), 'DELTA'] = 0.02
confirmed_deltas.loc[confirmed_deltas.index.str.startswith('US'), 'DELTA'] = 0.24
confirmed_deltas.loc[confirmed_deltas.index=='Turkey-', 'DELTA'] = 0.23
confirmed_deltas.loc[confirmed_deltas.index=='Brazil-', 'DELTA'] = 0.23
confirmed_deltas.loc[confirmed_deltas.index=='Mexico-', 'DELTA'] = 0.23
confirmed_deltas.loc[confirmed_deltas.index=='Romania-', 'DELTA'] = 0.25
confirmed_deltas.loc[confirmed_deltas.index=='Singapore-', 'DELTA'] = 0.07
confirmed_deltas.loc[confirmed_deltas.index=='Japan-', 'DELTA'] = 0.05
confirmed_deltas.loc[confirmed_deltas.index=='Korea South-', 'DELTA'] = 0.03
confirmed_deltas.loc[confirmed_deltas.index=='China-Hong Kong', 'DELTA'] = 0.12
confirmed_deltas.loc[confirmed_deltas.index=='Iran-', 'DELTA'] = 0.14
confirmed_deltas.loc[confirmed_deltas.index=='Italy-', 'DELTA'] = 0.14
confirmed_deltas.loc[confirmed_deltas.index.str.endswith('Princess'), 'DELTA'] = 0.01
confirmed_deltas


# In[ ]:


daily_log_confirmed = dfs.pivot('Location', 'Date', 'LogConfirmed').reset_index()
daily_log_confirmed = daily_log_confirmed.sort_values('2020-03-24', ascending=False)
daily_log_confirmed.to_csv('daily_log_confirmed.csv', index=False)

for i, d in tqdm(enumerate(pd.date_range('2020-03-25', '2020-04-24'))):
    new_day = str(d).split(' ')[0]
    last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)
    last_day = last_day.strftime(DATEFORMAT)
#     print(i, new_day, last_day)
    for loc in confirmed_deltas.index:
        confirmed_delta = confirmed_deltas.loc[confirmed_deltas.index == loc, 'DELTA'].values[0]
        daily_log_confirmed.loc[daily_log_confirmed.Location == loc, new_day] = daily_log_confirmed.loc[daily_log_confirmed.Location == loc, last_day] +             confirmed_delta * DECAY ** i


# In[ ]:


daily_log_confirmed.head(30)


# In[ ]:


confirmed_deltas = train.groupby('Location')[['Id']].count()

confirmed_deltas['DELTA'] = 0.18
confirmed_deltas.loc[confirmed_deltas.index.str.startswith('China'), 'DELTA'] = 0.02
confirmed_deltas.loc[confirmed_deltas.index.str.startswith('US'), 'DELTA'] = 0.2
confirmed_deltas.loc[confirmed_deltas.index=='Turkey-', 'DELTA'] = 0.23
confirmed_deltas.loc[confirmed_deltas.index=='Brazil-', 'DELTA'] = 0.23
confirmed_deltas.loc[confirmed_deltas.index=='Mexico-', 'DELTA'] = 0.23
confirmed_deltas.loc[confirmed_deltas.index=='Romania-', 'DELTA'] = 0.25
confirmed_deltas.loc[confirmed_deltas.index=='Singapore-', 'DELTA'] = 0.07
confirmed_deltas.loc[confirmed_deltas.index=='Japan-', 'DELTA'] = 0.05
confirmed_deltas.loc[confirmed_deltas.index=='Korea South-', 'DELTA'] = 0.03
confirmed_deltas.loc[confirmed_deltas.index=='China-Hong Kong', 'DELTA'] = 0.12
confirmed_deltas.loc[confirmed_deltas.index=='Iran-', 'DELTA'] = 0.14
confirmed_deltas.loc[confirmed_deltas.index=='Italy-', 'DELTA'] = 0.14
confirmed_deltas.loc[confirmed_deltas.index.str.endswith('Princess'), 'DELTA'] = 0.01
confirmed_deltas
confirmed_deltas.describe()


# In[ ]:


daily_log_confirmed[:25].set_index('Location').T.plot()
plt.grid()
plt.show();


# # Fatalities

# In[ ]:


death_deltas = dfs[np.logical_and(
        dfs.Fatalities > 0,
        ~dfs.Location.str.startswith('China')
)].dropna().sort_values(by='LogFatalitiesDelta', ascending=False)


# In[ ]:


death_deltas


# In[ ]:


daily_confirmed_death_deltas = pd.concat([
    death_deltas.groupby('Date')[['LogFatalitiesDelta']].mean(),
    death_deltas.groupby('Date')[['LogFatalitiesDelta']].std(),
    death_deltas.groupby('Date')[['LogFatalitiesDelta']].count()
], axis=1)


daily_confirmed_death_deltas.columns = ['avg', 'std', 'cnt']

plt.scatter(daily_confirmed_death_deltas.index, daily_confirmed_death_deltas.avg, s=daily_confirmed_death_deltas.cnt)
plt.xticks(rotation=90)
plt.grid()
plt.show();


# In[ ]:


loc_confirmed_death_deltas = pd.concat([
    death_deltas.groupby('Location')[['LogFatalitiesDelta']].mean(),
    death_deltas.groupby('Location')[['LogFatalitiesDelta']].std(),
    death_deltas.groupby('Location')[['LogFatalitiesDelta']].count(),
    death_deltas.groupby('Location')[['LogFatalities']].max()
], axis=1)
loc_confirmed_death_deltas.columns = ['avg', 'std', 'cnt', 'max']
loc_confirmed_death_deltas = loc_confirmed_death_deltas[loc_confirmed_death_deltas.cnt >= 3]

loc_confirmed_death_deltas.sort_values(by='avg').head(10)
loc_confirmed_death_deltas.sort_values(by='avg').tail(10)
loc_confirmed_death_deltas.to_csv('loc_confirmed_death_deltas.csv')


# In[ ]:


death_deltas = train.groupby('Location')[['Id']].count()

death_deltas['DELTA'] = 0.17
death_deltas.loc[death_deltas.index.str.startswith('China'), 'DELTA'] = 0.01
death_deltas.loc[death_deltas.index.str.startswith('US'), 'DELTA'] = 0.15
death_deltas.loc[death_deltas.index=='Turkey-', 'DELTA'] = 0.23
death_deltas.loc[death_deltas.index=='Brazil-', 'DELTA'] = 0.23
death_deltas.loc[death_deltas.index=='Mexico-', 'DELTA'] = 0.23
death_deltas.loc[death_deltas.index=='Singapore-', 'DELTA'] = 0.07
death_deltas.loc[death_deltas.index=='Japan-', 'DELTA'] = 0.05
death_deltas.loc[death_deltas.index=='Korea South-', 'DELTA'] = 0.05
death_deltas.loc[death_deltas.index=='Taiwan*-', 'DELTA'] = 0.05
death_deltas.loc[death_deltas.index=='China-Hong Kong', 'DELTA'] = 0.08
death_deltas.loc[death_deltas.index=='Iran-', 'DELTA'] = 0.14
death_deltas.loc[death_deltas.index=='Italy-', 'DELTA'] = 0.12
death_deltas.loc[death_deltas.index=='Spain-', 'DELTA'] = 0.14
death_deltas.loc[death_deltas.index.str.endswith('Princess'), 'DELTA'] = 0.01
death_deltas


# In[ ]:


death_deltas.describe()


# In[ ]:


daily_log_deaths = dfs.pivot('Location', 'Date', 'LogFatalities').reset_index()
daily_log_deaths = daily_log_deaths.sort_values('2020-03-24', ascending=False)
daily_log_deaths.to_csv('daily_log_deaths.csv', index=False)

for i, d in tqdm(enumerate(pd.date_range('2020-03-25', '2020-04-24'))):
    new_day = str(d).split(' ')[0]
    last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)
    last_day = last_day.strftime(DATEFORMAT)
    for loc in death_deltas.index:
        death_delta = death_deltas.loc[death_deltas.index == loc, 'DELTA'].values[0]
        daily_log_deaths.loc[daily_log_deaths.Location == loc, new_day] = daily_log_deaths.loc[daily_log_deaths.Location == loc, last_day] +             death_delta * DECAY ** i


# In[ ]:


daily_log_deaths[:25].set_index('Location').T.plot()
plt.grid()
plt.show();


# # Create submission file

# In[ ]:


submission.head()


# In[ ]:


confirmed = []
fatalities = []
for id, d, loc in tqdm(test.values):
#     print(id, d, loc)
    c = to_exp(daily_log_confirmed.loc[daily_log_confirmed.Location == loc, d].values[0])
    f = to_exp(daily_log_deaths.loc[daily_log_deaths.Location == loc, d].values[0])
    confirmed.append(c)
    fatalities.append(f)


# In[ ]:


test


# In[ ]:


my_submission = test.copy()
my_submission['ConfirmedCases'] = confirmed
my_submission['Fatalities'] = fatalities
my_submission.head()
my_submission.shape


# In[ ]:


my_submission.groupby('Date').sum().tail()


# In[ ]:


my_submission.groupby('Date').sum().plot()


# In[ ]:


plt.semilogy(my_submission.groupby('Date').sum()['ConfirmedCases'].values)


# In[ ]:


plt.semilogy(my_submission.groupby('Date').sum()['Fatalities'].values)


# In[ ]:


my_submission[[
    'ForecastId', 'ConfirmedCases', 'Fatalities'
]].to_csv('submission.csv', index=False)
print(DECAY)
my_submission.head()
my_submission.tail()
my_submission.shape


# In[ ]:




