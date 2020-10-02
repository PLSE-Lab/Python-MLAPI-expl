#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %matplotlib inline
import pandas as pd
import datetime as dt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[ ]:


plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff


# # Numerai
# 
# Numerai is a weekly data science competition. Predictions submitted by users steer Numerai's hedge fund together. A weekly prize pool is paid out to the top performing users who stake.
# 
# You could learn more here: https://numer.ai/learn

# In[ ]:


start = dt.datetime.now()
T = 166
FIRST_ROUND = 94
ROUNDS = range(FIRST_ROUND, T + 1)
STAKING_ROUNDS = range(FIRST_ROUND, T + 1)
THRESHOLD = 0.693


# # Numeraire
# 
# 
# Numeraire (NMR) is an Ethereum ERC20 token created by Numerai. It is could be used in the weekly staking tournaments and it is also used to pay some part of the prize pool.
# It has its ups and downs. Last year it dropped quite a lot with BTC and other cryptos.

# In[ ]:


nmr_price = pd.read_csv('../input/CoinMarketCapNMR')
nmr_price['Date'] = [dt.datetime.strptime(d, '%b-%d-%Y') for d in nmr_price.Date.values]
nmr_price['Date'] = [pd.Timestamp(d).date() for d in nmr_price.Date.values]
nmr_price = nmr_price[['Date', 'High']]
nmr_price.columns = ['Date', 'NMRUSD']
nmr_price = nmr_price[nmr_price.Date >= dt.date(2018, 2, 1)]


# In[ ]:


data = []
trace = go.Scatter(
    x = nmr_price.Date.values,
    y = nmr_price.NMRUSD.values,
    mode = 'lines',
    name = 'NMR - USD',
    line=dict(width=4)
)
data.append(trace)
layout= go.Layout(
    title= 'NMR historical price',
    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='NMR USD', ticklen=5, gridwidth=2),
    showlegend=True
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='nmr_usd')


# # Weekly tournament
# 
# Each new round comes with a unique data zip. This zip includes all the data you will need to compete on Numerai. You have 50 noisy features and one binary target for each of the five current tournaments to predict. The training data is actually the same but you have to predict the future for each week and when the tournament resolves four weeks later your submission is judged by its actual Logloss score.
# 
# The goal is simple beat the benchmark 0.693. It sounds easy right?
# 
# Well it is not always that easy. There are rounds when only a few percent survives.

# In[ ]:


competitions = pd.read_csv('../input/T{}_competitions.csv'.format(T), low_memory=False)
lb_with_stakes = pd.read_csv('../input/T{}_leaderboards.csv'.format(T), low_memory=False)
lb_with_stakes = lb_with_stakes[lb_with_stakes['number'] >= FIRST_ROUND]


# In[ ]:


tournament_results = lb_with_stakes.drop([
    'tournament_id', 'consistency', 'concordance.value', 'better_than_random', 'stake.insertedAt', 'stakeResolution.paid',
], axis=1)
tournament_results = tournament_results.merge(competitions, how='left', on=['number'])
tournament_results = tournament_results.drop([
    'datasetId', 'participants', 'prizePoolNmr', 'prizePoolUsd'], axis=1)
tournament_results['ResolveDate'] = [pd.Timestamp(d).date() for d in tournament_results.resolveTime.values]
tournament_results['OpenDate'] = [pd.Timestamp(d).date() for d in tournament_results.openTime.values]
tournament_results = tournament_results.merge(nmr_price, how='left', left_on='ResolveDate', right_on='Date')
tournament_results = tournament_results.merge(nmr_price, how='left', left_on='OpenDate', right_on='Date',
                                              suffixes=['Resolve', 'Open'])
tournament_results = tournament_results.drop([
    'openTime', 'resolveTime', 'DateResolve', 'DateOpen', 'resolvedGeneral', 'resolvedStaking'], axis=1)
resolved = tournament_results[~tournament_results.liveLogloss.isna()]


# In[ ]:


tour_performance_median = resolved.groupby('number').median()
fig, ax = plt.subplots()
plt.plot(tour_performance_median.index, tour_performance_median.liveLogloss, c='r', lw=5, label='medianliveLogloss')
plt.plot(tour_performance_median.index, tour_performance_median.validationLogloss, c='b', lw=5, label='medianvalidationLogloss')
plt.plot(tour_performance_median.index, [np.log(2)] * tour_performance_median.shape[0], 'k:', lw=2, label='ln(2)')
plt.plot(tour_performance_median.index, [THRESHOLD] * tour_performance_median.shape[0], 'k-', lw=2, label='Threshold: 0.693')
plt.scatter(resolved.number.values, resolved.liveLogloss.values, color='g', s=20, alpha=0.2, label='liveLogloss')
plt.grid()
plt.title('Live Logloss Results')
plt.xlabel('Round')
plt.ylabel('Logloss')
plt.legend(loc=0)
plt.ylim(0.69, 0.7)
plt.show();


# # User stats
# 
# Once you submitted your predictions and passed a few sanity checks. You are ready to participate in the staking tournament. If you think your model will beat the live logloss benchmark, you should stake on your submission for a chance to win some of the prize pool.
# 
# Roughly there are 500 accounts in each round and 100-200 of them stakes. One tricky part is that you are allowed to have 3 accounts.

# In[ ]:


users_with_submission = tournament_results.groupby('OpenDate')[['username']].nunique()
users_with_stake = tournament_results[~tournament_results['stake.value'].isna()].groupby('OpenDate')[['username']].nunique()
user_counts = pd.merge(users_with_submission, users_with_stake, left_index=True, right_index=True).reset_index()
user_counts.columns = ['OpenDate', 'UserswithSubmission', 'UserswithStake']


# In[ ]:


data = [
go.Scatter(
    x = user_counts.OpenDate.values,
    y = user_counts.UserswithSubmission.values,
    mode = 'lines',
    name = '#Users with submission',
    line=dict(width=4)
),
go.Scatter(
    x = user_counts.OpenDate.values,
    y = user_counts.UserswithStake.values,
    mode = 'lines',
    name = '#Users with stake',
    line=dict(width=4)
),
]
layout= go.Layout(
    title= 'User stats',
    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of users', ticklen=5, gridwidth=2),
    showlegend=True
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='users')


# # Skin in the game
# 
# The amount of NMR you are willing to put down for the duration of the round. The more you stake, the more you stand to earn.
# 
# Be aware if you do not beat the benchmark your stake will be burned!

# In[ ]:


total_stake_amount = tournament_results.groupby('OpenDate')[['stake.value']].sum().reset_index()
data = [
go.Bar(
    x = total_stake_amount.OpenDate.values,
    y = total_stake_amount['stake.value'].values,
    name = 'Stake',
),
]
layout= go.Layout(
    title= '"Skin in the game"',
    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Total stakes (NMR)', ticklen=5, gridwidth=2),
    showlegend=True
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='skin')


# The staking rules and payment structure changes quite often.
# 
# Currently you need to be confident in your prediction to have a chance for the prize pool otherwise your stake is returned. If you are selected you could win additional NMR and ETH with your stake or burn it completely.

# In[ ]:


stakes = resolved[~resolved['stake.value'].isna()]
stakes = stakes.fillna(0)
stakes['Burned'] = 1.0 * stakes['stakeResolution.destroyed'] * stakes['stake.value']
stakes['Winning'] = 1.0 * stakes['stakeResolution.successful'] * stakes['stake.value']
stakes['Unused'] = stakes['stake.value'] - stakes['Winning'] - stakes['Burned']
stake_type = stakes.groupby('OpenDate')[['Burned', 'Winning', 'Unused']].sum().reset_index()


# In[ ]:


total_stake_amount = tournament_results.groupby('OpenDate')[['stake.value']].sum().reset_index()
data = [
    go.Bar(x=stake_type.OpenDate.values, y=stake_type['Burned'].values,
           marker=dict(color='red'), name = 'Burn',),
    go.Bar(x=stake_type.OpenDate.values, y=stake_type['Winning'].values,
           marker=dict(color='green'), name = 'Winning',),
    go.Bar(x=stake_type.OpenDate.values, y=stake_type['Unused'].values,
           marker=dict(color='grey'), name = 'Unused',),
]
layout= go.Layout(
    barmode='stack',
    title= 'Stakes',
    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Stake (NMR)', ticklen=5, gridwidth=2),
    showlegend=True
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stake-type')


# # Prize pool
# 
# The prize depends on the NMR price and the weekly burn rate. 

# In[ ]:


stakes['PrizeUSD'] = stakes['paymentStaking.nmrAmount'] * stakes['NMRUSDResolve'] + stakes['paymentStaking.usdAmount']
stakes['PrizeUSD5'] = stakes['paymentStaking.nmrAmount'] * 5 + stakes['paymentStaking.usdAmount']
stakes['ProfitUSD'] = stakes['PrizeUSD'] - stakes['Burned'] * stakes['NMRUSDOpen'] + stakes['Winning'] * (stakes['NMRUSDResolve'] - stakes['NMRUSDOpen'])
stakes = stakes.fillna(0)
PrizeUSD = stakes.groupby('OpenDate')[['PrizeUSD', 'ProfitUSD', 'PrizeUSD5']].sum().reset_index()


# In[ ]:


data = [
go.Bar(
    x = PrizeUSD.OpenDate.values,
    y = PrizeUSD['PrizeUSD'].values,
    name = 'PrizeUSD',
)]
layout= go.Layout(
    title= 'Actual prize pool',
    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Paid Prize (USD)', ticklen=5, gridwidth=2),
    showlegend=True
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='prize')


# In[ ]:


data = [
go.Bar(
    x = PrizeUSD.OpenDate.values,
    y = PrizeUSD['ProfitUSD'].values,
    name = 'ProfitUSD',
)]
layout= go.Layout(
    title= 'Profit',
    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Profit (USD)', ticklen=5, gridwidth=2),
    showlegend=True
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='profit')


# # Tournaments
# 
# There are seven different targets you could predict:
# 
# * Bernie
# * Elizabeth
# * Ken
# * Jordan
# * Charles
# * Frank
# * Hillary
# 
# Note there are huge variance even between consecutive weeks.

# In[ ]:


nt = stakes.groupby(['number', 'tournament_name'])[['better_than_threshold']].mean().reset_index()
ntp = nt.pivot('number', 'tournament_name', 'better_than_threshold')
ntp = ntp[ntp.index >= 110]


# In[ ]:


data = [
go.Scatter(x = ntp.index.values, y = ntp[tname].values, mode='lines+markers',
           marker=dict(size=10), name=tname, line=dict(width=2), opacity=0.7)
    for tname in ntp.columns
]
layout= go.Layout(
    title= 'Live Success Rate',
    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='P(better than threshold)', ticklen=5, gridwidth=2),
    showlegend=True
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='consistency')


# In[ ]:


end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))


# # Acknowledgements
# 
# 1. The dataset was collected with [NumerApi](https://github.com/uuazed/numerapi)
# 2. The daily currency info is from [CoinMarketCap](https://coinmarketcap.com/currencies/numeraire/)
