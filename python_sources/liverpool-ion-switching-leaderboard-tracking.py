#!/usr/bin/env python
# coding: utf-8

# # Liverpool Ion Public Leaderboard Tracking

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.linear_model import LinearRegression
import datetime
import colorlover as cl
plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

# Format the dataa
df = pd.read_csv('../input/liverpoolionpubliclb/liverpool-ion-switching-publicleaderboard_05102020.csv')
df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
df = df.set_index(['TeamName','SubmissionDate'])['Score'].unstack(-1).T
df.columns = [name for name in df.columns]

FIFTEENTH_SCORE = df.max().sort_values(ascending=False)[15]
FIFTYTH_SCORE = df.max().sort_values(ascending=False)[50]
TOP_SCORE = df.max().sort_values(ascending=False)[0]


# In[ ]:


# Scores of top teams over time
plt.rcParams["font.size"] = "12"
ALL_TEAMS = df.columns.values
ALL_TEAMS = [x for x in ALL_TEAMS if type(x) == str]
df[ALL_TEAMS].ffill()     .T.sample(1000).T     .plot(figsize=(20, 10),
                           color=color_pal[0],
                           legend=False,
                           alpha=0.05,
                           ylim=(0.92, 0.947),
                           title='All Teams Public LB Scores over Time')
#df.ffill().max(axis=1).plot(color=color_pal[1], label='1st Place Public LB', legend=True)
plt.show()


# In[ ]:


team_over_time = df.ffill()     .count(axis=1)

lr = LinearRegression()
_ = lr.fit(np.array(pd.to_numeric(team_over_time.index).tolist()).reshape(-1, 1),
           team_over_time.values)

teamcount_df = pd.DataFrame(team_over_time)

teamcount_pred_df = pd.DataFrame(index=pd.date_range('04-20-2020','05-29-2020'))
teamcount_pred_df['teamcount_predict'] = lr.predict(np.array(pd.to_numeric(teamcount_pred_df.index).tolist()).reshape(-1, 1))

lr = LinearRegression()
_ = lr.fit(np.array(pd.to_numeric(team_over_time[-1000:].index).tolist()).reshape(-1, 1),
           team_over_time[-1000:].values)

teamcount_pred_df['teamcount_predict_recent'] = lr.predict(np.array(pd.to_numeric(teamcount_pred_df.index).tolist()).reshape(-1, 1))

plt.rcParams["font.size"] = "12"
ax =df.ffill()     .count(axis=1)     .plot(figsize=(20, 8),
          title='Forecasting the Final Number of Teams',
         color=color_pal[5], lw=5,
         xlim=('02-29-2020','05-29-2020'),
         label='Acutal Team Count by Date')
ax.set_ylabel('Number of Teams')
teamcount_pred_df['teamcount_predict'].plot(ax=ax, style='.-.', alpha=0.5, label='Regression Using All Data')
teamcount_pred_df['teamcount_predict_recent'].plot(ax=ax, style='.-.', alpha=0.5, label='Regression Using last 1000 observations')
plt.legend()
plt.axvline(pd.to_datetime('05-18-2020'), color='orange', linestyle='-.')
plt.text(pd.to_datetime('05-18-2020'), 500,'Merger Deadline',rotation=-90)
plt.axvline(pd.to_datetime('05-25-2020'), color='orange', linestyle='-.')
plt.text(pd.to_datetime('05-25-2020'), 500,'Final Deadline',rotation=-90)
plt.show()


# In[ ]:


import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
plt.rcParams["font.size"] = "10"
n_weeks = (datetime.date(2020, 5, 10) - datetime.date(2020, 2, 27)).days / 7 # Num days of the comp
n_weeks = int(n_weeks)
#n_weeks = 5
fig, axes = plt.subplots(n_weeks, 1, figsize=(15, 20), sharex=True)
#plt.subplots_adjust(top=8, bottom=2)
for x in range(n_weeks):
    date2 = df.loc[df.index.date == datetime.date(2020, 2, 28) + datetime.timedelta(x*7+1)].index.min()
    num_teams = len(df.ffill().loc[date2].dropna())
    max_cutoff = df.ffill().loc[date2] < 5
#     df.ffill().loc[date2].loc[max_cutoff].plot(kind='hist',
#                                bins=500,
#                                ax=axes[x],
#                                title='{} ({} Teams)'.format(date2.date().isoformat(),
#                                                             num_teams),
#                                               xlim=(0.93, 0.95))
    df.ffill().loc[date2].loc[max_cutoff]         .where(df.ffill().loc[date2].loc[max_cutoff] > 0.9)         .dropna().plot(kind='hist', bins=100, ax=axes[x],
                       title='{} ({} Teams)'.format(date2.date().isoformat(), num_teams))
#     pd.Series(df.ffill().loc[date2].loc[max_cutoff] \
#               .round(4) \
#               .value_counts()) \
#     .sort_index() \
#     .plot(ax=axes[x],
#           kind='bar',
#           title='{} ({} Teams)'.format(date2.date().isoformat(), num_teams))
    y_axis = axes[x].yaxis
    y_axis.set_label_text('')
    y_axis.label.set_visible(False)
    axes[x].grid(False)

