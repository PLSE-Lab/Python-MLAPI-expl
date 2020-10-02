#!/usr/bin/env python
# coding: utf-8

# # Tracking Santas Helpers

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.linear_model import LinearRegression
import datetime
import colorlover as cl
import numpy as np
plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

# Format the data
df = pd.read_csv('../input/santaworkshoplb/01072020-santa-workshop-tour-2019-publicleaderboard.csv')
df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
df = df.set_index(['TeamName','SubmissionDate'])['Score'].unstack(-1).T
df.columns = [name for name in df.columns]

FIFTEENTH_SCORE = df.min().sort_values(ascending=True)[15]
FIFTYTH_SCORE = df.min().sort_values(ascending=True)[50]
TOP_SCORE = df.min().sort_values(ascending=True)[0]


# In[ ]:


plt.style.use('bmh')
plt.rcParams["font.size"] = "12"
ALL_TEAMS = df.columns.values
df_ffill = df[ALL_TEAMS[1:]].ffill()

# This is broken
df_ffill.plot(figsize=(20, 10),
              color=color_pal[0],
              legend=False,
              alpha=0.05,
              ylim=(68850-1000, 80850),
              title='All Teams Public LB Scores over Time')

df.ffill().min(axis=1).plot(color=color_pal[1], label='1st Place Public LB', legend=True)
plt.show()


# In[ ]:


plt.style.use('bmh')
plt.rcParams["font.size"] = "25"
team_over_time = df.ffill()     .count(axis=1)

lr = LinearRegression()
_ = lr.fit(np.array(pd.to_numeric(team_over_time.index).tolist()).reshape(-1, 1),
           team_over_time.values)

teamcount_df = pd.DataFrame(team_over_time)

teamcount_pred_df = pd.DataFrame(index=pd.date_range('11-20-2019','01-20-2020'))

lr = LinearRegression()
_ = lr.fit(np.array(pd.to_numeric(team_over_time[-100:].index).tolist()).reshape(-1, 1),
           team_over_time[-100:].values)

teamcount_pred_df['Forecast Using Recent Data'] = lr.predict(np.array(pd.to_numeric(teamcount_pred_df.index).tolist()).reshape(-1, 1))

plt.rcParams["font.size"] = "12"
ax =df.ffill()     .count(axis=1)     .plot(figsize=(20, 8),
          title='Forecasting the Final Number of Teams',
         color=color_pal[5], lw=5,
         xlim=('11-01-2019','01-20-2020'))
teamcount_pred_df['Forecast Using Recent Data'].plot(ax=ax, style='.-.', alpha=0.5, label='Regression Using last 1000 observations')
ax.set_ylabel('Number of Teams')
teamcount_pred_df.plot(ax=ax, style='.-.', alpha=0.5)
plt.axvline('1-8-2020', color='orange', linestyle='-.')
plt.text('1-8-2020', 900,'Merger Deadline',rotation=-90)
plt.axvline('1-15-2020', color='orange', linestyle='-.')
plt.text('1-15-2020', 500,'Deadline',rotation=-90)
plt.show()

