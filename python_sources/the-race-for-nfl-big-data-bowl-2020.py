#!/usr/bin/env python
# coding: utf-8

# # Tracking the Public Leaderboard for the NFL Competition

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

# Format the data
df = pd.read_csv('../input/nflbigdatabowl2020lb/nfl-big-data-bowl-2020-publicleaderboard_11_16_2019.csv')
df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
df = df.set_index(['TeamName','SubmissionDate'])['Score'].unstack(-1).T
df.columns = [name for name in df.columns]

FIFTEENTH_SCORE = df.min().sort_values(ascending=True)[15]
FIFTYTH_SCORE = df.min().sort_values(ascending=True)[50]
TOP_SCORE = df.min().sort_values(ascending=True)[0]


# In[ ]:


# Interative Plotly
mypal = cl.scales['9']['div']['Spectral']
colors = cl.interp( mypal, 15 )
annotations = []
init_notebook_mode(connected=True)
TOP_TEAMS = df.min().loc[df.min() < FIFTEENTH_SCORE].index.values
df_filtered = df[TOP_TEAMS].ffill()
df_filtered = df_filtered.iloc[df_filtered.index >= '10-01-2019']
team_ordered = df_filtered.min(axis=0)     .sort_values(ascending=True).index.tolist()

data = []
i = 0
for col in df_filtered[team_ordered].columns:
    data.append(go.Scatter(
                        x = df_filtered.index,
                        y = df_filtered[col],
                        name=col,
                        line=dict(color=colors[i], width=2),)
               )
    i += 1

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='NFL Big Data Bowl Leaderboard',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))

layout = go.Layout(yaxis=dict(range=[TOP_SCORE-0.0001, 0.015]),
                   hovermode='x',
                   plot_bgcolor='white',
                  annotations=annotations,
                  )
fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    legend=go.layout.Legend(
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2,
    )
)

fig.update_layout(legend_orientation="h")
fig.update_layout(template="plotly_white")
#fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
fig.update_xaxes(showgrid=False)

iplot(fig)


# In[ ]:


# Scores of top teams over time
plt.rcParams["font.size"] = "12"
ALL_TEAMS = df.columns.values
df_ffill = df[ALL_TEAMS].ffill()

# This is broken
df_ffill.plot(figsize=(20, 10),
                           color=color_pal[0],
                           legend=False,
                           alpha=0.05,
                           ylim=(TOP_SCORE-0.0001, 0.02),
                           title='All Teams Public LB Scores over Time')

df.ffill().min(axis=1).plot(color=color_pal[1], label='1st Place Public LB', legend=True)
plt.show()


# ## Teams By Date

# In[ ]:


plt.rcParams["font.size"] = "13"
ax = df.ffill()     .count(axis=1)     .plot(figsize=(20, 8),
          title='Number of Teams in the Competition by Date',
          color=color_pal[5], lw=5)
ax.set_ylabel('Number of Teams')
#ax.set_ylim('2019-10-01','2019-11-30')
plt.axvline('11-20-2019', color='orange', linestyle='-.')
#plt.text('11-20-2019', 0.1,'Merger Deadline',rotation=-90)
plt.axvline('11-27-2019', color='orange', linestyle='-.')
#plt.text('11-27-2019', 40,'Deadline',rotation=-90)
plt.show()


# In[ ]:


plt.style.use('ggplot')
plt.rcParams["font.size"] = "25"
team_over_time = df.ffill()     .count(axis=1)

lr = LinearRegression()
_ = lr.fit(np.array(pd.to_numeric(team_over_time.index).tolist()).reshape(-1, 1),
           team_over_time.values)

teamcount_df = pd.DataFrame(team_over_time)

teamcount_pred_df = pd.DataFrame(index=pd.date_range('10-09-2019','11-30-2019'))
teamcount_pred_df['Forecast Using All Data'] = lr.predict(np.array(pd.to_numeric(teamcount_pred_df.index).tolist()).reshape(-1, 1))

lr = LinearRegression()
_ = lr.fit(np.array(pd.to_numeric(team_over_time[-100:].index).tolist()).reshape(-1, 1),
           team_over_time[-100:].values)

teamcount_pred_df['Forecast Using Recent Data'] = lr.predict(np.array(pd.to_numeric(teamcount_pred_df.index).tolist()).reshape(-1, 1))

plt.rcParams["font.size"] = "12"
ax =df.ffill()     .count(axis=1)     .plot(figsize=(20, 8),
          title='Forecasting the Final Number of Teams',
         color=color_pal[5], lw=5,
         xlim=('10-01-2019','11-30-2019'))
teamcount_pred_df['Forecast Using All Data'].plot(ax=ax, style='.-.', alpha=0.5, label='Regression Using All Data')
teamcount_pred_df['Forecast Using Recent Data'].plot(ax=ax, style='.-.', alpha=0.5, label='Regression Using last 1000 observations')
ax.set_ylabel('Number of Teams')
teamcount_pred_df.plot(ax=ax, style='.-.', alpha=0.5)
plt.axvline('11-20-2019', color='orange', linestyle='-.')
plt.text('11-20-2019', 900,'Merger Deadline',rotation=-90)
plt.axvline('11-27-2019', color='orange', linestyle='-.')
plt.text('11-27-2019', 500,'Deadline',rotation=-90)
plt.show()


# # Top Leaderboard Scores

# In[ ]:


plt.rcParams["font.size"] = "12"
# Create Top Teams List
TOP_TEAMS = df.min().loc[df.min() < FIFTYTH_SCORE].index.values
df[TOP_TEAMS].min().sort_values(ascending=False).plot(kind='barh',
                                       xlim=(TOP_SCORE-0.0005, 0.014),
                                       title='Top 50 Public LB Teams',
                                       figsize=(12, 15),
                                       color=color_pal[3])
plt.show()


# In[ ]:


plt.rcParams["font.size"] = "12"
df[TOP_TEAMS].nunique().sort_values().plot(kind='barh',
                                           figsize=(12, 15),
                                           color=color_pal[1],
                                           title='Count of Submissions improving LB score by Team')
plt.show()


# In[ ]:


plt.rcParams["font.size"] = "7"
n_weeks = (datetime.date.today() - datetime.date(2019, 10, 10)).days #/ 7 # Num days of the comp
n_weeks = int(n_weeks)
fig, axes = plt.subplots(n_weeks, 1, figsize=(15, 25), sharex=True)
#plt.subplots_adjust(top=8, bottom=2)
for x in range(n_weeks):
    date2 = df.loc[df.index.date == datetime.date(2019, 10, 10) + datetime.timedelta(x+1)].index.min()
    num_teams = len(df.ffill().loc[date2].dropna())
    max_cutoff = df.ffill().loc[date2] < 0.019
    df.ffill().loc[date2].loc[max_cutoff].plot(kind='hist',
                               bins=50,
                               ax=axes[x],
                               title='{} ({} Teams)'.format(date2.date().isoformat(),
                                                            num_teams),
                                              xlim=(0.012, 0.019))
    y_axis = axes[x].yaxis
    y_axis.set_label_text('')
    y_axis.label.set_visible(False)

