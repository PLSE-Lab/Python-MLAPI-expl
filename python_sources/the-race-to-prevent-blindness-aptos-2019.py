#!/usr/bin/env python
# coding: utf-8

# # This kernel is forked from [this kernel](https://www.kaggle.com/robikscube/the-race-to-predict-molecular-properties). Be sure to upvote it!
# 
# With the competition coming to and end, it would be interesting to see how the community progressed throughout the competition.
# As I didn't see a kernel like this for this competition, I decided to publish one myself. 

# # Lets track the Public LB Standings
# ## For the APTOS 2019 Blindness Detection 
# Last updated **September 7, 2019** (end of competition)

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
df = pd.read_csv('../input/aptos2019-public-lb/publicleaderboarddata/aptos2019-blindness-detection-publicleaderboard.csv')
df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
df = df.set_index(['TeamName','SubmissionDate'])['Score'].unstack(-1).T
df.columns = [name for name in df.columns]

FIFTEENTH_SCORE = df.max().sort_values(ascending=False)[15]
FIFTYTH_SCORE = df.max().sort_values(ascending=False)[50]
TOP_SCORE = df.max().sort_values(ascending=False)[0]


# # Public LB Scores of Top Teams over time

# In[ ]:


# Interative Plotly
mypal = cl.scales['9']['div']['Spectral']
colors = cl.interp( mypal, 15 )
annotations = []
init_notebook_mode(connected=True)
TOP_TEAMS = df.max().loc[df.max() > FIFTEENTH_SCORE].index.values
df_filtered = df[TOP_TEAMS].ffill()
team_ordered = df_filtered.loc[df_filtered.index.max()]     .sort_values(ascending=True).index.tolist()

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
                              text='APTOS 2019 Leaderboard Tracking',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))

layout = go.Layout(yaxis=dict(range=[0.6,TOP_SCORE]),
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


# # All competitors LB Position over Time

# In[ ]:


# Scores of all teams over time
plt.rcParams["font.size"] = "12"
ALL_TEAMS = df.columns.values
df[ALL_TEAMS[1:]].ffill().plot(figsize=(20, 10),
                           color=color_pal[0],
                           legend=False,
                           alpha=0.05,
                           ylim=(TOP_SCORE-0.5, 1),
                           title='All Teams Public LB Scores over Time')
df.ffill().max(axis=1).plot(color=color_pal[1], label='1st Place Public LB', legend=True)
plt.show()


# # Number of teams by Date

# In[ ]:


plt.rcParams["font.size"] = "12"
ax =df.ffill()     .count(axis=1)     .plot(figsize=(20, 8),
          title='Number of Teams in the Competition by Date',
         color=color_pal[5], lw=5)
ax.set_ylabel('Number of Teams')
plt.show()


# In[ ]:


team_over_time = df.ffill()     .count(axis=1)

lr = LinearRegression()
_ = lr.fit(np.array(pd.to_numeric(team_over_time.index).tolist()).reshape(-1, 1),
           team_over_time.values)

teamcount_df = pd.DataFrame(team_over_time)

teamcount_pred_df = pd.DataFrame(index=pd.date_range('06-03-2019','09-06-2019'))
teamcount_pred_df['teamcount_predict'] = lr.predict(np.array(pd.to_numeric(teamcount_pred_df.index).tolist()).reshape(-1, 1))

lr = LinearRegression()
_ = lr.fit(np.array(pd.to_numeric(team_over_time[-1000:].index).tolist()).reshape(-1, 1),
           team_over_time[-1000:].values)

teamcount_pred_df['teamcount_predict_recent'] = lr.predict(np.array(pd.to_numeric(teamcount_pred_df.index).tolist()).reshape(-1, 1))

plt.rcParams["font.size"] = "12"
ax =df.ffill()     .count(axis=1)     .plot(figsize=(20, 8),
          title='Forecasting the Final Number of Teams',
         color=color_pal[5], lw=5,
         xlim=('06-03-2019','09-07-2019'),
         label='Acutal Team Count by Date')
ax.set_ylabel('Number of Teams')
teamcount_pred_df['teamcount_predict'].plot(ax=ax, style='.-.', alpha=0.5, label='Regression Using All Data')
teamcount_pred_df['teamcount_predict_recent'].plot(ax=ax, style='.-.', alpha=0.5, label='Regression Using last 1000 observations')
plt.legend()
plt.axvline('08-29-2019', color='orange', linestyle='-.')
plt.text('08-29-2019', 1000,'Merger Deadline',rotation=-90)
plt.axvline('09-07-2019', color='orange', linestyle='-.')
plt.text('09-07-2019', 1000,'Final Deadline',rotation=-90)
plt.show()


# # Top LB Scores
# (Larger bar is better)

# In[ ]:


plt.rcParams["font.size"] = "12"
# Create Top Teams List
TOP_TEAMS = df.max().loc[df.max() > FIFTYTH_SCORE].index.values
df[TOP_TEAMS].max().sort_values(ascending=True).plot(kind='barh',
                                       xlim=(TOP_SCORE-0.1,FIFTYTH_SCORE+0.1),
                                       title='Top 50 Public LB Teams',
                                       figsize=(12, 15),
                                       color=color_pal[3])
plt.show()


# # Count of LB Submissions that improved score
# This is the count of times the person submitted and got the fun "Your score improved" notification. This is not the total submission count.

# In[ ]:


plt.rcParams["font.size"] = "12"
df[TOP_TEAMS].nunique().sort_values().plot(kind='barh',
                                           figsize=(12, 15),
                                           color=color_pal[1],
                                           title='Count of Submissions improving LB score by Team')
plt.show()


# # Distribution of Scores over time

# In[ ]:


plt.rcParams["font.size"] = "7"
n_days = (datetime.date.today() - datetime.date(2019, 6, 28)).days # Num days of the comp
fig, axes = plt.subplots(n_days, 1, figsize=(15, 10), sharex=True)
plt.subplots_adjust(top=8, bottom=2)
for x in range(n_days):
    date2 = df.loc[df.index.date == datetime.date(2019, 6, 28) + datetime.timedelta(x)].index.max()
    num_teams = len(df.ffill().loc[date2].dropna())
    max_cutoff = df.ffill().loc[date2] < 5
    df.ffill().loc[date2].loc[max_cutoff].plot(kind='hist',
                               bins=100,
                               ax=axes[x],
                               title='{} ({} Teams)'.format(date2.date().isoformat(),
                                                            num_teams))
    y_axis = axes[x].yaxis
    y_axis.set_label_text('')
    y_axis.label.set_visible(False)

