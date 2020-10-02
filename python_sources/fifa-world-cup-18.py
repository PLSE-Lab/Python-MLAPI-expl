#!/usr/bin/env python
# coding: utf-8

# ##**FIFA World Cup 2018**
#                                                                                                          
#   ![](http://www.lfm-radio.com/wp-content/uploads/2018/06/444823-fifaworldcup-article_image_d-1.jpg)                                                                                                                         

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

import plotly

import plotly.plotly as py
import plotly.graph_objs as go


# In[ ]:


data=pd.read_csv("../input/FIFA 2018 Statistics.csv")

data.head()


# In[ ]:


data.dtypes


# In[ ]:


# Plotting total goal attempts by teams
attempts=data.groupby('Team')['Attempts'].sum().reset_index().sort_values(by=('Attempts'),ascending=False)

plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Attempts", data=attempts)

plot1.set_xticklabels(attempts['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total attempts')
plot1.set_title('Total goal attempts by teams')


# In[ ]:


# Plotting total goals by teams
goals_by_team=data.groupby('Team')['Goal Scored'].sum().reset_index().sort_values(by=('Goal Scored'),ascending=False)

plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Goal Scored", data=goals_by_team)

plot1.set_xticklabels(goals_by_team['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total goals scored')
plot1.set_title('Total goals scored by teams')


# In[ ]:


# Plotting mean ball possession for teams

ball_possession=data.groupby('Team')['Ball Possession %'].mean().reset_index().sort_values(by=('Ball Possession %'),ascending=False)
ball_possession 

plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Ball Possession %", data=ball_possession)

plot1.set_xticklabels(ball_possession['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Ball possession')
plot1.set_title('Mean ball possession')


# In[ ]:


# Plotting total Man of the Match awards for teams

# Encoding the values for the column man of the Match
mom_1={'Man of the Match':{'Yes':1,'No':0}}
data.replace(mom_1,inplace=True)

# Converting column datatype to int
data['Man of the Match']=data['Man of the Match'].astype(int)

mom=data.groupby('Team')['Man of the Match'].sum().reset_index().sort_values(by=('Man of the Match'),ascending=False)

plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Man of the Match", data=mom)

plot1.set_xticklabels(mom['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total Man of the Matches')
plot1.set_title('Most Man of the Match awards')


# In[ ]:


# Plot of Total On-target and Off-target and blocked attempts by teams

group_attempt = data.groupby('Team')['On-Target','Off-Target','Blocked'].sum().reset_index()

# Changing the dataframe for plotting
group_attempt_sorted = group_attempt.melt('Team', var_name='Target', value_name='Value')

# Plotting the new dataset created above
plt.figure(figsize = (16, 10), facecolor = None)

sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Value", hue="Target", data=group_attempt_sorted)

plot1.set_xticklabels(group_attempt_sorted['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total Attempts')
plot1.set_title('Total On-Target, Off-Target and Blocked attempts by teams')


# In[ ]:


# Plotting Most saves by teams

saves=data.groupby('Team')['Saves'].sum().reset_index().sort_values(by=('Saves'),ascending=False)

plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
plot1 = sns.barplot(x="Team", y="Saves", data=saves)

plot1.set_xticklabels(saves['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total Saves')
plot1.set_title('Most Saves')


# In[ ]:


# Plotting Teams who did Own goals against themselves

own_goal=data.groupby('Opponent')['Own goals'].sum().reset_index().sort_values(by=('Own goals'),ascending=False)
own_goal=own_goal[own_goal['Own goals']!=0]

plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
plot1 = sns.barplot(x="Opponent", y="Own goals", data=own_goal)

plot1.set_xticklabels(own_goal['Opponent'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Own goals')
plot1.set_title('Own goals against themselves')


# In[ ]:


# Plot of total corners, free kicks and offsides for teams

corners_offsides_freekicks = data.groupby('Team')['Corners','Offsides','Free Kicks'].sum().reset_index()
corners_offsides_freekicks

# Changing the dataframe for plotting
corners_offsides_freekicks_sort = corners_offsides_freekicks.melt('Team', var_name='Target', value_name='Value')

# Plotting the new dataset created above
plt.figure(figsize = (16, 10), facecolor = None)

sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Value", hue="Target", data=corners_offsides_freekicks_sort)

plot1.set_xticklabels(corners_offsides_freekicks_sort['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Totals')
plot1.set_title('Total Corners, free kicks and offsides for teams')


# In[ ]:


# Plot of total goals conceded by teams

# Most goals conceded by teams
goals_conceded = data.groupby('Opponent')['Goal Scored'].sum().reset_index().sort_values(by=('Goal Scored'), ascending=False)

plt.figure(figsize = (16, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Opponent", y="Goal Scored", data=goals_conceded)

plot1.set_xticklabels(goals_conceded['Opponent'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total goals conceded')
plot1.set_title('Total goals conceded')


# In[ ]:


# Plot of total goals conceded by teams

# Most Yellow Cards by teams
yellow_cards = data.groupby('Team')['Yellow Card'].sum().reset_index().sort_values(by=('Yellow Card'), ascending=False)

plt.figure(figsize = (16, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Yellow Card", data=yellow_cards)

plot1.set_xticklabels(yellow_cards['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total yellow cards')
plot1.set_title('Total yellow cards')


# In[ ]:


# Lables for the Radar plot

labels=np.array(['Goal Scored', 'Attempts', 'Corners', 'Offsides', 'Free Kicks', 'Saves', 'Fouls Committed', 'Yellow Card'])

# Radar data for the Finals, "France vs Croatia"

data1=data.loc[126,labels].values
data2=data.loc[127,labels].values

# Radar data for Semi-Final 1 - "France vs Belgium"
data3=data.loc[120,labels].values
data4=data.loc[121,labels].values

# Radar data for Semi-Final 2 - "Croatia vs England"

data5=data.loc[122,labels].values
data6=data.loc[123,labels].values


# In[ ]:


# Create a radar plot for Semi-Final 2 using plotly

plotly.offline.init_notebook_mode(connected=True)

data = [
    go.Scatterpolar(
      r = data3,
      theta = labels,
      mode = 'lines',
      fill = 'toself',
      name = 'France'
    ),

    go.Scatterpolar(
      r = data4,
      theta = labels,
      mode = 'lines',
      fill = 'toself',
      name = 'Belgium'
    )
]

layout = go.Layout(
    title='Semi-Final 1 - "France vs Belgium"',
    polar = dict(
        radialaxis = dict(
          visible = True,
          range = [0, 20]
        )
      ),
    showlegend = True
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)


# In[ ]:


# Create a radar plot for Semi-Final 1 using plotly

plotly.offline.init_notebook_mode(connected=True)

data = [
    go.Scatterpolar(
      r = data5,
      theta = labels,
      mode = 'lines',
      fill = 'toself',
      name = 'Croatia'
    ),

    go.Scatterpolar(
      r = data6,
      theta = labels,
      mode = 'lines',
      fill = 'toself',
      name = 'England'
    )
]

layout = go.Layout(
    title='Semi-Final 2 - "Croatia vs England"',
    polar = dict(
        radialaxis = dict(
          visible = True,
          range = [0, 30]
        )
      ),
    showlegend = True
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)


# In[ ]:


# Create a radar plot for Finals using plotly

plotly.offline.init_notebook_mode(connected=True)

data = [
    go.Scatterpolar(
      r = data1,
      theta = labels,
      mode = 'lines',
      fill = 'toself',
      name = 'France'
    ),

    go.Scatterpolar(
      r = data2,
      theta = labels,
      mode = 'lines',
      fill = 'toself',
      name = 'Croatia'
    )
]

layout = go.Layout(
    title='Finals - "France vs Croatia"',
    polar = dict(
        radialaxis = dict(
          visible = True,
          range = [0, 20]
        )
      ),
    showlegend = True
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)

