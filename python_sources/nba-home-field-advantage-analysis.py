#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import linear_model
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


nba = pd.read_csv('../input/NBA Game Data 2016-2017.csv')


# In[ ]:


weekday = nba.groupby(by=['Week Day', 'Home/Away'])
weekday_home_away = weekday['Tm'].mean()
weekday_home_away


# In[ ]:


weekday_home_away = weekday_home_away.unstack()
weekday_home_away


# The data frame below shows the average amount the home team outscores the away team by day of the week.

# In[ ]:


weekday_home_away["Home Team Margin"] = weekday_home_away['Home']-weekday_home_away['Away']
weekday_home_away


# In[ ]:


weekday_home_away.reset_index(inplace = True)
weekday_home_away


# DayID is necessary to arrange the days of the week properly, rather than alphabetically

# In[ ]:


weekday_home_away["DayID"] = [5,1,6,0,4,2,3]


# In[ ]:


weekday_home_away.sort_values("DayID")


# In[ ]:


weekday_home_away.set_index("DayID", inplace = True)
weekday_home_away.sort_index(inplace = True)


# Bar plot of the hometeams advantage by day of the week

# In[ ]:


weekday_home_away.plot(x='Week Day', y='Home Team Margin', kind="Bar", color="Red");


# In[ ]:


nba["Net Points"] = nba["Tm"] - nba["Opp"]
nba.head()


# In[ ]:


team_summary = nba.groupby("Team")
net_points = team_summary["Net Points"].sum()
wins = team_summary["Result"].value_counts().unstack()


# In[ ]:


wins_net_points = wins.join(net_points)
wins_net_points


# Here we see that there is a very strong linear correlation between net points and the number of wins an NBA team has. Linear regression is appropriate to predict wins from net points.

# In[ ]:


wins_net_points[["W", "Net Points"]].corr()["W"]["Net Points"]


# In[ ]:


lm = linear_model.LinearRegression()
x = wins_net_points["Net Points"].values.reshape(-1, 1)
y = wins_net_points["W"].values.reshape(-1, 1)
model = lm.fit(x,y)


# This is equal to the correlation squared. It is the percent of variation in wins we can explain with net points.

# In[ ]:


lm.score(x,y)
# R^2


# The intercept is equal to 41. This is logical as a team with 0 net points is exactly average and show win half of its 82 regular season games. With a slope of approximately 0.03, we see that each additional point the offense scores, (or point prevented by the defense), that a single net point is worth about 3% of a game.

# In[ ]:


intercept = lm.intercept_
coef = lm.coef_


# Now we will find how many games each team played at home and away on each day of the week. Some teams may have an advantage from scheudling, e.g. one team gets a lot of home games on Saturday.

# In[ ]:


team_week_day = nba.groupby(["Team","Home/Away","Week Day"]).count()["G"]


# In[ ]:


weekday_home_away.set_index('Week Day', inplace = True)


# In[ ]:


team_home_away = pd.DataFrame(team_week_day).join(pd.DataFrame(weekday_home_away['Home Team Margin']))


# In[ ]:


team_home_away.reset_index(inplace = True)


# In[ ]:


team_home_away.loc[team_home_away["Home/Away"] == "Away",["Home Team Margin"]] = team_home_away.loc[team_home_away["Home/Away"] == "Away", ["Home Team Margin"]] * -1


# In[ ]:


team_home_away["Impact"]=team_home_away['G']*team_home_away["Home Team Margin"]


# In[ ]:


team_impact=team_home_away.groupby("Team")


# In[ ]:


point_impact = team_impact["Impact"].sum()


# In[ ]:


game_impact = point_impact * coef[0][0]


# In the 2016-2017 NBA regular season, each team won or lost some amount of games due to a difference in home field advantage by day of the week, presented below. We can see that the Phoenix Suns lost an entire game due to poor scheduling, partially due to playing 9 games away on Saturday, and only 3 at home.

# In[ ]:


game_impact.sort_values().plot(kind="Bar", color="Blue", figsize=(10,5));


# In[ ]:




