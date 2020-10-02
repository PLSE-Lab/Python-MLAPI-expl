#!/usr/bin/env python
# coding: utf-8

# # What do you need to get into the top 4 of the Premier League?

# ![](http://https://i0.wp.com/metro.co.uk/wp-content/uploads/2018/04/ars3.jpg?quality=90&strip=all&zoom=1&resize=644%2C428&ssl=1)

# ![Arsene Wenger](https://i0.wp.com/metro.co.uk/wp-content/uploads/2018/04/ars3.jpg?quality=90&strip=all&zoom=1&resize=644%2C428&ssl=1)

# Arsene Wenger once said that finishing 4th in the English Premier League is a trophy ([Source](https://www.theguardian.com/football/2012/feb/19/arsene-wenger-arsenal-fourth-place). With the riches and prestige that comes with a 4th place (and thus Champions League entry) finish means that it has become a common target for teams at the beginning of a season. 
# 
# A common question about who might "break into the top 4" is also often asked. So, using Python I have started to investigate what it would take for a team to break into the top 4. 
# I have started with an existing Kaggle dataset of the final Premier League tables for season 2007-8 to 2016-17. I've then created in the same format the tables for season 2017-18 & season 2018-19 and imported them into this notebook.
# 
# To begin with, I've concatenated all 12 tables into one pandas dataframe. This means the final position column ('#') will read 1 to 20, 1 to 20, 1 to 20 and so on. I also added a 'season' column so that in analysis that involves picking a subset of rows, I can see which season's the rows are from.
# 
# I've printed the first 30 rows (i.e. to show the entire first table and half the second) as an example.
# 
# Key: 
# * '#' = Final position in table
# * MP = matches played
# * W = Wins
# * D = Draws
# * L = Losses
# * F = Goals for
# * A = Goals against
# * GD = Goal difference (goals for minus goals against)
# * P = Points (3 points for a win, 1 for a draw, 0 for a loss)
# 

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

season0708 = pd.read_csv('../input/english-premier-league-tables/epl20072008.csv')
season0809 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20082009.csv')
season0910 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20092010.csv')
season1011 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20102011.csv')
season1112 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20112012.csv')
season1213 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20122013.csv')
season1314 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20132014.csv')
season1415 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20142015.csv')
season1516 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20152016.csv')
season1617 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20162017.csv')
season1718 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20172018.csv')
season1819 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20182019.csv')


all_data = pd.concat([season0708, season0809, season0910, season1011, season1112, season1213, season1314, season1415, season1516, season1617, season1718, season1819], ignore_index=True)
all_data.rename(columns={"D.1": "GD"}, inplace=True)
print(all_data.head(30))


# ## How difficult is it to break into the top 4? 
# Over the past 12 seasons, only 7 teams have managed to finish in the top 4 as shown below along with the amount of times each team has managed this feat:

# In[ ]:


top_4_teams = (all_data.loc[all_data['#'] < 5])

print(top_4_teams.iloc[:, 0:2].groupby('Team').count())
print('\n')
print('Only ' + str(all_data['Team'].nunique()) + ' teams have competed in the Premier League over the past 12 seasons')
print('Therefore only ' + "%.2f%%" % (7/38 * 100) + ' of the teams that have competed have made the top 4.')


# ## Exploratory analysis of previous 12 Premier League seasons
# An initial exploratory analysis of the averages across all our data categories produced some clear trends. The table below shows the average for each factor in teams finishing in one of the top 5 places (top 5 shown to distinguish distance in performance between top 4 and 5th place).
# The "MP" column can be dropped as every team, every season has played 38 games. 

# In[ ]:


seasons = ['2007/08', '2008/09', '2009/10', '2010/11', '2011/12', '2012/13', '2013/14', '2014/15', '2015/16', '2016/17', '2017/18', '2018/19']

all_data_means = all_data.groupby('#').mean()
all_data_means = all_data_means.drop(['MP'], axis=1)
all_data_described = all_data.groupby('#').describe()
all_data_described = all_data_described.drop(['MP'], axis=1)
print(all_data_means.head(5))


# By looking at the averages across the top 5 positions, we can see that, across the past 12 seasons, top 4 teams have on average:
# * Always scored more than 72 goals in a season
# * Always conceded less than 39 goals in a season
# * Always achieved more than 70 points in a season
# 
# As table position is determined by points (where points are equal then goal difference is used as a tiebreaker), we can look at the number of instances where a team has achieved more than the average points needed to finish 4th and yet has finished 5th or below. 

# In[ ]:


top_4_points_mean = all_data_means.iloc[3,5]

print(sum((all_data['P'] > top_4_points_mean) & (all_data['#'] > 4)))


# Turns out that's only happened once.
# I guess we now want to know who that team was...

# In[ ]:


print(all_data.loc[(all_data["P"] > top_4_points_mean) & (all_data["#"] > 4)])


# Bad luck Arsenal fans.

# # Last season
# If we look at last season, we can see that even though our 3 conditions that (on average) are met, they do not necessarily all need to be. 
# 

# In[ ]:


print(all_data[all_data['season'] == '2018_19'].head())


# For instance:
# * Neither Chelsea nor Tottenham met the average of 70+ goals scored, however they did meet or were within 1% of meeting the other two conditions of goals conceded and points
# * We can see that despite meeting the goals scored condition and being close on the points condition, Arsenal's goals conceded was heavily above average (34% above); limiting them to a 5th place finish. 
# 
# The visualisations below display the averages of our three conditions across all final positions. A 4th visualisation shows the goals scored by a 2nd placed team against the average goals scored for a 1st place team to show another example of where conditions can be heavily above average for the respective finishing position.

# In[ ]:


goals_scored_mean = all_data_means.iloc[0,3]

all_2nd_place = all_data[all_data['#'] == 2].reset_index()
goals_scored_mean_1st = [goals_scored_mean] * 12

plt.figure(figsize=(15,10))
ax1 = plt.subplot(2, 2, 1)
plt.plot(all_data_means['P'], color='red')
ax1.set_title('Average Points total by final position')
ax1.set_xticks(range(1,21))
ax1.set_xticklabels(range(1,21))
plt.ylabel('Total Points')
plt.xlabel('Final position')
plt.grid(True)

ax2 = plt.subplot(2, 2, 2)
plt.plot(all_data_means['F'])
ax2.set_title('Average goals scored by final position')
ax2.set_xticks(range(1,21))
ax2.set_xticklabels(range(1,21))
plt.ylabel('Total Goals scored')
plt.xlabel('Final position')
plt.grid(True)

ax3 = plt.subplot(2, 2, 3)
plt.plot(all_data_means['A'], color='green')
ax3.set_title('Average goals conceded by final position')
ax3.set_xticks(range(1,21))
ax3.set_xticklabels(range(1,21))
plt.ylabel('Total Goals conceded')
plt.xlabel('Final position')
plt.grid(True)

ax4 = plt.subplot(2, 2, 4)
plt.plot(goals_scored_mean_1st, ls='dashed')
plt.plot(all_2nd_place['F'])
ax4.set_xticks(range(0, 12))
ax4.set_xticklabels(seasons, rotation=45)
ax4.legend(['Mean goals scored 1st place team', 'Goals scored 2nd place team'],loc="upper left", prop={'size': 8})
plt.grid(True)
plt.subplots_adjust(bottom=0.15, wspace=0.5, hspace=0.5)
plt.show()


# # Next season's contenders

# Looking at the top half of last year's table, we can see how difficult it will be for new teams to break into the top 4. Looking at positions 6, 7 & 8, the need would be:
# 
# * Manchester United to concede 15 less goals
# * Wolves to score 23 - 25 more goals & concede 6 fewer
# * Everton to score 16 - 20 more goals & concede 6 fewer

# In[ ]:


print(all_data[all_data['season'] == '2018_19'].head(10))


# # Using machine learning to predict finish position

# We can validate our thinking against the entire dataset by running a mutiple linear regression analysis to predict the finishing position. 
# I am aware that due to the amount of features we are using to predict outcome (2 features; goals for and goals against), this model is extremely limited.
# 
# The below shows the predictions for Manchester United, Wolves and Everton should they meet the conditions mentioned above resulting in:
# 
# * Manchester United - For: 65 Against: 39
# * Wolves - For: 72 Against: 39
# * Everton - For: 74 Against: 39

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = all_data[['F', 'A']]

y = all_data[['#']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
mlr = LinearRegression()
model=mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)

united_team_prediction = [[65, 39]]
wolves_team_prediction = [[72, 39]]
everton_team_prediction = [[74, 39]]
predict_united = mlr.predict(united_team_prediction)
predict_wolves = mlr.predict(wolves_team_prediction)
predict_everton = mlr.predict(everton_team_prediction)

print("Manchester United predicted finishing position: " '%.1f' % predict_united)
print("Wolves predicted finishing position: " '%.1f' % predict_wolves)
print("Everton predicted finishing position: " '%.1f' % predict_everton)


# Again, a limited of this model being that as this is a league format, for one team to score more goals, another has to concede more. So the hypothesis here is not that all 3 of the above teams could break into the top 4, simply showing what they may need to do to get there. 
# As we can see, predictions would need to revised to potentially up the required goals needed to a minimum of 74 not 70 as we can see from our position estimates. 
# 
# Possibly due to the limited features used on this model, we end up with a training set score some .08 higher than the testing set. 

# In[ ]:


print(model.score(x_train,y_train))
print(model.score(x_test, y_test))
print(model.coef_)


# Finally, if we plot goals scored against finishing position we can unsurprisingly see that there is a correlation between the amount of goals scored and finishing position (groundbreaking insight there I know!)

# In[ ]:


plt.scatter(all_data[['F']], all_data[['#']], alpha=0.4)
plt.show()


# # So how do you get into the top 4?
# Simple:
# * Score 75 goals
# * Conceded less than 40
# * Get 70+ points
