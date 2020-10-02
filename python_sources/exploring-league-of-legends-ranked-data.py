#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Processing League of Legends Data
# Looking over League of Legends statistics confirms whether or not certain game perspectives/knowledge is validated by data. This can potentially debunk factors that contribute to winning games.

# ![](https://ddragon.leagueoflegends.com/cdn/img/champion/splash/Aphelios_1.jpg)

# # Data Exploration:

# In[ ]:


filepath = '/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv'
lol_df = pd.read_csv(filepath, index_col='gameId')


# In[ ]:


lol_df.head()


# **Checking Available Columns to Explore:**

# In[ ]:


lol_df.columns


# **Filtering Column Names that only concern the Blue Side of the map.**

# In[ ]:


blueCols = lol_df.columns[lol_df.columns.str.contains(pat = 'blue')] 


# In[ ]:


abs(lol_df.corr())['blueWins'][blueCols].sort_values(ascending=False)


# I found it unusual that Wards has a low correlation considering league players find vision control around objectives to be essential. I also need to take note that the data comes from only ten minutes into each match. This means that many of the wards placed at that time watch the river side of the map for potential jungle invades/ganks. This also implies that players in high diamond/master have other game fundamentals down that offset the lack of skills in warding. There are many more explanations for the low correlation score but I just listed a few insights from the top of my head.
# 
# Perhaps looking at its correlation with other factors would help me dig deeper on why this is the case.

# In[ ]:


abs(lol_df.corr())['blueWardsPlaced'][blueCols].sort_values(ascending=False)


# In[ ]:


sns.set_style("white")
sns.distplot(lol_df['blueWardsPlaced'])


# In[ ]:


sns.boxplot(lol_df.blueWardsPlaced)


# The boxplot of wards placed on the bluside shows a significant number of outliers. 

# In[ ]:


lol_df.blueWardsPlaced.describe()


# In[ ]:


q1 = lol_df.blueWardsPlaced.describe()[4]
q3 = lol_df.blueWardsPlaced.describe()[6]
iqr = q3 - q1


# In[ ]:


lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr) 
print('Range of Values that are not considered outliers:')
print(str(lower_bound) +' - ' + str(upper_bound))


# In[ ]:


outlier_condition = (lol_df['blueWardsPlaced'] > upper_bound) | (lol_df['blueWardsPlaced'] < lower_bound)
blueWard_outliers = len(lol_df['blueWardsPlaced'][outlier_condition])


# In[ ]:


print('Percentage of Blue Wards in the data that are outliers: ')
print(str(round((blueWard_outliers * 100) / lol_df.shape[0], 2)) + '%')


# Outliers of blueWardsPlaced are less than 1/4th of the data. I wanted to see if the reason behind blueWards not having a strong correlation with blueWins is cause of its variability. Maybe warding really isn't important compared to other game fundamentals such as last-hitting minions.

# Now let's move on to exploring data that are more closely related to the blue side winning more games. The primary column that we would be looking into would be the amount of Gold gathered in the match for the blueside.

# In[ ]:


sns.set_style("white")
sns.distplot(lol_df['blueGoldPerMin'])


# In[ ]:


sns.scatterplot(data=lol_df, x='blueGoldPerMin', y='blueKills', hue='blueWins')


# blueGoldPerMin and blueKills do not have clear boundaries in relation to the chances of blue side winning the game. Perhaps looking at average values and finding differences between data that contributes to winning a game and data that points to a loss would make it easier for me to make insights.

# In[ ]:


pd.pivot_table(lol_df, ['blueGoldPerMin', 'blueWardsPlaced', 'blueTotalExperience'], 'blueWins')


# In[ ]:


print('Average Gold Difference between winning and losing:')
print(round(abs(1586.411113 - 1714.526389)))

print('Average Exp. Difference between winning and losing:')
print(round(abs(17453.47161 - 18404.57789)))


# The average gold/exp advantages of the blue team turns out to have a thin line between winning and losing. 128 gold is not enough to buy a single component in-game. This just amounts to a few consumables such as health potions or control wards.
# 
# 951 exp is just enough to make an individual player reach from around level 8 to 9 or 9 to 10 
# 
# (Check this out: https://leagueoflegends.fandom.com/wiki/Experience_(champion))
# 
# This also proves that in general (or at least in the high-diamond - master elo), power gained from levelling up in a match holds more power on average than gold.

# In[ ]:


abs(lol_df.corr())['blueGoldPerMin'][blueCols].sort_values(ascending=False)


# Another aspect of data I want to explore is the impact of a the jungler in winning games.
# 
# Just a few things to note when looking over the data:
# 
# **Junglers have assorted playstyles. **
#  - Powerfarming
#  - Early Pressure/Ganking
#  - A mix of both
#  
# This means I can look at columns like the dragons and rift heralds taken before ten minutes and the amount of gold generated by the jungler versus an average laner.
# 
# Also, I can use the number of kills each team gets. This is under the assumption that "high-elo" games (high diamond-master) has lesser solo-kills in lane than their low elo counterparts.

# In[ ]:


lol_df.head()


# In[ ]:


#4 Since Supports don't aim to get gold early on in the game.
lol_df['avg_blue_gpm'] = lol_df['blueGoldPerMin'] / 4
lol_df['avg_blue_gpm'].head()


# Jungle Fullclear 24cs, total of 610 gold https://leagueoflegends.fandom.com/wiki/Jungling
# 
# 25.41 gold per cs

# In[ ]:


avg_gold_per_jungle_cs = 25.41
lol_df['avg_blue_jungle_farm'] = lol_df['blueTotalJungleMinionsKilled'] * avg_gold_per_jungle_cs


# In[ ]:


lol_df[['blueWins', 'blueTotalJungleMinionsKilled', 'avg_blue_jungle_farm']].head()


# Jungle farm data has not yet been scaled to jungle farm per minute. I'm using one of skillcapped's latest videos as basis for jungle time clears. Link is below.
# 
# https://www.youtube.com/watch?v=VzZwaVuZuYk
# 
# Ballpark estimate for diamond junglers is around 3 mins and 40 seconds average for one full clear. Which means dividng the gold by this time would give the values that I want.

# In[ ]:


avgFullClearTime = 3.67
lol_df['avg_blue_jungle_farm_per_min'] = (lol_df['blueTotalJungleMinionsKilled'] * avg_gold_per_jungle_cs) / (avgFullClearTime)


# In[ ]:


lol_df[['blueWins', 'blueTotalJungleMinionsKilled', 'avg_blue_jungle_farm_per_min']].head()


# Divided the average lane farm by 4 players since supports don't hoard gold during the early game.

# In[ ]:


lol_df['avg_blue_lane_farm_per_min'] = (lol_df['blueGoldPerMin'] - lol_df['avg_blue_jungle_farm_per_min']) / 4
lol_df[['blueGoldPerMin', 'avg_blue_lane_farm_per_min', 'avg_blue_jungle_farm_per_min']].head()


# In[ ]:


ax1 = lol_df.plot(kind='scatter', x='avg_blue_lane_farm_per_min', y='blueGoldPerMin', color='r')    
ax2 = lol_df.plot(kind='scatter', x='avg_blue_jungle_farm_per_min', y='blueGoldPerMin', color='g')    


# In[ ]:


lol_df[['blueGoldPerMin', 'avg_blue_lane_farm_per_min', 'avg_blue_jungle_farm_per_min']].head()


# In[ ]:


lol_df['blue_lane_jg_gold_diff_per_min'] = lol_df['avg_blue_lane_farm_per_min'] - lol_df['avg_blue_jungle_farm_per_min']
lol_df['blue_jg_lane_gold_diff_per_min'] = lol_df['blue_lane_jg_gold_diff_per_min'] * -1
lol_df['blue_lane_jg_gold_diff_per_min'].head()


# In[ ]:


sns.scatterplot(data=lol_df, y='blueGoldPerMin', x='blue_lane_jg_gold_diff_per_min')


# In[ ]:


print(lol_df['blueGoldPerMin'].corr(lol_df['blue_lane_jg_gold_diff_per_min']))
print(lol_df['blue_jg_lane_gold_diff_per_min'].corr(lol_df['blueKills']))


# In[ ]:


lol_df[['blueGoldPerMin', 'blue_jg_lane_gold_diff_per_min']][lol_df['blueWins'] == 1]


# A negative correlation of -0.48 confirms the fact that junglers (most likely ganking/mixed playstyle) heaviy rely on laners who are ahead to gain more gold advantages for the team. This explains the "never gank a losing lane" dogma thrown around in League of Legends.

# In[ ]:


sns.lineplot(data=lol_df, x='avg_blue_jungle_farm_per_min', y='blueGoldPerMin')


# In[ ]:


sns.lineplot(data=lol_df, x='avg_blue_lane_farm_per_min', y='blueGoldPerMin')


# # Model Creation

# Selecting X and y variables...

# In[ ]:


lol_df[blueCols].columns


# In[ ]:


abs(lol_df.corr())['blueWins'][blueCols].sort_values(ascending=False)


# In[ ]:


X_columns = lol_df[blueCols].columns[blueCols != 'blueWins']
X_list = list(X_columns)
X_list.append('avg_blue_lane_farm_per_min')
X = lol_df[X_list].copy()
y = lol_df['blueWins']

X_list


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# First, using Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
first_model = LogisticRegression(max_iter = 2000)
first_model.fit(X_train, y_train)

predictions = first_model.predict(X_test)


# In[ ]:


logistic_regression_results = pd.DataFrame(predictions)
logistic_regression_results.rename(columns={0: 'Predictions'})
logistic_regression_results['Actual'] = y_test.values
logistic_regression_results


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test.values, predictions)


# Second, DecisionTreeRegressor

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
second_model = DecisionTreeClassifier(criterion='entropy')
second_model.fit(X_train, y_train)
predictions = second_model.predict(X_test)


# In[ ]:


dectree_regression_results = pd.DataFrame()
dectree_regression_results['Predictions'] = predictions
dectree_regression_results['Actual'] = y_test.values
dectree_regression_results


# In[ ]:


accuracy_score(y_test.values, predictions)


# Third, RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

third_model = RandomForestClassifier()
third_model.fit(X_train, y_train)
predictions = third_model.predict(X_test)


# In[ ]:


randtree_regression_results = pd.DataFrame()
randtree_regression_results['Predictions'] = predictions
randtree_regression_results['Actual'] = y_test.values
randtree_regression_results


# In[ ]:


accuracy_score(predictions, y_test)

