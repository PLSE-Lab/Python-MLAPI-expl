#!/usr/bin/env python
# coding: utf-8

# <a id='Top'></a>
# <center>
# <h1><u>PUBG - EDA, XGBoost</u></h1>
# <h3>Author: Robert Kwiatkowski</h3>
# </center>
# 
# ---
# ![Imgur](https://i.imgur.com/NmskNuo.jpg)
# 
# **PUBG (Player Unknown's Battlegrounds)** is a hugely successful and popular online shooter game. It's of so-called "battle royale" type - the game ends when the last team stays alive on a map.  The difference to the normal deathmatch is that after you are killed in battle royale game you're not re-spawned anymore (perma-death). Here is the [official game site](https://www.pubg.com/).
# At the moment this competition was launched there were only two maps: "Erangel" and "Miramar". Currently, there is "Vikendi" as well but it is not included in our dataset.
# 
# There were few datasets regarding this game on Kaggle before. If you want for example to see my non-parametric Survival Analysis (Kaplan-Meier method) click [here](https://www.kaggle.com/datark1/pubg-survival-analysis-kaplan-meier).
# 
# This kernel is mostly EDA oriented but we will look for some anomalies as well ( possibly cheaters).
# 
# ### CONTENT:  
# 1. [Database description](#description)<br>
# 2. [Exploratory Data Analysis](#Exploratory Data Analysis)<br>
#    2.1 [Match types](#Match types)<br>
#    2.2 [Kills and damage dealt](#Kills and damage dealt)<br>
#    2.3 [Maximum distances](#Maximum distances)<br>
#    2.4 [Driving vs. Walking](#Driving vs. Walking)<br>
#    2.5 [Weapons acquired](#Weapons acquired)<br>
#    2.6 [Correlation map](#Correlation map)<br>
# 3. [Analysis of TOP 10% of players](#TOP10%)<br>
# 4. [Baseline XGBoost and features importance](#XGB)<br>
# 
# <a id='description'></a>
# ## 1. Database description <a href='#Top' style="text-decoration: none;">^</a><br>
# 
# OK, let's see what's inside. I will load some basic libraries first.

# In[ ]:


import numpy as np                    # linear algebra
import pandas as pd                   # database manipulation
import matplotlib.pyplot as plt       # plotting libraries
import seaborn as sns                 # nice graphs and plots
import warnings                       # libraries to deal with warnings
warnings.filterwarnings("ignore")

print("pandas version: {}".format(pd.__version__))
print("numpy version: {}".format(np.__version__))
print("seaborn version: {}".format(sns.__version__))


# Reading raw training data.

# In[ ]:


train = pd.read_csv('../input/train_V2.csv')

print('There are {:,} rows and {} columns in our dataset.'.format(train.shape[0],train.shape[1]))


# The first glance at the data. Below the first 5 rows:

# In[ ]:


train.head()


# In[ ]:


train.info()


# 
# For better understanding of database below there is a columns descriptions:
# 
# *     **groupId** - Players team ID
# *     **matchId** - Match ID
# *     **assists** - Number of assisted kills. The killed is actually scored for the another teammate.
# *     **boosts** - Number of boost items used by a player. These are for example: energy dring, painkillers, adrenaline syringe.
# *     **damageDealt** - Damage dealt to the enemy
# *     **DBNOs** - Down But No Out - when you lose all your HP but you're not killed yet. All you can do is only to crawl.
# *     **headshotKills** - Number of enemies killed with a headshot
# *     **heals** - Number of healing items used by a player. These are for example: bandages, first-aid kits
# *     **killPlace** - Ranking in a match based on kills.
# *     **killPoints** - Ranking in a match based on kills points.
# *     **kills** - Number of enemy players killed.
# *     **killStreaks** - Max number of enemy players killed in a short amount of time.
# *     **longestKill** - Longest distance between player and killed enemy.
# *     **matchDuration** - Duration of a mach in seconds.
# *     **matchType** - Type of match. There are three main modes: Solo, Duo or Squad. In this dataset however we have much more categories.
# *     **maxPlace** - The worst place we in the match.
# *     **numGroups** - Number of groups (teams) in the match.
# *     **revives** - Number of times this player revived teammates.
# *     **rideDistance** - Total distance traveled in vehicles measured in meters.
# *     **roadKills** - Number of kills from a car, bike, boat, etc.
# *     **swimDistance** - Total distance traveled by swimming (in meters).
# *     **teamKills** - Number teammate kills (due to friendly fire).
# *     **vehicleDestroys** - Number of vehicles destroyed.
# *     **walkDistance** - Total distance traveled on foot measured (in meters).
# *     **weaponsAcquired** - Number of weapons picked up.
# *     **winPoints** - Ranking in a match based on won matches.
# 
# And our target column:
# *     **winPlacePerc** - Normalised placement (rank). The 1st place is 1 and the last one is 0.
# 
# 

# Let's create some basic descriptive statistics for each column. These will be usefull to set the visualisation parameters, to filter out the outliers and to get the feeling about the ranges/scales.

# In[ ]:


train.describe()


# Now, let's check if there are any missing data.

# In[ ]:


missing_data = train.isna().sum().to_frame()
missing_data.columns=["Missing data"]


# <a id='Exploratory Data Analysis'></a>
# ## 2 Exploratory Data Analysis <a href='#Top' style="text-decoration: none;">^</a><br>
# 
# Nice - it looks we do not have any missing values. That's a perfect starting point for EDA and for ML as well.

# <a id='Match types'></a>
# ### 2.1 Match types <a href='#Top' style="text-decoration: none;">^</a><br>

# In[ ]:


no_matches = train.loc[:,"matchId"].nunique()
print("There are {} matches registered in our database.".format(no_matches))


# In[ ]:


m_types = train.loc[:,"matchType"].value_counts().to_frame().reset_index()
m_types.columns = ["Type","Count"]
m_types


# In PUBG there are essentially three main modes of game: **Solo**, **Duo** and **Squad**. 
# 
# In a squad mode, you play in a group of 4 players. Here we can see that the match types are further broken down taking into account view modes:
# * FPP - First Person Perspective
# * TPP - Thirst Peron Perspective
# * Normal - you can switch between views during a game
# However, I am not able to identify what flare- and crash- types of matches are.

# In[ ]:


plt.figure(figsize=(15,8))
ticks = m_types.Type.values
ax = sns.barplot(x="Type", y="Count", data=m_types)
ax.set_xticklabels(ticks, rotation=60, fontsize=14)
ax.set_title("Match types")
plt.show()


# The graph above shows that the most popular game modes are squad and duo. Next I will aggregate all these individual types into three main categories (squad, duo and solo).

# In[ ]:


m_types2 = train.loc[:,"matchType"].value_counts().to_frame()
aggregated_squads = m_types2.loc[["squad-fpp","squad","normal-squad-fpp","normal-squad"],"matchType"].sum()
aggregated_duos = m_types2.loc[["duo-fpp","duo","normal-duo-fpp","normal-duo"],"matchType"].sum()
aggregated_solo = m_types2.loc[["solo-fpp","solo","normal-solo-fpp","normal-solo"],"matchType"].sum()
aggregated_mt = pd.DataFrame([aggregated_squads,aggregated_duos,aggregated_solo], index=["squad","duo","solo"], columns =["count"])
aggregated_mt


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(5, 5))
labels = ['squad', 'duo', 'solo']

wedges, texts, autotexts = ax1.pie(aggregated_mt["count"],textprops=dict(color="w"), autopct='%1.1f%%', startangle=90)

ax1.axis('equal')
ax1.legend(wedges, labels,
          title="Types",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=12, weight="bold")
plt.show()


# The pie chart above shows that over 54% of all the matches was played in squad mode.

# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.distplot(train["numGroups"])
ax.set_title("Number of groups")
plt.show()


# The graph allows to clearly notice distribution three spikes referring (from left) to: squad games, duo games and solo games.

# <a id='Kills and damage dealt'></a>
# ### 2.2 Kills and damage dealt <a href='#Top' style="text-decoration: none;">^</a><br>

# In[ ]:


plt.figure(figsize=(15,8))
ax1 = sns.boxplot(x="kills",y="damageDealt", data = train)
ax1.set_title("Damage Dealt vs. Number of Kills")
plt.show()


# There is an obvious correlation between number of kills and damage dealt. We see also that there are some outliers, more in the lower range. As the number of kills increases number of outliers reduces - these players rather kill than wound enemies. The maximum kills is 72 which is much bigger than the wast majority of players scores.
# 

# Let's look at our kills masters:

# In[ ]:


train[train['kills']>60][["Id","assists","damageDealt","headshotKills","kills","longestKill"]]


# Now let's see at headshoots statistics as this is one of the most satisfying thing you can score during a game. Players without any headshoot kills are filtered out.

# In[ ]:


headshots = train[train['headshotKills']>0]
plt.figure(figsize=(15,5))
sns.countplot(headshots['headshotKills'].sort_values())
print("Maximum number of headshots that the player scored: " + str(train["headshotKills"].max()))


# DBNO - Down But Not Out. How many enemies DBNOs an average player scores.

# In[ ]:


headshots = train[train['DBNOs']>0]
plt.figure(figsize=(15,5))
sns.countplot(headshots['DBNOs'].sort_values())
print("Mean number of DBNOs that the player scored: " + str(train["DBNOs"].mean()))


# Is there a correlation between DBNOs and kills?

# In[ ]:


plt.figure(figsize=(15,8))
ax2 = sns.boxplot(x="DBNOs",y="kills", data = train)
ax2.set_title("Number of DBNOs vs. Number of Kills")
plt.show()


# It seems that DBNOs are correlated with kills. That makes sense as usually if player is not killed by headshoot yu have to finish him while he's in DBNO state. Interesting is the first observation in the plot - apparently there is a number of players who scored a kill without DBNOs - this is usually a headshot or a vechicle explosion.

# In[ ]:


plt.figure(figsize=(15,8))
ax3 = sns.boxplot(x="killStreaks",y="kills", data = train)
ax3.set_title("Number of kill streaks vs. Number of Kills")
plt.show()


# <a id='Maximum distances'></a>
# ### 2.3 Maximum distances <a href='#Top' style="text-decoration: none;">^</a><br>
# 
# Range is filtered to a resonable kill distance, e.g. 200 meters. To give you the feeling about distances in the game I prepared a small comparison in the picture below. On the left side the building I'm aiming at is approximately 100m away, on the right side around 200m.
# 
# ![Imgur](https://i.imgur.com/js8kQpU.jpg)

# In[ ]:


dist = train[train['longestKill']<200]
plt.rcParams['axes.axisbelow'] = True
dist.hist('longestKill', bins=20, figsize = (16,8))
plt.show()


# In[ ]:


print("Average longest kill distance a player achieve is {:.1f}m, 95% of them not more than {:.1f}m and a maximum distance is {:.1f}m." .format(train['longestKill'].mean(),train['longestKill'].quantile(0.95),train['longestKill'].max()))


# Longest kill of 1094m seems a bit unrealistic (cheater?) but from another side with a 8x scope, a static target, very good position and a lot of luck it is possible.
# 
# To get a scale the entire Miramar map is 8x8km and 1300 meters is about like shooting from La Bendita crater to Impala city. Below the picture showing this in practice.
# ![Imgur](https://i.imgur.com/7WzRzkQ.jpg)

# <a id='Driving vs. Walking'></a>
# ### 2.4 Driving vs. Walking <a href='#Top' style="text-decoration: none;">^</a><br>
# 
# I filtered data to exclude for players who don't ride at all and don't walk.

# In[ ]:


walk0 = train["walkDistance"] == 0
ride0 = train["rideDistance"] == 0
swim0 = train["swimDistance"] == 0
print("{} of players didn't walk at all, {} players didn't drive and {} didn't swim." .format(walk0.sum(),ride0.sum(),swim0.sum()))


# Above numbers indicate that there is a significant number of players who didn't walk at all. We should think how to interpret these record. It is obvious that you have to walk just a little bit in order to play this game (to get to a car at least). Are this disconnected players? If yes they shouldn't score any points. Let's check this.

# In[ ]:


walk0_rows = train[walk0]
print("Average place of non-walking players is {:.3f}, minimum is {} and the best is {}, 95% of players has a score below {}." 
      .format(walk0_rows["winPlacePerc"].mean(), walk0_rows["winPlacePerc"].min(), walk0_rows["winPlacePerc"].max(),walk0_rows["winPlacePerc"].quantile(0.95)))
walk0_rows.hist('winPlacePerc', bins=40, figsize = (16,8))
plt.show()


# As we see most of the non-walking players score only last places. However, few of them got better places and a few even the top ones. This may be indication of presence of famous **cheaters**! Let's print couple of suspicious row.

# In[ ]:


suspects = train.query('winPlacePerc ==1 & walkDistance ==0').head()
suspects.head()


# In[ ]:


print("Maximum ride distance for suspected entries is {:.3f} meters, and swim distance is {:.1f} meters." .format(suspects["rideDistance"].max(), suspects["swimDistance"].max()))


# Interestingly, all of the columns connected to travelling are zero.

# In[ ]:


ride = train.query('rideDistance >0 & rideDistance <10000')
walk = train.query('walkDistance >0 & walkDistance <4000')
ride.hist('rideDistance', bins=40, figsize = (15,10))
walk.hist('walkDistance', bins=40, figsize = (15,10))
plt.show()


# Plots above show that players mostly walk during a game. That's obvious when you think that vehicles are usually used just to loot more locations and to get a more strategic positions for attack and defend.

# Now let's create a sum of walking, driving and swimming distances for each row.

# In[ ]:


travel_dist = train["walkDistance"] + train["rideDistance"] + train["swimDistance"]
travel_dist = travel_dist[travel_dist<5000]
travel_dist.hist(bins=40, figsize = (15,10))
plt.show()


# <a id='Weapons acquired'></a>
# ### 2.5 Weapons acquired <a href='#Top' style="text-decoration: none;">^</a><br>

# In[ ]:


print("Average number of acquired weapons is {:.3f}, minimum is {} and the maximum {}, 99% of players acquired less than weapons {}." 
      .format(train["weaponsAcquired"].mean(), train["weaponsAcquired"].min(), train["weaponsAcquired"].max(), train["weaponsAcquired"].quantile(0.99)))
train.hist('weaponsAcquired', figsize = (20,10),range=(0, 10), align="left", rwidth=0.9)
plt.show()


# <a id='Correlation map'></a>
# ### 2.6 Correlation map <a href='#Top' style="text-decoration: none;">^</a><br>

# In[ ]:


ax = sns.clustermap(train.corr(), annot=True, linewidths=.6, fmt= '.2f', figsize=(20, 15))
plt.show()


# <a id='TOP10'></a>
# ## 3. Analysis of TOP 10% of players <a href='#Top' style="text-decoration: none;">^</a><br>

# In[ ]:


top10 = train[train["winPlacePerc"]>0.9]
print("TOP 10% overview\n")
print("Average number of kills: {:.1f}\nMinimum: {}\nThe best: {}\n95% of players within: {} kills." 
      .format(top10["kills"].mean(), top10["kills"].min(), top10["kills"].max(),top10["kills"].quantile(0.95)))


# In[ ]:


plt.figure(figsize=(15,8))
ax3 = sns.boxplot(x="DBNOs",y="kills", data = top10)
ax3.set_title("NUmber of DBNOs vs. Number of Kills")
plt.show()


# Let's see their way of travelling and comare this to the overall population.

# In[ ]:


fig, ax1 = plt.subplots(figsize = (15,10))
walk.hist('walkDistance', bins=40, figsize = (15,10), ax = ax1)
walk10 = top10[top10['walkDistance']<5000]
walk10.hist('walkDistance', bins=40, figsize = (15,10), ax = ax1)

print("Average walking distance: " + str(top10['walkDistance'].mean()))


# In[ ]:


fig, ax1 = plt.subplots(figsize = (15,10))
ride.hist('rideDistance', bins=40, figsize = (15,10), ax = ax1)
ride10 = top10.query('rideDistance >0 & rideDistance <10000')
ride10.hist('rideDistance', bins=40, figsize = (15,10), ax = ax1)
print("Average riding distance: " + str(top10['rideDistance'].mean()))


# What about the longest distances at which they scored their kills?

# In[ ]:


print("On average the best 10% of players have the longest kill at {:.3f} meters, and the best score is {:.1f} meters." .format(top10["longestKill"].mean(), top10["longestKill"].max()))


# Let's see now the correlations between the variables

# In[ ]:


ax = sns.clustermap(top10.corr(), annot=True, linewidths=.5, fmt= '.2f', figsize=(20, 15))
plt.show()


# Comparison of both clustertmap, for all and TOP 10% shows that the same columns seems to be of significant importance (I assume above 0.6 or below -0.6):
# * rankPoints vs killPoints
# * kills vs. damage dealt/DBNOs/headshotKills/killPlace
# * killStreaks vs. damageDealt/killPlace/kills
# * longestKill vs. damageDealt/kills
# * walkDistance vs. killPlace
# * winPoints vs. killPonts/rankPoints
# * winPlacePerc vs. boosts/killPlace/walkDistance/weaponsAquired

# <a id='XGB'></a>
# ## 4. Baseline XGBoost and features importance <a href='#Top' style="text-decoration: none;">^</a><br>

# In[ ]:


import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

train.dropna(subset=["winPlacePerc"], inplace=True) # droping rows with missing labels

X = train.drop(["Id","groupId","matchId","matchType","winPlacePerc"], axis=1)
y = train["winPlacePerc"]

col_names = X.columns

transformer = Normalizer().fit(X)
X = transformer.transform(X)


# In[ ]:


X = pd.DataFrame(X, columns=col_names)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)


# In[ ]:


param = {
    'eta': 0.15, 
    'max_depth': 5,  
    'num_class': 2} 

steps = 20  # The number of training iterations
model = xgb.train(param, D_train, steps)


# In[ ]:


fig, ax1 = plt.subplots(figsize=(8,15))
xgb.plot_importance(model, ax=ax1)
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("MSE = {}".format(mean_squared_error(Y_test, best_preds)))


# 
# <!-- Start of Unsplash Embed Code - Centered (Embed code by @BirdyOz)-->
# <div style="width:50%; margin: 20px 20% !important;">
#     <img src="https://images.unsplash.com/photo-1557461762-e08315322e3d?ixlib=rb-1.2.1&amp;q=80&amp;fm=jpg&amp;crop=entropy&amp;cs=tinysrgb&amp;w=720&amp;fit=max&amp;ixid=eyJhcHBfaWQiOjEyMDd9" class="img-responsive img-fluid img-med" alt="grey helmet keychain selective focus photography " title="grey helmet keychain selective focus photography ">
#     <div class="text-muted" style="opacity: 0.5">
#         <small><a href="https://unsplash.com/photos/5zvziFY-yj8" target="_blank">Photo</a> by <a href="https://unsplash.com/@clintbustrillos" target="_blank">@clintbustrillos</a> on <a href="https://unsplash.com" target="_blank">Unsplash</a>, accessed 26/02/2020</small>
#     </div>
# </div>
# <!-- End of Unsplash Embed code -->
#                 
# **THE END**
