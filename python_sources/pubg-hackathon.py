#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
import seaborn as sns
import matplotlib as ml
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
ml.style.use('ggplot')

init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pbg = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
pbg.head()


# In[ ]:


pbg.tail(50)


# In[ ]:


pbg.info()


# In[ ]:


pbg.isnull().sum()


# ##### THE DATASET SEEMS WELL-MADE, SINCE IT SHOWS NO NULL VALUES EXCEPT FOR IN winPlacePerc.

# In[ ]:


pbg.describe()


# In[ ]:


# creating a copy
pg = pbg.copy()


# ##### QUALITY ISSUES :
# *  Missing value in winPlacePerc.
# 
# ##### TIDINESS ISSUES :
# *  None as of now.
# 
# ##### COMPLETENESS ISSUES
# ##### DEFINE
# *  Remove row with missing value in winPlacePerc, because it is a user specific value and we cannot estimate/guess it.
# 
# ##### CODE

# In[ ]:


pg[pg.winPlacePerc.isnull()]


# In[ ]:


pg.dropna(axis=0,inplace = True)


# ##### TEST

# In[ ]:


pg.isnull().sum()


# ##### ALL CHECKS COMPLETE. FINAL CHECK :

# In[ ]:


pbg = pg
pbg.isnull().sum()


# ##### EDA 
# Points of analysis :
# 
# 
# 
# 
# 
# 

# ##### HEATMAP TO GET A ROUGH IDEA ABOUT CORRELATED FEATURES

# In[ ]:


plt.figure(figsize=(30,20))
sns.heatmap(pbg.corr(),annot=True)
plt.show()


# ##### UNIVARIATE ANALYSIS

# In[ ]:


pbg.head(20)


# In[ ]:


pbg.columns


# ##### NUMERICAL COLUMNS :
# All columns from 'assists' to 'winPoints'.
# 
# ##### CATEGORICAL COLUMNS :
# 1. 'winPlacePerc'. 1 = Win, 0 = Lose

# ##### SOME OBSERVATIONS :
# *  'kills' is highly related with 'damageDealt', and 'killPlace', 'DBNOs', 'headshotKills', 'killStreaks', 'longestKills', and very moderately related to 'winPlacePerc'.
# *  'winPlacePerc' is highly correlated to 'walkDistance', 'weaponsAcquired', 'boosts' and moderately related to 'damageDealt', 'kills' and 'heals'.
# *  'killPoints' is highly correlated to 'winPoints'.

# In[ ]:


# REMOVE HIGHLY CORRELATED FEATURES. HIGHLY CORRELATED TO 'KILLS'. POORLY CORRELATED TO 'WINPLACEPERC'.
pbg.drop(columns=['killPlace','headshotKills','killStreaks','longestKill'],axis=1,inplace=True)
pbg.info()


# In[ ]:


pbg.head(40)


# ##### UNIVARIATE ANALYSIS

# In[ ]:


wins_modr_best = []
for val in list(pbg.winPlacePerc.unique()):
    if val > 0.45:
        wins_modr_best.append(val)
    else:
        continue
print(pbg.winPlacePerc.nunique())
print(pd.Series(wins_modr_best).nunique())


# ##### A LITTLE MORE THAN HALF OF THE WIN PERCENTAGES OBSERVED AMONG THE PEOPLE ARE THE CLOSEST TO AN ACTUAL WIN (CLOSER TO 1, 1 SIGNIFIES WIN, 0 LOSS)
# ##### DEFINE :
# How many people either won or lost ?
# ##### CODE

# In[ ]:


winner = pbg[pbg.winPlacePerc==1]
loser = pbg[pbg.winPlacePerc==0]

fig = go.Figure(data=[go.Pie(labels=['Won','Lost','Drew/Others'],
                             values=[winner.shape[0],loser.shape[0],pbg.shape[0]-(winner.shape[0]+loser.shape[0])])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)))
fig.show()


# ##### DEFINE :
# ##### TYPES OF MATCHES AND THEIR PREFERRENCE WEIGHTS.
# ##### CODE

# In[ ]:


match_types = list(pbg.matchType.value_counts().values)
labels = list(pbg.matchType.value_counts().index)

# Plot a pie chart to show which game type is more popular
fig = go.Figure(data=[go.Pie(labels=labels, values=match_types, hole=.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)))
fig.show()


# ##### CONCLUSION :
# ### So we see that Squad-FPP is the most popular match type, followed by Duo-FPP. Normal-Duo is the least played game type.
# 

# In[ ]:


pw = pbg[pbg['winPlacePerc'] == 1]
pl = pbg[pbg['winPlacePerc'] == 0]


# ##### MULTIVARIATE ANALYSIS

# ##### DEFINE :
# ##### DOES TYPE OF MATCH PLAYED HAVE ANY EFFECT ON WIN PERCENTAGE ?
# ##### CODE

# In[ ]:


for_win = list(pw.matchType.value_counts().values)
for_loss = list(pl.matchType.value_counts().values)

fig = go.Figure(data=[
    go.Bar(name='WON', x=list(pw.matchType.values), y=for_win),
    go.Bar(name='LOST', x=list(pl.matchType.values), y=for_loss)
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# ##### CONCLUSIONS :
# *  Even though Squad-FPP is the most popular match type, it has more losses to its name.
# *  Duo-FPP is the second most popular match type and has listed no cases of match loss.
# *  Squad is the 3rd most popular match type, but records more wins than Squad-FPP.
# *  Duo is an unpopular match type and is justified by the fact that it accounts to most number of losses.

# ##### RELATION BETWEEN KILLING AND WINNING : ARE THEY DEPENDENT ?
# Points of analysis:
# *  Should you kill more to win ?
# *  Is damage proportional to killing ? Or an alternative ?
# *  Does more damage mean a better win ?
# 
# We'll find a relation, if it exists, between the Killers and the Win percentage. 
# Assert or deny the following conclusion drafts :
# *  We need more kills to win.
# *  More kill means more damage.
# *  Less kills but more damage gives a better win OR the opposite.
# 
# ##### DEFINE :
# ##### KILL COUNTS FOR WIN AND LOSS
# ##### CODE

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='kills',data=pw)
plt.title("NO. OF MATCHES WON V/S NO. OF KILLS")
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='kills',data=pl)
plt.title("NO. OF MATCHES LOST V/S NO. OF KILLS")
plt.show()


# ##### CONCLUSIONS :
# We observe :
# *  A major portion of the win percentage has very low kill count(0-3)
# *  A major portion of the loss percentage has kill count == 0.
# *  However, for no kills made during the entire match, Losing the match is more common than winning it.
# *  As we move to higher values of kill count, we see that, we need a decent no. of kills to win a match (kills > 3). Kills <= 2 have been more prone to a loss.
# 
# ### So, we can conclude that no. of kills has a low but definite correlation with win/loss. Kills <= 2 are more prone to be a lost match.

# ##### DEFINE :
# Find the relation between 'kills' and 'wins'
# ##### CODE

# In[ ]:


sns.jointplot(x="winPlacePerc", y="kills", data=pbg, height=10, ratio=3, color="g")
plt.show()


# ##### CONCLUSION :
# *  winPlacePerc and kills are moderately correlated. Hence information gained from the heatmap is justified.
# *  In order to win it is absolutely necessary to fetch some kills (threshold = 3), and not lay down to cover and hide.
# *  One must not aim for lower kills. Higher kills justify a player's skill and guarantees a higher chance of winning.

# ##### DEFINE :
# ##### IF KILL COUNTS ARE SO LOW FOR WINNERS, THEN DO THEY FOCUS ON DAMAGE TO SCORE ?
# *  Find Damage metrics for kills.
# *  Find Damage metrics for losses.
# 
# ##### CODE

# In[ ]:


plt.figure(figsize=(30,20))
sns.distplot(pw['damageDealt'])
sns.distplot(pl['damageDealt'])
plt.legend(['WON','LOST'])
plt.show()


# ##### CONCLUSIONS :
# We observe :
# *  For least damages dealt, the possility of losing a match is higher than winning it.
# *  For matches that were won, the highest damage dealt went till around 6700.
# *  For matches that were lost, the highest damage dealt went till around 2900.
# *  Players are more prone to losing a match if they deal less damages.
# 
# ### So, we can conclude that amount of damage is low to moderately correlated to win.

# ##### DEFINE :
# Find relation between 'damageDealt' and 'winPlacePrec'
# ##### CODE

# In[ ]:


sns.jointplot(x="winPlacePerc", y="damageDealt", data=pbg, height=10, ratio=3, color="r")
plt.show()


# ##### CONCLUSION :
# *  winPlacePerc and damageDealt are moderately correlated. Hence information gained from the heatmap is justified.
# *  In order to win it is absolutely necessary to inflict damage as it is one of the most vital scoring areas of PUBG. It reveals the skills of a player.
# *  Players are more prone to losing a game if they do not inflict enough damage to fetch them scores.
# 
# ##### FURTHER STATISTICS
# ##### DEFINE :
# What percentage of players won their games even after zero kills and zero damages ? Is hiding still a strong competitior to conventional strategy of damage-kill-cover ?
# ##### CODE

# In[ ]:


# Percentage of zero kills winners
colors1 = ['maroon','green']
colors2 = ['yellow']
fig = go.Figure(data=[go.Pie(labels=['ZERO KILLS','OTHERS'],
                             values=[pw[pw.kills==0].shape[0],(pw.shape[0]-pw[pw.kills==0].shape[0])])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors1,line=dict(color='#000000', width=2)))
fig.show()

# Percentage of zero damage winners
fig = go.Figure(data=[go.Pie(labels=['ZERO DAMAGE','OTHERS'],
                             values=[pw[pw.damageDealt==0].shape[0],(pw.shape[0]-pw[pw.damageDealt==0].shape[0])])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors2,line=dict(color='#000000', width=2)))
fig.show()


# ##### CONCLUSION :
# *  Only 13.1% of players acheive victory with zero kills, which is not a convincing number.
# *  Only 3.74% of players acheive victory with zero damage, which is not a convincing number at all.
# *  It is the best to follow the threshold and try to play by acheiving kills and inflicting enough damage to ensure victory.

# ##### DEFINE :
# ##### COMPARISON BETWEEN RUNNING, DRIVING AND SWIMMING : DOES MODE OF TRAVEL/TRANSPORT AFFECT WIN PROBABILITY ? WHICH ONE TO AVOID ?

# ##### RUNNING/WALKING
# ##### DEFINE :
# Find walking distance distribution for victory and loss.
# ##### CODE

# In[ ]:


plt.figure(figsize=(30,20))
sns.distplot(pw['walkDistance'])
sns.distplot(pl['walkDistance'])
plt.legend(['WON','LOST'])
plt.show()


# ##### CONCLUSIONS :
# We observe :
# *  Maximum matches were lost when walkingDistance = 0. This can mean two things :
#     -  The player got killed even before he/she could leave.
#     -  The player was probably trying to hide instead of venturing out and got killed.
# *  Clearly, maximum wins were recorded for walking distances greater than a threshold(walkingDistance > 2000) and of course, GREATER THAN ZERO.
# *  The data goes hand in hand. The wins start peaking where the losses fade out. So, as walking distance increases :
#     -  Possibility of winning increases.
#     -  Possibility of losing a match is lesser.
# 
# ### So, we can conclude that walking distance is a good measure of determining wins. Above a certain threshold (> 2000) chances of winning a game increases. Staying idle/hidden from the very beginning of the game is not a good strategy and makes a player vulnerable to getting knocked down.

# ##### DEFINE :
# Find relation between walkingDistance and winPlacePerc
# ##### CODE

# In[ ]:


sns.jointplot(x='winPlacePerc', y='walkDistance', data=pbg, height=10, ratio=3, color="maroon")
plt.show()


# ##### CONCLUSION :
# *  winPlacePerc and walkDistance are highly correlated. Hence information gained from the heatmap(0.81) is justified.
# *  In order to win it is absolutely necessary to move and walk/run in order to take part in killing, damage and upgrade weapons and vests from fallen enemies.
# *  Players are more prone to losing a game if they do not take a step or walk short distances(very close to zero).
# 
# ##### RIDING/DRIVING
# ##### DEFINE :
# Find riding distribution for victory and loss.
# ##### CODE

# In[ ]:


plt.figure(figsize=(30,20))
sns.distplot(pw['rideDistance'])
plt.title('DISTRIBUTION OF RIDING DISTANCE OF WINNERS')
plt.show()


# ##### CONCLUSION : 
# *  We observe that majority of the winners show zero rideDistance. So the possible conclusion could be:
#     -  Going by the decreasing trend, it is evident that number of wins decrease with increase in rideDistance. This is expected because :
#         (a) There's the risk of roadKills from the oponent's perspective. So riding a vehicle is far more dangerous than walking.
#        So the reson for such an abnormal surge in zero rideDistance is due to players not opting for riding any vehicle.
#        
# ### So, we can conclude that riding distance is not a good measure of determining wins. Trend of victories decrease with increase in rideDistance.
# ##### DEFINE :
# Find/Reject relation between 'winPlacePerc' and 'rideDistance'
# ##### CODE

# In[ ]:


sns.jointplot(x='winPlacePerc', y='rideDistance', data=pbg, height=10, ratio=3, color="y")
plt.show()


# ##### CONCLUSION :
# *  winPlacePerc and rideDistance are lowly correlated. Hence information gained from the heatmap(0.34) is justified.
# 
# ##### SWIMMING
# ##### DEFINE :
# Find swimming distribution for victory and loss.
# ##### CODE

# In[ ]:


plt.figure(figsize=(30,20))
sns.distplot(pw['swimDistance'],kde=False)
sns.distplot(pl['swimDistance'],kde=False)
plt.legend(['WON','LOST'])
plt.show()


# ##### CONCLUSION :
# *  Almost no one swims.
# *  Even those who do swim are more prone to losing than winning.
# *  Chances of winning decrease as swimDistance increases.
# 
# ### So, swimming is not at all a good factor to determine a win. Chances of winning decrease as swimDistance increases.
# ##### DEFINE :
# Find/Reject relation between 'winPlacePerc' and 'swimDistance'
# ##### CODE

# In[ ]:


sns.jointplot(x='winPlacePerc', y='swimDistance', data=pbg, height=10, ratio=3, color="g")
plt.show()


# ##### CONCLUSION :
# *  As expected swimDistance and winPlacePerc have poor correlation, justifying heatmap data(0.15).
# 
# ##### FURTHER STATISTICS
# ##### DEFINE :
# What percentage of players won their games even after zero walkDistance, rideDistance and awimDistance ? Does victory come with choice of commutation ?
# ##### CODE

# In[ ]:


# Percentage of zero walk distance
colors1 = ['maroon','green']
colors2 = ['yellow']
colors3 = ['grey','red']
fig = go.Figure(data=[go.Pie(labels=['ZERO WALK DISTANCE','OTHERWISE'],
                             values=[pw[pw.walkDistance==0].shape[0],(pw.shape[0]-pw[pw.walkDistance==0].shape[0])])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors1,line=dict(color='#000000', width=2)))
fig.show()

# Percentage of zero riding distance
fig = go.Figure(data=[go.Pie(labels=['ZERO RIDE DISTNACE','OTHERWISE'],
                             values=[pw[pw.rideDistance==0].shape[0],(pw.shape[0]-pw[pw.rideDistance==0].shape[0])])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors2,line=dict(color='#000000', width=2)))
fig.show()

# Percentage of zero swimming distance
fig = go.Figure(data=[go.Pie(labels=['ZERO SWIM','OTHERWISE'],
                             values=[pw[pw.swimDistance==0].shape[0],(pw.shape[0]-pw[pw.swimDistance==0].shape[0])])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors3,line=dict(color='#000000', width=2)))
fig.show()


# ##### CONCLUSIONS :
# *  Most wins are recorded when walk distance > 0 (99.2%)
# *  Almost half of match losses are recorded when ride distance = 0 (49.5%)
# *  A considerate amount of losses are recorded when swim distance = 0 (15.1%)
# *  So the best strategy is to walk > swim >> ride a vehicle
# 
# ##### DEFINE :
# ##### HOW DOES BOOSTS AND HEALS AFFECT WIN PERCENTAGE ?
# ##### CODE

# In[ ]:


plt.figure(figsize=(30,20))
sns.pointplot(x='heals',y='winPlacePerc',data=pbg,color='red',alpha=0.8)
sns.pointplot(x='boosts',y='winPlacePerc',data=pbg,color='blue',alpha=0.8)
plt.legend(['BOOSTS'])
plt.xlabel('NUMBER OF HEALING/BOOSTING ITEMS USED')
plt.ylabel('Win Percentage')
plt.title('HEALS V/S BOOSTS')
plt.grid()
plt.show()


# ##### CONCLUSIONS :
# *  Boosts show an almost increasing trend. As boosts increase, win percentage also genereally increase.
# *  However, that is not the case for Healing items. They show random fluctuations, so we are unsure about its exact relation with victory.
# 
# ##### DEFINE :
# Find relation between boosts, healing items and winPlacePerc.
# ##### CODE

# In[ ]:


sns.jointplot(x='winPlacePerc', y='heals', data=pbg, height=10, ratio=3, color="maroon")
plt.show()

sns.jointplot(x='winPlacePerc', y='boosts', data=pbg, height=10, ratio=3, color="y")
plt.show()


# ##### CONCLUSION :
# *  So, even though healing and boosting are both correlated to winning, Boosts are more highly correlated(0.63) than heals(0.43).
# *  Boosts are definitely an excellent way to score more.
# 
# ##### DEFINE :
# Do weapons acquired and enemies killed affect the winPlacePerc ?
# ##### CODE

# In[ ]:


plt.figure(figsize=(30,20))
sns.pointplot(x='weaponsAcquired',y='winPlacePerc',data=pbg,color='red',alpha=0.8)
plt.xlabel('NUMBER OF  WEAPONS ACQUIRED')
plt.ylabel('Win Percentage')
plt.title('Weapons Acquired')
plt.grid()
plt.show()


# ##### The graph is too fluctuating.

# In[ ]:


# WEAPONS ACQUIRED
sns.jointplot(x='winPlacePerc', y='weaponsAcquired', data=pbg, height=10, ratio=3, color="b")
plt.show()

# DBNOs
sns.jointplot(x='winPlacePerc', y='DBNOs', data=pbg, height=10, ratio=3, color="violet")
plt.show()


# ###### CONCLUSION :
# *  weaponsAcquired has low-moderate correlation with winning. Though it is not a vital factor affecting vcitory, but an existent one.
# *  DBNOs has a low correlation with winning. Probably because some battles are against stronger opponents.
# 

# ### ENDING CONCLUSIONS :
# 
# PUBG is a worldwide famous multiplayer game that has taken the world by a storm. Every PUBG player will agree to the immense, unparalleled happiness washing through on winning a victory(a 'Chicken dinner', in PUBG lingo). As much as it is relaxing, it is addictive too. So in this notebook, I have derived conclusions on some of the most important factors affecting victory in a PUBG match.
# 
# A. WHAT NOT TO DO AT PUBG ?
# 
# *  Never opt for Squad-FPP matches unless very sure of your communication skills. Analysis shows that Squad-FPP players tend to lose more games. A major reason behind them could be miscommunication, as Squad FPPs form random teams with players across the globe.
# *  Never opt for a duo match, as analysis shows none of the players under observation won in a duo match.
# *  Never stop at just one kill. Analytics shows that players with less than 2 kills got gunned down.
# *  Never hide from the very beginning of the game. Analaysis shows that people with zero walking distance lose more.
# *  Never take a ride, as you are in risk of a road kill. Also, avoid swimming too. Analysis shows they contribiute very feebly to victory.
# *  Never pick on a battle with a stronger opponent. The number of opponents you kill is not much related to winning.
# 
# B. WHAT TO DO AT PUBG TO WIN ?
# *  Opt for Squad matches. Analysis shows they are the 3rd most popular match type, yet guarantee more victory than Squad-FPP. This is because, in squad matches, you get to form or choose your own team and communication is thus easier if you are playing with your friends.
# *  Opt for Duo-FPP matches, as analysis says that they offer the highest victory rate.
# *  Kill as many people as you can without getting gunned down. Analysis shows that number of kills and win percentages are moderately correlated.  If you feel the situation is safe, kill strategically. Like, if you have a shotgun, fight someone with a shotgun and not a higher level weapon. And the more you kill, the more vests and weapons you collect ;)
# *  As long as you are not in danger, analysis shows that inflicting more damage showcases a player's skills and usually ensures a good score.
# *  Prefer walking or running than swimming or riding. Data tells us that the peak of victory comes with moderately high walking distance.
# *  Indulge in picking up more boosts and healing items, as both are strongly correlated to victory.
# *  Pick battles wisely. Fight enemies you are sure you can gun down.
# 
# THANK YOU :)

# In[ ]:




