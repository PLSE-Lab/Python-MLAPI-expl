#!/usr/bin/env python
# coding: utf-8

# # Paris Saint-Germain : PSG Qatar version 
# ### **Yassine Ghouzam, PhD**
# #### 21/08/2017
# 
# * **1 Introduction**
# * **2 Load and check data**
# * **3 Bet smartly**
# * **4 PSG Qatar version**    

# ## 1. Introduction
# 
# This kernel is a data exploration analysis of european soccer games from 2008 to 2016. I focused on the PSG football club with some statistics displayed about the Qatari period.
# 
# If you have other ideas, you can add a comment :)

# ## 2. Load and check data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import sqlite3
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')

sns.set(style='whitegrid', context='notebook', palette='deep')

np.random.seed(2)


# In[ ]:


con = sqlite3.connect("../input/database.sqlite")

matches = pd.read_sql_query("SELECT * FROM Match;",con)
teams = pd.read_sql_query("SELECT * FROM Team;",con)
#leagues = pd.read_sql_query("SELECT * FROM League;",con)
countries = pd.read_sql_query("SELECT * FROM Country;",con)


# In[ ]:


matches.head()


# Our matches dataset is composed of 115 columns and 25979 games. 
# Firstly, i'll drop all the columns i don't need.

# In[ ]:


matches = matches.drop(labels = ['home_player_X1','home_player_X2','home_player_X3',
                       'home_player_X4','home_player_X5','home_player_X6',
                       'home_player_X7','home_player_X8','home_player_X9',
                       'home_player_X10','home_player_X11',
                       'home_player_Y1','home_player_Y2','home_player_Y3',
                       'home_player_Y4','home_player_Y5','home_player_Y6',
                       'home_player_Y7','home_player_Y8','home_player_Y9',
                       'home_player_Y10','home_player_Y11','home_player_1',
                       'home_player_2','home_player_3','home_player_4',
                       'home_player_5','home_player_6','home_player_7',
                       'home_player_8','home_player_9','home_player_10',
                       'home_player_11','away_player_X1','away_player_X2',
                       'away_player_X3','away_player_X4',
                       'away_player_X5','away_player_X6','away_player_X7',
                       'away_player_X8','away_player_X9','away_player_X10',
                       'away_player_X11','away_player_Y2','away_player_Y3',
                       'away_player_Y4','away_player_Y5','away_player_Y6',
                       'away_player_Y7','away_player_Y1',
                       'away_player_Y8','away_player_Y9','away_player_Y10',
                       'away_player_Y11','away_player_1',
                       'away_player_2','away_player_3','away_player_4',
                       'away_player_5','away_player_6','away_player_7',
                       'away_player_8','away_player_9','away_player_10',
                       'away_player_11'],axis = 1)

matches = matches.drop(labels = ['goal','shoton','shotoff','foulcommit','card',
                                 'cross','corner','possession'],axis = 1)


# In[ ]:


matches.isnull().sum()


# Features from B365H to BSA are bet odds from different betting sites. XXH are bet odd for home team win, XXD are for draw game bet odd and XXA are bet odd for away team win.
# 
# Ill drop bet features with too much NAs (>3500)

# In[ ]:


matches = matches.drop(labels = ['GBH','GBD','GBA','BSH','BSD','BSA','SJH','SJD','SJA',
                       'PSH','PSD','PSA'],axis = 1)


# Then i drop rows (games) with NAs.

# In[ ]:


matches = matches.dropna(axis = 0).reset_index(drop=True)

matches.shape


# We have 22432 games, which is larger enough for robust statistics.

# In[ ]:


matches.isnull().sum()


# The dataset is ready, no missing value remains. 
# 
# ## 3. Bet smartly
# 
# Before analysing the PSG stats, i wanted to clarify something.
# 
# With friends i used to debate about the most predictible league. 
# French, Spain, German, English, Italian ? To anwser this question i'll compare the bet odds with the real results and look at the accuracy.

# In[ ]:


matches = matches.merge(countries,left_on="league_id",right_on="id")

matches = matches.drop(labels = ["id_y","league_id","country_id"],axis=1)

matches = matches.rename(columns={'name':'league_country'})


# I merged the countries dataset with matches dataset in order to get leagues countries.

# In[ ]:


matches["result"] = (matches["home_team_goal"] - matches["away_team_goal"]).map(
         lambda s : 0 if s > 0 else 1 if s == 0 else 2 )


# For practical reasons, i made a result feature with the game result (0 = home team victory, 1 = Draw, 2 = Away team victory)

# For each Bet site B365, BW, IW, LB ... 3 odds are available. 
# 
# The minimum odd refers to the most probable victory according to the betting site, for example for PSG-OM if we have B365H = 1.8 , B365D = 3.2 and  B365A = 4.5, the team home win (here PSG) is the most probable result according to Bet365.
# 
# I took the minimum odd for each betting site (0 = predicted home team victory, 1 = predicted Draw, 2 = predicted Away team victory)

# In[ ]:


matches["B365"] = np.argmin(matches[["B365H","B365D","B365A"]].values,axis = 1)
matches["BW"] = np.argmin(matches[["BWH","BWD","BWA"]].values,axis = 1)
matches["IW"] = np.argmin(matches[["IWH","IWD","IWA"]].values,axis = 1)
matches["LB"] = np.argmin(matches[["LBH","LBD","LBA"]].values,axis = 1)
matches["WH"] = np.argmin(matches[["WHH","WHD","WHA"]].values,axis = 1)
matches["VC"] = np.argmin(matches[["VCH","VCD","VCA"]].values,axis = 1)


# Let's see now what is the best betting site.

# In[ ]:


# The most accurate betting sites

accuracy_score(matches["result"],matches["B365"])


# In[ ]:


accuracy_score(matches["result"],matches["BW"])


# In[ ]:


accuracy_score(matches["result"],matches["IW"])


# In[ ]:


accuracy_score(matches["result"],matches["LB"])


# In[ ]:


accuracy_score(matches["result"],matches["WH"])


# In[ ]:


accuracy_score(matches["result"],matches["VC"])


# They are very similar, bet365 is the most accurate so i'll keep this one.

# In[ ]:


# Compute accuracy in each group in the groupby pandas objects
def acc_group(y_true_desc,y_pred_desc):
    def inner(group):
        return accuracy_score(group[y_true_desc],group[y_pred_desc])
    inner.__name__ = 'acc_group'
    return inner


# I define this little function to compute the accuracy inside the groups.

# In[ ]:


matches.groupby("league_country").apply(acc_group("result","B365"))

league_seasons_accuracies = matches.groupby(("league_country","season")).apply(acc_group("result","B365"))

league_seasons_accuracies = league_seasons_accuracies.reset_index()
league_seasons_accuracies = league_seasons_accuracies.rename(columns={0:'accuracy'})


# In[ ]:


selected_countries = ["France","Spain","England","Germany","Italy"]

Five_leagues = league_seasons_accuracies[league_seasons_accuracies['league_country'].isin(selected_countries)]

g = sns.factorplot(x="season",y="accuracy",hue="league_country",data=Five_leagues,size=6,aspect=1.5)
g.set_xticklabels(rotation=45)
sns.plt.suptitle('Bet 365 accuracy for the 5 biggest soccer leagues')


# So, the french league (Ligue 1) seems to be the less predictable, followed by the German league (Bundesliga).
# 
# The Spanish league (La Liga) is the most predictable. (I have tried with the other betting sites, the results are very close to this).
# 
# @yonilev has tried a different method to compute the 'predictability' (entropy of the bet odds) (https://www.kaggle.com/yonilev/the-most-predictable-league) but we both have similar results and conclusions.
# 
# So if you're a gambler bet on French league.
# If you want the minimal risk bet on Spanish league.

# ## 4. PSG Qatar version
# 
# Now i'll show you the difference between PSG before and after the Qatar investment.

# In[ ]:


PSG_id = teams.loc[teams["team_short_name"] == 'PSG','team_api_id'].ravel()[0]


# Find the PSG id.

# In[ ]:


PSG_games = matches.loc[((matches["home_team_api_id"] == PSG_id) | (matches["away_team_api_id"] == PSG_id))]


# Then i select the PSG games.

# In[ ]:


PSG_games["date"] = pd.to_datetime(PSG_games["date"])
seasons = list(PSG_games["season"].unique())
PSG_games["season"] = pd.to_numeric(PSG_games["season"].map(lambda s: s.split("/")[0]))


# In[ ]:


PSG_goals = []
Opp_goals = []

for team_home, hg, ag in PSG_games[["home_team_api_id","home_team_goal","away_team_goal"]].values:
    if team_home == PSG_id:
        PSG_goals.append(hg)
        Opp_goals.append(ag)
    else :
        PSG_goals.append(ag)
        Opp_goals.append(hg)

PSG_games["PSG_goals"] = PSG_goals
PSG_games["Opp_goals"] = Opp_goals


# I count PSG  and Opponent goals and add them to the dataset as new features.

# In[ ]:


# 0 for win , 1 Draw and 2 for loss
PSG_games["PSG_result"] = (PSG_games["PSG_goals"] - PSG_games["Opp_goals"]).map(
         lambda s : 0 if s > 0 else 1 if s == 0 else 2)


# Just as the 'result' feature, i create a 'PSG_result' feature which refers to the PSG match result (0 == Win, 1 == Draw, 2 == Loss).

# In[ ]:


PSG_games['Investor'] = PSG_games["season"].map(lambda s: 'Qatar' if s>=2011 else 'Other')


# Qatari investors have been investing in the PSG since 2011. So i have created a new feature that reports this information.
# 
# Let's see now the difference.

# In[ ]:


results_counts = (PSG_games.groupby(['Investor'])['PSG_result']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index())

sns.set_style("darkgrid")
p = sns.barplot(x="PSG_result", y="percentage", hue="Investor", data=results_counts,alpha=0.7)
p.set_ylabel("Percentage")
p = p.set_xticklabels(["Win","Draw","Loss"])


# The Qatari investments make an important difference. PSG won almost 70% of their games since 2011 !!
# 
# Let's see now statistics during the games.

# In[ ]:


fig,axs = plt.subplots(1,2,figsize=(8,5))
g = sns.barplot(y="PSG_goals",x="Investor", data=PSG_games,ax=axs[0],alpha=0.7)
g.set_ylabel("Goals/game")
g = sns.barplot(y="Opp_goals",x="Investor", data=PSG_games,ax=axs[1],alpha=0.7)
g.set_ylabel("Goals conceded /game")
plt.tight_layout()
plt.show()
plt.gcf().clear()


# Since Qatari investments , PSG scores more than 2 goals / games and concedes less than 1 goal/games ! Impressive !
# 
# Let's see the results in the long term with the records.
# 
# I wanted to compute the best  winning streak, undefeat streak ...

# In[ ]:


#Victory_series :
def get_best_streak(PSG_results,result_match):
    best_streak = 0
    max_streak = 0
    for i in PSG_results:
        if best_streak > max_streak:
            max_streak = best_streak
        
        if i in result_match:
            best_streak += 1
        else :
            best_streak = 0
    return max_streak


# In[ ]:


# 0 == Win , 1 == Draw, 2 == Loss
PSG_streaks = []
PSG_streaks.append(get_best_streak(PSG_games["PSG_result"],result_match=[0,1]))
PSG_streaks.append(get_best_streak(PSG_games["PSG_result"],result_match=[0]))
PSG_streaks.append(get_best_streak(PSG_games["PSG_result"],result_match=[2,1]))
PSG_streaks.append(get_best_streak(PSG_games["PSG_result"],result_match=[2]))

PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] == "Qatar","PSG_result"],result_match=[0,1]))
PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] == "Qatar","PSG_result"],result_match=[0]))
PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] == "Qatar","PSG_result"],result_match=[2,1]))
PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] == "Qatar","PSG_result"],result_match=[2]))

PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] != "Qatar","PSG_result"],result_match=[0,1]))
PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] != "Qatar","PSG_result"],result_match=[0]))
PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] != "Qatar","PSG_result"],result_match=[2,1]))
PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] != "Qatar","PSG_result"],result_match=[2]))

Investors = ["All"]*4
Investors.extend(["Qatar"]*4)
Investors.extend(["Other"]*4)
Streak_type = ["Undefeat_streak","Winning_streak","Nowin_streak","Loss_streak"]*3

PSG_streaks = pd.DataFrame({'Investor':Investors,'PSG_streak':PSG_streaks,'Streak_type':Streak_type})

g = sns.barplot(x="Streak_type",hue="Investor",y="PSG_streak",data=PSG_streaks,alpha=0.8,saturation=1)
g.set_ylabel("Games")
g = g.set_xticklabels(["Undefeat streak","Winning streak","No winning streak","Losing streak"])


# - Longest winning run in Ligue 1: 10 matches . During the Qatari period.
# - Longest unbeaten run in Ligue 1: 36 matches. During the Qatari period.
# - Longest no winning run in Ligue 1: 6 matches. Outside the Qatari period.
# - Longest losing run in Ligue 1: 4 matches. Outside the Qatari period.
# 
# PSG beats all their records in ligue 1, during the qatari pariod.

# In[ ]:


colors = ['r','r','r','g','g','g','g','g']
g = sns.barplot(y="PSG_goals",x="season", data=PSG_games,estimator=sum,palette=sns.color_palette(colors),
                label='Before Quatar',saturation=1,alpha=0.7,ci=0)
g.set_ylabel("Goals / season")
g.set_xticklabels(seasons,rotation = 45)
g.legend()


# PSG scores a lot more goals per seasons since the Qatar investments.
# 
# - The PSG 2015/2016 season is the french record of scored goals (102 goals scored).

# In[ ]:


g = sns.barplot(y="Opp_goals",x="season", data=PSG_games,estimator=sum,palette=sns.color_palette(colors),
                label='Before Quatar',saturation=1,alpha=0.7,ci=0)
g.set_ylabel("Goals conceded / season")
g.set_xticklabels(seasons,rotation = 45)
g.legend()


# PSG concedes much less goals per seasons since the Qatar investments.
# 
# - The PSG 2015/2016 season is the french record (19 goals conceded).

# In[ ]:


PSG_bets = PSG_games.groupby(("season","Investor")).apply(acc_group("result","B365"))

PSG_bets = PSG_bets.reset_index()
PSG_bets = PSG_bets.rename(columns={0:'accuracy'})

g = sns.factorplot(x='season',y="accuracy",hue="Investor",data=PSG_bets,size=5,aspect=1.5)
g.set_xticklabels(seasons,rotation = 45)
g = sns.plt.suptitle('Bet 365 accuracy of PSG games')


# Since the qatari investments, it is better for you to bet on PSG. The team results are much more reliable. Bet365 had 80% of accuracy during the season 2015/2016 !! You could have made some money $$ :p.
# 
# It is now clear that the PSG became a very competitive team thanks to the qatar investments.
# 
# ## I really enjoyed writing this kernel, and explain it. So if it is helpful for you (i hope) or you liked it (i hope too), some upvotes would be very much appreciated - That will keep me motivated :)

# In[ ]:




