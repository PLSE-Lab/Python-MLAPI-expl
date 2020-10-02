#!/usr/bin/env python
# coding: utf-8

# Welcome to our kernel! We ([Travis Buhrow](https://www.kaggle.com/travisbuhrow/account) and [Joel Rosenberg](https://www.kaggle.com/j7rose)) hope to take you on a journey that explains the thought process we went through as we developed a rule change recommendation for this competition. While subtle on the surface, we believe our recommendation would significantly decrease concussion rates on NFL punt plays, while still maintaining the overall spirit and excitement of the punt play.

# In[ ]:


# Import packages
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
sns.set()


# Import files
player_info = pd.read_csv('../input/player_punt_data.csv')
video_review = pd.read_csv('../input/video_review.csv', na_values='Unclear', dtype={'Primary_Partner_GSISID': np.float64}) 
video_injury_input = pd.read_csv('../input/video_footage-injury.csv')
play_info = pd.read_csv('../input/play_information.csv')
punt_player_info = pd.read_csv('../input/play_player_role_data.csv')
game_info = pd.read_csv('../input/game_data.csv')


# In[ ]:


# The next set of code is all prework for analysis

# Below contains some code from Chris Oakeson's 'Watch the Videos Showing Injury' kernel, with modifications

# For players in data with more than one position, group data field to show multiple positions
grouped_player_info = player_info.groupby('GSISID').agg({
    'Number': lambda x: ','.join(x.replace(to_replace='[^0-9]', value='', regex=True).unique()), 
    'Position': lambda x: ','.join(x.unique())})

video_injury = pd.merge(video_review, video_injury_input, left_on=['PlayID','GameKey'], right_on=['playid','gamekey'])
video_injury = pd.merge(video_injury, grouped_player_info, how='left', left_on='GSISID', right_on='GSISID')
# Merge in player positions
video_injury.rename({'Number': 'Injured Player Number(s)', 'Position': 'Injured Player Position'}, axis=1, inplace=True)
video_injury = pd.merge(video_injury, grouped_player_info, how='left', left_on='Primary_Partner_GSISID', right_on='GSISID')
video_injury.rename({'Number': 'Other Player Number(s)', 'Position': 'Other Player Position'}, axis=1, inplace=True)
# Merge in player roles for punts
video_injury = pd.merge(video_injury, punt_player_info, how='left', left_on=['GSISID', 'PlayID', 'GameKey', 'Season_Year'],
                        right_on=['GSISID', 'PlayID', 'GameKey', 'Season_Year'])
video_injury.rename({'Role': 'Injured Player Role'}, axis=1, inplace=True)
video_injury = pd.merge(video_injury, punt_player_info, how='left', left_on=['Primary_Partner_GSISID', 'PlayID', 'GameKey', 'Season_Year'],
                        right_on=['GSISID', 'PlayID', 'GameKey', 'Season_Year'])
video_injury.rename({'Role': 'Other Player Role'}, axis=1, inplace=True)
# Merge in game data
video_injury = pd.merge(video_injury, game_info, how= 'left', on=['GameKey','Season_Year', 'Week', 'Visit_Team'])
# Merge Game_Day from game data to play data
play_info = pd.merge(play_info, game_info[['GameKey', 'Season_Year', 'Game_Day']], how='left', on=['GameKey', 'Season_Year'])

# Remove unneeded columns
video_injury.drop(['gamekey', 'playid', 'season', 'PlayDescription', 'GSISID_y', 'Game_Date', 'Season_Type'], axis=1, inplace=True)
video_injury = video_injury.rename({'GSISID_x':'GSISID'}, axis=1)

# End of Chris Oakeson's 'Watch the Videos Showing Injury' kernel

#Define whether a punt actually occurred, and if so, how many yards
PuntYards = play_info['PlayDescription'].str.split("punts ", n = 1, expand = True)
PuntYards = PuntYards.astype('str')
PuntYards = PuntYards[1].str.split(" yard", n = 1, expand = True)
play_info['Punt_Yards'] = PuntYards[0]
play_info['Punt_Indicator'] = play_info['Punt_Yards'].apply(lambda x: True if x != 'None' else False)
play_info.loc[play_info.Punt_Yards == 'None', 'Punt_Yards'] = 0
play_info['Punt_Yards'] = play_info['Punt_Yards'].astype(int)

#Setup other indicator variables, set to false for now... will use later
play_info['Return_Indicator'] = False
play_info['PreSnapPenalty_Indicator'] = False
play_info['BlockedPunt_Indicator'] = False
play_info['AbortedPlay_Indicator'] = False
play_info['FakePunt_Indicator'] = False
play_info['Review_Indicator'] = False
play_info['CallUpheld_Indicator'] = False
play_info['CallReversed_Indicator'] = False
play_info['Muff_Indicator'] = False
play_info['PostSnapPenalty_Indicator'] = False
play_info['ReplayDown_Indicator'] = False
play_info['FairCatch_Indicator'] = False
play_info['OB_Indicator'] = False
play_info['Downed_Indicator'] = False
play_info['Touchback_Indicator'] = False
play_info['Fumble_Indicator'] = False
play_info['Turnover_Indicator'] = False

#Split play_info dataset into two: plays where a punt occurred, and those where a punt did not occur
punt_info = play_info[play_info['Punt_Indicator'] == True].reset_index().drop('index', axis=1)
no_punt_info = play_info[play_info['Punt_Indicator'] == False].reset_index().drop('index', axis=1)

#Work on plays where no punt occurred

#Identify what happened if there was no punt - either a Pre-Snap Penalty, a Blocked Punt, or a Fake Punt

#Identify pre-snap penalty
PreSnap = no_punt_info['PlayDescription'].str.split("[\\(||\\)]Punt formation[\\(||\\)]", n = 1, expand = True)
PreSnap = PreSnap[1].str.split(" on", n = 1, expand = True)
no_punt_info['PreSnapStart'] = PreSnap[0]
no_punt_info['PreSnapPenalty_Indicator'] = no_punt_info['PreSnapStart'].str.contains('penalty', case=False)

#Identify blocked punt
no_punt_info['BlockedPunt_Indicator'] = no_punt_info['PlayDescription'].str.contains('blocked', case=False)

#Identify aborted play - essentially, fumbles on the snap
no_punt_info['AbortedPlay_Indicator'] = no_punt_info['PlayDescription'].str.contains('aborted', case=False)

#Identify fake punt - essentially, if none of the above
no_punt_info.loc[(no_punt_info.BlockedPunt_Indicator == False) & (no_punt_info.PreSnapPenalty_Indicator == False) & (no_punt_info.AbortedPlay_Indicator == False), 'FakePunt_Indicator'] = True

#Drop data that's no longer needed
no_punt_info = no_punt_info.drop('PreSnapStart', axis=1)
del PreSnap 
del PuntYards

#Define categorical variable to show play result
no_punt_info.loc[no_punt_info.BlockedPunt_Indicator == True, 'PlayResult'] = 'Blocked Punt'
no_punt_info.loc[no_punt_info.PreSnapPenalty_Indicator == True, 'PlayResult'] = 'Pre-Snap Penalty'
no_punt_info.loc[no_punt_info.AbortedPlay_Indicator == True, 'PlayResult'] = 'Aborted Play'
no_punt_info.loc[no_punt_info.FakePunt_Indicator == True, 'PlayResult'] = 'Fake Punt'

#Work on plays where punt did occur

#Identify what happened after the punt - return, fair catch, muffed punt, or untouched punt
#Work on plays where punt did occur

#Identify what happened after the punt - return, fair catch, muffed punt, or untouched punt
#Identify if a review occurred - either brought on by replay official, or by challenge
punt_info['Review_Indicator'] = punt_info['PlayDescription'].str.contains('challenged|reviewed', case=False)
#Was the call upheld, or reversed?
punt_info['CallUpheld_Indicator'] = punt_info['PlayDescription'].str.contains('upheld', case=False)
punt_info['CallReversed_Indicator'] = punt_info['PlayDescription'].str.contains('reversed', case=False)

#Was the punt muffed?
punt_info['Muff_Indicator'] = punt_info['PlayDescription'].str.contains('muff', case=False)

#Was there some sort of fumble on the play? 
punt_info['Fumble_Indicator'] = punt_info['PlayDescription'].str.contains('fumble', case=False)

#Was there ultimately a turnover on the play? 
FumbleRecovery = punt_info['PlayDescription'].str.split("recovered by (?i)", n = 1, expand = True)
FumbleRecovery = FumbleRecovery.astype('str')
FumbleRecovery = FumbleRecovery[1].str.split("-", n = 1, expand = True)
punt_info['FumbleRecoveryPre'] = FumbleRecovery[0]
punt_info.loc[(punt_info.FumbleRecoveryPre == punt_info.Poss_Team) & (punt_info.Fumble_Indicator == True), 'Turnover_Indicator'] = True

#Drop data that's no longer needed
del FumbleRecovery
punt_info = punt_info.drop('FumbleRecoveryPre', axis=1)

#Was there a penalty during play, or after?
punt_info['PostSnapPenalty_Indicator'] = punt_info['PlayDescription'].str.contains('penalty', case=False)

#If there was a penalty, was the down replayed? 
punt_info['ReplayDown_Indicator'] = punt_info['PlayDescription'].str.contains('no play', case=False)

#Identify if a return occurred
#Was there a fair catch?
punt_info['FairCatch_Indicator'] = punt_info['PlayDescription'].str.contains('fair catch', case=False)
#Did the punt go out of bounds?
punt_info['OB_Indicator'] = punt_info['PlayDescription'].str.contains('out of bounds', case=False)
#Was the punt downed?
punt_info['Downed_Indicator'] = punt_info['PlayDescription'].str.contains('downed', case=False)
#Did the punt result in a touchback?
punt_info['Touchback_Indicator'] = punt_info['PlayDescription'].str.contains('touchback', case=False)

#If none of the four above - there must have been a return play of some sort 
punt_info.loc[(punt_info.FairCatch_Indicator == False) & (punt_info.OB_Indicator == False) & (punt_info.Downed_Indicator == False)
              & (punt_info.Touchback_Indicator == False), 'Return_Indicator'] = True

#Define categorical variable to show play result
punt_info.loc[punt_info.FairCatch_Indicator == True, 'PlayResult'] = 'Fair Catch'
punt_info.loc[(punt_info.Downed_Indicator == True) | (punt_info.OB_Indicator == True) | (punt_info.Touchback_Indicator),'PlayResult'] = 'No Catch'
#If not a fair catch or no catch, calling everything else a return play - so includes real returns, returns called back by penalty, and muffed punts
punt_info['PlayResult'] = punt_info['PlayResult'].replace(np.nan, 'Return')

#Append datasets back together
play_info = punt_info.append(no_punt_info, sort=True).reset_index().drop('index', axis=1)

#For those plays where a return occurred, identify how long the return was
ReturnYards = play_info['PlayDescription'].str.split("for ", n = 1, expand = True)
ReturnYards = ReturnYards.astype('str')
ReturnYards = ReturnYards[1].str.split(" yard", n = 1, expand = True)
play_info['Return_Yards'] = ReturnYards[0]
play_info.loc[(play_info.Return_Indicator == False),'Return_Yards'] = 0
play_info.loc[(play_info.Return_Yards.str.len() > 2),'Return_Yards'] = 0
play_info['Return_Yards'] = play_info['Return_Yards'].astype(int)

#Drop data that's no longer needed
del ReturnYards

#Create return categories, for describing the distribution of return yardages
play_info.loc[(play_info.Return_Yards < 6),'Return_Category'] = '00-05'
play_info.loc[(play_info.Return_Yards > 5) & (play_info.Return_Yards < 11),'Return_Category'] = '06-10'
play_info.loc[(play_info.Return_Yards > 10) & (play_info.Return_Yards < 16),'Return_Category'] = '11-15'
play_info.loc[(play_info.Return_Yards > 15) & (play_info.Return_Yards < 21),'Return_Category'] = '16-20'
play_info.loc[(play_info.Return_Yards > 20) & (play_info.Return_Yards < 31),'Return_Category'] = '21-30'
play_info.loc[(play_info.Return_Yards > 30),'Return_Category'] = '30+'

#Define where Punt starts, where it is caught/downed (if applicable), and where it is returned to (if applicable)
play_info['YardLineStart'] = play_info['YardLine']
YardLineStart = play_info['YardLine'].str.split(" ", n = 1, expand = True)
play_info['YardLineStartFieldSide'] = YardLineStart[0]
play_info['YardLineStartPre'] = YardLineStart[1].astype(int)
play_info.loc[(play_info.YardLineStartFieldSide == play_info.Poss_Team),'YardLineStart'] = play_info['YardLineStartPre']
play_info.loc[(play_info.YardLineStartFieldSide != play_info.Poss_Team),'YardLineStart'] = 100 - play_info['YardLineStartPre']
play_info['YardLineCatch'] = play_info['YardLineStart'] + play_info['Punt_Yards']
play_info['YardLineEnd'] = play_info['YardLineCatch'] - play_info['Return_Yards']

#Drop data that's no longer needed
del YardLineStart
play_info = play_info.drop(['YardLineStartFieldSide', 'YardLineStartPre'], axis=1)

#Create categorical variable for describing the distribution of yardages where the punt play began
play_info.loc[(play_info.YardLineStart < 11),'YardLineStart_Category'] = '00-10'
play_info.loc[(play_info.YardLineStart > 10) & (play_info.YardLineStart < 21),'YardLineStart_Category'] = '11-20'
play_info.loc[(play_info.YardLineStart > 20) & (play_info.YardLineStart < 31),'YardLineStart_Category'] = '21-30'
play_info.loc[(play_info.YardLineStart > 30) & (play_info.YardLineStart < 41),'YardLineStart_Category'] = '31-40'
play_info.loc[(play_info.YardLineStart > 40) & (play_info.YardLineStart < 51),'YardLineStart_Category'] = '41-50'
play_info.loc[(play_info.YardLineStart > 50),'YardLineStart_Category'] = '50+'

#Create categorical variable for describing the distribution of yardages where the punt was caught
play_info.loc[(play_info.YardLineCatch > 89),'YardLineCatch_Category'] = '00-10'
play_info.loc[(play_info.YardLineCatch > 79) & (play_info.YardLineCatch < 90),'YardLineCatch_Category'] = '11-20'
play_info.loc[(play_info.YardLineCatch > 69) & (play_info.YardLineCatch < 80),'YardLineCatch_Category'] = '21-30'
play_info.loc[(play_info.YardLineCatch > 59) & (play_info.YardLineCatch < 70),'YardLineCatch_Category'] = '31-40'
play_info.loc[(play_info.YardLineCatch > 49) & (play_info.YardLineCatch < 60),'YardLineCatch_Category'] = '41-50'
play_info.loc[(play_info.YardLineCatch < 50),'YardLineCatch_Category'] = '50+'

#Merge this data to injury data - what type of plays do concussions occur on?
injury = video_injury.merge(play_info, how='left', on=['Season_Year', 'GameKey', 'PlayID', 'Week', 'Game_Day'])


# Let's start by cutting right to the chase - what are we recommending, and what do we think will happen as a result?

# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

# Play types across all punts
sns.countplot(play_info.PlayResult.sort_values(), ax=axarr[0]).set_title("Play Types Across All Punts", fontsize=14)
# Play types across concussion plays
axarr[1] = sns.countplot(injury.PlayResult.sort_values()).set_title("Play Types Across Concussion Plays", fontsize=14)


# From the above - punt plays that include some sort of punt return action by the receiving team have a concussion rate (1.07%) that is **_almost 7 times greater_** than that of every other type of punt play (0.16%). Our takeaway - limit returns, and you will limit concussions.
# 
# With this insight as the motivation, our rule change recommendation is the following:
# 
# **If a fair catch is completed on a punt, the ball is advanced 10 yards from the spot of the fair catch, prior to the next snap.** 
# 
# We call it the "Fair Catch Advancement Rule". We would recommend that the proposed rule change be added to the language in Rule 10, Section 2, Article 4b of the NFL Rulebook.
# 
# Why 10 yards? See the charts below.

# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

# Return yardage distributions for all punt plays
sns.countplot(play_info[play_info['Return_Indicator']==True].Return_Category.sort_values(), ax=axarr[0]).set_title("Return Yardage Distribution - All Punt Plays", fontsize=14)
# Return yardage distributions for concussion plays
axarr[1] = sns.countplot(injury[injury['Return_Indicator']==True].Return_Category.sort_values()).set_title("Return Yardage Distribution - Concussion Plays", fontsize=14)


# Approximately 2/3 of all punt returns result in a return that is 10 yards or less. Our rule change recommendation would prevent the vast majority of these returns from occuring. 
# 
# From the injury dataset, we see that 18 of the 37 concussions occurred on these short returns. Thus, we predict that our proposed rule change would **decrease the concussion rate on punt plays by over** **_40 percent_**. 
# 
# The rest of our kernel can be divided into 2 parts: 
# 
# 1. [Data Exploration](#1) - How we came to our rule change recommendation
# 2. [NFL Hypotheticals](#2) - If our rule change recommendation was put into place, what would happen?

# # 1. Data Exploration <a id='1'></a>

# In our data exploration, we tested a number of general hypotheses we had about punt plays and concussions as we came to our rule recommendation. The highlights of our journey into the punt data are below.
# 
# We started at the highest level - how many punt plays were ran from 2016-2017? How many concussions occurred on those plays?

# In[ ]:


# Print some descriptive stats - total punt plays, total concussions, concussion rate
print("Total number of punt plays:")
print(len(play_info.index))
print("Total number of concussions on punt plays:")
print(len(video_injury.index))
print("Concussion rate:")
print("{:.2%}".format(len(video_injury.index)/len(play_info.index)))


# So, on average, from 2016-2017 we saw about 1 concussion for every 200 punt plays.
# 
# How does this compare to the average concussion rate across all plays in the NFL? 
# 
# We had a hard time finding an all-encompassing total plays statistic in the public arena, but from [Pro Football Reference](https://www.pro-football-reference.com/years/NFL/index.htm) we can use a proxy that will get us pretty close: average offensive plays run per game. Per Pro Football Reference, for 2016 and 2017, those metrics were:
# 
# 2016 - 63.4 Offensive Plays per game, per team
# 2017 - 63.1 Offensive Plays per game, per team
# 
# Let's be conservative and just say that a typical NFL game has 63+63 = 126 total plays per game. Multiply that out for the 4 * 16 = 64 preseason games per season, plus the 16 * 16 = 256 regular season games per season, plus the 11 postseason games per season, and here's what we get:
# 
# 126 * (64 + 256 + 11) * 2 = 83,412
# 
# 83,412 total NFL plays over the 2016 and 2017 seasons - and a conservative number, at that.
# 
# From the NFL's [Play Smart Play Safe](https://www.playsmartplaysafe.com/newsroom/reports/2017-injury-data/) website, we gather that there were 45 + 172 = 217 in-game concussions in 2016, and 46 + 189 = 235 in-game concussions in 2017. So, the total play concussion rate would be:
# 
# (217 + 235) / 83,412 = 0.54%
# 
# Pretty similar to the rate we saw from the punt data. However, recall that our estimate of the total plays ran over the last two years (the denominator) is conservative. Also, note that this rate includes concussions from kickoffs, which the NFL has noted were showing a likelihood of concussion that was 4-5 times greater than a typical run or pass play ([source](http://www.espn.com/nfl/story/_/id/22944176/nfl-competition-committee-says-kickoffs-made-safer-recommend-eliminating-it)).  
# 
# The NFL made great strides after the 2017 season in implementing changes to kickoff rules, and early results are encouraging ([source](http://www.espn.com/blog/nflnation/post/_/id/287320/the-nfl-rule-tweaks-saving-kickoff-from-extinction)).
# 
# With those two things in mind, we can safely say that the concussion rate on punts is showing to be higher than that of a typical play, perhaps significantly so. 
# 
# Next, we looked at concussion plays with respect to season timing. Are there any points in the season where concussions occur more frequently?

# In[ ]:


#When did concussion plays occur - regular season, preseason, or postseason?
print("Concussion Instances")
print("All Punt Plays -", len(injury.index))
print("Preseason Games -", len(injury[injury['Season_Type']=='Pre'].index))
print("Regular Season Games -", len(injury[injury['Season_Type']=='Reg'].index))
print("Concussion Rates")
print("All Punt Plays -", "{:.2%}".format(len(injury.index)/len(play_info.index)))
print("Preseason Games -", "{:.2%}".format(len(injury[injury['Season_Type']=='Pre'].index)/len(play_info[play_info['Season_Type']=='Pre'].index)))
print("Regular Season Games -", "{:.2%}".format(len(injury[injury['Season_Type']=='Reg'].index)/len(play_info[play_info['Season_Type']=='Reg'].index)))


# In[ ]:


#What week of the season are concussions occurring 
fig, ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(x="Week", hue= "Season_Type", data=injury).set_title("Concussion Occurances by Week of Season", fontsize=14)


# Here, we saw the concussion rate week over week and noticed a pretty strong climb as the season moves along, with week 15 showing the most concussion occurances. This could be attributed to the volume of intense games during that week, but with such a small sample size it's hard to say. Either way, it appears that the concussion rate grows as the season goes on, which leads us to believe that fatigue comes into play here. 
# 
# Ultimately, we could not see a way to develop a reasonable rule change to account for this trending, but we think this is something the NFL should keep in mind in general.
# 
# We also noted that concussion rates in the preseason games are higher than in the regular season games. We suspect this might be due to the nature of the preseason games - often, the players getting the most playing time in these games are players who are not sure whether they will be on the regular season roster or not, and thus these players might be willing to be a little riskier in their play, in the hopes of making a spectacular play that may lead to a spot on the roster (and a nice contract). 
# 
# While we would recommend that the NFL consider this insight as one of many in their evaluation of the effectiveness of the NFL preseason, for this competition we will not be advocating any rule changes specificially in regards to preseason play.
# 
# Next, we looked at traditional player positions as it relates to the concussion plays - what types of positions are getting hurt? What about their roles within the punt play? 

# In[ ]:


fig, axarr = plt.subplots(2, 2, figsize=(20, 12))

#Concussion Occurances by Injured Player Postion & Primary Impact Type
sns.countplot(x="Injured Player Position", hue="Primary_Impact_Type", ax=axarr[0][0], data=injury.sort_values("Injured Player Position")).set_title("Concussion Occurances by Injured Player Position & Primary Impact Type", fontsize=14)
#Concussion Occurances by Other Player Position & Primary Impact Type
sns.countplot(x="Other Player Position", hue="Primary_Impact_Type", ax=axarr[0][1], data=injury.sort_values("Injured Player Position")).set_title("Concussion Occurances by Other Player Position & Primary Impact Type", fontsize=14)

#Concussion Occurances by Injured Player Postion & Injured Player Activity
sns.countplot(x="Injured Player Position", hue="Player_Activity_Derived", ax=axarr[1][0], data=injury.sort_values("Injured Player Position")).set_title("Concussion Occurances by Injured Player Position & Activity", fontsize=14)
#Concussion Occurances by Other Player Position & Other Player Activity
axarr[1][1] = sns.countplot(x="Other Player Position", hue="Primary_Partner_Activity_Derived", data=injury.sort_values("Injured Player Position")).set_title("Concussion Occurances by Other Player Position & Activity", fontsize=14)


# The above visualizations highlight the distribution of concussions by position as to who received the concussion, as well as who else was involved in the impact that led to the concussion. Wide receiver, the most common position of the punt returner, showed high concussion instances, both as the injured player and primary partner. In our opinion, this is expected, as the player returning the punt is the one the punting team is sprinting towards to stop, thus leading to the punt returner being involved in many of the highest-impact collisions that occur during punt plays.
# 
# We also saw higher concussion instances from linebackers and tight ends - tight ends receive quite a few concussions, while LBs cause quite a few. We do not find this surprising, as these players tend to be some of the largest on the field for punt plays, and also tend to be some of the more aggressive.
# 
# We noted that concussions seem to occur pretty equally across multiple impact types (head-to-head and head-to-body), as well as across multiple activities (being tackled, tackling, being blocked, and blocking), with concussions occuring slightly more often in activities where the concussed player is the one who creates the contact - tackling and blocking.

# In[ ]:


#Injured Player Activity on Concussion Plays
print("Concussion Instances")
print("Tackling -", len(injury[injury['Player_Activity_Derived']=='Tackling'].index))
print("Blocking -", len(injury[injury['Player_Activity_Derived']=='Blocking'].index))
print("Tackled -", len(injury[injury['Player_Activity_Derived']=='Tackled'].index))
print("Blocked -", len(injury[injury['Player_Activity_Derived']=='Blocked'].index))


# We found these insights to be interesting and informative, but we did not see anything here that could be used as a means to a rule change that would limit concussions, while still keeping the structure and integrity of the punt play intact. 
# 
# The major takeaway we had from the above is that concussions occur in many ways, across many positions, in many unique situations.
# 
# What about the specific player roles in the punt play?

# In[ ]:


#Set any role with multiple options (PDL1, PDL2, PDL3, etc) to one same role, for charts
injury.loc[injury['Injured Player Role']=='PDL1', 'Injured Player Role'] = 'PDL'
injury.loc[injury['Injured Player Role']=='PDL2', 'Injured Player Role'] = 'PDL'
injury.loc[injury['Injured Player Role']=='PDL3', 'Injured Player Role'] = 'PDL'
injury.loc[injury['Injured Player Role']=='PDR1', 'Injured Player Role'] = 'PDR'
injury.loc[injury['Injured Player Role']=='PDR2', 'Injured Player Role'] = 'PDR'
injury.loc[injury['Injured Player Role']=='PDR3', 'Injured Player Role'] = 'PDR'
injury.loc[injury['Injured Player Role']=='PLL1', 'Injured Player Role'] = 'PLL'
injury.loc[injury['Injured Player Role']=='PLL2', 'Injured Player Role'] = 'PLL'
injury.loc[injury['Injured Player Role']=='PLL3', 'Injured Player Role'] = 'PLL'

injury.loc[injury['Other Player Role']=='PDL1', 'Other Player Role'] = 'PDL'
injury.loc[injury['Other Player Role']=='PDL2', 'Other Player Role'] = 'PDL'
injury.loc[injury['Other Player Role']=='PDL3', 'Other Player Role'] = 'PDL'
injury.loc[injury['Other Player Role']=='PDR1', 'Other Player Role'] = 'PDR'
injury.loc[injury['Other Player Role']=='PDR2', 'Other Player Role'] = 'PDR'
injury.loc[injury['Other Player Role']=='PDR3', 'Other Player Role'] = 'PDR'
injury.loc[injury['Other Player Role']=='PLL1', 'Other Player Role'] = 'PLL'
injury.loc[injury['Other Player Role']=='PLL2', 'Other Player Role'] = 'PLL'
injury.loc[injury['Other Player Role']=='PLL3', 'Other Player Role'] = 'PLL'

fig, axarr = plt.subplots(2, 2, figsize=(20, 12))

#Concussion Occurances by Injured Player Postion & Primary Impact Type
sns.countplot(x="Injured Player Role", hue="Primary_Impact_Type", ax=axarr[0][0], data=injury.sort_values("Injured Player Role")).set_title("Concussion Occurances by Injured Player Role & Primary Impact Type", fontsize=14)
#Concussion Occurances by Other Player Position & Primary Impact Type
sns.countplot(x="Other Player Role", hue="Primary_Impact_Type", ax=axarr[0][1], data=injury.sort_values("Injured Player Role")).set_title("Concussion Occurances by Other Player Role & Primary Impact Type", fontsize=14)

#Concussion Occurances by Injured Player Postion & Injured Player Activity
sns.countplot(x="Injured Player Role", hue="Player_Activity_Derived", ax=axarr[1][0], data=injury.sort_values("Injured Player Role")).set_title("Concussion Occurances by Injured Player Role & Activity", fontsize=14)
#Concussion Occurances by Other Player Position & Other Player Activity
axarr[1][1] = sns.countplot(x="Other Player Role", hue="Primary_Partner_Activity_Derived", data=injury.sort_values("Injured Player Role")).set_title("Concussion Occurances by Other Player Role & Activity", fontsize=14)


# Similar to the looks by position, what we found most interesting to note in the above is that punt returners show a higher occurance of being the other player in the concussion collision, than they do being the concussed player. The PLG and PRG positions also show higher concussion rates - these players tend to be the biggest players in punt formations (LBs, TEs), which explains the higher rates due to the reasoning mentioned earlier for these position players. 
# 
# We didn''t see much else in these views - again, concussions occur in a variety of ways, for a variety of player types. 
# 
# We also looked at when games are played in the context of the week, as well - do Thursday night games show something different than Sunday games? 

# In[ ]:


fig, axarr = plt.subplots(figsize=(20,6))

# Concussions by Game Day
axarr = sns.countplot(x="Game_Day", hue="Season_Type", data=injury.sort_values("Game_Day")).set_title("Counts of Concussion Occurances by Date of Week", fontsize=14)


# In[ ]:


print("Concussion Rates")
print("All Punt Plays -", "{:.2%}".format(len(injury.index)/len(play_info.index)))
print("Preseason Games -", "{:.2%}".format(len(injury[injury['Season_Type']=='Pre'].index)/len(play_info[play_info['Season_Type']=='Pre'].index)))
print("Regular Season Games -", "{:.2%}".format(len(injury[injury['Season_Type']=='Reg'].index)/len(play_info[play_info['Season_Type']=='Reg'].index)))
print("Thursday Games - Preseason", "{:.2%}".format(len(injury[(injury['Game_Day']=='Thursday') & (injury['Season_Type']=='Pre')].index)/len(play_info[(play_info['Game_Day']=='Thursday') & (play_info['Season_Type']=='Pre')].index)))
print("Friday Games - Preseason", "{:.2%}".format(len(injury[(injury['Game_Day']=='Friday') & (injury['Season_Type']=='Pre')].index)/len(play_info[(play_info['Game_Day']=='Friday') & (play_info['Season_Type']=='Pre')].index)))
print("Saturday Games - Preseason", "{:.2%}".format(len(injury[(injury['Game_Day']=='Saturday') & (injury['Season_Type']=='Pre')].index)/len(play_info[(play_info['Game_Day']=='Saturday') & (play_info['Season_Type']=='Pre')].index)))
print("Thursday Games - Regular Season", "{:.2%}".format(len(injury[(injury['Game_Day']=='Thursday') & (injury['Season_Type']=='Reg')].index)/len(play_info[(play_info['Game_Day']=='Thursday') & (play_info['Season_Type']=='Reg')].index)))
print("Saturday Games - Regular Season", "{:.2%}".format(len(injury[(injury['Game_Day']=='Saturday') & (injury['Season_Type']=='Reg')].index)/len(play_info[(play_info['Game_Day']=='Saturday') & (play_info['Season_Type']=='Reg')].index)))
print("Sunday Games - Regular Season", "{:.2%}".format(len(injury[(injury['Game_Day']=='Sunday') & (injury['Season_Type']=='Reg')].index)/len(play_info[(play_info['Game_Day']=='Sunday') & (play_info['Season_Type']=='Reg')].index)))
print("Monday Games - Regular Season", "{:.2%}".format(len(injury[(injury['Game_Day']=='Monday') & (injury['Season_Type']=='Reg')].index)/len(play_info[(play_info['Game_Day']=='Monday') & (play_info['Season_Type']=='Reg')].index)))


# We again noted the preseason impact mentioned earlier.
# 
# We also saw that Thursday night games show a considerably higher rate of concussion injury on punt plays than the average, even if only considering the regular season games. A likely reason for this would be the shortened week of preparation time for players when they have a Thursday game coming up. 
# The data here is a bit alarming, but could be attributed to a small sample size. We are not advocating for a rule change based on this insight, as the NFL would have many other aspects to consider before making scheduling adjustments. However, we do think that the NFL should carefully monitor this moving forward.
# 
# Next, we looked  into the types of plays that can occur in a punt situation. We grouped plays by the following definitions:
# 
# 1. Return - This includes any play were the receiving team makes contact with the ball and initiates a return. This could be a normal return play, a muffed catch, or a touched ball by another member of the receiving team.
# 2. No Catch - This includes any play where there is a punt, but the receiving team does not touch the ball. The punt could go out-of-bounds, be a touchback, or the ball could be downed by the punting team.
# 3. Fair Catch - This includes any play where the receiving team fair catches the ball, ending the play.
# 4. Fake Punt - This includes any play where the punting team does not punt and instead chooses to attempt to gain a first down from the punt formation.
# 5. Blocked Punt - This includes any play where the receiving team blocks the punt.
# 6. Pre-Snap Penalty - This includes any play where a penalty occurs before the snap, so no action occurs.
# 7. Aborted Play - This includes any play where the punt team loses control of the ball prior to a punt occurring. This could be a bad snap or a fumble by the punter.
# 
# Note that for any play that includes a punt action (Return, No Catch, or Fair Catch), plays where penalties occur during play are included, even those that cause the punt down to be replayed. We have intentionally defined the plays in this way, because although the penalty might cause the play to "not count", the action that occurs on those plays is similar to that of a play that would count, so the potential for a concussion injury is similar.

# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

# Play types across all punts
sns.countplot(play_info.PlayResult.sort_values(), ax=axarr[0]).set_title("Play Types Across All Punts", fontsize=14)
# Play types across concussion plays
axarr[1] = sns.countplot(injury.PlayResult.sort_values()).set_title("Play Types Across Concussion Plays", fontsize=14)


# In[ ]:


# Print some descriptive stats - concussion rate on different play results
print("Concussion Rates")
print("All Punt Plays -", "{:.2%}".format(len(injury.index)/len(play_info.index)))
print("Returns -", "{:.2%}".format(len(injury[injury['PlayResult']=='Return'].index)/len(play_info[play_info['PlayResult']=='Return'].index)))
print("No Catch -", "{:.2%}".format(len(injury[injury['PlayResult']=='No Catch'].index)/len(play_info[play_info['PlayResult']=='No Catch'].index)))
print("Fair Catch -", "{:.2%}".format(len(injury[injury['PlayResult']=='Fair Catch'].index)/len(play_info[play_info['PlayResult']=='Fair Catch'].index)))
print("Fake Punt -", "{:.2%}".format(len(injury[injury['PlayResult']=='Fake Punt'].index)/len(play_info[play_info['PlayResult']=='Fake Punt'].index)))
print("Blocked Punt -", "{:.2%}".format(len(injury[injury['PlayResult']=='Blocked Punt'].index)/len(play_info[play_info['PlayResult']=='Blocked Punt'].index)))
print("Pre-Snap Penalty -", "{:.2%}".format(len(injury[injury['PlayResult']=='Pre-Snap Penalty'].index)/len(play_info[play_info['PlayResult']=='Pre-Snap Penalty'].index)))
print("Aborted Play -", "{:.2%}".format(len(injury[injury['PlayResult']=='Aborted Play'].index)/len(play_info[play_info['PlayResult']=='Aborted Play'].index)))
print("All Punt Plays - Excluding Returns", "{:.2%}".format(len(injury[injury['PlayResult']!='Return'].index)/len(play_info[play_info['PlayResult']!='Return'].index)))


# Here, we saw what we believe to be the most important insight of this analysis - punt plays where some sort of return action occurs see a concussion rate (1.07%) that is almost 7 times greater than that of every other type of punt play (0.16%). ***7 times!***
# 
# We noted that while fake punts show a high concussion rate, this is due to a small sample size - we only observe 1 concussion in a fake punt play, but there are only 39 fake punt plays observed in the dataset.
# 
# Thus, we saw an idea emerge. It's clear that if we can implement a rule change to either reduce or fundamentally change the action that occurs on return plays, we will see a meaningful decrease to the overall concucssion rate on punt plays - across all players, not just specific player positions. This insight is the motivation behind our rule change recommendation.
#  
# In our exploration, we also wanted to consider the Next Gen tracking data, as there may be some insights to be had from player position, direction, and speed during the play.
# 
# As we explored the Next Gen data, we decided to focus on player movement throughout the play, and less on the specific speed of the player. The reasoning for this was that we could not see a path to a reasonable rule change that would be based on player speed, yet still be something that the NFL could implement. Surely, high speed collisions are a common theme among concussion occurances, but what's the NFL going to do - put a limit on how fast a player can run on the field? 
# 
# The charts below look at the movement of both the injured player and the primary partner player throughout the course of each play in the injury dataset.

# In[ ]:


#Import Next Gen data - append together, make sure to delete import files to save memory
dtypes = {"Season_Year": np.int16, "GameKey": np.int16, "PlayID": np.int16, "GSISID": np.float32, "Time": np.float32, "x": np.float32,
          "y": np.float32, "o": np.float32, "dir": np.float32, "event": str}
parse_dates = ["Time"]

ngs_2016_pre = pd.read_csv('../input/NGS-2016-pre.csv', dtype=dtypes, parse_dates=parse_dates)
ngs_2016_reg1 = pd.read_csv('../input/NGS-2016-reg-wk1-6.csv', dtype=dtypes, parse_dates=parse_dates)
ngs = ngs_2016_pre.append(ngs_2016_reg1, ignore_index=True)
del ngs_2016_pre
del ngs_2016_reg1
ngs_2016_reg2 = pd.read_csv('../input/NGS-2016-reg-wk7-12.csv', dtype=dtypes, parse_dates=parse_dates)
ngs = ngs.append(ngs_2016_reg2, ignore_index=True)
del ngs_2016_reg2
ngs_2016_reg3 = pd.read_csv('../input/NGS-2016-reg-wk13-17.csv', dtype=dtypes, parse_dates=parse_dates)
ngs = ngs.append(ngs_2016_reg3, ignore_index=True)
del ngs_2016_reg3
ngs_2016_post = pd.read_csv('../input/NGS-2016-post.csv', dtype=dtypes, parse_dates=parse_dates)
ngs = ngs.append(ngs_2016_post, ignore_index=True)
del ngs_2016_post
ngs_2017_pre = pd.read_csv('../input/NGS-2017-pre.csv', dtype=dtypes, parse_dates=parse_dates)
ngs = ngs.append(ngs_2017_pre, ignore_index=True)
del ngs_2017_pre
ngs_2017_reg1 = pd.read_csv('../input/NGS-2017-reg-wk1-6.csv', dtype=dtypes, parse_dates=parse_dates)
ngs = ngs.append(ngs_2017_reg1, ignore_index=True)
del ngs_2017_reg1
ngs_2017_reg2 = pd.read_csv('../input/NGS-2017-reg-wk7-12.csv', dtype=dtypes, parse_dates=parse_dates)
ngs = ngs.append(ngs_2017_reg2, ignore_index=True)
del ngs_2017_reg2
ngs_2017_reg3 = pd.read_csv('../input/NGS-2017-reg-wk13-17.csv', dtype=dtypes, parse_dates=parse_dates)
ngs = ngs.append(ngs_2017_reg3, ignore_index=True)
del ngs_2017_reg3
ngs_2017_post = pd.read_csv('../input/NGS-2017-post.csv', dtype=dtypes, parse_dates=parse_dates)
ngs = ngs.append(ngs_2017_post, ignore_index=True)
del ngs_2017_post


# In[ ]:


# Merge ngs data with concussion data, to get tracking data on injured player
injury_injured = injury.merge(ngs, how='inner', on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'])

# Do same for 'primary partner'
injury_partner = injury.drop('GSISID', axis=1)
injury_partner = injury_partner.merge(ngs, how='inner', left_on=['Season_Year', 'GameKey', 'PlayID', 'Primary_Partner_GSISID'],
                                                 right_on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'])

# Append datasets together
injury_ngs = injury_injured.append(injury_partner, ignore_index=True, sort=True)

# Delete dataframes, no longer needed
del injury_injured
del injury_partner


# In[ ]:


# Code below is a slight modification of the code created in Kevin Mader's 'Previewing the Games' kernel
# Code shows the patterns of injured (red) and 'primary partner' (blue) players on a given play 

football_field = lambda : Rectangle(xy=(10, 0), 
                                    width=100, 
                                    height=53.3, 
                                    color='g',
                                   alpha=0.10)

plt.style.use('ggplot')

fig, m_axs = plt.subplots(7, 6, figsize=(20, 20))
for (play_id, play_rows), c_ax in zip(injury_ngs.groupby(['GameKey', 'PlayID', 'PlayResult']), 
                      m_axs.flatten()):
    c_ax.add_patch(football_field())
    for player_id, player_rows in play_rows.groupby('GSISID'):
        player_rows = player_rows.sort_values('Time')
        c_ax.plot(player_rows['x'], player_rows['y'], 
                  label=player_id)
    c_ax.set_title(play_id)
    c_ax.set_aspect(1)
    c_ax.set_xlim(0, 120)
    c_ax.set_ylim(-10, 63)


# Upon review of the charts above, we did not see any easily identifiable trends. Some plays involve both players spending much of the play side-by-side - we suspect that these are plays where the concussion occurs on impact created through a block. Other plays, the relationship isn't as clear - perhaps the returner is being tackled, or maybe a player is coming across the field to put a block on an unsuspecting player. Again, we feel this lends to our theory that concussions on punt plays occur in a variety of ways.
# 
# What about the patterns of the punt returners on these plays?

# In[ ]:


# Merge punt role information with ngs & video injury 
punt_player_info_PR = punt_player_info[punt_player_info['Role'] == 'PR']
injury_returner = injury.drop('GSISID', axis=1)
injury_returner = injury_returner.merge(punt_player_info_PR, how='inner', left_on=['Season_Year', 'GameKey', 'PlayID'],
                                                 right_on=['Season_Year', 'GameKey', 'PlayID'])

injury_returner = injury_returner.merge(ngs, how='inner', left_on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'],
                                                 right_on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'])
# Delete dataframe, no longer needed
del punt_player_info_PR


# In[ ]:


fig, m_axs = plt.subplots(7, 6, figsize=(20, 20))
for (play_id, play_rows), c_ax in zip(injury_returner.groupby(['GameKey', 'PlayID', 'PlayResult']), 
                      m_axs.flatten()):
    c_ax.add_patch(football_field())
    for player_id, player_rows in play_rows.groupby('GSISID'):
        player_rows = player_rows.sort_values('Time')
        c_ax.plot(player_rows['x'], player_rows['y'], 
                  label=player_id)
    c_ax.set_title(play_id)
    c_ax.set_aspect(1)
    c_ax.set_xlim(0, 120)
    c_ax.set_ylim(-10, 63)


# There is one trend we noted from the punt returner views - there seems to be quite a bit more east-west (side-to-side) movement on the field by the punt returner, than there is north-south (endzone-to-endzone) movement. This makes sense to us, as punt returners spend a lot of the return trying to avoid potential tacklers. Besides that, we didn't see much for identifiable trends.
# 
# This brings us to the end of our data exploration. Our findings can be summed up like this:
# 
# **_Concussions occur in a variety of ways, but the common link for most is that they occur during the return portion of the play_**.

# # 2. NFL Hypotheticals <a id='2'></a>

# Recall our rule change proposal:
# 
# The Fair Catch Advancement Rule: **If a fair catch is completed on a punt, the ball is advanced 10 yards from the spot of the fair catch, prior to the next snap.** 
# 
# We note that our intention with this proposal is that this rule would **not apply** in situations where a fair catch is signaled, **but the catch is not completed** - this includes touchbacks, punts that go out of bounds, punts that are downed by the punting team, and muffed punts where the catch is never completed. This rule would only apply in situations where the fair catch is signaled, the ball is caught, and as a result the play is called dead by the official.
# 
# We believe that this rule change would ultimately reduce concussion instances on NFL punt plays by over **_40 percent_**. Of course, this reduction doesn't just appear out of thin air! The behaviors of various NFL stakeholders would change in some way. The question is, how?
# 
# For our rule change recommendation, we asked ourselves the same questions the NFL Competition Committee asks when discussing potential rule changes ([source](https://operations.nfl.com/the-rules/2018-rules-changes-and-points-of-emphasis/)):
# 
# 1. Does the change improve the game? 
# 2. How will it be officiated?
# 3. How will it be coached?
# 4. How can the player play by the rule?    
# 
# We address these questions below in order.
# 
# **Does the change improve the game?** 
# 
# For this competition, we think of this question primarily in the following context: Does the rule change fundamentally decrease concussion occurances in punt plays, while still maintaining the spirit of the play?
# 
# Consider the charts below:

# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

# Return yardage distributions for all punt plays
sns.countplot(play_info[play_info['Return_Indicator']==True].Return_Category.sort_values(), ax=axarr[0]).set_title("Return Yardage Distribution - All Punt Plays", fontsize=14)
# Return yardage distributions for concussion plays
axarr[1] = sns.countplot(injury[injury['Return_Indicator']==True].Return_Category.sort_values()).set_title("Return Yardage Distribution - Concussion Plays", fontsize=14)


# Here, we see the distributions of return length across the entire 2016-2017 sample of punts that included some sort of return, as well as the distribution across the 31 concussion plays that included some sort of return. We find this look to be more insightful than a straight average return of these plays, as an average return metric can be greatly influenced by outliers, such as long returns for touchdowns.
# 
# We notice above that the two sets of distributions are quite similar, so in this context we find the concussion plays to be a good representation of punt return plays in general. From the concussion plays, we see that 18 of the 31 plays resulted in a return of 10 yards or less.
# 
# If we assume that punt returners have perfect expectations of the upcoming return as they receive a punt, then we could safely say that with a rule that gives a 10 yard advancement after a fair catch on a punt, all 18 of those returns that were 10 yards or less would instead have been fair caught. 
# 
# We manually tracked every video in the concussion dataset, and found that on 1 of those 18 total return plays, the impact that led to the concussion occurred prior to the ball being caught; thus, this concussion would not have been avoided if a fair catch had been exectued. However, the other 17 would have been avoided. Thus, in this context we would argue that 17 of the 37 concussions could have been avoided under this rule framework, which indicates a decrease in concussion instances of 46%.
# 
# Of course, we should not expect punt returners to have perfect information. However, we do think it is reasonable to assume that punt returners are skilled at "feeling" the potential yardage to be had with a given return, as they see the other players moving down the field prior to the catch. One could argue that punt returners are likely more confident in their ability to return punts than they should be, and thus they might typically expect punt returns to go for slightly more yardage than should be expected. 
# 
# Even if we adjust for this, though, an expected decrease in concussion rates of around 40% due to the Fair Catch Advancement Rule is still very reasonable. Also, as you'll see below, we believe that coaches will influence their punt returners to be generally conservative when making the decision of whether to return or fair catch the punt. Thus, from this perspective, we conclude that our rule change would certainly improve the game.
# 
# We wanted to make sure we considered the other perspectives involved in this overarching question, as well. One perspective we considered was that of the spirit and integrity of the play. Phrased as a question, we asked - does our rule change fundamentally change the punt play? 
# 
# In our opinion, it does not. Our proposed rule maintains all of the parts that currently make up the punt play - all it does is shift some of the incentives for players and coaches (further explained below). While our proposed rule intends to limit punt returns, our proposed rule leaves intact all of the potential outcomes that can come from a punt play today. Returns would still occur, just at a much lower rate.
# 
# Last, we also considered the perspective of the fan. How does our proposed rule change impact the overall fan experience? In total, we believe that our proposed rule change would have a positive impact on fan experience. From a pure punt return perspective, we acklowedge that at best, our rule change would be net neutral, and perhaps a slight negative on fan experience, because the rule would effectively limit one of the most exciting plays that can occur on a punt, the long return/return for a touchdown. However, we see two other impacts at play here. First, fans are showing increased awareness in concussions in general. See the chart from Google Trends below.
# 
# <img src="https://imgur.com/RSPKR4i.jpg"/>
# 
# We suspect that the substantial impact on concussion instances that would occur from our proposed rule would create a positive impact on the NFL's fan base, as they appreciate the NFL's efforts to make the game safer. 
# 
# Also, our proposed rule change would create slightly better field positions for NFL offenses, and would thus slightly increase scoring across the league. Given that fans tend to [enjoy offense](http://www.espn.com/nfl/story/_/id/25670107/nfl-television-ratings-rose-5-percent-2017), we believe this would also create a positive impact for the game.
# 
# **How will it be officiated?** This is the easiest of the four questions to answer, as this rule change requires minimal education to NFL officials. The only thing that changes from an official's point of view is that after a completed fair catch on a punt, the ball must be advanced 10 yards from the spot of the catch. With respect to punt plays, we would also argue that this makes an official's job slightly easier overall, as there would be fewer return plays to officiate, which tend to be plays with a higher rate of penalties.
# 
# **How will it be coached?** Coaches would coach punt returners to be generally conservative if this rule were to be implemented, by encouraging returners to execute the fair catch in most circumstances. Why do we think this? One reason is the fumble rate on punt returns - see below.

# In[ ]:


print("Fumble Rate on Punt Returns -", "{:.2%}".format(len(play_info[(play_info['PlayResult']=='Return') & (play_info['Fumble_Indicator']== True)].index)/len(play_info[play_info['PlayResult']=='Return'].index)))
print("Turnover Rate on Punt Returns -", "{:.2%}".format(len(play_info[(play_info['PlayResult']=='Return') & (play_info['Turnover_Indicator']== True)].index)/len(play_info[play_info['PlayResult']=='Return'].index)))


# About 1 in every 58 returns, the punt returner fumbles the ball. Of those, about 1 in every 2.5 results in a turnover. While these plays tend to be fairly uncommon, a turnover on a punt return can be devastating, as the ball is often recovered deep in scoring territory for the recovering team, which then often leads to points scored. 
# 
# This, along with seeing that over 2/3 of all punt returns go for less than 10 yards from the distributions above, lead us to believe that coaches will tell their returners to play things conservatively. There are likely a few exceptions to this general idea - one would include treatment of the best punt returners, as coaches of these players may tell them to be more aggresive in making returns. Another exception might be certain game situations, as there may be situations where a team wants to be aggresive in trying to execute a punt return, in hopes of a game-shifting play.
# 
# Overall, though, we would expect coaching philosophy to be more in line with what is emerging with the kickoff, where most coaches are adopting a conservative strategy ([source](http://www.espn.com/blog/nflnation/post/_/id/287320/the-nfl-rule-tweaks-saving-kickoff-from-extinction)).
# 
# For coaching as it pertains to the rest of the players on the field, as well as the strategies of the play, we'd expect things to stay relatively similar to what's current. Return teams still have to account for potential fake punts, so we wouldn't expect much of a shift towards increased punt block attempts. Similarly, we wouldn't expect much of a shift in punt protection strategies. 
# 
# From the punt team's perspective, we think that there could be a slight increase in attempts at 4th down conversions, either through traditional offensive formations or punt fakes. This is due to the expected decrease in the net field change from the propsed rule change. We cover this in more detail below when we discuss the punter's incentives.
# 
# **How can the player play by the rule?** We consider 3 separate groups of players: punt returners, punters, and everybody else on the field. 
# 
# Punt returners would become more conservative in general, opting for more fair catches. We also think that punt returners will be willing to fair catch the ball closer to their own end zone, knowing the ball will be advanced 10 yards. Often, we currently observe punt returners letting the ball hit the ground if it is inside the 10 yard line, hoping the ball will bounce into the endzone for a touchback. Our belief is that returners would be willing to catch the ball back to the 5 yard line, and perhaps even closer to the endzone in some circumstances.
# 
# The other non-punter players involved in the play would continue to do the same things they do currently - their duties as it pertains to the punt play would not change.
# 
# Last, we look to the punters. Our proposed rule change, by definition, is a detriment to the punting team, as it causes the net yardage change that results from the punt to decrease in many instances, and on plays that would have resulted in a fair catch anyways, the proposed rule change decreases that net yardage change by 10 yards. Given that the average punt length and average return length in these datasets are..

# In[ ]:


print("Average Punt Length -", "{:.3}".format(play_info[play_info['Punt_Indicator']==True].Punt_Yards.mean()), "yards")
print("Average Return Length -", "{:.3}".format(play_info[play_info['Punt_Indicator']==True].Return_Yards.mean()), "yards")
print("Average Return Length -", "{:.3}".format(play_info[play_info['Return_Indicator']==True].Return_Yards.mean()), "yards")


# If we assumed every punt was fair catch under this new rule, the effective return on each punt would be 10 yards, and the average net yardage change from a punt would go from about 41 yards, down to 35 yards - a significant change. Of course, not every punt will be fair caught - some will still be returned, and others will go out of bounds, or into the endzone for a touchback - still, we think it's reasonable to expect that our proposed rule change would shift field position by at least a few yards, in favor of offenses.
# 
# How would punters respond? As noted in this [SBNation article](https://www.sbnation.com/2018/12/11/18058128/nfl-punting-art-form-golden-age-evolution), NFL punters are as good today as they have ever been, and they only continue to get better. We think punters would be able to adjust successfully to the new incentives created by our proposed rule change.
# 
# One way we think punters would adjust is by punting the ball out of bounds more, particularly when the punt is coming from around midfield and doesn't require a full kick from the punter to reach the endzone. Punting the ball out of bounds, rather than straight down the center of the field, usually means a small sacrifice of field position. With our proposed rule change, though, that small loss in punt yardage would be better for the punting team than the 10 yards advancement on a fair catch.
# 
# Another way that punters would adjust is by punting the ball closer to the endzone. Currently, punters often aim to punt the ball slightly inside the 10 yard line, hoping to either spin the ball in such a way that it bounces to a stop short of the endzone, or to have a teammate down the ball prior to the endzone. With the proposed rule change in play, we believe punters would try to kick the ball slightly closer to the endzone, perhaps the 5 yard line. This is because the relative penalty of a touchback (ball being placed at the 25 yard line) would be lower than it is now - a fair catch at the 5 yard line with our proposed rule is effectively the same as a fair catch at the 15 yard line today.
# 
# **Why 10 Yards?**
# 
# As we decided on a recommendation of a 10 yard advancement, we also considered both a 5 yard advancement and a 15 yard advancement. The tradeoff to consider between the three advancement options is the difference in impact on concussion instances versus the difference in impact on game strategy. 
# 
# For impact on concussion instances, we revisited the following charts:

# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

# Return yardage distributions for all punt plays
sns.countplot(play_info[play_info['Return_Indicator']==True].Return_Category.sort_values(), ax=axarr[0]).set_title("Return Yardage Distribution - All Punt Plays", fontsize=14)
# Return yardage distributions for concussion plays
axarr[1] = sns.countplot(injury[injury['Return_Indicator']==True].Return_Category.sort_values()).set_title("Return Yardage Distribution - Concussion Plays", fontsize=14)


# From the concussion chart on the right, if we use our general rule of thumb described above for estimating the decrease in concussion instances, we'd expect the following decreases (recall that on one of the punt return plays, the concussion occurred prior to the return):
# 
# 5 yard rule - 10/37 - 27%
# 10 yard rule - 17/37 - 46%
# 15 yard rule - 24/37 - 65%
# 
# For evaluating game strategy changes, we considered all of the incentives discussed earlier, under each of the 5 yard/10 yard/15 yard rule frameworks. 
# 
# Ultimately, we concluded that a 5 yard rule would not create enough incentive for returners to change their actions in a meaningful way. On the other side of the spectrum, we concluded that a 15 yard rule would create too much change in player actions - we speculate that returns would almost entirely go away under this framework, and the change in net field position would be too substantial. Thus, we found a 10 yard rule framework to be the best balance of meaningful impact on concussion instances and small shifts in player actions.
# 
# The last area we explored was situations where the punt is fielded around midfield. The concern there is that our proposed rule may further shorten an already short field for the receiving team. Is it fair that a fair catch around midfield could automatically put the receiving team in range for a long field goal? What if this were to happen near the end of a close game? In that instance, the proposed rule could have a large impact on the outcome of the game.

# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

# Punt starting field position distributions for all punt plays
sns.countplot(play_info[(play_info['Return_Indicator']==True)|(play_info['FairCatch_Indicator']==True)].YardLineStart_Category.sort_values(), ax=axarr[0]).set_title("Punt Starting Field Position - All Punt Plays", fontsize=14)
# Field position of catch for concussion plays
axarr[1] = sns.countplot(play_info[(play_info['Return_Indicator']==True)|(play_info['FairCatch_Indicator']==True)].YardLineCatch_Category.sort_values()).set_title("Field Position of Catch - All Punt Plays", fontsize=14)


# In[ ]:


print("Percent of Punts Started Inside 10 Yard Line -", "{:.2%}".format(len(play_info[((play_info['Return_Indicator']==True)
                                                                                       |(play_info['FairCatch_Indicator']==True))
                                                                                      &(play_info['YardLineStart_Category']=='00-10')].index)/
                                                                        len(play_info[((play_info['Return_Indicator']==True)
                                                                                       |(play_info['FairCatch_Indicator']==True))].index)))
print("Percent of Punts Caught Beyond 40 Yard Line -", "{:.2%}".format(len(play_info[((play_info['Return_Indicator']==True)
                                                                                       |(play_info['FairCatch_Indicator']==True))
                                                                                      &((play_info['YardLineCatch_Category']=='41-50')
                                                                                        |(play_info['YardLineCatch_Category']=='50+'))].index)/
                                                                        len(play_info[((play_info['Return_Indicator']==True)
                                                                                       |(play_info['FairCatch_Indicator']==True))].index)))


# The charts above look at both field position to begin the punt (from the punt team's side of the field), as well as field position at the point of the catch (from the receiving team's side of the field). We see that about 6% of punts that are either fair caught or returned start inside the punting team's 10 yard line, while about 8% of those same punts are caught beyond the receiving team's 40 yard line. 
# 
# We find these distributions to be small enough to be considered negligable. As punters continue to improve, these distributions will grow even smaller. The proposed rule change would also further limit these situations, as we expect a decrease in punts downed inside the 10 yard line - meaning there would be fewer instances where a team is pinned deep in their own territory, and forced to punt from out of their own endzone.
# 
# With the biggest concern here being the potential for a fair catch around midfield late in a close game - we think that these situations will be infrequent enough to safely ignore.
# 
# This brings us to the end of our kernel. We'll leave you with our proposal one last time: 

# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

# Play types across all punts
sns.countplot(play_info.PlayResult.sort_values(), ax=axarr[0]).set_title("Play Types Across All Punts", fontsize=14)
# Play types across concussion plays
axarr[1] = sns.countplot(injury.PlayResult.sort_values()).set_title("Play Types Across Concussion Plays", fontsize=14)


# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

# Return yardage distributions for all punt plays
sns.countplot(play_info[play_info['Return_Indicator']==True].Return_Category.sort_values(), ax=axarr[0]).set_title("Return Yardage Distribution - All Punt Plays", fontsize=14)
# Return yardage distributions for concussion plays
axarr[1] = sns.countplot(injury[injury['Return_Indicator']==True].Return_Category.sort_values()).set_title("Return Yardage Distribution - Concussion Plays", fontsize=14)


# The Fair Catch Advancement Rule: **If a fair catch is completed on a punt, the ball is advanced 10 yards from the spot of the fair catch, prior to the next snap.** 
# 
# We predict that the Fair Catch Advancement Rule would **decrease the concussion rate on punt plays by over** **_40 percent_**. 
# 
# We hope you have found our story to be engaging and easy to follow. Whether our rule is picked up by the NFL or not, we are honored to have been able to participate in a competition that will help to ultimately increase the longevity of NFL players, not just in their NFL careers, but also in their livelihoods, as the work to limit concussions and head trauma continues.
