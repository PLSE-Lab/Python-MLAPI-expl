#!/usr/bin/env python
# coding: utf-8

# # Q1: Cost of Turf Field
# 
# ## Abstract
# 
# In this notebook, we examine the costs of playing on artificial turf as compared natural turf (grass). Our assumption is that playing on artificial turf causes relatively more injuries than playing on grass. We would like to calculate how many more games players are expected to miss per season by playing on artificial turf as compared to grass, then translate that difference in games played into a dollar amount by using per game player salaries.
# 
# Our conclusion is based on NFL teams playing 8 home games and 8 away games, meaning that a team's surface decision will impact their players for 8 games per year.
# 
# Our analysis indicates that injuries are about 1.6 times more likely to occur on artifical turf as compared to grass. We find the total cost of having an artifical turf at home to be about $2.95m / season in missed game salaries more than having a grass field at home. 

# ## Preparation
# 
# ### Import data
# 
# For this analysis, we do not need player track data, as do not look at specific intra-play movement.

# In[ ]:


import pandas as pd
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

"""
Import datasets 
https://www.kaggle.com/c/nfl-playing-surface-analytics/data
"""

KAGGLE = True
if not KAGGLE:
    IR_data = pd.read_csv('InjuryRecord.csv')
    PL_data = pd.read_csv('PlayList.csv')
else:
    IR_data = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
    PL_data = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

plt.rcParams['figure.dpi'] = 200
plt.rcParams.update({'font.size': 14})


# ### Clean IR data
# 
# Some of the injuries do not have a play associated with them. Lets use the last play they played as the injury play. For this analysis, it does not matter which play they were injured on, as our analysis is based on game-level variables.

# In[ ]:


def cleanIRdata(IR_data):

    # Mark whether we need to set a play for each row
    IR_data['PlayBool'] = [False if pd.isnull(i) else True for i in IR_data['PlayKey']] 

    # Get maximum play count for each play
    IR_data['HighPlay'] = [PL_data[PL_data['GameID'] == i]['PlayerGamePlay'].max() for i in IR_data['GameID']] 

    # Create a new playkey
    PlayKey_adj = []
    PlayKey = list(IR_data['PlayKey'])
    HighPlay = list(IR_data['HighPlay'])
    PlayBool = list(IR_data['PlayBool'])
    GameID = list(IR_data['GameID'])
    for i in range(len(IR_data)):
        if PlayBool[i]:
            PlayKey_adj.append(PlayKey[i])
        else:
            PlayKey_adj.append(GameID[i] + '-' + str(HighPlay[i]))

    IR_data['PlayKey_adj'] = PlayKey_adj
    return IR_data
    
IR_data = cleanIRdata(IR_data)
IR_data.head()


# ### Merge data
# 
# We start with two datasets:
# 
# - PL_data: lists every player-play for the 2017 and 2018 season for 250 players
# - IR_data: lists every injury at a play level (or if not available, a game level)
# 
# We will merge these datasets to create one dataset with every player play, and an indication of whether that player was injured on that play or not.

# In[ ]:


# Merge data
PL_data['PlayKey_adj'] = PL_data['PlayKey']  
df = pd.merge(PL_data, IR_data, on = 'PlayKey_adj', how='left').fillna(0)
df.head()


# ### In terms of "games missed"
# 
# Our original dataset tells us whether the player missed 1, 7, 28, or 42 days of gametime. For simplicity, we are going to create a new variable called "games missed." We assume that missing 1 day means that the player got injured during the game and did not return, but was able to return the following week. We therefore assign this as 0.5 games missed. 
# 
# Since all injuries happened during a game, we assume that an injury results in 0.5 games missed for the game that caused the injury, as well as X number of games missed for the DM/7 following days missed.
# 
# - DM_M1   = 0.5 games missed
# - DM_M7	  = They missed this game and the next game. 1.5 games missed
# - DM_M28  = 4 weeks, so 4.5 games.
# - DM_M42  = 6 weeks, so 6.5 games

# In[ ]:


# This function calculates the number of games missed and appends it to our dataframe
def countGamesMissed(df):
    
    GM_array = []
    for i, row in df.iterrows():

        DM1  = row['DM_M1']
        DM7  = row['DM_M7']
        DM28 = row['DM_M28']
        DM42 = row['DM_M42']

        if DM42 == 1:
            GM_array.append(6.5)
        elif DM28 == 1:
            GM_array.append(4.5)
        elif DM7 == 1:
            GM_array.append(1.5)
        elif DM1 == 1:
            GM_array.append(.5)
        else:
            GM_array.append(0)

    df['GamesMissed'] = GM_array
    return df


# In[ ]:


df = countGamesMissed(df)
df[df['GamesMissed'] > 0][['PlayKey_adj', 'DM_M1','DM_M7','DM_M28','DM_M42','GamesMissed']].head()


# ## Data Analysis
# 
# We start by taking a look at our dataset to see how many plays we have, how many injuries we have, and what the breakdown looks like between position group.

# In[ ]:


PGs = ['LB', 'QB', 'DL', 'OL', 'SPEC', 'TE', 'WR', 'RB', 'DB']
injurySummary = pd.DataFrame(columns=['PG', 'Plays', 'Injuries'])
for i in range(len(PGs)):
    PG = PGs[i]
    injurySummary.loc[i] = [PG, len(df[df['PositionGroup'] == PG]), len(df[(df['PositionGroup'] == PG) & (df['DM_M1'] == 1)])]
    
injurySummary


# We do not have any injuries in our dataset for QBs and SPEC -- therefore we will use the average injury rate for our analysis for those position groups.

# ### Natural v. Artificial Injury Rates
# 
# At a play level, is there a higher rate of injuries for plays on artifical terf?

# In[ ]:


syntheticPlays    = df[df['FieldType'] == 'Synthetic']
syntheticInjuries = df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Synthetic')]
naturalPlays      = df[df['FieldType'] == 'Natural']
naturalInjuries   = df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Natural')]
plays             = df
injuries          = df[df['DM_M1'] == 1]

print('There are', len(IR_data), 'plays in the full dataset')
print('There are', len(plays), 'plays in the merged dataset')
print(len(injuries), 'of', len(plays), "(%.4f" % (len(injuries) / len(plays) * 100) + '%)', 'of plays in the merged dataset are injuries')
print(len(syntheticInjuries), 'of', len(syntheticPlays), "(%.4f" % (len(syntheticInjuries) / len(syntheticPlays) * 100) + '%)', 'sythetic injuries')
print(len(naturalInjuries), 'of', len(naturalPlays), "(%.4f" % (len(naturalInjuries) / len(naturalPlays) * 100) + '%)', 'natural injuries')

# P-Test
natural_sample = [1] * len(naturalInjuries) + [0] * (len(naturalPlays) - len(naturalInjuries))
synthetic_sample = [1] * len(syntheticInjuries) + [0] * (len(syntheticPlays) - len(syntheticInjuries))
t_stat, p_val = stats.ttest_ind(natural_sample, synthetic_sample, equal_var=False)
print('The p-value that synthetic is worse than natural is', "%.5f" % p_val)


# The answer is yes -- our p-value is significantly lower than 0.05.
# 
# Let's run the same analysis at a game level.

# In[ ]:


SYN_INJURIES = len(set(df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Synthetic')]['GameID_x']))
SYN_GAMES = len(set(df[(df['FieldType'] == 'Synthetic')]['GameID_x']))
NAT_INJURIES = len(set(df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Natural')]['GameID_x']))
NAT_GAMES = len(set(df[(df['FieldType'] == 'Natural')]['GameID_x']))

print(SYN_INJURIES, 'of', SYN_GAMES, "(%.4f" % (SYN_INJURIES / SYN_GAMES * 100) + '%)', 'of synthetic games had injuries')
print(NAT_INJURIES, 'of', NAT_GAMES, "(%.4f" % (NAT_INJURIES / NAT_GAMES * 100) + '%)', 'of natural games had injuries')
print("%.4f" %  ((SYN_INJURIES / SYN_GAMES) / (NAT_INJURIES / NAT_GAMES)), 'higher injury rate on turf')


# P-Test
natural_sample = [1] * NAT_INJURIES + [0] * (NAT_GAMES - NAT_INJURIES)
synthetic_sample = [1] * SYN_INJURIES + [0] * (SYN_GAMES - SYN_INJURIES)
t_stat, p_val = stats.ttest_ind(natural_sample, synthetic_sample, equal_var=False)
print('The p-value that synthetic is worse than natural (on a game level) is', "%.5f" % p_val)


# Taking a similar approach to above, we see that (as expected) synthetic games have a higher rate of injury than natural grass games.

# ### Comparison in terms of games missed
# 
# Our previous test shows that artifical term is associated with a higher risk of injury. Is one set of injuries worse than an other set in terms of resulting games missed?

# In[ ]:


syntheticGM       = sum(df[df['FieldType'] == 'Synthetic']['GamesMissed'])
naturalGM         = sum(df[df['FieldType'] == 'Natural']['GamesMissed'])
SYN_INJURY_AVG = round(syntheticGM / SYN_INJURIES,2)
NAT_INJURY_AVG = round(naturalGM / NAT_INJURIES,2)
print('Each synthetic injury averages', SYN_INJURY_AVG, 'games missed')
print('Each natural injury averages', NAT_INJURY_AVG, 'games missed')
print('Synthetic injuries miss', round(SYN_INJURY_AVG / NAT_INJURY_AVG,3), 'more games')

GM_NAT_LIST = list(df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Natural')]['GamesMissed'])
GM_SYN_LIST = list(df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Synthetic')]['GamesMissed'])

t_stat, p_val = stats.ttest_ind(GM_NAT_LIST, GM_SYN_LIST, equal_var=False)
print('The p-value that synthetic injuries are worse (in terms of games missed) than natural injuries is', "%.3f" % p_val)


# It appears that synthetic injuries are no more dangerous than injuries that happened on grass.

# ## Regression
# 
# We will now run a regression to determine the expected number of games missed per play. We will break players down by their position group to account for signficiantly different injury probabilities for different groups.
# 
# ### Clean data
# 
# To hopefully mitigate any other explanatory variables, we create a few variables for weather such as
# 
# - Is_Rain: whether it was raining or had a chance of rain
# - Is_Snow: whether it was snowing
# - Temperature_Adj: The temperature during the game. This is set as room temperature (72F) for games that happened indoors.

# In[ ]:


def convertVariablesForRegression(tmp):

    def isIn(item, word):
        return item in str(word).lower()

    # Convert some fields to boolean
    tmp['Is_Synthetic'] = tmp['FieldType'] == 'Synthetic'
    tmp['Is_Rain'] = [True if isIn('rain', i) or isIn('shower', i) else False for i in tmp['Weather']]
    tmp['Is_Snow'] = [True if isIn('snow', i) else False for i in tmp['Weather']]

    # Clean some variables
    tmp['Temperature_Adj'] = [i if i != -999 else 72 for i in tmp['Temperature']]
    
    return tmp
    
df = convertVariablesForRegression(df)

display(df[['PlayKey_adj', 'PositionGroup', 'PlayerDay', 'PlayerGamePlay', 'Is_Synthetic', 'Temperature_Adj', 'Is_Rain',     'Is_Snow', 'GamesMissed']].head())


# ### Run regression
# 
# We can now run a generalized linear model using a Poisson distribution. We want to make a seperate model for each position group.
# 
# 
# We have no data for some position groups -- SPEC and QB. We will just use the average value for their position.

# In[ ]:


X = 'GamesMissed'
Y = ['C(PositionGroup) * Is_Synthetic', 'Is_Synthetic', 'PlayerDay', 'PlayerGamePlay', 'Temperature_Adj', 'Is_Rain', 'Is_Snow']
Y = ' + '.join(Y)

model = sm.GLM.from_formula(X + ' ~ ' + Y, data=df, family=sm.families.Poisson()).fit()
# model = smf.ols(X + ' ~ ' + Y, data = df).fit()
print(model.summary())


# ### Predictions
# 
# We can use our regression results to predict how many expected games a player in each position group will miss per play on synthetic and natural turf.

# In[ ]:


# Create testing dataframe to predict on
td = pd.DataFrame()
PG = list(set(df['PositionGroup']))

# Get average plays per game by position
PPG = []
for i in PG:
    POS_PLAYS = len(df[df['PositionGroup'] == i])
    POS_GAMES = len(set(df[df['PositionGroup'] == i]['GameID_x']))
    PPG.append(int(POS_PLAYS / POS_GAMES))

# Use mean for each group
def getMeanForPG(stat, PG=PG, df=df):
    return [df[df['PositionGroup'] == pos_group][stat].median() for pos_group in PG]

td['PositionGroup']     = PG
td['Is_Synthetic']      = [False] * len(PG)
td['PlayerDay']         = getMeanForPG('PlayerDay')
td['PlayerGamePlay']    = getMeanForPG('PlayerGamePlay')
td['Temperature_Adj']   = getMeanForPG('Temperature_Adj')
td['Is_Rain']           = [False] * len(PG)
td['Is_Snow']           = [False] * len(PG)
td['PPG']               = PPG
                    
# Predict
td['GM_Natural']        = model.predict(td)

# Predict on artificial
td['Is_Synthetic']      = [True]  * len(PG)
td['GM_Synthetic']      = model.predict(td)

# Set average values for QB and SPEC since insufficient data (0 injuries)
GM_Avg = {}
for fieldType in ['Natural', 'Synthetic']:
    GM_Avg[fieldType] = sum(df[df['FieldType'] == fieldType]['GamesMissed']) / len(df[df['FieldType'] == fieldType])
    for PG in ['QB', 'SPEC']:
        td.at[list(td[td['PositionGroup'] == PG].index)[0], 'GM_' + fieldType] = GM_Avg[fieldType]

# Differences
td['SyntheticDelta']    = td['GM_Synthetic'] - td['GM_Natural']
td['SyntheticRatio']    = td['GM_Synthetic'] / td['GM_Natural']      

# GM / Game
td['DeltaGMperGame'] = list(td['SyntheticDelta'] * td['PPG'])

# GM / other units
td['GM_Natural_Game'] = list(td['GM_Natural'] * td['PPG'])
td['GM_Synthetic_Game'] = list(td['GM_Synthetic'] * td['PPG'])
td['GM_Natural_HomeSeason'] = [i*8 for i in list(td['GM_Natural'] * td['PPG'])]
td['GM_Synthetic_HomeSeason'] = [i*8 for i in list(td['GM_Synthetic'] * td['PPG'])]  
        
# Remove "missing data" player group
td = td[td['PositionGroup'] != 'Missing Data'].reset_index(drop=True)

# Display
td.sort_values('SyntheticRatio', ascending=False)


# ### Results
# 
# From here, we can see which position groups are most likely to get injured per play and which position groups are more  (or less) likely to get injured on synthetic turf as compared to grass.

# In[ ]:


def plotGamesMissedPG(PerGame = False, PerHomeSeason=False):

    # Example from here https://pythonspot.com/matplotlib-bar-chart/
    fig, ax = plt.subplots(figsize=(10,5))
    index = np.arange(len(td))
    bar_width = 0.35
    opacity = 0.75
    
    nat_bars = list(td['GM_Natural'])
    syn_bars = list(td['GM_Synthetic'])
    timeframe = 'play'
        
    if PerGame:
        nat_bars = td['GM_Natural_Game']
        syn_bars = td['GM_Synthetic_Game']
        timeframe = 'game'
        
    if PerHomeSeason:
        nat_bars = td['GM_Natural_HomeSeason']
        syn_bars = td['GM_Synthetic_HomeSeason']
        timeframe = 'home season'
    
    
    plt.bar(
                x      = index, 
                height = nat_bars, 
                width  = bar_width,
                alpha  = opacity,
                color  = '#48BB78', # color from tailwindcss
                label  = 'Natural'
           )

    plt.bar(
                x      = index + bar_width, 
                height = syn_bars, 
                width  = bar_width,
                alpha  = opacity,
                color  = '#2B6CB0', # color from tailwindcss
                label  = 'Synthetic'
            )

    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.xlabel('Position')
    plt.ylabel('Expected Games Missed (per %s)' % timeframe)
    plt.title('Natural v. Synthetic Turf - Games Missed Due to Injury', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(index + bar_width, list(td['PositionGroup']))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
plotGamesMissedPG(PerGame=True)


# ### Home-Season Level
# 
# Let's make the same chart that shows the expected games missed difference in playing 8 games at home on natural grass compared to 8 games at home on artifical turf.

# In[ ]:


plotGamesMissedPG(PerHomeSeason=True)


# ## Salary-Injury Dollars
# 
# We have already converted expected missed games per play into expected games missed per season. We can then use that result along with player salary to determine the salary value of the missed games.
# 
# For example, if a QB is paid \\$16m / season, or \\$1m per game, having 0.5 additional expected games has a salary loss value of \\$500,000.
# 
# This somewhat represents the cost of replacing a player, or "lost productivity cost" associated with the injury. Of course, there are certainly other non quantifiable costs associated with injuries, such as emotional and distress costs to that player. There are also quantifiable costs, such as healthcare costs, and increased chances of reaggravating the injury.
# 
# 
# ### Methodology
# 
# We start with our first finding of the change in missed games per play.
# 
# <br>
# <br>
# $$\:\frac{Missed\:Games\:Artificial}{Play}\:- \:\frac{Missed\:Games\:Grass}{Play}\:=\:\frac{\Delta Missed\:Games}{Play}\:$$
# <br>
# <br>
# We can then use this $\frac{\Delta Missed Games}{Play}$ number to calcuate the effect over a season. As an example, we will look at the WR position group.
# <br>
# <br>
# $$\:\frac{\Delta Missed\:Games}{Play}*\:\frac{Plays}{WR}*\:\frac{WR}{Game}*\:\frac{\\$}{Game}*\:\frac{Games}{Season}\:=\:\frac{Cost}{Season}\:$$
# <br>
# <br>
# We can reduce simplify this equation a bit by just finding the average number of WRs plays per game, instead of the number of WRs multiplied by average plays 
# <br>
# <br>
# $$\frac{Plays}{WR}*\:\frac{WR}{Game}=\:\frac{WR\:Plays}{Game}\:$$
# <br>
# <br>
# Additionally, we can just look at $\frac{Dollar}{Season}$ salary, instead of at a per-game level, resulting in the following equation that calculates the associated injury-cost for the WR group that playing 16 games on artificial turf would as compared to grass
# <br>
# <br>
# $$\:\frac{\Delta Missed\:Games}{WR\:Play}*\:\:\frac{WR\:Plays}{Game}\:*\:\frac{\$}{Season}\:=\:\frac{\$}{Season}\:$$
# <br>
# <br>
# We then want to convert this value to a per-season-team level. We can do this by summing the $\frac{Dollar}{Season}$ cost $C$ for each position group
# <br>
# <br>
# $${C}_{LB}\;+\;{C}_{QB}\;+{C}_{DL}\;+\;{C}_{OL}\;+{C}_{SPEC}\;+\;{C}_{TE}\;+\:{C}_{WR}\;+\;{C}_{RB}\;+\;{C}_{DB}\:=\:{C}_{Team}$$
# <br>
# <br>
# Finally, we can convert the increased injury cost per season into a more meaningful stat, the injury cost for having 8 home games on artificial turf.
# <br>
# <br>
# $$\:\:\frac{{\$}_{\:Team}}{Season}\:*\:\frac{8\:Home\:Games}{16\:Games}\:=\:\frac{\$_{\:Team\:Home}}{Season}\:$$

# ### Static variables
# 
# To convert expected missed games into a dollar amount, we have two variables that we need to calculate at a group level. We'll need to obtain some NFL data from outside our given dataset for this. The variables we need are $\frac{Plays}{Game}\:$ and $\frac{$}{Game}\:$ at a player group level.
# 
# #### Plays Per Game 
# 
# As we have expected games missed per play (by position group), we can convert this into expected games missed per game (by position group) by multiplying $\frac{Missed\:Games}{Play}\:*\frac{Plays}{Game}\:$
# 
# We want to find how many snaps per game players of each position group play. This is going to be higher for [DBs (defensive backs)](https://en.wikipedia.org/wiki/American_football_positions) as each normal defensive play has 5 DBs, unlike QBs which only have 1 player per play. 
# 
# We found snap count data for all players on 8 teams from the 2018 season on Pro-Football-Reference.com. We then collapse then collapsed the table by player position, summing snap counts, and average the data out to find the per game / per team snap count per position group.
# <br><br>
# *Sports Reference LLC. ["New England Patriots Snap."](https://www.pro-football-reference.com/teams/nwe/2018-snap-counts.htm#snap_counts::none) Pro-Football-Reference.com - Pro Football Statistics and History. [PRF](https://www.pro-football-reference.com/). Dec 25, 2019. [Citation Link](https://www.pro-football-reference.com/about/contact.htm)*

# In[ ]:


# Read in all snap data
teams = ['car', 'cle', 'crd', 'gnb', 'nwe', 'pit', 'tam', 'was']

if KAGGLE:
    PATH = '/kaggle/input/nflpfrdata/PFR_data/'
else:
    PATH = 'PFR_DATA/'

snaps = pd.read_csv(PATH + 'snap_count/pit.csv', header=1)
for team in teams[1:]:
    teamSnaps = pd.read_csv(PATH + '/snap_count/%s.csv' % team, header=1)
    snaps = pd.concat([snaps, teamSnaps])
    
snaps[['Player', 'Pos','Num']].head()


# #### Data Cleaning
# 
# The position group labels are more granular then the groups we want to work with. So we have to manually map the specific positions to their position groups 

# In[ ]:


# We need to conert all positions to our position groups
# https://en.wikipedia.org/wiki/American_football_positions
positionGroupConvert = {
    'C'    : 'OL', # C
    'CB'   : 'DB', # C
    'CBDB' : 'DB', # C
    'DB'   : 'DB', # C
    'DE'   : 'DL', # Could also be LB
    'DT'   : 'DL', # C
    'FB'   : 'RB', # C
    'FS'   : 'DB', # C
    'FSS'  : 'DB', # C
    'FSSS' : 'DB', # C
    'FSSSS': 'DB', # C
    'G'    : 'OL', # C
    'K'    : 'SPEC', # c
    'LB'   : 'LB', # C
    'LS'   : 'SPEC', # Long snapper
    'NT'   : 'DL', # C
    'P'    : 'SPEC', # C
    'QB'   : 'QB', # C
    'RB'   : 'RB', # C
    'S'    : 'DB', # C
    'SS'   : 'DB', # C
    'SSS'  : 'DB', # C
    'T'    : 'OL', # C
    'TE'   : 'TE', # C
    'WR'   : 'WR', # C
    
    # Extras from salary dataset
    'DL'   : 'DL', # C
    'EDGE' : 'DL', # Same as defensive end
    'HB'   : 'RB', # Halfback
    'ILB'  : 'LB', # C
    'LB-DE': 'DL', # Suggs (DE)
    'LG'   : 'OL', # C
    'LT'   : 'OL', #C
    'NT'   : 'DL', # Nose tackle, this is center on defense
    'OG'   : 'OL', # C
    'OL'   : 'OL', # C
    'OLB'  : 'LB', # C 
    'OT'   : 'OL', # C 
    'QB/TE': 'TE', # Logan Thomas (TE)
    'RB-WR': 'RB', # Ty Montgomery (RB)
}


# #### Findings
# 
# After summing the table and averaging it out to a per game level, we were able to find the number of position groups snaps per game. The numbers intuitively make sense -- there are about 65-70 offensive players per team per game, and for QB we have 67 team snaps per game.

# In[ ]:


# Group and count
snaps['PositionGroup'] = [positionGroupConvert[i] for i in snaps['Pos']]
snaps.rename(columns={'Num': 'Off', 'Num.1': 'Def', 'Num.2':'Spec'}, inplace=True)
snaps['Snaps'] = snaps['Off'] + snaps['Def'] + snaps['Spec']
snaps[['Pos', 'Snaps', 'PositionGroup']]
snaps['TeamSnapsPerGame'] = round(snaps['Snaps'] / 16 / 8, 2)
groupedSnaps = snaps.groupby('PositionGroup').sum()
groupedSnaps


# ### Salary data
# 
# We want to calculate the average salary for a player in each position group play. More accurately, we want a weighted average salary by position group snap count. Since higher paid players are more likely to play more snaps per game (and therefore more likely to get injured), we don't really want an average of all players in a position group, but rather we want to weight it towards players that are playing significant snap counts.
# 
# To further clarify, a team could have 3 quarterbacks, paid \\$20m, \\$1m, and \\$1m each, but playing 90\%-9\%-1\% snaps. If our model showed that QBs miss one game per average per season, the correct calulation would be closer to \$20m * 1/16 as compared to (\\$20m + \\$1m + \\$1m)/3 * 1/16.
# 
# #### Dataset
# 
# We found salary data on the 2019 season from Pro-Football-Reference.com. 
# 
# Observations:
# - Data is from 2019, while we would ideally want to average 2017-2018 data
# - Salary does does not appear to include signing bonuses: Le'Veon Bell is listed at \\$2m
# 
# *Sports Reference LLC. ["NFL 2019 Player Salaries."](https://www.pro-football-reference.com/players/salary.htm) Pro-Football-Reference.com - Pro Football Statistics and History. [PRF](https://www.pro-football-reference.com/). (Dec 25, 2019). [Citation Link](https://www.pro-football-reference.com/about/contact.htm)*

# In[ ]:


# Load data, preview
if KAGGLE:
    PATH = '/kaggle/input/nflpfrdata/PFR_data/'
else:
    PATH = 'PFR_DATA/'
salary = pd.read_csv(PATH + 'salary/salary.csv', header=0)
salary['PositionGroup'] = [i if pd.isnull(i) else positionGroupConvert[i] for i in salary['Pos']]
salary['Salary'] = [int(i.replace('$', '')) for i in salary['Salary']]
salary[['Player', 'Salary', 'PositionGroup']].head()


# #### Average position group salary 
# 
# We want to calculate the average position group salary for players who are playing meaningful snap counts.
# 
# For QBs, we assume that about 32 different players will play 80%+ of all QB snaps in a season. This would be 100% if QBs didn't get injured or benched. Taking the average salary of the top 32 QBs would probably be appropriate -- or at least relatively close to the average position group salary.
# 
# However, a team might have 4 WRs on the field at one time. So it wouldn't really make sense to take the average salary of the top 32 WRs -- that would likely be a top 20% salary for starting WRs. If indeed teams do play an average of 4 WRs, we could reasonably then take the average of the top 32 * 4 highest paid WRs.
# 
# This means we first have to calculate how many players are played at each position group. What we can do is look at the total snap counts on offense, defense, and special teams to see what percent each position group had. We can then look at what proportion of players on the field were in that position group, and multiply that by 11 to find how many players are typically on the field from that group at one time

# In[ ]:


# Calculate Number of Men from each group on field
for i in ['Off', 'Def', 'Spec']:
    groupedSnaps[i + '_men'] = round(11 * groupedSnaps[i] / sum(groupedSnaps[i]), 2)
    
groupedSnaps['Men'] = groupedSnaps[['Off_men','Def_men','Spec_men']].max(axis=1)
groupedSnaps


# This tells us that the average number of men in each group on average. It makes sense that QB is always 1, WR is about 2.5 and OL is about 5.
# 
# We can now use these numbers to calculate how many top salaries we should look at in each group. To do this, we will take 32 teams * the average number of players that play in that group (i.e. 'Men').

# In[ ]:


# Calculate average position salary
positionSalary = []
for pos in list(groupedSnaps.index):
    numPosition = int(32 * groupedSnaps['Men'][pos])
    positionSalary.append(salary[salary['PositionGroup'] == pos][0:numPosition]['Salary'].mean())
    
groupedSnaps['Salary'] = positionSalary
groupedSnaps['SalaryFormatted'] = ['${:,.2f}m'.format(i/1000000) for i in positionSalary]

groupedSnaps.sort_values(by=['Salary'], ascending=False)


# These results show that QB have the highest average salary while SPEC (punters, kickers) have the lowest.

# ### Calculations
# 
# We now have all the values needed to make our calculations. To review, we have the following data:

# In[ ]:


res = pd.merge(td, groupedSnaps, on = 'PositionGroup', how='left').fillna(0)
res[['PositionGroup', 'SyntheticDelta', 'TeamSnapsPerGame', 'Salary']]


# And can use the following equation to find the injury lost salary per game by position group:
# <br>
# <br>
# $$\:\frac{\Delta Missed\:Games}{Play}*\:\:\frac{Plays}{Game}\:*\:\frac{$}{Season}\:=\:\frac{$}{Season}\:$$
# <br>
# <br>
# And then convert this to home games:
# <br><br>
# $$\:\:\frac{{$}_{\:Group}}{Season}\:*\:\frac{8\:Home\:Games}{16\:Games}\:=\:\frac{$_{\:Home\:Group}}{Season}\:$$

# In[ ]:


# Calculations
HOME_GAMES = 8 / 16
res['InjuryHomeCostNatural'] = res['GM_Natural'] * res['TeamSnapsPerGame'] * res['Salary'] * HOME_GAMES
res['InjuryHomeCostSynthetic'] = res['GM_Synthetic'] * res['TeamSnapsPerGame'] * res['Salary'] * HOME_GAMES
res['InjuryHomeCostDelta'] = res['SyntheticDelta'] * res['TeamSnapsPerGame'] * res['Salary'] * HOME_GAMES

res['TeamSnapsPerGame_F'] = [int(i) for i in res['TeamSnapsPerGame']]
res['Salary_F'] = ['${:,.2f}m'.format(i/1000000) for i in res['Salary']]

res['InjuryHomeCostDelta_F'] = ['${:,.0f}k'.format(i/1000) for i in res['InjuryHomeCostDelta']]


res[['PositionGroup', 'TeamSnapsPerGame_F', 'Salary_F',     'SyntheticDelta', 'InjuryHomeCostDelta_F' ]]


# ### Results
# 
# We can plot our results to look at the cost breakdown by position group

# In[ ]:


# Example from here https://pythonspot.com/matplotlib-bar-chart/
fig, ax = plt.subplots(figsize=(10,5))
index = np.arange(len(td))
bar_width = 0.35
opacity = 0.75

totalNaturalCost = '${:,.2f}m'.format(float(sum(res['InjuryHomeCostNatural']) / 1000000))
totalSyntheticCost = '${:,.2f}m'.format(float(sum(res['InjuryHomeCostSynthetic']) / 1000000))
plt.bar(
            x      = index, 
            height = list(res['InjuryHomeCostNatural']), 
            width  = bar_width,
            alpha  = opacity,
            color  = '#48BB78', # color from tailwindcss
            label  = 'Natural (Total Cost of Injuries: ' + totalNaturalCost + ')'
       )

plt.bar(
            x      = index + bar_width, 
            height = list(res['InjuryHomeCostSynthetic']), 
            width  = bar_width,
            alpha  = opacity,
            color  = '#2B6CB0', # color from tailwindcss
            label  = 'Sythetic (Total Cost: ' + totalSyntheticCost + ')'
        )

ax.set_yticklabels(['${:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
plt.xlabel('Position Group')
plt.ylabel('Cost to Team')
plt.title('Home Games on Natural v. Synthetic Turf - Injury Cost', fontsize=14, fontweight='bold', pad=20)
plt.xticks(index + bar_width, list(res['PositionGroup']))
plt.legend()

plt.tight_layout()
plt.show()

