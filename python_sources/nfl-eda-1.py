#!/usr/bin/env python
# coding: utf-8

# # EDA Part 1
# 
# This notebook is basically a stream of consciousness data exploration. I check out each of the features one at a time to see if I need to clean them up, determine how useful they might be, or anticipate difficulties they might give me when I go to start modelling. I don't check out how yardage various with many of them, saving that for another notebook.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from math import cos
from math import sin
from math import radians

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


raw_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')


# In[ ]:


raw_data.head()


# In[ ]:


raw_data.dtypes


# ## GameID
# 
# Each game gets one. Presumably every play in the game gets a row in the dataset, as long as a player touches the ball and it's a run.

# In[ ]:


plt.figure(figsize=(20,10))
plt.title("Plays per Game")
raw_data['GameId'].value_counts().plot(kind='bar')


# Is this a reasonable number of plays per game? Qquick Google search shows teams have way less than 100 plays per game typically. What gives? Well, it's not one entry per play! It's one entry *per player* per play. Let's make sure this checks out and see that each play ID has 22 entries.

# In[ ]:


raw_data.groupby(['PlayId']).count().describe()


# Yes! Each PlayID has 22 players in it, which is good. We also notice that some of the columns are missing data!
# 
# I've also discovered that using DataFrames rather than Series takes a long, long, long time. Use Series whenever possible! Let's check and see how many games we have.

# In[ ]:


len(raw_data['GameId'].unique())


# There are 256 games in a season, so that means we must have 2 years worth of games here. 

# ## Play ID
# 
# The ID of the play. We looked at it a little bit above, so I'll just move on. However, I do want to write a function that will get a random play's (GameID, PlayID) combo

# In[ ]:


def randomPlay():
    '''
    Returns a random valid (GameId,PlayId) pair
    '''
    N = raw_data.shape[0]
    i = np.random.randint(0,N)
    return (raw_data.iloc[i,:].GameId, raw_data.iloc[i,:].PlayId)

randomPlay()


# ## Team
# 
# Home or away. Not so important probably, but we could investigate the degree of home field advantage! Let's just make sure the counts are correct:

# In[ ]:


raw_data['Team'].value_counts()


# ## X, Y
# 
# The coordinates of the player. Let's see if we can make a visual of the locations of the players.

# In[ ]:


def graphPlay1(GameId, PlayId):
    play_data = raw_data[(raw_data['GameId'] == GameId) & (raw_data['PlayId'] == PlayId)]
    colors = {'away':'red', 'home':'blue'}
    for i in play_data.index:
        plt.plot(play_data.loc[i,'X'], play_data.loc[i,'Y'], 'ro', markersize=10, color=colors[play_data.loc[i,'Team']])
    plt.show()
    return None


# In[ ]:


graphPlay1(*randomPlay())

Nice! What are the distributions of X and Y?
# In[ ]:


ax_x = raw_data['X'].plot(kind='hist', bins = int(np.max(raw_data['X'])-np.min(raw_data['X'])))
plt.title('Distribution of X')
ax_x.axvline(10, color='r')
ax_x.axvline(110, color='r')
ax_x.axvline(35, color='y')
plt.show()


# The red lines indicate the endzones. The data is symmetric, which is good. What are the odd spikes doing there? Well, after a touchback the ball is placed on the 25 yardline, which I've marked in yellow. As hoped, the spike in handoffs occurs right behind that line. What about handoffs in the endzone? To be expected, since sometimes the offense has to snap from its own 1 or 2 yard line. (Poor guys.)

# In[ ]:


ax_y = raw_data['Y'].plot(kind='hist', bins = int(np.max(raw_data['Y'])-np.min(raw_data['Y'])))
plt.title('Distribution of Y')
hashdist = 23 + 1/3 + 9/36 # how far the hashes are from the sideline
ax_y.axvline(hashdist, color='y')
ax_y.axvline(53.3-hashdist, color='y')


# The hash marks are yellow here. The interesting thing here is that there are small local peaks around 10 yards from the sidelines. Presumably that's where tosses get caught?

# ## S, Dir, A
# 
# we can make our picture nicer by including arrows indicating the speed and direction of each.

# In[ ]:


def graphPlay2(GameId, PlayId):
    '''
    Makes a visual depiction of the play.
    The ball carrier is marked yellow.
    The arrows indicate the direction of movement of the player. 
    The length of the black arrow is proportional to the speed.
    The length of the orange arrow is proportional to the accel. 
    The direction of accel is not provided, and so it is plotted in the direction of motion.
    '''
    play_data = raw_data[(raw_data['GameId'] == GameId) & (raw_data['PlayId'] == PlayId)]
    colors = {'away':'red', 'home':'blue'}
    print(play_data.shape)
    play_data.index = range(22)
    print(play_data.loc[0,'OffenseFormation'])
    print(play_data.loc[0,'PlayDirection'])
    print(play_data.loc[0,'TimeHandoff'])
    print(play_data.loc[0,'Yards'])
    for i in play_data.index:
        playerInfo = play_data.loc[i,:]
        if playerInfo['NflId'] != playerInfo['NflIdRusher']:
            plt.plot('X', 'Y', 'ro', markersize=10, color=colors[play_data.loc[i,'Team']], data = playerInfo)
        else:
            plt.plot('X', 'Y', 'ro', markersize=10, color='yellow', data = playerInfo)
        plt.arrow(playerInfo['X'], playerInfo['Y'], playerInfo['S']*sin(radians(playerInfo['Dir'])), playerInfo['S']*cos(radians(playerInfo['Dir'])), head_width=0.5)
        plt.arrow(playerInfo['X'], playerInfo['Y'], playerInfo['A']*sin(radians(playerInfo['Dir'])), playerInfo['A']*cos(radians(playerInfo['Dir'])),color='orange',head_width=0.5)
    plt.show()
    return None


# In[ ]:


graphPlay2(*randomPlay())


# In[ ]:


raw_data.groupby(['GameId','PlayId']).count()


# Now, I want to check out the distributions of the these quantities.

# In[ ]:


raw_data['S'].plot(kind='hist', bins=50)
plt.title('Speeds of players')
plt.show()

raw_data['A'].plot(kind='hist', bins=50)
plt.title('Accels of players')
plt.show()

raw_data['Dir'].plot(kind='hist', bins=50)
plt.title('Angle of motion of players')


# The suprising thing to see here is that the main peaks for angle of motion is not along the length of the field, but rather towards the sidelines. I guess this indicates that many of the running plays are not up the middle, but rather out towards the sides. This maybe agrees with the observation that many handoffs occur near the sideline.

# ## Orientation
# 
# This is the direction the player is facing? I'll plot it and compare it to the direction of motion to make sure everything seems sound.
# 

# In[ ]:


raw_data['Orientation'].plot(kind='hist', bins= 50, alpha=.9)
raw_data['Dir'].plot(kind='hist', bins= 50, alpha=.4)
plt.legend(bbox_to_anchor = (1.1,0.5), bbox_transform = plt.gcf().transFigure)
plt.xlabel('Angle')
plt.title('Player orientation and movement direction')


# Interesting. Players are looking towards up and downfield much more often than they're actually moving up and down the field. 

# ## Offense vs Defense
# 
# I wonder how being on offense or defense affects the data? There's not a column that indicates if the player is offense or defense, so let's make a mask for that. (I initially tried to add a column, but the way I did it took too long. This is faster.)

# In[ ]:


def abbrConv(abbr):
    '''
    convert from the XTeamAbbr to PossesionTeam
    see code cell below for why we need this
    '''
    if abbr == 'ARI':
        return 'ARZ'
    elif abbr == 'BAL':
        return 'BLT'
    elif abbr == 'CLE':
        return 'CLV'
    elif abbr == 'HOU':
        return 'HST'
    else:
        return abbr

def isOffense(row):
    offense = row['PossessionTeam']
    side = row['Team']
    if side == 'away':
        key = 'VisitorTeamAbbr'
    else:
        key = 'HomeTeamAbbr'
    if offense == abbrConv(row[key]):
        return True
    else:
        return False
    
offenseMask = raw_data.apply(isOffense, axis=1)
        


# In[ ]:


print(2*offenseMask.sum())
len(raw_data.index)


# In[ ]:



# annoying inconsistency: There are different sets of abbreviations for the two teams! GRR.
team_data = pd.DataFrame(raw_data['PossessionTeam'].value_counts())
team_data = pd.concat([team_data, raw_data['HomeTeamAbbr'].value_counts(), raw_data['VisitorTeamAbbr'].value_counts()], axis = 1)
team_data


# In[ ]:


# now that we have the offense mask, let's replot some of the old data with it
raw_data['S'][offenseMask].plot(kind='hist', bins=50, color='r', alpha=.5)
raw_data['S'][~offenseMask].plot(kind='hist', bins=50,alpha=.5)
plt.title('Speeds of players')
plt.legend(['Offense','Defense'])
plt.show()

raw_data['A'][offenseMask].plot(kind='hist', bins=50,color='r', alpha=.5)
raw_data['A'][~offenseMask].plot(kind='hist', bins=50, alpha=.5)
plt.title('Accels of players')
plt.legend(['Offense','Defense'])
plt.show()

raw_data['Dir'][offenseMask].plot(kind='hist', bins=50, color='r',alpha=.5)
raw_data['Dir'][~offenseMask].plot(kind='hist', bins=50, alpha=.5)
plt.title('Angle of motion of players')
plt.legend(['Offense','Defense'])
plt.show()


# Interesting! Contrary to what I thought, the offensive players are accelerating more and moving faster at the time of handoff. They also account for the spike in motion to the sidelines.

# ## NFL id, DisplayName, Jersey Number
# 
# These are all attributes of the individual player. Some questions come to mind: 
# 1. Does the NFL id stay consistent with the player from season to season?
# 2. Does the DisplayName change? Are there variations in it?

# In[ ]:


id_name_pairs = raw_data[['NflId', 'DisplayName']].drop_duplicates()
id_name_counts = id_name_pairs.groupby('NflId').count().reset_index().rename(columns = {'DisplayName':'Count'})
id_name_counts.describe()


# Aha! So somebody has more than one name. Who?

# In[ ]:


double_name_IDs = id_name_counts[id_name_counts['Count'] > 1]['NflId']
id_name_pairs[id_name_pairs['NflId'].isin(double_name_IDs)].sort_values('NflId')


# Long story short is to use the IDs and not the player names for identifying players! And just for fun...

# In[ ]:


raw_data['JerseyNumber'].plot(kind='hist', bins=50)
# popular jersy numbers, weighted by number of plays!


# ## Season
# 
# Which year the game took place: 2017 or 2018. Somewhere I read that the measurement data is slightly different from year to year, so this feature might be important

# In[ ]:


raw_data['Season'].value_counts().plot(kind='bar')


# ## Yardline, FieldPosition
# 
# This should be incorporated into the visualization! It will probably also be important in modelling, since the relative distance the players are from the yardline is probably a key factor.

# In[ ]:


raw_data['YardLine'].plot(kind='hist', bins=50)


# Notice that they only go from 0 to 50, so we'll have to do a little more work to convert the yardline into an X coordinate for the modelling. (The 25 yardline from the touchback is clear.) As specified in the competition's rule, X=0 corresponds to the home team's endzone. Hence we need to use "FieldPosition" to correct it.

# In[ ]:


raw_data['FieldPosition'].value_counts().plot(kind='bar')


# They're not "home" and "visitor"... they're the abbreviations! We can fix this. Let's copy and paste the abbrevation code from before to see what sorts of abbreviations these are.

# In[ ]:



field_pos_data = pd.DataFrame(raw_data['FieldPosition'].value_counts())
field_pos_data = pd.concat([team_data, raw_data['HomeTeamAbbr'].value_counts(), raw_data['VisitorTeamAbbr'].value_counts()], axis = 1)
field_pos_data


# So we need to convert 'FieldPosition' to compare to home-visitor team abbrevations. 

# In[ ]:


def yardline_to_X(row):
    poss_team = row['PossessionTeam']
    field_team = row['FieldPosition']
    yardline = row['YardLine']
    print("Poss:", poss_team)
    print("Field:", field_team)
    print("Yardline:", yardline)
    print(row['PlayDirection'])
    if row['PlayDirection'] == 'left':
        # <<<
        if poss_team == field_team:
            return 110-yardline
        else:
            return 10+yardline
    else:
        # >>>
        if poss_team == field_team:
            return 10+yardline
        else:
            return 110-yardline
        
    
#raw_data.apply(yardLine_to_X, axis=1).plot(kind='hist', bins=100)


# The spikes at X=35 and X=85 are the expected touchback spikes. The spikes at the endzones are also acceptable. What's a bit confusing is the giant gap at the 50 yard line! What's that all about?

# In[ ]:


def graphPlay3(GameId, PlayId):
    '''
    Makes a visual depiction of the play.
    The ball carrier is marked yellow.
    The arrows indicate the direction of movement of the player. 
    The length of the black arrow is proportional to the speed.
    The length of the orange arrow is proportional to the accel. 
    The direction of accel is not provided, and so it is plotted in the direction of motion.
    The dotted blue line is the line of scrimmage
    '''
    play_data = raw_data[(raw_data['GameId'] == GameId) & (raw_data['PlayId'] == PlayId)]
    colors = {'away':'red', 'home':'blue'}
    print(play_data.shape)
    play_data.index = range(22)
    print(play_data.loc[0,'OffenseFormation'])
    print(play_data.loc[0,'PlayDirection'])
    print(play_data.loc[0,'TimeHandoff'])
    print(play_data.loc[0,'Yards'])
    ax = plt.gca()
    scrimmX = yardline_to_X(play_data.loc[0,:])
    ax.axvline(scrimmX, ls='--')
    
    if play_data.loc[0,'PlayDirection'] == 'left':
        downline = scrimmX-play_data.loc[0,'Distance']
    else:
        downline = scrimmX+play_data.loc[0,'Distance']
    ax.axvline(downline, ls='--', color='orange')
    for i in play_data.index:
        playerInfo = play_data.loc[i,:]
        if playerInfo['NflId'] != playerInfo['NflIdRusher']:
            plt.plot('X', 'Y', 'ro', markersize=10, color=colors[play_data.loc[i,'Team']], data = playerInfo)
        else:
            plt.plot('X', 'Y', 'ro', markersize=10, color='yellow', data = playerInfo)
        plt.arrow(playerInfo['X'], playerInfo['Y'], playerInfo['S']*sin(radians(playerInfo['Dir'])), playerInfo['S']*cos(radians(playerInfo['Dir'])), head_width=0.5)
        plt.arrow(playerInfo['X'], playerInfo['Y'], playerInfo['A']*sin(radians(playerInfo['Dir'])), playerInfo['A']*cos(radians(playerInfo['Dir'])),color='orange',head_width=0.5)
    plt.show()
    return None


# In[ ]:


graphPlay3(*randomPlay())


# ## Quarter, GameClock

# In[ ]:


raw_data['GameClock'].value_counts()


# Given the fact that a quater starts with 15 mins and there's a 2 min warning, this seems OK. The games with 0 on the clock seem... troubling.

# In[ ]:


raw_data[raw_data['GameClock']=='00:00:00']


# In[ ]:


graphPlay3(2017110600, 20171106003613)


# This is a real play that happened at 00:00:00! If you check out [this play](https://www.nfl.com/gamecenter/2017110600/2017/REG9/lions@packers) you'll find that the play before there was an interference, so presumably the offence got another crack at scoring. 

# In[ ]:


raw_data['Quarter'].value_counts()

def gameClock(PlayId):
    row = raw_data[raw_data['PlayId'] == PlayId]
    res = 'Q' + str(row.iloc[0]['Quarter']) + ' - ' + str(row.iloc[0]['GameClock'])
    return res
    
gameClock(randomPlay()[1])


# ## Down
# 
# I don't expect much interesting here. But I'll make a handy function to print out the down and distance for a play!

# In[ ]:


raw_data['Down'].value_counts().plot(kind='bar')


# Not many runs on fourth down! I'm sort of curious about how fourth down runs actually do...

# In[ ]:


for i in range(1,5):
    plays = raw_data[raw_data['Down'] == i]
    plays['Yards'].plot(kind='hist', bins=20)
    plt.title('Yard on down ' + str(i))
    print('Average', plays['Yards'].mean(), 'yards on down', i)
    plt.show()


# In[ ]:


def downAndDist(playId):
    row = raw_data[raw_data['PlayId'] == playId].iloc[0]
    return str(row['Down']) + ' and ' + str(row['Distance'])

downAndDist(randomPlay()[1])


# In[ ]:


def fieldPos(playId):
    row = raw_data[raw_data['PlayId'] == playId].iloc[0]
    return row['FieldPosition'] + ' ' + str(row['YardLine'])

fieldPos(randomPlay()[1])


# ## HomeScoreBeforePlay, VisitorScoreBeforePlay
# 
# Scores before the play starts.

# In[ ]:


raw_data['HomeScoreBeforePlay'].value_counts().plot(kind='bar')
plt.show()

raw_data['VisitorScoreBeforePlay'].value_counts().plot(kind='bar')
plt.show()


# In[ ]:


def scoreString(playId):
    row = raw_data[raw_data['PlayId'] == playId].iloc[0]
    return row['HomeTeamAbbr'] + ' ' + str(row['HomeScoreBeforePlay']) + '\n' + row['VisitorTeamAbbr'] + ' ' + str(row['VisitorScoreBeforePlay'])

print(scoreString(randomPlay()[1]))
                                     


# ## Offense Formation, Offense Personnel
# 
# These are a couple of columns that tell us about how things looked before the snap. Let's check them out.

# In[ ]:


raw_data['OffenseFormation'].value_counts()


# In[ ]:


raw_data['OffensePersonnel'].value_counts()


# The data in this entry is rather messy: I think if there are 5 OL it doesn't bother to put them, and leaves 1 QB out as well. If those values change, then they're included. Occasionally a defensive player is thrown into the mix. If this data is processed a bit to standardize it, maybe it could be useful.

# In[ ]:


print(raw_data['DefendersInTheBox'].value_counts())

raw_data[['DefendersInTheBox', 'Yards']].groupby('DefendersInTheBox').mean().plot(kind='bar')


# Based on this histogram, defenders in the box isn't a bad feature to use for prediction?

# In[ ]:


raw_data['DefensePersonnel'].value_counts()


# Unlike offensive personnel, this data column doesn't have quite as much variability. It seems the numbers always add up to 11 as expected. Occasionally a random OL is thrown in.

# ## Play Direction
# 
# Something I've been wanting to do for a while now is standardize all the plays to be going to the right. I think I'll do this in the next iteration of EDA.

# ## TimeSnap, TimeHandoff
# 
# My gut says that alone, neither of these datapoints are too interesting. However, perhaps together they can tell us something.

# In[ ]:


def handoffDelay(playId):
    row = raw_data[raw_data['PlayId']==playId].iloc[0]
    snap = pd.Timestamp(row['TimeSnap'])
    handoff = pd.Timestamp(row['TimeHandoff'])
    return (handoff-snap).seconds
    
def handoffDelayFromRow(row):
    snap = pd.Timestamp(row['TimeSnap'])
    handoff = pd.Timestamp(row['TimeHandoff'])
    return (handoff-snap).seconds

handoffs = raw_data.apply(handoffDelayFromRow, axis=1)


# In[ ]:


handoffDelay(randomPlay()[1])


# In[ ]:


handoffs.value_counts().plot(kind='bar')


# ## Player info
# 
# There is a column for player height, player weight, player birthdate, and player college name. Height and weight seem be more useful for predicting than college name and birthdate. I'll probably exlude them from the first models I build. Just for fun, though...

# In[ ]:


raw_data['PlayerWeight'].plot(kind='hist', bins=50)
plt.show()

raw_data['PlayerHeight'].value_counts().plot(kind='bar')
plt.show()

print(raw_data['PlayerCollegeName'].value_counts())

# make a graph of player's ages
raw_data['PlayerBirthDate'].apply(lambda x: int((pd.Timestamp.now() - pd.Timestamp(x)).days/365)).value_counts().plot(kind='bar')


# ## Week
# 
# Which week of the season the game is being played during. Let's see if it correlates with yardage...

# In[ ]:





# In[ ]:


sigma = raw_data['Yards'].std()
mu = raw_data['Yards'].mean()
errors = sigma/np.sqrt(raw_data.groupby('Week')['Yards'].count())

ax = raw_data.groupby('Week')['Yards'].mean().plot(kind='bar', yerr=2*errors)
ax.axhline(raw_data['Yards'].mean())
plt.show()


# So I've marked the error bars according to what I should be getting for a SRS of the given size of that week (I hope...) and we find that for some reason, clumping games week by week makes it behave very differently from a simple random sample. In fact, let's experiment a little:

# In[ ]:


means = []
errs = []
plt.gca().axhline(raw_data['Yards'].mean())
for i in range(17):
    sample = raw_data.sample(frac=1/17)
    means.append(sample['Yards'].mean())
    errs.append(2*sigma/np.sqrt(sample['Yards'].count()))
plt.bar(range(17), means, yerr=errs)


# So yeah, the week-by-week grouping behaves very differently from the random grouping! Maybe it's because the plays are grouped in block of 22?

# ## Stadium Type, Turf

# In[ ]:


raw_data['StadiumType'].value_counts()


# In[ ]:


raw_data['Turf'].value_counts()


# Both of these will need cleaning before they'll be useful.

# ## Weather Stats

# In[ ]:


pd.set_option('display.max_rows', 1000)
print(raw_data['GameWeather'].value_counts())
pd.set_option('display.max_rows', 10)


# In[ ]:


raw_data['Temperature'].plot(kind='hist',bins=20)


# In[ ]:


raw_data['Humidity'].plot(kind='hist', bins=20)


# Scatter plots of temp and humidity make me think they're not so important.

# In[ ]:


raw_data['WindSpeed'].value_counts()


# So windspeed is a weird column that contains not only speeds but... directions?

# In[ ]:


raw_data['WindDirection'].value_counts()


# Aha. Look how there's a random numeric value here... someone must have swapped speed and direction?

# ## Cleanup Summary
# 
# After exploring all these variables, here are the operations I'd like to do to each before running any algs on them:
# 
# **Ready-To-Go Features**
# The following features should be considered for models and kept with only a typecast or simple recoding.
# 
# 1. GameId, PlayId
# 2. Team
# 3. S, A
# 3. Dis
# 3. NflId
# 4. JerseyNumber
# 5. Season
# 6. Quarter
# 7. Down
# 8. Distance
# 9. HomeScoreBeforePlay
# 10. VisitorScoreBeforePlay
# 11. DefendersInTheBox
# 12. PlayerHeight
# 13. PlayerWeight
# 14. Week
# 15. Temperature
# 16. Humidity
# 
# **Position and Direction Changes**
# 
# I think it'd be best to make all plays going towards the right and to only record the player's X distance relative to the line of scrimmage or ball carrier, rather than absolute X. 
# 
# 1. X - shift and flip
# 2. Y - shift
# 3. Orientation - change angle
# 4. Dir - change angle
# 5. Yardline - must be altered to an appropriate X value
# 
# **Drops**
# 
# Remove these features.
# 
# 1. Displayname - already accountedfor in NflId
# 2. NflIdRusher - instead, make the rusher a specific column of the data entry
# 3. FieldPosition - incorporated into the yardline after it's been shifted
# 4. PlayerCollegeName - it seems unlikely this is important compared to other stats
# 5. PlayDirection - used to flip other vars, then not needed
# 
# **Conversions and Recodings**
# 
# Standardize the following.
# 
# These variables have different conventions.
# 1. HomeTeamAbbr
# 2. VisitorTeamAbbr
# 3. PossessionTeam
# 
# Convert the gameclock to raw seconds
# 1. Gameclock
# 
# Convert birthdate to age
# 1. PlayerBirthDate
# 
# Convert these to a more standard data format or delete
# 1. TimeSnap
# 2. TimeHandoff
# 
# Recode for consistency
# 1. OffenseFormation
# 2. GameWeather
# 3. StadiumType
# 4. Turf
# 5. Location
# 6. WindDirection/WindSpeed - Swap the odd value out
# 7. Stadium
# 
# 
# 

# In[ ]:


raw_data['Dis'].plot(kind='hist', bins=50)


# In[ ]:




