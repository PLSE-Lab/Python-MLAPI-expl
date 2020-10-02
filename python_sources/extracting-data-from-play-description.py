#!/usr/bin/env python
# coding: utf-8

# # **Data Exploration, Extraction, and Visualization using Pandas, RE, and Matplotlib**

# I wanted to try and make the biggest and most informational dataset using the NGS and play information provided in this competition. Using pandas, matplotlib, and some regular expressions, I attempted to extract as much data form the PlayDescription field as I could and join this new table with more data provided through NGS analysis tables. Hopefully it proves to be helpful. Enjoy.

# In[ ]:


# workspace prep 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# First, let's get our data in here.

# In[ ]:


# import data & look at data
play = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')
player_role =pd.read_csv("../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv") 
player = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')


# Next, we need to check our data quality. Missing values and other data imperfections can prove to be quite the pain.

# In[ ]:


# data quality check
dfs = ([play, player_role, player])

for df in dfs:
    
    print("dataframe information")
    nan_count = df.apply(lambda x: x.count(), axis=0)
    if sum(nan_count) == len(df)*len(df.columns):
        print('No Missing Values')
    elif nan_count != len(df):
        print(nan_count)
    
    print(df.shape)
    print(df.info())
    print(df.head())


# Great! Seems we were provided with some high quality data. Let's move on. Next, we will conduct some simple joins using the all powerful pandas package. Using the unique fields og GSISID, GameKey, and PlayID, we can create a very imformative data set from which we can gain some insights on the injuries during punt plays.  

# In[ ]:


# join on proper keys - no missing data
# first two on GSISID
# then on GameKey and PlayID to get the full data set for each player    
full_players = player.merge(player_role, left_on='GSISID',right_on='GSISID',how = 'left')
full_set = full_players.merge(play, left_on=['GameKey','PlayID'],
                              right_on = ['GameKey','PlayID'],
                              how = 'left')
print(full_set.info())
print(full_set.isna().sum())
full_set.head()


# Now we can drop the null values to ensure data quality.

# In[ ]:


#drop the null values
df=full_set.dropna()


# ## Working with Play Description

# Since I wanted to work with the text in the certain columns, some coloumns need to become strings to do so.

# In[ ]:


# need to split the score column and also the home and away column into 4 
# diff columns 
df['Home_Team_Visit_Team'] = df['Home_Team_Visit_Team'].astype(str)
df['Score_Home_Visiting'] = df['Score_Home_Visiting'].astype(str)


# We can now split the comlums we turned into strings into seperate variables using string split, and we can also quickly change the date format.

# In[ ]:


# splits
df=df.join(df['Home_Team_Visit_Team'].str.split('-', 1, expand=True).rename(columns={0:'Home',1:'Away'}))
df=df.join(df['Score_Home_Visiting'].str.split(' - ', 1, expand=True).rename(columns={0:'Home_score',1:'Away_score'}))

# Date
df["Game_Date"] = pd.to_datetime(df["Game_Date"], format = '%m/%d/%Y')

# drop columns that were split
df = df.drop(['Home_Team_Visit_Team'], axis = 1)
df = df.drop(['Score_Home_Visiting'], axis = 1)


# In[ ]:


df.head()


# With those few lines of code, we now have more data than we did before that we can isolate and begin working with. Let's do the same for Play Description to get dummy variables and numerical variables for punt length, return length, fair catch, injury, penatly, a downed punt, fumbles, muffed punts, touchdowns, and touchbacks. This will allow for us to have the most amount of data possible for analysis later on. We can do this using a combination of for loops, regular expression, and basic string selection in Python. 

# In[ ]:


# Extract key information from the Play Description string variable using re
df['PlayDescription'] = df['PlayDescription'].astype(str)

# punt length 
import re 
punt_length = []
for row in df['PlayDescription']:
    match = re.search('punts (\d+)', row)
    if match:
        punt_length.append(match.group(1))
    elif match is None:
        punt_length.append(0)
        
# return length
return_length = []
for row in df['PlayDescription']:
    match = re.search('for (\d+)', row)
    if match:
        return_length.append(match.group(1))
    elif match is None:
        return_length.append(0)
            
# fair catch
fair_catch = []
for row in df['PlayDescription']:
    match = re.search('fair catch', row)
    if match:
        fair_catch.append(1)
    elif match is None:
        fair_catch.append(0)

# injury
injury = []
for row in df['PlayDescription']:
    match = re.search('injured', row)
    if match:
        injury.append(1)
    elif match is None:
            injury.append(0)

# penalty         
penalty = []
for row in df['PlayDescription']:
    if 'Penalty' in row.split():
        penalty.append(1)
    elif 'PENALTY' in row.split():
        penalty.append(1)
    elif 'Penalty' not in row.split():
        penalty.append(0)
    elif 'PENALTY' not in row.split():
        penalty.append(0)
        

# downed
downed = []
for row in df['PlayDescription']:
    match = re.search('downed', row)
    if match:
        downed.append(1)
    elif match is None:
        downed.append(0)

# fumble
fumble = []
for row in df['PlayDescription']:
    match = re.search('FUMBLES', row)
    if match:
        fumble.append(1)
    elif match is None:
        fumble.append(0)

# muff
muff = []
for row in df['PlayDescription']:
    match = re.search('MUFFS', row)
    if match:
        muff.append(1)
    elif match is None:
        muff.append(0)

# Touchback
touchback = []
for row in df['PlayDescription']:
    match = re.search('Touchback', row)
    if match:
        touchback.append(1)
    elif match is None:
        touchback.append(0)

# Touchdown
touchdown = []
for row in df['PlayDescription']:
    match = re.search('TOUCHDOWN', row)
    if match:
        touchdown.append(1)
    elif match is None:
        touchdown.append(0)

# add new columns to the df 
df["punt_length"] = punt_length
df["return_length"] = return_length
df["fair_catch"] = fair_catch
df["injury"] = injury
df["penalty"] = penalty
df["downed"] = downed
df["fumble"] = fumble
df['muff'] = muff
df['touchback'] = touchback
df['touchdown'] = touchdown


# Take a look at your new and imporved dataframe, ready to tell the full story about each player for every punt play provided in the data.

# In[ ]:


df.head()


# I wanted to make this new dataset as robust and informational as possible so it seemed right to add the corresponding NGS data as well. Due to the sheer size of all the data, I used this amazing and helpful [kernal](http://www.kaggle.com/kmader/convert-to-feather-for-use-in-other-kernels/) by the great [Kevin Mader](http://www.kaggle.com/kmader). Using Apache Feather by the Pandas Father Wes McKinney, the NGS data file is cut nearly into a quarter of its original size, reducing its impact on the disk when read in by pandas, allowing for the kernal to survive the import. The NGS file contains all the NGS data as the serperate files share column names, making the concat process seamless.

# ## More Joining using the feathered data file

# In[ ]:


import feather
df_final = feather.read_dataframe('../input/feathered-ngs/ngs.feather')


# Let's take a peak at the data we just loaded in.

# In[ ]:


print(df_final.shape)
df_final.head()


# Lots of data there but we will soon trim it down. Let's do some joining with pandas again.

# In[ ]:


new_df = df.merge(df_final.drop_duplicates(subset=['GSISID','GameKey','PlayID']), how='left',
                  left_on=['GSISID','GameKey','PlayID','Season_Year_x'], right_on = ['GSISID','GameKey','PlayID','Season_Year'])
del df_final


# In[ ]:


new_df.head()


# We are beginning to see the power of proper joining, giving us a very insightful data set. We are not done yet though. Let do some final touches and trim down the columns.

# In[ ]:


# game data
game = pd.read_csv('../input/NFL-Punt-Analytics-Competition/game_data.csv')

game_with_new = new_df.merge(game, how = 'left', left_on = "GameKey", 
                             right_on = "GameKey")
# columns to keep 
keep = ['GSISID', 'Number', 'Position','Season_Year_x', 'GameKey', 'PlayID',
       'Role', 'Game_Date_x', 'Week_x',
       'Game_Clock', 'YardLine', 'Quarter', 'Play_Type', 'Poss_Team',
       'Home', 'Away', 'Home_score', 'Away_score',
       'punt_length', 'return_length', 'fair_catch', 'injury', 'penalty',
       'downed', 'fumble', 'muff', 'touchback','touchdown','x',
       'y', 'dis', 'o', 'dir', 'Event', 'Season_Type_y',
       'Game_Day', 'Game_Site', 'Start_Time',
       'Home_Team', 'Visit_Team', 'Stadium',
       'StadiumType', 'Turf', 'GameWeather', 'Temperature', 'OutdoorWeather'
       ]
df_clean = game_with_new[keep]
del game_with_new

# rename columns
headers = ['GSISID', 'Number', 'Position', 'Season_Year','Season_Year_x', 'GameKey', 'PlayID',
       'Role', 'Game_Date', 'Week',
       'Game_Clock', 'YardLine', 'Quarter', 'Play_Type', 'Poss_Team',
       'Home', 'Away', 'Home_score', 'Away_score',
       'punt_length', 'return_length', 'fair_catch', 'injury', 'penalty',
       'downed', 'fumble', 'muff', 'touchback','touchdown','x',
       'y', 'dis', 'o', 'dir', 'Event', 'Season_Type',
       'Game_Day', 'Game_Site', 'Start_Time',
       'Home_Team', 'Visit_Team', 'Stadium',
       'StadiumType', 'Turf', 'GameWeather', 'Temperature', 'OutdoorWeather'
       ]
df_clean.columns = headers


# In[ ]:


print(df_clean.dtypes)
df_clean[["punt_length", "return_length"]] = df_clean[["punt_length", "return_length"]].apply(pd.to_numeric)
df_clean = df_clean.drop(columns='Season_Year_x')
df_clean.head()


# We now have a data set that gives us in depth looks at every play a player was out for a punt for the last two years. We can now begin to subset this data based on injuries and begin to gain some insights to injuries on punt plays in the NFL.

# ## Injury Exploration

# In[ ]:


# lets look at how many games and punts there are 
games = len(df_clean['GameKey'].unique().tolist())
print('There are ' + str(games) + ' games in the dataset.')
punts = len(df_clean['PlayID'].unique().tolist())
print('There are ' + str(punts) + ' punts in the dataset.')
print('On average, there are ' + str(punts/games) + ' punts per game.')


# In[ ]:


# let's start with the injury field
no_injuries = df_clean.loc[df_clean['injury'] == 0]
injuries = df_clean.loc[df_clean['injury'] == 1]


# We now have two differnt data set that contain injury plays and none injury plays and the data for each player on the field. We will need to be careful when doing aggregation to use unique vales when appropriate. We can need to quickly define an avergae function for lists since .nunique() will be returning lists.

# In[ ]:


# average function 
def avg(lst):
    return sum(lst)/len(lst)

# Number of injuries
print('There are ' + str(len(injuries['PlayID'].unique().tolist())) + ' injuries in the dataset.')

# lets look at the average punt length and return lenth for both new dfs
print('The average punt length for a play with an injury is ' + str(avg(injuries['punt_length'].unique().tolist())))
print('The average punt length for a play without an injury is ' + str(avg(no_injuries['punt_length'].unique().tolist())))
print('The average punt return for a play with an injury is ' + str(avg(injuries['return_length'].unique().tolist())))
print('The average punt return for a play without an injury is ' + str(avg(no_injuries['return_length'].unique().tolist())))


# The difference between the average punt return lengths for plays with and without injuries is jarring. This is likely due to the returner being stopped abruptly by the coverage team with a bone-crushing tackle.

# In[ ]:


#injuries by gameday
total_injuries = injuries.groupby('Game_Day')['PlayID'].nunique()
total_no_injuries = no_injuries.groupby('Game_Day')['PlayID'].nunique()
print('On Fridays, injuires occured on ' + str(3/203) + ' percent of punt plays.')
print('On Mondays, injuires occured on ' + str(4/312) + ' percent of punt plays.')
print('On Saturdays, injuires occured on ' + str(7/602) + ' percent of punt plays.')
print('On Sundays, injuires occured on ' + str(56/2648) + ' percent of punt plays.')
print('On Thursdays, injuires occured on ' + str(16/871) + ' percent of punt plays.')


# While Sunday is the most represented day in the data set, it still seems the odds of being hurt on a punt play on Sunday are still the highest. Keep your head on a swivel!

# Let's now do some quick but fun bar charts and histograms to wrap up.

# In[ ]:


# injuries by game site
injuries.groupby('Game_Site')['PlayID'].nunique().plot(kind='bar',figsize=(18, 16))
plt.xlabel('Week')
plt.ylabel('Injuries')
plt.title('Injuries per Location')
plt.show()


# Miami Garden seems to be a pretty rough place to play or the Dolphins are getting it doen on special teams. This probably deserves more analysis.

# In[ ]:


# injuries by season year
injuries.groupby('Season_Year')['PlayID'].nunique().plot(kind='bar',figsize=(12, 10))
plt.xlabel('Year')
plt.ylabel('Injuries')
plt.title('Injuries (2016-2017)')
plt.show()


# Even with increased awareness of player safety, there were more injuries in 2017 than 2016.

# In[ ]:


# injuries by muff
data = injuries.groupby('muff')['PlayID'].nunique().plot(kind='bar', figsize=(12, 10))
plt.xlabel('muff')
plt.ylabel('Injuries')
plt.title('Injuries on Muffs')
plt.show()


# Muffs do not seem to be a cause of injury.

# In[ ]:


# injuries by fumble
data = injuries.groupby('fumble')['PlayID'].nunique().plot(kind='bar', figsize=(12, 10))
plt.xlabel('Fumble')
plt.ylabel('Injuries')
plt.title('Injuries on Fumbles')
plt.show()


# On fumble plays, the focus shifts from the player to the ball on the ground, therefore decreasing injuries dramatically.

# In[ ]:


# injuries by touchdown
data = injuries.groupby('touchdown')['PlayID'].nunique().plot(kind='bar',figsize=(12, 10))
plt.xlabel('Touchdowns')
plt.ylabel('Injuries')
plt.title('Injuries on Touchdowns')
plt.show()


# 6 points don't seem to be a cause for injury.

# In[ ]:


# injuries by week
data = injuries.groupby('Week')['PlayID'].nunique().plot(kind='bar',figsize=(18, 16))
plt.xlabel('Week')
plt.ylabel('Injuries')
plt.title('Injuries per Week')
plt.show()


# Week 5 has more injuries than any other week during both seasons. This deserves further analysis.

# In[ ]:


# injuries by quarter
injuries.groupby('Quarter')['PlayID'].nunique().plot(kind='bar',figsize=(12, 10))
plt.xlabel('Week')
plt.ylabel('Injuries')
plt.title('Injuries per Quarter')
plt.show()


# As the game carries on, injuries begin to happen more. This can be attributed to several different factors. As the game wears on player's become tired, fatigued and often crucial plays are made by these world class athletes on coverage teams resulting big hits.

# In[ ]:


# injuries per season type
injuries.groupby('Season_Type')['PlayID'].nunique().plot(kind='bar', figsize=(12, 10))
plt.xlabel('Season Type')
plt.ylabel('Injuries')
plt.title('Injuries in Pre, Post, and Regular Season Games')
plt.show()


# There are more regular season games through the year, therefore seeing most injuires occuring in the regular season makes sense. That pregame number looks high and probably deserves further analysis.

# In[ ]:


# lets look at teams who have the most injuries
fig, axes = plt.subplots(nrows=1, ncols=2, sharey = True)

injuries.groupby('Home')['PlayID'].nunique().plot(figsize=(18, 16),ax=axes[0],kind='bar')
plt.ylabel('Injuries')
plt.suptitle('Frequency of Injuries by Home (Left) and Away (Right)')
injuries.groupby('Away')['PlayID'].nunique().plot(figsize=(18, 16),ax=axes[1],kind='bar')
plt.show()


# As we saw earlier Miami Gardens has a high number of injuries and we now see that most of those injuries are suffered by the home team, the Dolphins :/

# In[ ]:


# lets look at punt length
cols = ['GameKey', 'PlayID','punt_length','injury']
punt_length = df_clean[cols]
punt_length = punt_length.drop_duplicates()

# histogram for punt length on injuires
fig, axes = plt.subplots(nrows=1, ncols=2)

punt_length['punt_length'].loc[punt_length['injury']==1].plot(ax=axes[0],kind='hist', bins = 10, color = 'red', edgecolor = 'black', figsize=(18, 16))
punt_length['punt_length'].loc[punt_length['injury']==0].plot(ax=axes[1],kind='hist', bins = 10, edgecolor = 'black', figsize=(18, 16))
plt.suptitle('Frequency of Injuries (Red) and Non-Injuries (Blue) by Return length')
plt.show()


# We see that most of the innjuries happen right around the 50-55 yard range. This is likely due to the fact that punts of this length give sufficent time for theplayers on the coverage team to cover the play properly, and deliver blows to the punt returner almost immediately. Punts of this length also allow for the returner to sometimes return the ball if blocked correctly, leading to blindside hits and more opportunites for coverage players to deliver blows as well.

# In[ ]:


# same process for return length
cols = ['GameKey', 'PlayID','return_length','injury']
return_length= df_clean[cols]
return_length= return_length.drop_duplicates()

# histogram for return length on injuires
fig, axes = plt.subplots(nrows=1, ncols=2)

return_length['return_length'].loc[return_length['injury']==1].plot(ax=axes[0],kind='hist', bins = 15, color = 'red', edgecolor='black',figsize=(18, 16))
return_length['return_length'].loc[return_length['injury']==0].plot(ax=axes[1],kind='hist', bins = 15, edgecolor='black',figsize=(18, 16))
plt.suptitle('Frequency of Injuries (Red) and Non-Injuries (Blue) by Return length')
plt.show()


# Concurrent with our prior analysis of punt length, it seems that most of the injuries take place almost immedately after the return process begins. Let's double check with a quick bar graph of fair catches just to be certain.

# In[ ]:


# injuries by fair catch
injuries.groupby('fair_catch')['PlayID'].nunique().plot(kind='bar', figsize=(12, 10))
plt.xlabel('Fair Catch')
plt.ylabel('Injuries')
plt.title('Injuries on Fair Catches')
plt.show()


# It seems we were corrent, most injures are happening directly after the returner begins the return process.

# ## Conclusion 

# Further analysis is needed but it seems that most of the injuries that are taking place on punt plays are happening during the regular season, late in the game, on punts of rougly 50-55 yards, immedately after the returner catches the ball and starts the return process. I will post another kernal if time permits of my analysis of the video data provided as well. I hope you enjoyed this analysis.

# ## Cheers!
