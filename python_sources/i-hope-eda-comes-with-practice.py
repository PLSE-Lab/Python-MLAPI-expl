#!/usr/bin/env python
# coding: utf-8

# ## About this notebook
# 
# This is the first time I'm trying to write a proper EDA kernel. If you have the time, please, leave your suggestions, corrections, tips and tricks.

# ![](https://static.nfl.com/static/content/public/static/img/share/shield.jpg)

# ## Loading data

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# ## General look at our dataset
# 
# It's good to start by having a quick look at how our data looks like, this is what we're going to do in this section.

# ### Original features

# In[ ]:


features_dict = {
    'GameId': "a unique game identifier",
    'PlayId': "a unique play identifier",
    'Team': "home or away",
    'X': "player position along the long axis of the field. See figure below.",
    'Y': "player position along the short axis of the field. See figure below.",
    'S': "speed in yards/second",
    'A': "acceleration in yards/second^2",
    'Dis': "distance traveled from prior time point, in yards",
    'Orientation': "orientation of player (deg)",
    'Dir': "angle of player motion (deg)",
    'NflId': "a unique identifier of the player",
    'DisplayName': "player's name",
    'JerseyNumber': "jersey number",
    'Season': "year of the season",
    'YardLine': "the yard line of the line of scrimmage",
    'Quarter': "game quarter (1-5, 5 == overtime)",
    'GameClock': "time on the game clock",
    'PossessionTeam': "team with possession",
    'Down': "the down (1-4)",
    'Distance': "yards needed for a first down",
    'FieldPosition': "which side of the field the play is happening on",
    'HomeScoreBeforePlay': "home team score before play started",
    'VisitorScoreBeforePlay': "visitor team score before play started",
    'NflIdRusher': "the NflId of the rushing player",
    'OffenseFormation': "offense formation",
    'OffensePersonnel': "offensive team positional grouping",
    'DefendersInTheBox': "number of defenders lined up near the line of scrimmage, spanning the width of the offensive line",
    'DefensePersonnel': "defensive team positional grouping",
    'PlayDirection': "direction the play is headed",
    'TimeHandoff': "UTC time of the handoff",
    'TimeSnap': "UTC time of the snap",
    'Yards': "the yardage gained on the play (you are predicting this)",
    'PlayerHeight': "player height (ft-in)",
    'PlayerWeight': "player weight (lbs)",
    'PlayerBirthDate': "birth date (mm/dd/yyyy)",
    'PlayerCollegeName': "where the player attended college",
    'Position': "the player's position (the specific role on the field that they typically play)",
    'HomeTeamAbbr': "home team abbreviation",
    'VisitorTeamAbbr': "visitor team abbreviation",
    'Week': "week into the season",
    'Stadium': "stadium where the game is being played",
    'Location': "city where the game is being player",
    'StadiumType': "description of the stadium environment",
    'Turf': "description of the field surface",
    'GameWeather': "description of the game weather",
    'Temperature': "temperature (deg F)",
    'Humidity': "humidity",
    'WindSpeed': "wind speed in miles/hour",
    'WindDirection': "wind direction",
}

pd.DataFrame.from_dict(features_dict, orient='index', columns=['Description']).style.set_properties(**{'text-align': 'left'})


# As [Texas Tom](https://www.kaggle.com/tombresee) said, it helps to break the 49 features into groups such as player information, game information, etc. The hidden code below does only that.

# In[ ]:


game_information = ['GameId', 'Team', 'Season', 'Week' ,'HomeTeamAbbr', 'VisitorTeamAbbr', 'Stadium', 'Location', 'StadiumType', 'Turf']
player_information = ['NflId', 'DisplayName', 'JerseyNumber', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate', 'PlayerCollegeName', 'Position', 'NflIdRusher']
weather_information = ['GameWeather', 'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']
formation_information = ['OffenseFormation', 'OffensePersonnel', 'DefendersInTheBox', 'DefensePersonnel']
stats_information = ['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir']
play_information = ['PlayId', 'Yards', 'Down', 'Quarter', 'YardLine', 'GameClock', 'PossessionTeam', 'Distance', 'TimeHandoff', 'TimeSnap', 'NflIdRusher', 'FieldPosition', 
                    'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'Yards', 'PlayDirection']


# ### Instance example
# 
# Let's take a look at a single example of our dataset.

# In[ ]:


df.sample(1).T


# If you like a summary of our data, here it is! I usually can't get much information out of it.

# In[ ]:


df.describe().T


# Now that we have a general idea of how our dataset looks like, let's begin to look at each one of the features and see if we can extract any information.
# 
# We're gonna start by trying to relate the features to Yards (our target variable),

# ## Game & Field information

# Let's start with the game related features.

# In[ ]:


game_information_dict = {key:features_dict[key] for key in game_information}
pd.DataFrame.from_dict(game_information_dict, orient='index', columns=['Description']).style.set_properties(**{'text-align': 'left'})


# We are not gonna talk about some features, here's why:
# * GameId: unique value for each game, can't help us.
# * Team: only two unique values, will not use it for now. (in the future maybe compare the two values somehow)
# * Season: year of the season can't help us, we are predicting for a year that is not present in our training set.
# * HomeTeamAbbr and VisitorTeamAbbr: both have the same unique values, nothing to worry about.
# * Week: I have no idea about how  to look at this feature.
# * Location: we already know the stadium in which the game is played, I don't think the location can help us a lot. Maybe I'll use it for visualization only.

# ### Stadium Type

# Quick look at the percentage of each value of StadiumType in our dataset.

# In[ ]:


df['StadiumType'].value_counts(normalize=True)


# Let's look at some yard gain statistics by type of stadium.

# In[ ]:


df.groupby('StadiumType')['Yards'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(15, 10), title='Yards mean by Stadium Type')


# Maybe it's a good idea to group some types into something like: outside/inside or closed/open. For example, we have the values Outdoor, Outdoors, Oudoor, Indoors, Indoor and they all have a different mean.  

# In[ ]:


def clean_stadium_type(txt):
    if pd.isna(txt):
        return np.nan
    txt=txt.lower()# lower case
    txt=txt.strip()# return a copy
    txt=txt.replace("outdoors","outdoor")
    txt=txt.replace("oudoor","outdoor")
    txt=txt.replace("ourdoor","outdoor")
    txt=txt.replace("outdor","outdoor")
    txt=txt.replace("outddors","outdoor")
    txt=txt.replace("outside","outdoor")
    txt=txt.replace("indoors","indoor")
    txt=txt.replace("retractable ","retr")
    return txt

def transform_stadium_type(txt):
    if pd.isna(txt):
        return np.nan
    if 'outdoor' in txt or 'open' in txt:
        return 1
    if 'indoor' in txt or 'closed' in txt:
        return 0
    
    return np.nan

df["StadiumType"] = df["StadiumType"].apply(clean_stadium_type)
df["StadiumType"] = df["StadiumType"].apply(transform_stadium_type)


# Let's take another look at the percentage of each value of StadiumType, but now after grouping.

# In[ ]:


df['StadiumType'].value_counts(normalize=True)


# In[ ]:


df.groupby('StadiumType')['Yards'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(15, 5), title='Yards mean by Stadium Type')


# In[ ]:


plt.figure(figsize=(15, 10))
sns.boxplot(x=df['StadiumType'], y = df['Yards'], data=df, showfliers=False)


# After grouping the stadium types, it looks like it doesn't matter that much to the amount of yards gained.

# ### Turf

# In[ ]:


df['Turf'].value_counts(normalize=True)


# In[ ]:


df.groupby('Turf')['Yards'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(15, 10), title='Yards mean by Turf')


# Again, let's group them into two groups: Natural and Artificial

# In[ ]:


turf_groups = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 
        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 
        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 
        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 


# In[ ]:


df['Turf'] = df['Turf'].replace(turf_groups)
df['Turf'].value_counts(normalize=True)


# In[ ]:


df.groupby('Turf')['Yards'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(15, 5), title='Yards mean by Turf')


# Just like it happened with the stadium types, grouping the turf doesn't seem to matter that much to the amount of yards gained.

# ### Stadium

# In[ ]:


df['Stadium'].value_counts(normalize=True)


# In[ ]:


df.groupby('Stadium')['Yards'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(15, 13), title='Yards mean by Stadium')


# The mean amount of yards gained in a play seems to change drastically when we look at each one of the stadiums. However, this might have something to do with the teams that play in those stadiums. Needs further investigation.
# 
# After all, it looks like the game related features can't give us much information. What do you think? Are those features *helping your model*?

# ## Play information

# In[ ]:


play_information_dict = {key:features_dict[key] for key in play_information}
pd.DataFrame.from_dict(play_information_dict, orient='index', columns=['Description']).style.set_properties(**{'text-align': 'left'})


# Again, we're not going to talk about some features:
# 
# * PlayId
# * GameClock
# * PossesionTeam
# * TimeHandoff
# * TimeSnap
# * NflIdRusher
# * FieldPosition
# * HomeScoreBeforePlay
# * VisitorScoreBeforePlay
# * PlayDirection
# 
# The main reason is that I don't know how to extract information out of them...

# ### Yards, our happy little target

# Let's start by plotting the variable distribution. **The red line is the mean.**

# In[ ]:


plt.figure(figsize=(15, 10))
sns.distplot(df['Yards'], kde=False).set_title("Yards gained distribution")
plt.axvline(df['Yards'].mean(), linewidth=4, color='r')


# Let's also take a look at **the most commom values.**

# In[ ]:


df['Yards'].value_counts()[:15]


# ### Down
# 
# Let's take a look at the distribution of Yard gain by down.

# In[ ]:


plt.figure(figsize=(15,10))
sns.boxplot(x='Down', y='Yards', data=df, showfliers=False).set_title('Gained yards by down')


# I don't know what conclusions to take from this. The downs with most gained yards are the first and the second. I assumed it would be the other way around.

# ### Quarter
# 
# I'll take a look at this feature out of curiosity. Do players get tired and lose performance throughout the game?

# In[ ]:


plt.figure(figsize=(15,10))
sns.boxplot(x='Quarter', y='Yards', data=df, showfliers=False).set_title('Gained yards by quarter')


# Well, not much, apparently.

# ### YardLine
# 
# I'll take a look at this later since I have no idea what to do with it.

# ### Distance
# 
# This is the distance that needs to be covered to get a first down.

# In[ ]:


plt.figure(figsize=(15,10))
sns.boxplot(x='Down', y='Distance', data=df, showfliers=False)


# It makes perfect sense. At the first down, you always need to cover 10 yards, and that number goes down as you move forward.

# At this point I'm starting to believe I'm doing it all wrong. Most of the features I looked at so far doesn't seem to provide much information about tha amount of yards gained.

# ## This is a work in progress...
# 
# I'm currently working on learning how to do a good data exploration. I would love to hear what you have to say.
