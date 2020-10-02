#!/usr/bin/env python
# coding: utf-8

# ## Let's Start new Competition!
# 
# 

# A new contest has begun!!
# 
# How about seeing all the features before entering the competition?

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<a href="http://a.video.nfl.com//films/vodzilla/153321/Lechler_55_yd_punt-lG1K51rf-20181119_173634665_5000k.mp4"> \n    (2:57) (Punt formation) S.Lechler punts 48 yards to TEN 16, Center-J.Weeks. A.Jackson pushed ob at TEN 32 for 16 \n    yards (J.Jenkins).\n</a> \n<img src="https://s3.amazonaws.com/nonwebstorage/headstrong/animation_585_733_3.gif" width="650">')


# image from https://www.kaggle.com/jpmiller/nfl-punt-analytics

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("Set2")
import os
print(os.listdir('../input/nfl-big-data-bowl-2020'))


# In[ ]:


from kaggle.competitions import nflrush
env = nflrush.make_env()


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)\ntrain_df.head()")


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe(include='all')


# ## Data
# 
# - GameId - a unique game identifier
# - PlayId - a unique play identifier
# - Team - home or away
# - X - player position along the long axis of the field. See figure below.
# - Y - player position along the short axis of the field. See figure below.
# - S - speed in yards/second
# - A - acceleration in yards/second^2
# - Dis - distance traveled from prior time point, in yards
# - Orientation - orientation of player (deg)
# - Dir - angle of player motion (deg)
# - NflId - a unique identifier of the player
# - DisplayName - player's name
# - JerseyNumber - jersey number
# - Season - year of the season
# - YardLine - the yard line of the line of scrimmage
# - Quarter - game quarter (1-5, 5 == overtime)
# - GameClock - time on the game clock
# - PossessionTeam - team with possession
# - Down - the down (1-4)
# - Distance - yards needed for a first down
# - FieldPosition - which side of the field the play is happening on
# - HomeScoreBeforePlay - home team score before play started
# - VisitorScoreBeforePlay - visitor team score before play started
# - NflIdRusher - the NflId of the rushing player
# - OffenseFormation - offense formation
# - OffensePersonnel - offensive team positional grouping
# - DefendersInTheBox - number of defenders lined up near the line of scrimmage, spanning the width of the offensive line
# - DefensePersonnel - defensive team positional grouping
# - PlayDirection - direction the play is headed
# - TimeHandoff - UTC time of the handoff
# - TimeSnap - UTC time of the snap
# - Yards - the yardage gained on the play (you are predicting this)
# - PlayerHeight - player height (ft-in)
# - PlayerWeight - player weight (lbs)
# - PlayerBirthDate - birth date (mm/dd/yyyy)
# - PlayerCollegeName - where the player attended college
# - HomeTeamAbbr - home team abbreviation
# - VisitorTeamAbbr - visitor team abbreviation
# - Week - week into the season
# - Stadium - stadium where the game is being played
# - Location - city where the game is being player
# - StadiumType - description of the stadium environment
# - Turf - description of the field surface
# - GameWeather - description of the game weather
# - Temperature - temperature (deg F)
# - Humidity - humidity
# - WindSpeed - wind speed in miles/hour
# - WindDirection - wind direction

# In[ ]:


train_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import missingno as msno\nmsno.matrix(train_df)')


# ## GameId & NflId

# In[ ]:


get_ipython().run_cell_magic('time', '', 'print(len(train_df.GameId.unique()))\nprint(len(train_df.NflId.unique()))')


# Total 512 game describe in train dataset.

# ## PlayId & NflId

# In[ ]:


get_ipython().run_cell_magic('time', '', 'print(len(train_df.PlayId.unique()))\nprint(len(train_df.NflId.unique()))')


# In[ ]:


play_id_unique = pd.Series([0 for i in range(512)])
for idx, i in enumerate(train_df.GameId.unique()):
    play_id_unique[idx] = len(train_df[train_df['GameId']==i].PlayId.unique())

fig, ax = plt.subplots(1, 1, figsize=(15, 6))
sns.countplot(play_id_unique, ax=ax)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#https://www.kaggle.com/kmader/previewing-the-games\n\nfrom matplotlib.patches import Rectangle\nfootball_field = lambda : Rectangle(xy=(10, 0), width=100, height=53.3,  color='g',alpha=0.10)\n\nfig, axes = plt.subplots(5, 4, figsize=(20, 20))\nfor (play_id, play_rows), ax in zip(train_df.groupby('PlayId'), axes.flatten()):\n    ax.add_patch(football_field())\n    for player_id, player_rows in play_rows.groupby('NflId'):\n        player_rows = player_rows.sort_values('TimeSnap')\n        ax.scatter(player_rows['X'], player_rows['Y'])\n    ax.set_title(play_id)\n    ax.set_aspect(1)\n    ax.set_xlim(0, 120)\n    ax.set_ylim(-10, 63)\nplt.show()")


# ## X and Y 

# In[ ]:


print((train_df.X.min(), train_df.Y.min()))
print((train_df.X.max(), train_df.Y.max()))


# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1,1, figsize=(12, 5.3))\nsns.scatterplot(x='X', y='Y', data=train_df, alpha=0.3)\nplt.show()")


# without Outlier, it seams very normal or uniform distribution

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1,2, figsize=(18, 7))\nsns.distplot(train_df['X'], ax=ax[0])\nsns.distplot(train_df['Y'], ax=ax[1])\n\nplt.show()")


# ## S & A (speed and acceleration)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fig, ax = plt.subplots(1,3, figsize=(21, 7))\nsns.distplot(train_df[\'S\'], ax=ax[0], color="#4285f4")\nsns.distplot(train_df[\'A\'], ax=ax[1], color="#34a853")\nsns.distplot(train_df[\'Dis\'], ax=ax[2], color="#ea4335")\n\nplt.show()')


# It seems skew. (have to preprocessing)

# ## PlayerHeight, PlayerWeight

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1,2, figsize=(18, 7))\nsns.countplot(train_df['PlayerHeight'], ax=ax[0])\nsns.countplot(train_df['PlayerWeight'], ax=ax[1])\n\nplt.show()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "def ftoi(str):\n    a, b = map(int, str.split('-'))\n    return a * 12 + b\n\ntrain_df['PlayerHeight'] = train_df['PlayerHeight'].apply(ftoi)")


# In[ ]:


train_df['PlayerHeight'].head()
train_df['PlayerWeight'].head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1,2, figsize=(18, 7))\nsns.countplot(train_df['PlayerHeight'], ax=ax[0])\nsns.countplot(train_df['PlayerWeight'], ax=ax[1])\n\nplt.show()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1,2, figsize=(18, 7))\nsns.distplot(train_df['PlayerHeight'], ax=ax[0])\nsns.distplot(train_df['PlayerWeight'], ax=ax[1])\n\nplt.show()")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'sns.lmplot(x=\'PlayerHeight\', y=\'PlayerWeight\', data=train_df, palette="Set3")\nplt.show()')


# ## PlayerBirthDate

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1,1, figsize=(20, 6))\nsns.countplot(train_df['PlayerBirthDate'], ax=ax)\nplt.show()")


# In[ ]:


print(min(train_df['PlayerBirthDate']), max(train_df['PlayerBirthDate']))


# ## PlayerCollegeName

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1, 1, figsize=(20,15))\nsns.countplot(y='PlayerCollegeName', data=train_df, ax=ax)\nplt.show()")


# In[ ]:


len(train_df.PlayerCollegeName.unique())


# ## Location

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1, 1, figsize=(20,15))\nsns.countplot(y='Location', data=train_df, ax=ax)\nplt.show()")


# ## Stadium

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fig, ax = plt.subplots(1, 1, figsize=(20,13))\nsns.countplot(y=\'Stadium\', data=train_df, ax=ax)\nplt.title("{} Stadiums".format(len(train_df.Stadium.unique())))\nplt.show()')


# ## Stadium Type

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fig, ax = plt.subplots(1, 1, figsize=(15,8))\nsns.countplot(y=\'StadiumType\', data=train_df, ax=ax)\nplt.title("{} Stadium Types".format(len(train_df.StadiumType.unique())))\nplt.show()')


# ## GameWeather

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1, 1, figsize=(15,8))\nsns.countplot(y='GameWeather', data=train_df, ax=ax)\nplt.show()")


# It's hard to see.. How about WordCloud

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from wordcloud import WordCloud \n\nfig, ax = plt.subplots(1, 1, figsize=(15, 7))\ntrain_df[\'GameWeather\'] = train_df[\'GameWeather\'].apply(str)\nwordcloud = WordCloud(background_color=\'white\').generate(" ".join(train_df[\'GameWeather\']))\n\n\nplt.imshow(wordcloud, interpolation=\'bilinear\')\nplt.axis("off")\nplt.show()')


# ## Other Weather
# 
# - Temperature - temperature (deg F)
# - Humidity - humidity
# - WindSpeed - wind speed in miles/hour
# - WindDirection - wind direction

# In[ ]:


sns.set_palette('bright')
fig, ax = plt.subplots(2,1,figsize=(20,10))
for idx, elem in enumerate(["Temperature" , "Humidity"]):
    sns.distplot(train_df[elem].dropna(), ax=ax[idx])
plt.show()


# In[ ]:


train_df['WindSpeed'].unique()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def numeric_check(s):\n    s = str(s)\n    if s.isnumeric():\n        return int(s)\n    return None\n\ntrain_df["WindSpeed"] = train_df["WindSpeed"].apply(numeric_check)\n\nfig, ax = plt.subplots(1,1,figsize=(10,4))\nsns.distplot(train_df["WindSpeed"].dropna(),ax=ax)\nplt.show()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def direction_convert(s):\n    s = str(s)\n    for i in [\'North\', \'South\', \'East\', \'West\']:\n        s = s.replace(i, i[0])\n        s = s.replace(i.lower(), i[0])\n        s = s.replace(i.upper(), i[0])\n\n    for i in [\'-\', \'from\', \'From\', \'/\', \' \']:\n        s = s.replace(i,\'\')\n        \n    s.replace(\'s\',\'S\')\n    if s.isnumeric() or s==\'nan\':\n        s = "Calm"\n    return s \nfig, ax = plt.subplots(1,1,figsize=(30,4))\ntrain_df["WindDirection"] = train_df["WindDirection"] .apply(direction_convert)\nsns.countplot(train_df["WindDirection"] ,ax=ax)\nplt.show()')


# ## Heatmap(have to fill null data)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import pandas_profiling\ntrain_df.profile_report()')

