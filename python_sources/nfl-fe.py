#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
# from kaggle.competitions import nflrush

# # You can only call make_env() once, so don't lose it!
# env = nflrush.make_env()

pd.set_option('max_columns', 100)
data_path = "/kaggle/input/nfl-big-data-bowl-2020/"

train = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2020/train.csv", low_memory=False)
train.head(5)


# In[ ]:


# Initial features and data correction
train['ToLeft'] = train.PlayDirection == "left"
train['IsBallCarrier'] = train.NflId == train.NflIdRusher

train.loc[train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
train.loc[train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

train.loc[train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
train.loc[train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

train.loc[train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
train.loc[train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

train.loc[train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
train.loc[train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

train['Dir_rad'] = np.mod(90 - train.Dir, 360) * math.pi/180.0

cols_to_drop = ['GameId', 'PlayId', 'NflId', 'DisplayName', 'NflIdRusher']
# train.isnull().sum()


# In[ ]:


# def create_football_field(linenumbers=True,
#                           endzones=True,
#                           highlight_line=False,
#                           highlight_line_number=50,
#                           highlighted_name='Line of Scrimmage',
#                           fifty_is_los=False,
#                           figsize=(12*2, 6.33*2)):
#     """
#     Function that plots the football field for viewing plays.
#     Allows for showing or hiding endzones.
#     """
#     rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
#                              edgecolor='r', facecolor='darkgreen', zorder=0,  alpha=0.5)

#     fig, ax = plt.subplots(1, figsize=figsize)
#     ax.add_patch(rect)

#     plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
#               80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
#              [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
#               53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
#              color='white')
#     if fifty_is_los:
#         plt.plot([60, 60], [0, 53.3], color='gold')
#         plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
#     # Endzones
#     if endzones:
#         ez1 = patches.Rectangle((0, 0), 10, 53.3,
#                                 linewidth=0.1,
#                                 edgecolor='r',
#                                 facecolor='blue',
#                                 alpha=0.2,
#                                 zorder=0)
#         ez2 = patches.Rectangle((110, 0), 120, 53.3,
#                                 linewidth=0.1,
#                                 edgecolor='r',
#                                 facecolor='blue',
#                                 alpha=0.2,
#                                 zorder=0)
#         ax.add_patch(ez1)
#         ax.add_patch(ez2)
#     plt.xlim(0, 120)
#     plt.ylim(-5, 58.3)
#     plt.axis('off')
#     if linenumbers:
#         for x in range(20, 110, 10):
#             numb = x
#             if x > 50:
#                 numb = 120 - x
#             plt.text(x, 5, str(numb - 10),
#                      horizontalalignment='center',
#                      fontsize=20,  # fontname='Arial',
#                      color='white')
#             plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
#                      horizontalalignment='center',
#                      fontsize=20,  # fontname='Arial',
#                      color='white', rotation=180)
#     if endzones:
#         hash_range = range(11, 110)
#     else:
#         hash_range = range(1, 120)

#     for x in hash_range:
#         ax.plot([x, x], [0.4, 0.7], color='white')
#         ax.plot([x, x], [53.0, 52.5], color='white')
#         ax.plot([x, x], [22.91, 23.57], color='white')
#         ax.plot([x, x], [29.73, 30.39], color='white')

#     if highlight_line:
#         hl = highlight_line_number + 10
#         plt.plot([hl, hl], [0, 53.3], color='yellow')
#         plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
#                  color='yellow')
#     return fig, ax
# # create_football_field()


# In[ ]:


# def get_dx_dy(radian_angle, dist):
#     dx = dist * math.cos(radian_angle)
#     dy = dist * math.sin(radian_angle)
#     return dx, dy

# def show_play(play_id, train=train):
#     df = train[train.PlayId == play_id]
#     fig, ax = create_football_field()
#     ax.scatter(df.X, df.Y, cmap='rainbow', c=~(df.Team == 'home'), s=100)
#     rusher_row = df[df.NflIdRusher == df.NflId]
#     ax.scatter(rusher_row.X, rusher_row.Y, color='black')
#     yards_covered = rusher_row["Yards"].values[0]
#     x = rusher_row["X"].values[0]
#     y = rusher_row["Y"].values[0]
#     rusher_dir = rusher_row["Dir_rad"].values[0]
#     rusher_speed = rusher_row["S"].values[0]
#     dx, dy = get_dx_dy(rusher_dir, rusher_speed)

#     ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3, color='black')
#     left = 'left' if df.ToLeft.sum() > 0 else 'right'
#     plt.title(f'Play # {play_id} moving to {left}, yard distance is {yards_covered}', fontsize=20)
#     plt.legend()
#     plt.show()


# In[ ]:


# show_play(20170907000118)


# In[ ]:


print(train.shape)
empty_rows = train[train.GameId=='EMPTY'].index.values
train.drop(axis=0, index=empty_rows, inplace=True)
train.reset_index(inplace=True)
train.shape


# In[ ]:


# create home binary feature from Team, home=1 and away=0
train['home'] = train['Team'].map({'home': 1, 'away': 0})
# can drop 'Team' field


# ## Feature Engineering
# ### Post grouping at PlayId level

# In[ ]:


# Get game date feature from GameId field
# train['GameId'].astype(str).str.slice(stop=8).head()
train['GameDate'] = pd.to_datetime(train['GameId'].astype(str).str.slice(stop=8), format="%Y%m%d")
print(train['GameDate'].min(), train['GameDate'].max())
train['GameDate'].value_counts().head()


# In[ ]:


# Converting gameclock string to gametime timedelta
# Reversing the gametime to calculate the number of seconds passed in each quarter(instead of time remaining)
from datetime import datetime, timedelta
print(train['GameClock'].min(), train['GameClock'].max())
# train['GameClock'].str.split(':').head()
train['GameTime'] = train['GameClock'].str.split(':').apply(lambda x: timedelta(minutes=int(x[0]), 
                                                                                seconds=int(x[1])))
print(train['GameTime'].head())
train['GameTimeSec'] = train['GameTime'].apply(lambda x: timedelta(minutes=15)-x).dt.total_seconds()
print(train['GameTimeSec'].head())
train['GameTimeMins'] = train['GameTimeSec'] // 60
train['GameTimeMins'].value_counts()
# can drop 'GameClock' and 'GameTime' fields


# In[ ]:


# create fature is_home_offensive
train['is_home_offensive'] = (train['PossessionTeam'] == train['HomeTeamAbbr']).astype('int16')
train['is_home_offensive'].value_counts(dropna=False)


# In[ ]:


# create a feature explaining if the field position is on offensive / defensive team side
def update_field_pos(home_offensive, field_pos, home_team, visitor_team):
    if(field_pos==np.nan):
        return 'midway'
    elif(home_offensive == 1 and field_pos==home_team):
        return 'offensive'
    elif(home_offensive == 0 and field_pos==visitor_team):
        return 'offensive'
    else:
        return 'defensive'

train['FieldPosition'].value_counts(dropna=False)


# In[ ]:


# train[train['FieldPosition'].isnull()]['YardLine'].sum()/6424 
# FieldPosition is null as the yardline is 50yards (midway)
train['field_pos'] = train[['is_home_offensive','FieldPosition','HomeTeamAbbr','VisitorTeamAbbr']].apply(lambda x: update_field_pos(*x), axis=1)
train['field_pos'].value_counts()
# Can drop the field "FieldPosition"


# In[ ]:


# Normalizing season to 1,2,3,4 etc.,
train['Season'] = train['Season'] - 2016


# In[ ]:


train.OffenseFormation.value_counts(dropna=False)/22


# In[ ]:


# train.groupby(['OffensePersonnel'], as_index=False).agg({
#     'OffenseFormation': 'nunique'
# })
# # 1 RB, 2 TE, 3 WR
# # 2 RB, 3 TE, 1 WR


# In[ ]:


train.OffenseFormation.fillna('UNKNOWN', inplace=True)
train.loc[train["OffenseFormation"]=="ACE","OffenseFormation"] = 'UNKNOWN'
train.loc[train["OffenseFormation"]=="EMPTY","OffenseFormation"] = 'UNKNOWN'
train.OffenseFormation.value_counts(dropna=False)/22


# In[ ]:


train.DefendersInTheBox.value_counts(dropna=False)/22


# In[ ]:


train.DefendersInTheBox.fillna(4.0, inplace=True)
# train.DefendersInTheBox.map({'EMPTY': 4.0})
train.DefendersInTheBox = train.DefendersInTheBox.astype('int16')
train.loc[train.DefendersInTheBox < 4, 'DefendersInTheBox'] = 4
# train.DefendersInTheBox = train.DefendersInTheBox.astype('int16')
train.DefendersInTheBox.value_counts(dropna=False)/22


# In[ ]:


train['TimeHandoff'] = pd.to_datetime(train['TimeHandoff'], format="%Y-%m-%dT%H:%M:%S.%fZ")
train['TimeSnap'] = pd.to_datetime(train['TimeSnap'], format="%Y-%m-%dT%H:%M:%S.%fZ")
train['time_btw_handoff_snap'] = (train['TimeHandoff'] - train['TimeSnap']).astype('timedelta64[s]')
train['time_btw_handoff_snap'].value_counts()/22


# In[ ]:


# train['OffensePersonnel'].str.split("\, |\ ").head()
OffensePersonnelSet = set()
import re
for pos in train['OffensePersonnel'].unique():
    OffensePersonnelSet = OffensePersonnelSet | set(re.split(", |\d |,", pos))
OffensePersonnelSet = OffensePersonnelSet - {""}
OffensePersonnelSet


# In[ ]:


DefensePersonnelSet = set()
import re
for pos in train['DefensePersonnel'].unique():
    DefensePersonnelSet = DefensePersonnelSet | set(re.split(", |\d |,", pos))
DefensePersonnelSet = DefensePersonnelSet - {""}
DefensePersonnelSet


# In[ ]:


print("Before: ", train.shape)

for col in OffensePersonnelSet:
    train['off_'+col] = 0
for col in DefensePersonnelSet:
    train['def_'+col] = 0

print("After: ", train.shape)


# In[ ]:


# lst = [1, 'RB', 1, 'TE', 3, 'WR']
def get_pos(lst):
    pos_dict = {}
    for i in range(1, len(lst), 2):
        pos_dict[lst[i]] = int(lst[i-1])
    return pos_dict
# get_pos(lst)   
train['offense_dict'] = train['OffensePersonnel'].str.split("\, |\ ").apply(lambda x: get_pos(x))
train['offense_dict'].head()


# In[ ]:


def get_pos(lst):
    pos_dict = {}
    for i in range(1, len(lst), 2):
        pos_dict[lst[i]] = int(lst[i-1])
    return pos_dict
# get_pos(lst)   
train['defense_dict'] = train['DefensePersonnel'].str.split("\, |\ ").apply(lambda x: get_pos(x))
train['defense_dict'].head()


# In[ ]:


def off_explode_map(playid, pos_dict):
    for k,v in pos_dict.items():
        train[train['PlayId']==playid, 'off_'+k] = v
    pass

def def_explode_map(playid, pos_dict):
    for k,v in pos_dict.items():
        train[train['PlayId']==playid, 'def_'+k] = v
    pass

# train[['PlayId','defense_dict']].apply(lambda x: def_explode_map(*x), axis=1)
# train[['PlayId','offense_dict']].apply(lambda x: off_explode_map(*x), axis=1)


# In[ ]:


train[['Stadium','Location','StadiumType','Turf','GameWeather','Temperature','Humidity','WindSpeed','WindDirection']].isnull().sum()/22


# ### StadiumType - imputation and feature grouping

# In[ ]:


train[train.StadiumType.isnull()].Stadium.value_counts()


# In[ ]:


train[train.Stadium=="MetLife Stadium"].StadiumType.value_counts(dropna=False) 
# Outdoor


# In[ ]:


train[train.Stadium=="StubHub Center"].StadiumType.value_counts(dropna=False) 
# Outdoor from internet
# Dignity Health Sports Park


# In[ ]:


train[train.Stadium=="TIAA Bank Field"].StadiumType.value_counts(dropna=False) 
# Outdoor


# In[ ]:


# StadiumType imputation
train.StadiumType.fillna('Outdoor', inplace=True)


# In[ ]:


train.StadiumType.value_counts()


# In[ ]:


# Distinct stadium types - outdoor, indoor, ret_roof_open, ret_roof_closed, dome_open, dome_closed
# ret_roof and dome can be grouped for open and closed
# train[train.StadiumType=="Domed, Open"].head()

# train[train.StadiumType.str.contains(r'roof[-|\s]?close', case=False)] = 'outdoor'
# train.StadiumType.str.contains(r'roof[-|\s]+close', case=False).sum()
# print(train.StadiumType.str.contains(r'roof[-|\s]+open', case=False).sum())

train.loc[train.StadiumType.str.contains(r'open', case=False), 'StadiumType'] = 'open'
train.loc[train.StadiumType.str.contains(r'close', case=False), 'StadiumType'] = 'closed'
train.loc[train.StadiumType == "Retractable Roof", 'StadiumType'] = 'closed'
train.loc[train.StadiumType.str.contains(r'Dome', case=False), 'StadiumType'] = 'closed'

train.loc[train.StadiumType.str.contains(r'in[a-z]?door', case=False), 'StadiumType'] = 'indoor'
train.loc[train.StadiumType.str.contains(r'ou[a-z]?d[o|d]?or', case=False), 'StadiumType'] = 'outdoor'
train.loc[train.StadiumType == "Heinz Field", 'StadiumType'] = 'outdoor'
train.loc[train.StadiumType == "Outside", 'StadiumType'] = 'outdoor'
train.loc[train.StadiumType == "Bowl", 'StadiumType'] = 'outdoor'
train.loc[train.StadiumType == "Cloudy", 'StadiumType'] = 'outdoor'


# In[ ]:


train.StadiumType.value_counts()/22


# ### Temperature imputation

# In[ ]:


train[train.Temperature.isnull()]['StadiumType'].value_counts()


# In[ ]:


train[train.StadiumType=='open']['Temperature'].mean()


# In[ ]:


for sdum in ['indoor', 'closed', 'open']:
    train.loc[(train.Temperature.isnull()) & (train.StadiumType==sdum), 'Temperature'] = train[train.StadiumType==sdum]['Temperature'].mean()


# In[ ]:


sns.distplot(train.Temperature);


# ### Humidity imputation

# In[ ]:


train[train.Humidity.isnull()]['StadiumType'].value_counts()


# In[ ]:


train[train.StadiumType=='indoor']['Humidity'].mean()


# In[ ]:


for sdum in ['indoor', 'closed']:
    train.loc[(train.Humidity.isnull()) & (train.StadiumType==sdum), 'Humidity'] = train[train.StadiumType==sdum]['Humidity'].mean()


# In[ ]:


sns.distplot(train.Humidity);


# ### Turf - feature grouping

# In[ ]:


train.Turf.value_counts()


# In[ ]:


train.loc[train.Turf=='DD GrassMaster', 'Turf'] = 'hybrid'
train.loc[train.Turf=='SISGrass', 'Turf'] = 'hybrid'
train.loc[train.Turf.str.contains(r'artific', case=False), 'Turf'] = 'artificial'
train.loc[train.Turf.str.contains(r'S5-M', case=False), 'Turf'] = 'artificial'
train.loc[train.Turf.str.contains(r'Field[\s]?Turf', case=False), 'Turf'] = 'artificial'
train.loc[train.Turf.str.contains(r'A-Turf', case=False), 'Turf'] = 'artificial'
train.loc[train.Turf.str.contains(r'Twenty-Four/Seven', case=False), 'Turf'] = 'artificial'
train.loc[train.Turf.str.contains(r'natural', case=False), 'Turf'] = 'natural'
train.loc[train.Turf.str.contains(r'grass', case=False), 'Turf'] = 'natural'


# In[ ]:


train.Turf.value_counts()/22


# ### Location - feature cleaning

# In[ ]:


train.Location.value_counts()/22


# In[ ]:


train.loc[train.Location.str.contains(r'Rutherford'), 'Location'] = 'Rutherford, NJ'
train['city'] = train.Location.str.split(r',|\.\sI|\sOh|\s[FN]').str[0]


# In[ ]:


train['city'].value_counts()


# ### Stadium - feature cleaning

# In[ ]:


import Levenshtein as lev
from fuzzywuzzy import fuzz


# In[ ]:


train['Stadium_lower'] = train.Stadium.str.lower().str.replace(" ","")                                     .str.replace("stadium","").str.replace("dome","")                                     .str.replace("super","").str.replace("link","")                                     .str.replace("field", "").str.replace("bank","")                                     .str.replace("stdium","").str.replace("-","")


# In[ ]:


stadium_lst = set(train.Stadium_lower.unique().tolist())
stadium_lst_cpy = set(train.Stadium_lower.unique().tolist())
len(stadium_lst), len(stadium_lst_cpy)


# In[ ]:


print(stadium_lst, end=", ")


# In[ ]:


# fuzz.token_sort_ratio('M&T Bank Stadium', 'Bank of America Stadium')
# stadium_map = dict()
# stadium_map.keys

# st1 = set([1,2,4,5,9,10,3])
# st1^set([4,3,7])


# In[ ]:


stadium_map = dict()
for elem in stadium_lst:
    mapped_set = set()
    for elm in stadium_lst_cpy:
        ratio_temp = fuzz.token_set_ratio(elem, elm)
        if (ratio_temp > 95) and (elm not in stadium_map.keys()) :
            stadium_map[elm] = elem
            mapped_set.add(elm)
            
    stadium_lst_cpy^mapped_set
print(len(stadium_map))        
stadium_map        


# In[ ]:


train['Stadium_lower'] = train.Stadium_lower.map(stadium_map)


# In[ ]:


# train.Stadium_lower.value_counts()/22


# In[ ]:


train.Stadium_lower.nunique()


# ### GameWeather feature engineering

# In[ ]:


train.GameWeather.value_counts(dropna=False)/22


# In[ ]:


train.GameWeather.unique()


# In[ ]:


train['GameWeather_lower'] = train.GameWeather.str.lower().str.replace("showers","rain")                                             .str.replace("coudy","cloudy").str.replace("clouidy","cloudy")                                             .str.replace("fair","clear").str.replace("controlled","indoor")                                             .str.replace("overcast","dull").str.replace("cold","snow")                                             .str.replace("hazy","dull").str.replace("t: 51","snow")


# In[ ]:


weather_lst = set(train.GameWeather_lower.unique().tolist())
weather_lst_cpy = set(train.GameWeather_lower.unique().tolist())
len(weather_lst), len(weather_lst_cpy)


# In[ ]:


weather_map = dict()
for elem in ['sunny','cloudy','windy','rain','clear','indoor','snow','dull']:
    mapped_set = set()
    for elm in weather_lst_cpy:
        ratio_temp = fuzz.token_set_ratio(elem, elm)
        if (ratio_temp > 60) and (elm not in weather_map.keys()) :
            weather_map[elm] = elem
            mapped_set.add(elm)
            
    weather_lst_cpy = weather_lst_cpy^mapped_set
print(len(weather_map), len(weather_lst_cpy))        
weather_map        


# In[ ]:


weather_lst_cpy


# In[ ]:


train['GameWeather_lower'] = train.GameWeather_lower.map(weather_map)


# In[ ]:


train.city.value_counts()/22


# In[ ]:


# train[train.GameWeather.isnull()]['Location'].value_counts()/22
train[train.GameWeather=='Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.'].sample(10)


# In[ ]:


for cty in train.city.unique():
    to_fill = train[train.city==cty]['GameWeather_lower'].value_counts().index[0]
    train.loc[(train.city==cty) & (train.GameWeather.isnull()), 'GameWeather_lower'] = to_fill


# In[ ]:


# train.loc[(train.Humidity.isnull()) & (train.StadiumType==sdum), 'Humidity'] = train[train.StadiumType==sdum]['Humidity'].mean()
train.GameWeather_lower.value_counts(dropna=False)/22


# ### Windspeed feature cleaning

# In[ ]:


for ws in ['SE','SSW','E']:
    win_spd = train[train.WindSpeed==ws]['WindDirection'].values[0]
    win_dir = train[train.WindSpeed==ws]['WindSpeed'].values[0]
    train.loc[train.WindSpeed==ws,'WindSpeed'] = win_spd
    train.loc[train.WindSpeed==ws,'WindDirection'] = win_dir


# In[ ]:


train['WindSpeed_crtd'] = train.WindSpeed.str.replace(r'mph',"",case=False).str.replace("Calm","1")                                             .str.replace(" ","").str.replace("gustsupto", "-")


# In[ ]:


# train.WindSpeed_crtd.dtype
train.loc[(train.WindSpeed_crtd.str.contains("-") & ~(train.WindSpeed_crtd.isnull())), 'WindSpeed_crtd'] = train.loc[(train.WindSpeed_crtd.str.contains("-") & ~(train.WindSpeed_crtd.isnull())), 'WindSpeed_crtd']         .str.split("-").str[0]#.apply(lambda x: (x[0]+x[1])/2)
train.WindSpeed_crtd = train.WindSpeed_crtd.astype('float64')


# In[ ]:


# train.WindSpeed_crtd = train.WindSpeed_crtd.astype('float64')
train.WindSpeed_crtd.value_counts(dropna=False)


# In[ ]:


train[train.WindSpeed_crtd.isnull()].sample(10)


# In[ ]:


train[train.WindSpeed_crtd.isnull()]['StadiumType'].value_counts()/22


# In[ ]:


train.loc[~(train.WindSpeed_crtd.isnull()) & (train.StadiumType=='outdoor') & (train.Season==2018), 'WindSpeed_crtd'].mean()


# In[ ]:


for yr in [2017,2018]:
    for st_typ in train.StadiumType.unique():
        temp_val = train.loc[~(train.WindSpeed_crtd.isnull()) & (train.StadiumType==st_typ) & (train.Season==yr), 'WindSpeed_crtd'].mean()
        train.loc[(train.WindSpeed_crtd.isnull()) & (train.StadiumType==st_typ) & (train.Season==yr), 'WindSpeed_crtd'] = temp_val


# In[ ]:


sns.distplot(train.WindSpeed_crtd);


# ### Wind direction

# In[ ]:


train['WindDirection_crtd'] = train.WindDirection.str.lower().str.replace(" ","")                                                 .str.replace(r'from',"",case=False).str.replace("-","")                                                 .str.replace("/","").str.replace("north","n")                                                 .str.replace("south","s").str.replace("east","e")                                                 .str.replace("west","w")


# In[ ]:


train.loc[~train.WindDirection_crtd.str.contains(r'[news]', na=True),'WindDirection_crtd'] = np.nan


# In[ ]:


train.WindDirection_crtd.value_counts(dropna=False)/22


# In[ ]:


train[train.WindDirection_crtd.isnull()]['city'].value_counts().index


# In[ ]:


for cty in train[train.WindDirection_crtd.isnull()]['city'].value_counts().index:
    try:
        to_fill = train[train.city==cty]['WindDirection_crtd'].value_counts().index[0]
        print(to_fill)
        train.loc[(train.city==cty) & (train.WindDirection_crtd.isnull()), 'WindDirection_crtd'] = to_fill
    except:
        print("City: ", cty, "\n", train[train.city==cty]['WindDirection_crtd'].value_counts())
        train.loc[(train.city==cty) & (train.WindDirection_crtd.isnull()), 'WindDirection_crtd'] = "indoor"


# In[ ]:


train.WindDirection_crtd.value_counts(dropna=False)/22


# ## At Player level

# In[ ]:


train.X.nunique(), train.Y.nunique(), train.S.nunique(), train.A.nunique(), train.Dis.nunique(), train.Orientation.nunique(), train.Dir.nunique()


# In[ ]:


train[['X','Y','S','A','Dis','Orientation','Dir','PlayerHeight','PlayerWeight','Position']].isnull().sum()


# In[ ]:


train[['X','Y','S','A','Dis','Orientation','Dir','PlayerHeight','PlayerWeight','Position']].dtypes


# In[ ]:


train[['X','Y','S','A','Dis','Orientation','Dir','PlayerHeight','PlayerWeight','Position']].head(10)


# In[ ]:


train.tail()


# In[ ]:




