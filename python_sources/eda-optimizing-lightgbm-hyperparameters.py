#!/usr/bin/env python
# coding: utf-8

# # NFL Big Data Bowl 2020 (used to be Official) Starter Notebook
# ## Introduction
# In this competition you will predict how many yards a team will gain on a rushing play in an NFL regular season game.  You will loop through a series of rushing plays; for each play, you'll receive the position, velocity, orientation, and more for all 22 players on the field at the moment of handing the ball off to the rusher, along with many other features such as teams, stadium, weather conditions, etc.  You'll use this information to predict how many yards the team will gain on the play as a [cumulative probability distribution](https://en.wikipedia.org/wiki/Cumulative_distribution_function).  Once you make that prediction, you can move on to the next rushing play.
# 
# This competition is different from most Kaggle Competitions in that:
# * You can only submit from Kaggle Notebooks, and you may not use other data sources, GPU, or internet access.
# * This is a **two-stage competition**.  In Stage One you can edit your Notebooks and improve your model, where Public Leaderboard scores are based on your predictions on rushing plays from the first few weeks of the 2019 regular season.  At the beginning of Stage Two, your Notebooks are locked, and we will re-run your Notebooks over the following several weeks, scoring them based on their predictions relative to live data as the 2019 regular season unfolds.
# * You must use our custom **`kaggle.competitions.nflrush`** Python module.  The purpose of this module is to control the flow of information to ensure that you are not using future data to make predictions for the current rushing play.  If you do not use this module properly, your code may fail when it is re-run in Stage Two.
# 
# ## In this Starter Notebook, I will simply apply a lightGBM for this kind of a table data. 

# ## In-depth Introduction
# First let's import the module and create an environment.

# In[ ]:


from kaggle.competitions import nflrush
import pandas as pd
import numpy as np
import os
import datetime
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import optuna
from scipy.stats import norm
import random
from sklearn.model_selection import KFold, train_test_split
import gc
from sklearn import preprocessing
import tqdm
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set_context("talk")
sns.set_style("white")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
import warnings
warnings.filterwarnings('ignore')
print("libraries imported!")
pd.set_option('max_columns', 100)


# ### Training data is in the competition dataset as usual

# In[ ]:


train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
print(train_df.shape)
train_df.head()


# The amount of data is not crazy, which is good!
# 
# Let's check what type of data are in the data.

# In[ ]:


print(train_df.dtypes)


# There are relatively a lot of object. We may need to convert them to numerical values somehow.
# 
# nans?

# In[ ]:


# search for missing data
import missingno as msno
msno.matrix(df=train_df, figsize=(14,14), color=(0.5,0,0))


# In[ ]:


# Which columns have nan?
for i in np.arange(train_df.shape[1]):
    n = train_df.iloc[:,i].isnull().sum() 
    if n > 0:
        print(list(train_df.columns.values)[i] + ': ' + str(n) + ' nans')


# There are relatively many nans for some columns such as "WindSpeed" and "WindDirection".
# 
# number of uniques in each column?

# In[ ]:


train_df.nunique()


# Let's see how numerical columns look like for now.

# In[ ]:


train_df.describe()


# In[ ]:


df_numeric = train_df.select_dtypes(exclude=['object'])
print(df_numeric.shape)


# In[ ]:


fig, ax = plt.subplots(5, 5, figsize=(20, 20))
ax = ax.flatten()

for i, c in enumerate(df_numeric.columns.values):
    sns.distplot(df_numeric[c].dropna(), ax=ax[i])
plt.tight_layout()


# We are to predict "Yards". Let's close up.

# In[ ]:


del df_numeric
gc.collect()


# In[ ]:


sns.distplot(train_df["Yards"].values, bins=100)


# Note that yards are int64!

# In[ ]:


train_df["Yards"].value_counts()


# 1, 2 and 3 yards are frequently observed.

# # Field
# Thanks to (https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position), we can draw players' positions on the field:D Please do upvote his kernel!

# In[ ]:


# Function to Create The Football Field 
def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax


# In[ ]:


# Adding Players For a Play (cyan = home, magenta = away), highlighting the line of scrimmage
playId = 20181230154157
yl = train_df.query("PlayId == @playId")['YardLine'].tolist()[0]
fig, ax = create_football_field(highlight_line=True,
                                highlight_line_number=yl+54)
train_df.query("PlayId == @playId and Team == 'away'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='magenta', s=30, legend='Away')
train_df.query("PlayId == @playId and Team == 'home'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='cyan', s=30, legend='Home')
plt.title('Play # 20170907000118')
plt.legend()
plt.show()


# # Making all-numeric data frame 
# Many of the following functions are from https://www.kaggle.com/marcovasquez/nfl-basic-eda-data-visualization. Thanks a lot! Please do upvote his kernel:D The feature engineering is hard and needs more work.

# In[ ]:


train_df.head()


# In[ ]:


#from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087
Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 
        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 
        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 
        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 


# In[ ]:


# from https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in train_df['PossessionTeam'].unique():
    map_abbr[abb] = abb


# In[ ]:


def uid_aggregation(comb, main_columns, uids, aggregations):
    X = pd.DataFrame()
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = col+'_'+main_column+'_'+agg_type
                temp_df = comb[[col, main_column]]
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                X[new_col_name] = comb[col].map(temp_df)
                del temp_df
                gc.collect()
    return X


# In[ ]:


def preprocess(df, labelEncoders=None):
    X = df.select_dtypes(exclude=['object'])
    def gameclock2min(x):
        clock = x.split(":")
        return 60 * int(clock[0]) + int(clock[1])
    def height2inch(x):
        height = x.split("-")
        return 12 * int(height[0]) + int(height[1])
    def birthday2day(x):
        days = x.split("/")
        return 30 * int(days[0]) + int(days[1]) + 365 * int(days[2])
    def timesnap2day(x):
        days = x.split("-")
        return 365 * int(days[0]) + 30 * int(days[1]) + int(days[2][:2])
    def utc2sec(x):
        return int(x.split("-")[2].split(":")[2].split(".")[0])
    def group_stadium_types(stadium):
        outdoor       = [
            'Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field',
            'Outdor', 'Ourdoor', 'Outside', 'Outddors',
            'Outdoor Retr Roof-Open', 'Oudoor', 'Bowl'
        ]
        indoor_closed = [
            'Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed',
            'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed',
        ]
        indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']
        dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']
        dome_open     = ['Domed, Open', 'Domed, open']

        if stadium in outdoor:
            return 0 #'outdoor'
        elif stadium in indoor_closed:
            return 3 # 'indoor closed'
        elif stadium in indoor_open:
            return 2 #'indoor open'
        elif stadium in dome_closed:
            return 4 #'dome closed'
        elif stadium in dome_open:
            return 1 #'dome open'
        else:
            return 5 #'unknown'

    def group_game_weather(weather):
        rain = [
            'Rainy', 'Rain Chance 40%', 'Showers',
            'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',
            'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain'
        ]
        overcast = [
            'Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',
            'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',
            'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',
            'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',
            'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',
            'Partly Cloudy', 'Cloudy'
        ]
        clear = [
            'Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',
            'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',
            'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',
            'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',
            'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',
            'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny'
        ]
        snow  = ['Heavy lake effect snow', 'Snow']
        none  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']

        if weather in rain:
            return -1 #'rain'
        elif weather in overcast:
            return 1 #'overcast'
        elif weather in clear:
            return 2 #'clear'
        elif weather in snow:
            return -2 #snow'
        elif weather in none:
            return 0 #'none'

    def clean_wind_speed(windspeed):
        """
        This is not a very robust function,
        but it should do the job for this dataset.
        """
        ws = str(windspeed)
        # if it's already a number just return an int value
        if ws.isdigit():
            return int(ws)
        # if it's a range, take their mean
        if '-' in ws:
            return (int(ws.split('-')[0]) + int(ws.split('-')[1]))/2
        # if there's a space between the number and mph
        if ws.split(' ')[0].isdigit():
            return int(ws.split(' ')[0])
        # if it looks like '10MPH' or '12mph' just take the first part
        if 'mph' in ws.lower():
            return int(ws.lower().split('mph')[0])
        else:
            return 0

    def clean_wind_direction(wind_direction):
        wd = str(wind_direction).upper()
        if wd == 'N' or 'FROM S' in wd:
            return 90 #'north'
        if wd == 'S' or 'FROM N' in wd:
            return 270 #'south'
        if wd == 'W' or 'FROM E' in wd:
            return 180 #'west'
        if wd == 'E' or 'FROM W' in wd:
            return 0 #'east'

        if 'FROM SW' in wd or 'FROM SSW' in wd or 'FROM WSW' in wd:
            return 45 #'north east'
        if 'FROM SE' in wd or 'FROM SSE' in wd or 'FROM ESE' in wd:
            return 135 #'north west'
        if 'FROM NW' in wd or 'FROM NNW' in wd or 'FROM WNW' in wd:
            return 315 #'south east'
        if 'FROM NE' in wd or 'FROM NNE' in wd or 'FROM ENE' in wd:
            return 225 #'south west'

        if 'NW' in wd or 'NORTHWEST' in wd:
            return 135 #'north west'
        if 'NE' in wd or 'NORTH EAST' in wd:
            return 45 #'north east'
        if 'SW' in wd or 'SOUTHWEST' in wd:
            return 225 #'south west'
        if 'SE' in wd or 'SOUTHEAST' in wd:
            return 315 #'south east'

    def clean_offenceformation(of):
        if of == "SHOTGUN":
            return 9
        elif of == "SINGLEBACK":
            return 8
        elif of == "JUMBO":
            return 6
        elif of == "PISTOL":
            return 5
        elif of == "I_FORM":
            return 4
        elif of == "ACE":
            return 3
        elif of ==  "WILDCAT":
            return 2
        elif of == "EMPTY":
            return 1
        else: 
            return 7
    X["Dir"] = np.mod(90 - df["Dir"].values, 360)
    X['Team'] = df['Team'].map({"home": 0, "away": 1})
    X['Turf'] = df['Turf'].map(Turf)
    X['Turf'] = X['Turf'].map({"Natural": 0,"Artificial": 1})
    df["HomeTeamAbbr"] = df["HomeTeamAbbr"].map(map_abbr)
    df["VisitorTeamAbbr"] = df["VisitorTeamAbbr"].map(map_abbr)
    df["Possession"] = df["PossessionTeam"].map(map_abbr)
    X['HomePossesion'] = 1 * (df['PossessionTeam'] == df['HomeTeamAbbr'])
    X["OffenseFormation"] = df['OffenseFormation'].apply(clean_offenceformation)
    X['OffenseFormation'] = X['OffenseFormation'].fillna(7)
    X['PassDuration'] = df['TimeHandoff'].apply(utc2sec) - df['TimeSnap'].apply(utc2sec)
    # from https://www.kaggle.com/zero92/best-lbgm-new-features
    X['Month'] = df['TimeHandoff'].apply(lambda x : int(x[5:7]))
    X['Year'] = df['TimeHandoff'].apply(lambda x : int(x[0:4]))
    X['Morning'] = df['TimeHandoff'].apply(lambda x : 1 if (int(x[11:13]) >=0 and int(x[11:13]) <12) else 0)
    X['Afternoon'] = df['TimeHandoff'].apply(lambda x : 1 if (int(x[11:13]) <18 and int(x[11:13]) >=12) else 0)
    X['Evening'] = df['TimeHandoff'].apply(lambda x : 1 if (int(x[11:13]) >= 18 and int(x[11:13]) < 24) else 0)
    X['MatchDay'] = df['TimeSnap'].apply(timesnap2day)
    X['PlayerBirthDate'] = df['PlayerBirthDate'].apply(birthday2day)
    X['PlayerAge'] = X['MatchDay'] - X['PlayerBirthDate']
    X['PlayDirection'] = df['PlayDirection'].map({'right': 1, 'left': -1})
    X['PlayerWeight'] = df['PlayerWeight']
    X['PlayerHeight'] = df['PlayerHeight'].apply(height2inch)
    X['BMI'] = X['PlayerWeight'] / X['PlayerHeight']
    X['GameClock'] = df['GameClock'].apply(gameclock2min)
    X['StadiumType'] = df['StadiumType'].apply(group_stadium_types)
    X['GameWeather'] = df['GameWeather'].apply(group_game_weather)
    X['WindSpeed'] = df['WindSpeed'].apply(clean_wind_speed)
    X['WindDirection'] = df['WindDirection'].apply(clean_wind_direction)
    X['WindDirection'] = 2 * np.pi * (90 - X['WindDirection']) / 360
    X['Humidity'] = df['Humidity'].fillna(df['Humidity'].median())
    X['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    X['DefendersInTheBox'] = df['DefendersInTheBox'].fillna(df['DefendersInTheBox'].median())
    
    posts1 = [['0_DL','1_DL','2_DL','3_DL','4_DL','5_DL','6_DL','7_DL'],
         ['0_LB','1_LB','2_LB','3_LB','4_LB','5_LB','6_LB'],['1_DB',
         '2_DB','3_DB','4_DB','5_DB','6_DB','7_DB','8_DB']]
    posts2 = [['1_RB','2_RB','3_RB','6_OL','7_OL','2_QB'],
         ['1_TE','2_TE','3_TE','1_RB','2_RB','3_RB','0_TE','1_TE','2_TE','3_TE','4_TE'],['1_DB',
         '1_WR','2_WR','3_WR','4_WR','5_WR','0_TE','1_TE','2_TE','3_TE','8_DB']]
    for k in range(0,3) :
        for col in posts1[k] :
            X[col] = df['DefensePersonnel'].str.replace(' ','_').str.split(',_').apply(lambda x : 1 if (x[k] == col) else 0)
            
    for k in range(0,3) :
        for col in posts2[k] :
            X[col] = df['OffensePersonnel'].str.replace(' ','_').str.split(',_').apply(lambda x : 1 if (x[k] == col) else 0)
    # from https://www.kaggle.com/ryches/model-free-benchmark
    X['Field_eq_Possession'] = 1 * (df['FieldPosition'] == df['PossessionTeam'])    
    X['is_rusher'] = 1 * (df['NflId'] == df['NflIdRusher'])
    X['seconds_need_to_first_down'] = (df['Distance']*0.9144) / (df['Dis'].values + 0.01)
    X['seconds_need_to_YardsLine'] = (df['YardLine']*0.9144) / (df['Dis'].values + 0.01)
    X['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']
    
    # based on https://www.kaggle.com/sryo188558/cox-proportional-hazard-model
    X["Start"] = X["YardLine"]
    X.loc[(X["Field_eq_Possession"] == 1) & (X["PlayDirection"] == 1), "Start"] = X.loc[(X["Field_eq_Possession"] == 1) & (X["PlayDirection"] == 1), 
                                                                                       "YardLine"] + 10
    X.loc[(X["Field_eq_Possession"] == 1) & (X["PlayDirection"] == -1), "Start"] = 120 - X.loc[(X["Field_eq_Possession"] == 1) & (X["PlayDirection"] == -1), 
                                                                                       "YardLine"] - 10
    X.loc[(X["Field_eq_Possession"] == 0) & (X["PlayDirection"] == 1), "Start"] = 120 - X.loc[(X["Field_eq_Possession"] == 0) & (X["PlayDirection"] == 1), 
                                                                                       "YardLine"] - 10
    X.loc[(X["Field_eq_Possession"] == 0) & (X["PlayDirection"] == -1), "Start"] = X.loc[(X["Field_eq_Possession"] == 0) & (X["PlayDirection"] == -1), 
                                                                                       "YardLine"] + 10
    X['Orientation'] = 2 * np.pi * (90 - X['Orientation']) / 360
    X['locX'] = (X['X'].values - X['Start'].values) * X['PlayDirection'].values
    X['locY'] = X['Y'].values - 53.3 / 2
    X['velX'] = X['S'].values * np.cos(X['Orientation'].values) * X['PlayDirection'].values
    X['velY'] = X['S'].values * np.sin(X['Orientation'].values)
    X['accX'] = X['A'].values * np.cos(X['Orientation'].values) * X['PlayDirection'].values
    X['accY'] = X['A'].values * np.sin(X['Orientation'].values)
    
    i_cols = ['VisitorScoreBeforePlay','HomeScoreBeforePlay','YardLine']
    uids = ['DisplayName']
    aggregations = ['mean','std','median', 'max', 'min']
    X_agg = uid_aggregation(df, i_cols, uids, aggregations)
    X = pd.concat([X, X_agg], axis=1)
    
    
#     X['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
#     X['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
#     X['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)
#     X['FieldPosition'] = np.where(df['YardLine'] == 50, df['PossessionTeam'], df['FieldPosition'])
#     for i in range(1,23) :
#         for j in ['X','Y','S','A','YardLine'] :
#             X[j+"_"+str(i)] = df[j].shift(i)

#     categorical = df.select_dtypes("object")
#     if labelEncoders == None:
#         labelEncoders = {}
#         for c in categorical.columns:
#             le = preprocessing.LabelEncoder()
#             X[c] = le.fit(df[c].values).transform(df[c].values)
#             labelEncoders[c] = le
#     else:
#         for c in categorical.columns:
#             le = labelEncoders[c]
#             X[c] = le.fit(df[c].values).transform(df[c].values)

    return X


# In[ ]:


# mydf = preprocess(train_df)
# print(mydf.shape)
# mydf.head()


# In[ ]:


train_df = preprocess(train_df)
print(train_df.shape)
train_df.head()


# In[ ]:


rm_cols = ['index','GameId','PlayId','NflId','FieldPosition', 
          'DisplayName','NflIdRusher']


# In[ ]:


features = [c for c in train_df.columns.values if c not in rm_cols]
train_df = train_df[features]
print(train_df.shape)
train_df.head()


# In[ ]:


# from https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm

train_data=np.zeros((509762//22, len(features)))
for i in tqdm.tqdm(range(0,509762,22)):
    count=0
    for c in features:
        train_data[i//22][count] = train_df[c][i]
        count+=1


# In[ ]:


y_train_ = np.array([train_df["Yards"][i] for i in range(0,509762,22)])


# In[ ]:


X_train = pd.DataFrame(data=train_data,columns=features)


# In[ ]:


features = [f for f in features if f not in ["Yards"]]
X_train = X_train[features]

print(X_train.shape)
X_train.head()


# In[ ]:


y_train = np.zeros(len(y_train_),dtype=np.float)
for i in range(len(y_train)):
    y_train[i]=(y_train_[i])

scaler = preprocessing.StandardScaler()
scaler.fit([[y] for y in y_train])
y_train = np.array([y[0] for y in scaler.transform([[y] for y in y_train])])
data = [0 for i in range(199)]
for y in y_train:
    data[int(y+99)]+=1
plt.plot([i-99 for i in range(199)],data)


# In[ ]:


missing = train_df.isnull().sum() # Sum of missing values
missing = missing[missing > 0]  
missing.sort_values(inplace=True)
missing


# Now ready for modeling!

# # Modeling
# Here we use lightGBM based on https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm. Please upvote his kernel!

# In[ ]:


print(X_train.shape)
print(y_train.shape)


# In[ ]:


# from https://www.kaggle.com/newbielch/lgbm-regression-view
def get_cdf_df(yards_array):
    pdf, edges = np.histogram(yards_array, bins=199,
                 range=(-99,100), density=True)
    cdf = pdf.cumsum().clip(0, 1)
    cdf_df = pd.DataFrame(data=cdf.reshape(-1, 1).T, 
                            columns=['Yards'+str(i) for i in range(-99,100)])
    return cdf_df
cdf = get_cdf_df(y_train).values.reshape(-1,)
dist_to_end_train = X_train.apply(lambda x:(100 - x.loc['YardLine']) if x.loc["Field_eq_Possession"]==1 else x.loc['YardLine'],axis=1)


# In[ ]:


def get_score(y_pred,cdf,w,dist_to_end):
    y_pred = int(y_pred)
#     y_pred = y_pred.astype(int)
    if y_pred ==w:
        y_pred_array = cdf.copy()
    elif y_pred - w >0:
        y_pred_array = np.zeros(199)
        y_pred_array[(y_pred-w):] = cdf[:(-(y_pred-w))].copy()
    elif w - y_pred >0:
        y_pred_array = np.ones(199)
        y_pred_array[:(y_pred-w)] = cdf[(w-y_pred):].copy()
    y_pred_array[-1]=1
    y_pred_array[(dist_to_end+99):]=1
    return y_pred_array    

def get_score_pingyi1(y_pred,y_true,cdf,w,dist_to_end):
    y_pred = int(y_pred)
    if y_pred ==w:
        y_pred_array = cdf.copy()
    elif y_pred - w >0:
        y_pred_array = np.zeros(199)
        y_pred_array[(y_pred-w):] = cdf[:(-(y_pred-w))].copy()
    elif w - y_pred >0:
        y_pred_array = np.ones(199)
        y_pred_array[:(y_pred-w)] = cdf[(w-y_pred):].copy()
    y_pred_array[-1]=1
    y_pred_array[(dist_to_end+99):]=1
    y_true_array = np.zeros(199)
    y_true_array[(y_true+99):]=1
    return np.mean((y_pred_array - y_true_array)**2)


def CRPS_pingyi1(y_preds,y_trues,w,cdf,dist_to_ends):
    if len(y_preds) != len(y_trues):
        print('length does not match')
        return None
    n = len(y_preds)
    tmp = []
    for a,b,c in zip(y_preds, y_trues, dist_to_ends):
        tmp.append(get_score_pingyi1(a,b,cdf,w,c))
    return np.mean(tmp)


# In[ ]:


# Initial LGB parameters are ...
lgbParams = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    "num_iterations": 1000, 
    "learning_rate": 0.05,
    "lambda_l1": 9,
    "lambda_l2": 0.9,
    "num_leaves": 42,
    "feature_fraction": 0.4,
    "bagging_fraction": 0.45,
    "bagging_freq": 7,
    "min_child_samples": 74,
    "random_state": 42
}


# ## Feature importance

# In[ ]:


## Visualize feature importance

# make a LightGBM dataset
trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
d_train = lgb.Dataset(trainX, trainY)
d_eval = lgb.Dataset(testX, testY, reference=d_train)

# model training
LGBmodel = lgb.train(lgbParams, d_train, valid_sets=d_eval, verbose_eval=1000)
# LGBmodel = lgb.train(lgbParams, d_train, valid_sets=d_eval, early_stopping_rounds=500, verbose_eval=1000)

# feature importance
importance = LGBmodel.feature_importance(importance_type="gain")
ranking = np.argsort(-importance)
fig, ax = plt.subplots(figsize=(20, 20))
sns.barplot(x=importance[ranking], y=X_train.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()


# Let's use the first X, for now...

# In[ ]:


features = X_train.columns.values[ranking][:30]
print(features)
X_train = X_train[features]


# ### lightGBM parameter tuning using Optuna
# Uncomment the following two cells to run hyperparameter optimization using [Optuna](https://optuna.org).

# In[ ]:


# # FYI: Objective functions can take additional arguments
# # (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
# def objective(trial):
    
#     # make a LightGBM dataset
#     trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
#     d_train = lgb.Dataset(trainX, trainY)

#     param = {
#         'objective': 'regression',
#         'metric': 'mae',
#         'verbosity': -1,
#         'boosting_type': 'gbdt',
#         'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#         'num_leaves': trial.suggest_int('num_leaves', 40, 256),
#         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#     }

#     gbm = lgb.train(param, d_train)
#     preds = gbm.predict(testX)
#     mae = mean_absolute_error(testY, preds)
#     return mae


# In[ ]:


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

# print('Number of finished trials: {}'.format(len(study.trials)))

# print('Best trial:')
# trial = study.best_trial

# print('  Value: {}'.format(trial.value))

# print('  Params: ')
# for key, value in trial.params.items():
#     print('    {}: {}'.format(key, value))


# So we use...

# In[ ]:


# lgbParams = trial.params
# lgbParams['objective'] = 'regression'
# lgbParams['metric'] = 'mae'
# lgbParams['verbosity'] = -1
# lgbParams['boosting_type'] = 'gbdt'
# lgbParams["learning_rate"] = 0.01
# lgbParams["num_iterations"] = 5000
# lgbParams["random_state"] = 1220
# print(lgbParams)


# In[ ]:


lgbParams = {'lambda_l1': 9.22654100630611, 'lambda_l2': 0.0014139850193561965, 'num_leaves': 64, 'feature_fraction': 0.5952892097040369,
             'bagging_fraction': 0.45160449675869496, 'bagging_freq': 3, 'min_child_samples': 91, 
             'objective': 'regression', 'metric': 'mae', 'verbosity': -1, 'boosting_type': 'gbdt', 
             'learning_rate': 0.01, 'num_iterations': 5000, 'random_state': 1220}


# In[ ]:


n_splits = 5
seed = 1220
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
y_valid = np.zeros(X_train.shape[0])
models = []

for train_idx, valid_idx in kf.split(X_train, y_train):
    trainX, trainY = X_train.iloc[train_idx, :], y_train[train_idx]
    validX, validY = X_train.iloc[valid_idx, :], y_train[valid_idx]
    
    d_train = lgb.Dataset(trainX, trainY)
    d_eval = lgb.Dataset(validX, validY, reference=d_train)
    
    LGBmodel = lgb.train(lgbParams, d_train, valid_sets=d_eval, 
                         early_stopping_rounds=500, 
                         learning_rates=lambda iter: 0.01 * (0.99 ** iter),
                         verbose_eval=1000)
    y_valid[valid_idx] += LGBmodel.predict(validX, num_iteration=LGBmodel.best_iteration)
    models.append(LGBmodel)
gc.collect()


# # Evaluation

# In[ ]:


cprs = CRPS_pingyi1(y_valid, y_train.astype(int), 4, cdf, dist_to_end_train.astype(int))
print("cprs = {}".format(cprs))


# In[ ]:


# y_pred = np.zeros((509762//22,199))
# y_ans = np.zeros((509762//22,199))

# for i,p in enumerate(np.round(scaler.inverse_transform(y_valid))):
#     p+=99
#     for j in range(199):
#         if j>=p+10:
#             y_pred[i][j]=1.0
#         elif j>=p-10:
#             y_pred[i][j]=(j+10-p)*0.05

# for i,p in enumerate(y_train):
#     p+=99
#     for j in range(199):
#         if j>=p:
#             y_ans[i][j]=1.0

# print("validation score:",np.sum(np.power(y_pred-y_ans,2))/(199*(509762//22)))


# In[ ]:


# _EvalFunction(y_train, y_valid)[1]


# In[ ]:


sns.distplot(y_valid)


# # Make a submission
# ## `iter_test` function
# 
# Generator which loops through each rushing play in the test set and provides the observations at `TimeHandoff` just like the training set.  Once you call **`predict`** to make your yardage prediction, you can continue on to the next play.
# 
# Yields:
# * While there are more rushing play(s) and `predict` was called successfully since the last yield, yields a tuple of:
#     * `test_df`: DataFrame with player and game observations for the next rushing play.
#     * `sample_prediction_df`: DataFrame with an example yardage prediction.  Intended to be filled in and passed back to the `predict` function.
# * If `predict` has not been called successfully since the last yield, prints an error and yields `None`.

# In[ ]:


# You can only call make_env() once, so don't lose it!
env = nflrush.make_env()


# In[ ]:


index = 0
for (test_df, sample_prediction_df) in tqdm.tqdm(env.iter_test()):
#     test, le = preprocess(test_df, labelEncoders=labelEncoders)
    test = preprocess(test_df)
    count=0
    dist_to_end_test = test.apply(lambda x:(100 - x.loc['YardLine']) if x.loc["Field_eq_Possession"]==1 else x.loc['YardLine'],axis=1)
    test_data = np.zeros((1,len(features)))
    for c in features:
        try:
            test_data[0][count] = test[c][index]
        except:
            test_data[0][count] = np.nan
        count+=1
        
#     print(test_data)
    y_pred = np.zeros(199)        
    y_pred_p = np.sum(np.round(scaler.inverse_transform(
        [model.predict(test_data) for model in models]))) / n_splits
#     y_pred = list(get_score(y_pred_p, cdf, 4, dist_to_end_test.astype(int).values[0]))
#     y_pred = np.array(y_pred).reshape(1,199)
#     print(y_pred)
    y_pred_p += 99
    for j in range(199):
        if j>=y_pred_p+10:
            y_pred[j]=1.0
        elif j>=y_pred_p-10:
            y_pred[j]=(j+10-y_pred_p)*0.05
#             y_pred[j]=norm_cumsum[max(min(j+10-y_pred_p),0)]
#     pred_target = pd.DataFrame(index = sample_prediction_df.index, \
#                                columns = sample_prediction_df.columns, \
#                                data = y_pred)
#     env.predict(pred_target)
    env.predict(pd.DataFrame(data=[y_pred],columns=sample_prediction_df.columns))
    index += 22
env.write_submission_file()

