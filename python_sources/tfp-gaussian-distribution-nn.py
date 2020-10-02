#!/usr/bin/env python
# coding: utf-8

# Inspiration from these 2:
# https://www.kaggle.com/coolcoder22/nn-19-features
# https://www.kaggle.com/kenmatsu4/nn-outputs-gaussian-distribution-directly

# Mixed the kernals together and added a couple TFP layers to the model!

# In[ ]:


#update gast to correct version
#!pip install gast==0.2.2


# For TFP to work, we need to use TensorFlow.Keras

# In[ ]:


import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime
from kaggle.competitions import nflrush
import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')
mpl.rcParams['figure.figsize'] = [15,10]

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

from matplotlib import animation as ani
from IPython.display import Image

plt.rcParams["patch.force_edgecolor"] = True
#rc('text', usetex=True)
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]

# pandas formatting
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = '{:,.5f}'.format

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# ## Feature engineering

# In[ ]:


def preprocess(train):
    ## GameClock
    train['GameClock_sec'] = train['GameClock'].apply(strtoseconds)
    train["GameClock_minute"] = train["GameClock"].apply(lambda x : x.split(":")[0]).astype("object")

    ## Height
    train['PlayerHeight_dense'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    ## Time
    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    ## Age
    seconds_in_year = 60*60*24*365.25
    train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    train["PlayerAge_ob"] = train['PlayerAge'].astype(np.int).astype("object")

    ## WindSpeed
    train['WindSpeed_ob'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    train['WindSpeed_dense'] = train['WindSpeed_ob'].apply(strtofloat)

    ## Weather
    train['GameWeather_process'] = train['GameWeather'].str.lower()
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    train['GameWeather_dense'] = train['GameWeather_process'].apply(map_weather)

    ## Rusher
    train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])
    train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")
    temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})
    train = train.merge(temp, on = "PlayId")
    train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

    ## dense -> categorical
    train["Quarter_ob"] = train["Quarter"].astype("object")
    train["Down_ob"] = train["Down"].astype("object")
    train["JerseyNumber_ob"] = train["JerseyNumber"].astype("object")
    train["YardLine_ob"] = train["YardLine"].astype("object")
    # train["DefendersInTheBox_ob"] = train["DefendersInTheBox"].astype("object")
    # train["Week_ob"] = train["Week"].astype("object")
    # train["TimeDelta_ob"] = train["TimeDelta"].astype("object")

    ## diff Score
    train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]
    train["diffScoreBeforePlay_binary_ob"] = (train["HomeScoreBeforePlay"] > train["VisitorScoreBeforePlay"]).astype("object")

    ## Turf
    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 
    train['Turf'] = train['Turf'].map(Turf)

    ## sort
#     train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index(drop = True)
    train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop = True)
    return train


# In[ ]:


def create_features(df, deploy=False):
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate

    def new_line(rush_team, field_position, yardline):
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return 10.0 + yardline
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return 60.0 + (50 - yardline)

    def new_orientation(angle, play_direction):
        if play_direction == 'left':
            new_angle = 360.0 - angle
            if new_angle == 360.0:
                new_angle = 0.0
            return new_angle
        else:
            return angle

    def euclidean_distance(x1,y1,x2,y2):
        x_diff = (x1-x2)**2
        y_diff = (y1-y2)**2

        return np.sqrt(x_diff + y_diff)

    def back_direction(orientation):
        if orientation > 180.0:
            return 1
        else:
            return 0

    def update_yardline(df):
        new_yardline = df[df['NflId'] == df['NflIdRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
        new_yardline = new_yardline[['GameId','PlayId','YardLine']]

        return new_yardline

    def update_orientation(df, yardline):
        df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
        df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
        df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

        df = df.drop('YardLine', axis=1)
        df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

        return df

    def back_features(df):
        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
        carriers = carriers.rename(columns={'X':'back_X',
                                            'Y':'back_Y'})
        carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']]

        return carriers

    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])                                         .agg({'dist_to_back':['min','max','mean','std']})                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                   'min_dist','max_dist','mean_dist','std_dist']

        return player_distance

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        defense = defense.groupby(['GameId','PlayId'])                         .agg({'def_dist_to_back':['min','max','mean','std']})                         .reset_index()
        defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

        return defense

    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
                                                            'YardLine','Quarter','Down','Distance','DefendersInTheBox']].drop_duplicates()
        static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))

        return static_features
    
    def split_personnel(s):
        splits = s.split(',')
        for i in range(len(splits)):
            splits[i] = splits[i].strip()

        return splits

    def defense_formation(l):
        dl = 0
        lb = 0
        db = 0
        other = 0

        for position in l:
            sub_string = position.split(' ')
            if sub_string[1] == 'DL':
                dl += int(sub_string[0])
            elif sub_string[1] in ['LB','OL']:
                lb += int(sub_string[0])
            else:
                db += int(sub_string[0])

        counts = (dl,lb,db,other)

        return counts

    def offense_formation(l):
        qb = 0
        rb = 0
        wr = 0
        te = 0
        ol = 0

        sub_total = 0
        qb_listed = False
        for position in l:
            sub_string = position.split(' ')
            pos = sub_string[1]
            cnt = int(sub_string[0])

            if pos == 'QB':
                qb += cnt
                sub_total += cnt
                qb_listed = True
            # Assuming LB is a line backer lined up as full back
            elif pos in ['RB','LB']:
                rb += cnt
                sub_total += cnt
            # Assuming DB is a defensive back and lined up as WR
            elif pos in ['WR','DB']:
                wr += cnt
                sub_total += cnt
            elif pos == 'TE':
                te += cnt
                sub_total += cnt
            # Assuming DL is a defensive lineman lined up as an additional line man
            else:
                ol += cnt
                sub_total += cnt

        # If not all 11 players were noted at given positions we need to make some assumptions
        # I will assume if a QB is not listed then there was 1 QB on the play
        # If a QB is listed then I'm going to assume the rest of the positions are at OL
        # This might be flawed but it looks like RB, TE and WR are always listed in the personnel
        if sub_total < 11:
            diff = 11 - sub_total
            if not qb_listed:
                qb += 1
                diff -= 1
            ol += diff

        counts = (qb,rb,wr,te,ol)

        return counts
    
    def personnel_features(df):
        personnel = df[['GameId','PlayId','OffensePersonnel','DefensePersonnel']].drop_duplicates()
        personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: split_personnel(x))
        personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: defense_formation(x))
        personnel['num_DL'] = personnel['DefensePersonnel'].apply(lambda x: x[0])
        personnel['num_LB'] = personnel['DefensePersonnel'].apply(lambda x: x[1])
        personnel['num_DB'] = personnel['DefensePersonnel'].apply(lambda x: x[2])

        personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: split_personnel(x))
        personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: offense_formation(x))
        personnel['num_QB'] = personnel['OffensePersonnel'].apply(lambda x: x[0])
        personnel['num_RB'] = personnel['OffensePersonnel'].apply(lambda x: x[1])
        personnel['num_WR'] = personnel['OffensePersonnel'].apply(lambda x: x[2])
        personnel['num_TE'] = personnel['OffensePersonnel'].apply(lambda x: x[3])
        personnel['num_OL'] = personnel['OffensePersonnel'].apply(lambda x: x[4])

        # Let's create some features to specify if the OL is covered
        personnel['OL_diff'] = personnel['num_OL'] - personnel['num_DL']
        personnel['OL_TE_diff'] = (personnel['num_OL'] + personnel['num_TE']) - personnel['num_DL']
        # Let's create a feature to specify if the defense is preventing the run
        # Let's just assume 7 or more DL and LB is run prevention
        personnel['run_def'] = (personnel['num_DL'] + personnel['num_LB'] > 6).astype(int)

        personnel.drop(['OffensePersonnel','DefensePersonnel'], axis=1, inplace=True)
        
        return personnel
    
    def process_two(t_):
        t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) - np.square(t_.Y.values))))
        t_['fe5'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values  # N
        t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
        t_['fe8'] = t_['S'].values / np.clip(t_['fe1'].values, 0.6, None)
        radian_angle = (90 - t_['Dir']) * np.pi / 180.0
        t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
        t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
        return t_

    def combine_features(relative_to_back, defense, static, personnel, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,personnel,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df
    
    df.loc[df['Season'] == 2017, 'S'] = (df['S'][df['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    static_feats = static_features(df)
    personnel = personnel_features(df)
    basetable = combine_features(rel_back, def_feats, static_feats, personnel, deploy=deploy)
    basetable = process_two(basetable)
    basetable = basetable.drop(['X','Y','S','A','Dis','Orientation','Dir','Yards','YardLine',
           'Quarter','Down','Distance','DefendersInTheBox'], axis=1)
    
    return basetable


# In[ ]:


#https://www.kaggle.com/rooshroosh/fork-of-neural-networks-different-architecture
def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans

def strtofloat(x):
    try:
        return float(x)
    except:
        return -1

def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0


# In[ ]:


get_ipython().run_cell_magic('time', '', "df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})\noutcomes = df[['GameId','PlayId','Yards']].drop_duplicates()\n\ntrain_gametable = preprocess(df)\ntrain_basetable = create_features(df, False)")


# In[ ]:


join_by = ['GameId','PlayId']
train = pd.merge(train_gametable, train_basetable, on=(join_by), how='left')
print(train.shape)


# In[ ]:


## DisplayName remove Outlier
v = train["DisplayName"].value_counts()
missing_values = list(v[v < 5].index)
train["DisplayName"] = train["DisplayName"].where(~train["DisplayName"].isin(missing_values), "nan")

## PlayerCollegeName remove Outlier
v = train["PlayerCollegeName"].value_counts()
missing_values = list(v[v < 10].index)
train["PlayerCollegeName"] = train["PlayerCollegeName"].where(~train["PlayerCollegeName"].isin(missing_values), "nan")


# In[ ]:


#join_by = ['DisplayName','Team','Quarter_ob']
#agg_history = train.groupby(join_by, as_index=False)['Yards'].agg('mean')
#agg_history.reset_index(inplace=True, drop=True)
#agg_history.columns = ['avg_' + c if c == 'Yards' else c for c in agg_history.columns.values]
#agg_history = agg_history.replace(np.nan, 0)

#agg_history.reset_index(inplace=True, drop=True)
#agg_history.columns = ['avg_' + c if c == 'Yards' else c for c in agg_history.columns.values]
#agg_history = agg_history.replace(np.nan, 0)

#train = pd.merge(train, agg_history, on=(join_by), how='left')


# In[ ]:


def drop(df):
    drop_cols = ["GameId", "GameWeather", "NflId", "Season", "NflIdRusher",'JerseyNumber'] 
    drop_cols += ['TimeHandoff', 'TimeSnap', 'PlayerBirthDate','PlayerHeight','diffScoreBeforePlay_binary_ob']
    drop_cols += ['Team','DisplayName','GameClock','PossessionTeam']
    drop_cols += ['PlayerCollegeName','Position','HomeTeamAbbr','FieldPosition']
    drop_cols += ['VisitorTeamAbbr','Stadium','Location','StadiumType','JerseyNumber_ob']
    drop_cols += ['PlayId','Yards','RusherTeam','YardLine_ob','WindSpeed','GameClock_minute']

    df = df.drop(drop_cols, axis = 1)
    return df


# In[ ]:


train = drop(train)
pd.to_pickle(train, "train.pkl")


# In[ ]:


cat_features = []
dense_features = []
for col in train.columns:
    if train[col].dtype =='object':
        cat_features.append(col)
        print("*cat*", col, len(train[col].unique()))
    else:
        dense_features.append(col)
        print("!dense!", col, len(train[col].unique()))


# ## categorical

# In[ ]:


train_cat = train[cat_features]
categories = []
most_appear_each_categories = {}
for col in tqdm_notebook(train_cat.columns):
    print(col)
    train_cat.loc[:,col] = train_cat[col].fillna("nan")
    train_cat.loc[:,col] = col + "__" + train_cat[col].astype(str)
    most_appear_each_categories[col] = list(train_cat[col].value_counts().index)[0]
    categories.append(train_cat[col].unique())
categories = np.hstack(categories)
print(len(categories))


# In[ ]:


le = LabelEncoder()
le.fit(categories)
for col in tqdm_notebook(train_cat.columns):
    train_cat.loc[:, col] = le.transform(train_cat[col])
num_classes = len(le.classes_)


# ## Dense

# In[ ]:


train_dense = train[dense_features]
sss = {}
medians = {}
for col in tqdm_notebook(train_dense.columns):
    print(col)
    medians[col] = np.nanmedian(train_dense[col])
    train_dense.loc[:, col] = train_dense[col].fillna(medians[col])
    ss = StandardScaler()
    train_dense.loc[:, col] = ss.fit_transform(train_dense[col].values[:,None])
    sss[col] = ss


# ## Divide features into groups

# In[ ]:


## dense features for play
dense_game_features = train_dense.columns[train_dense[:22].std() == 0]
## dense features for each player
dense_player_features = train_dense.columns[train_dense[:22].std() != 0]
## categorical features for play
cat_game_features = train_cat.columns[train_cat[:22].std() == 0]
## categorical features for each player
cat_player_features = train_cat.columns[train_cat[:22].std() != 0]


# In[ ]:


train_dense_game = train_dense[dense_game_features].iloc[np.arange(0, len(train), 22)].reset_index(drop = True).values
train_dense_game = np.hstack([train_dense_game, train_dense[dense_player_features][train_dense["IsRusher"] > 0]]) ## with rusher player feature

train_dense_players = [train_dense[dense_player_features].iloc[np.arange(k, len(train), 22)].reset_index(drop = True) for k in range(22)]
train_dense_players = np.stack([t.values for t in train_dense_players]).transpose(1, 0, 2)

train_cat_game = train_cat[cat_game_features].iloc[np.arange(0, len(train), 22)].reset_index(drop = True).values
train_cat_game = np.hstack([train_cat_game, train_cat[cat_player_features][train_dense["IsRusher"] > 0]]) ## with rusher player feature

train_cat_players = [train_cat[cat_player_features].iloc[np.arange(k, len(train), 22)].reset_index(drop = True) for k in range(22)]
train_cat_players = np.stack([t.values for t in train_cat_players]).transpose(1, 0, 2)


# In[ ]:


def return_step(x):
    temp = np.zeros(199)
    temp[x + 99:] = 1
    return temp

#train_y_raw = train["Yards"].iloc[np.arange(0, len(train), 22)].reset_index(drop = True)
train_y_raw = outcomes['Yards']
train_y = np.vstack(outcomes['Yards'].apply(return_step).values)


# In[ ]:


train_dense_game.shape, train_dense_players.shape, train_cat_game.shape, train_cat_players.shape, train_y.shape


# ## Model

# In[ ]:


from tensorflow.keras.callbacks import Callback
class CRPSCallback(Callback):
    
    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        
        print('validation shape',len(self.validation))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['CRPS_score_val'] = float('-inf')
            
        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * pd.DataFrame(X_valid).shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s
        


# In[ ]:


def crps(y_true, y_pred):
    loss = K.mean((y_pred - y_true)**2)
    return loss

# get the newest model file within a directory
def getNewestModel(model, dirname):
    from glob import glob
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model
    
def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
      n = kernel_size + bias_size
      c = np.log(np.expm1(1.))
      return tf.keras.Sequential([
          tfp.layers.VariableLayer(2 * n, dtype=dtype),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(  # pylint: disable=g-long-lambda
              tfd.Normal(loc=t[..., :n],
                         scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
              reinterpreted_batch_ndims=1)),])

def prior_trainable(kernel_size, bias_size=0, dtype=None):
      n = kernel_size + bias_size
      return tf.keras.Sequential([
          tfp.layers.VariableLayer(n, dtype=dtype),
          tfp.layers.DistributionLambda(
              lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1),  # pylint: disable=g-long-lambda
                                        reinterpreted_batch_ndims=1)),])


# In[ ]:


class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred = self.model.predict(X_train)
        y_true = np.clip(np.cumsum(y_train, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        tr_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_train[-1].shape[0])
        tr_s = np.round(tr_s, 6)
        logs['tr_CRPS'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid[-1].shape[0])
        val_s = np.round(val_s, 6)
        logs['val_CRPS'] = val_s
        print('tr CRPS', tr_s, 'val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)
            


# In[ ]:


def get_model(batch_size = 32, epochs = 10):
    keras.backend.clear_session()
    
    ## inputs
    input_dense_game = keras.layers.Input(shape=(train_dense_game.shape[1],), name = "numerical_general_inputs")
    input_dense_players = keras.layers.Input(shape=(train_dense_players.shape[1],train_dense_players.shape[2]), name = "numerical_players_inputs")
    input_cat_game = keras.layers.Input(shape=(train_cat_game.shape[1], ), name = "categorical_general_inputs")
    input_cat_players = keras.layers.Input(shape=(train_cat_players.shape[1], train_cat_players.shape[2]), name = "categorical_players_input")
    
    ## embedding
    embedding = keras.layers.Embedding(num_classes, 4, embeddings_regularizer=regularizers.l2(1e-4))
    emb_cat_game = embedding(input_cat_game)
    emb_cat_game = keras.layers.Flatten()(emb_cat_game)
    emb_cat_players = embedding(input_cat_players)
    emb_cat_players = keras.layers.Reshape((int(emb_cat_players.shape[1]), int(emb_cat_players.shape[2]) * int(emb_cat_players.shape[3])))(emb_cat_players)
    
    ## general game features
    game = keras.layers.Concatenate(name = "general_features")([input_dense_game, emb_cat_game])
    game = keras.layers.Dense(1024, activation="relu")(game)
    game = keras.layers.Dropout(0.5)(game)
    
    ## players features
    players = keras.layers.Concatenate(name = "players_features")([input_dense_players, emb_cat_players])
    n_unit = 256
    players_aves = []
    for k in range(3):
        players = keras.layers.Dense(n_unit, activation="relu")(players)
        players_aves.append(keras.layers.GlobalAveragePooling1D()(players))
    players = keras.layers.Concatenate(name = "deep_players_features")(players_aves)

    ### concat all
    x_concat = keras.layers.Concatenate(name = "general_and_players")([game, players])
    x_concats = []
    n_unit = 128
    decay_rate = 0.5
    for k in range(3):
        x_concat = keras.layers.Dense(n_unit, activation="relu")(x_concat)
        x_concats.append(x_concat)
        n_unit = int(n_unit * decay_rate)
    x_concat = keras.layers.Concatenate(name = "deep_features")(x_concats)
    x_concat = keras.layers.Dropout(0.5)(x_concat)

    out_soft = keras.layers.Dense(199, activation="softmax", name = "out_soft")(x_concat)
    model = keras.models.Model(inputs = [input_dense_game, input_dense_players, input_cat_game, input_cat_players],
                               outputs = out_soft)
    ## compile
    model.compile(optimizer=keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999), loss=crps, metrics=[])

    ## train
    tr_x = [train_dense_game[tr_inds], train_dense_players[tr_inds], train_cat_game[tr_inds], train_cat_players[tr_inds]]
    tr_y = train_y[tr_inds]
    val_x = [train_dense_game[val_inds], train_dense_players[val_inds], train_cat_game[val_inds], train_cat_players[val_inds]]
    val_y = train_y[val_inds]
    
    es = EarlyStopping(monitor='val_CRPS', 
                   mode='min',
                   restore_best_weights=True, 
                   verbose=1, 
                   patience=15)

    es.set_model(model)
    metric = Metric(model, [es], [(tr_x,tr_y), (val_x, val_y)])
    model.fit(tr_x,
              tr_y,
              batch_size=32,
              callbacks=[metric],
              epochs=epochs,
              verbose=1)
    return model


# In[ ]:


from sklearn.model_selection import train_test_split, KFold
tf.keras.backend.clear_session()
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras.layers import Concatenate, Layer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

losses = []
models = []
for k in range(2):
    kfold = KFold(5, random_state = 420 + k, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(train_y)):
        print("-----------")
        print("-----------")
        model, loss = get_model(1024, 200)
        models.append(model)
        print(k_fold, loss)
        losses.append(loss)
print("-------")
print(losses)
print(np.mean(losses))


# In[ ]:


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[ ]:


print(losses)
print(np.mean(losses))


# ## Prediction

# In[ ]:


def make_pred(test, sample, env, model):    
    test_gametable = preprocess(test)
    test_basetable = create_features(test, False)

    test = pd.merge(test_gametable, test_basetable, on=(join_by), how='left')
    #test = pd.merge(test, agg_history, on=join_by, how='left')
    test = drop(test)

    ### categorical
    test_cat = test[cat_features]
    for col in (test_cat.columns):
        test_cat.loc[:,col] = test_cat[col].fillna("nan")
        test_cat.loc[:,col] = col + "__" + test_cat[col].astype(str)
        isnan = ~test_cat.loc[:,col].isin(categories)
        if np.sum(isnan) > 0:
#             print("------")
#             print("test have unseen label : col")
            if not ((col + "__nan") in categories):
#                 print("not nan in train : ", col)
                test_cat.loc[isnan,col] = most_appear_each_categories[col]
            else:
#                 print("nan seen in train : ", col)
                test_cat.loc[isnan,col] = col + "__nan"
    for col in (test_cat.columns):
        test_cat.loc[:, col] = le.transform(test_cat[col])

    ### dense
    test_dense = test[dense_features]
    for col in (test_dense.columns):
        test_dense.loc[:, col] = test_dense[col].fillna(medians[col])
        test_dense.loc[:, col] = sss[col].transform(test_dense[col].values[:,None])

    ### divide
    test_dense_players = [test_dense[dense_player_features].iloc[np.arange(k, len(test), 22)].reset_index(drop = True) for k in range(22)]
    test_dense_players = np.stack([t.values for t in test_dense_players]).transpose(1,0, 2)

    test_dense_game = test_dense[dense_game_features].iloc[np.arange(0, len(test), 22)].reset_index(drop = True).values
    test_dense_game = np.hstack([test_dense_game, test_dense[dense_player_features][test_dense["IsRusher"] > 0]])
    
    test_cat_players = [test_cat[cat_player_features].iloc[np.arange(k, len(test), 22)].reset_index(drop = True) for k in range(22)]
    test_cat_players = np.stack([t.values for t in test_cat_players]).transpose(1,0, 2)

    test_cat_game = test_cat[cat_game_features].iloc[np.arange(0, len(test), 22)].reset_index(drop = True).values
    test_cat_game = np.hstack([test_cat_game, test_cat[cat_player_features][test_dense["IsRusher"] > 0]])

    test_inp = [test_dense_game, test_dense_players, test_cat_game, test_cat_players]
    
    ## pred
    pred = 0
    for model in models:
        _pred = model.predict(test_inp)
        #_pred = np.cumsum(_pred, axis = 1)
        pred += _pred
    pred /= len(models)
    pred = np.clip(pred, 0, 1)
    #pred = np.where(pred > 0.93,1,pred)
    env.predict(pd.DataFrame(data=pred,columns=sample.columns))
    return pred


# In[ ]:


env = nflrush.make_env()
preds = []
for test, sample in tqdm_notebook(env.iter_test()):
    pred = make_pred(test, sample, env, models)
    preds.append(pred)
env.write_submission_file()


# In[ ]:


preds = np.vstack(preds)
## check whether prediction is submittable
print(np.mean(np.diff(preds, axis = 1) >= 0) == 1.0)
print(np.mean(preds > 1) == 0)


# In[ ]:


print(losses)
print(np.mean(losses))


# In[ ]:




