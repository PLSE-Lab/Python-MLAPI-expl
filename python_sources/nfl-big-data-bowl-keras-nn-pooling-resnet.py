#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import sys
import numpy as np
import pandas as pd
import datetime
import random
import time as tm
import gc

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import tensorflow as tf
import keras
from keras.utils import plot_model
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, Concatenate, Reshape, GlobalAveragePooling1D, Activation, BatchNormalization, Add
from keras.initializers import glorot_normal
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint

import category_encoders as ce
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer

import tqdm
from tqdm import tqdm_notebook

from kaggle.competitions import nflrush

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = -1

def seed_everything(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

RANDOM_STATE = 313
seed_everything(seed=RANDOM_STATE)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


start = tm.time()


# In[ ]:


# choose mode
prod = 1
commit = 0


# In[ ]:


if prod == 1:
    train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
else:
    train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
    train = train[pd.to_datetime(train['TimeHandoff'], infer_datetime_format=True, utc=True) >= pd.to_datetime('2018-01-01', infer_datetime_format=True, utc=True)]
    train = train.iloc[0:22*100]

print(train.shape)
train = reduce_mem_usage(train, verbose=True)


# # Helper Functions

# In[ ]:


def handle_windspeed(speed):
    if pd.isna(speed) is True:
        return speed
    else:
        speed_lower = speed.lower().strip()
        try:
            candidate_speed = int(speed_lower)
            return candidate_speed
        except Exception:
            ss = speed.split(" ")
            sd = speed.split("-")
            if ss[-1] == "mph":
                return int(ss[0])
            elif ss[-1].isnumeric() is True:
                return int(float(ss[0])*0.80 + float(ss[-1])*0.20)
            elif sd[-1].isnumeric() is True:
                return int(float(sd[0])/2 + float(sd[-1])/2) 
            elif speed.isalpha() is True:
                return np.nan
            elif "mph" in speed and len(speed) <= 5:
                return int(speed.replace("mph", ""))
            
def handle_offence_personnel(x):
    map_ = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0, 'QB' : 0, 'RB' : 0, 'TE' : 0, 'WR' : 0}
    for formation in x.strip().split(","):
        item = formation.strip().split(" ")
        map_.update({item[1]: int(item[0])})
    return map_

def handle_defence_personnel(x):
    map_ = {'OL': 0, 'LB': 0, 'DL': 0, 'DB': 0}
    for formation in x.strip().split(","):
        item = formation.strip().split(" ")
        map_.update({item[1]: int(item[0])})
    return map_

def feature_set(orig_features=None, new_features=None, operation="add"):
    if operation == "add":
        return list(set(orig_features + new_features))
    else:
        return list(set(set(orig_features) - set(new_features)))

def handle_height(x):
    split_ = x.split('-')
    foot = int(split_[0])
    inch = int(split_[-1])
    return foot * 12 + inch

def handle_game_clock(x):
    split_ = x.split(":")
    minute = int(split_[0])
    second = int(split_[1])
    return np.around((minute * 60.0 + second)/60.0)

def tag_to_left(x):
    if x == "left":
        return True
    else:
        return False

def fix_team_abbr(x):
    if x == "ARI":
        return "ARZ"
    elif x == "BAL":
        return "BLT"
    elif x == "CLE":
        return "CLV"
    elif x == "HOU":
        return "HST"
    else:
        return x
    
def tag_team_on_offence(row):
    if row['PossessionTeam'] == row['HomeTeamAbbr']:
        return 'home'
    else:
        return 'away'

def tag_yards_from_own_goal(row):
    if row['FieldPosition'] == row['PossessionTeam']:
        return row['YardLine']
    else:
        return 50 + (50 - row['YardLine'])

def tag_X_std(row):
    if row['ToLeft'] is True:
        return (120 - row['X']) - 10
    else:
        return row['X'] - 10

def tag_Y_std(row):
    if row['ToLeft'] is True:
        return 160/3 - row['Y']
    else:
        return row['Y']

def tag_dir_std_1(row):
    if row["ToLeft"] is True and row['Dir'] < 90:
        return row['Dir'] + 360
    else:
        return row['Dir']

def tag_dir_to_right(row):
    if row["ToLeft"] is False and row["Dir"] > 270:
        return row["Dir"] - 360
    else:
        return row["Dir_std_1"]

def tag_dir_std_2(row):
    if row["ToLeft"] is True:
        return row["Dir_std_1"] - 180
    else:
        return row["Dir_std_1"]
    
def tag_orien_std_1(row):
    if row["ToLeft"] is True and row['Orientation'] < 90:
        return row['Orientation'] + 360
    else:
        return row['Orientation']

def tag_orien_to_right(row):
    if row["ToLeft"] is False and row["Orientation"] > 270:
        return row["Orientation"] - 360
    else:
        return row["Orien_std_1"]

def tag_orien_std_2(row):
    if row["ToLeft"] is True:
        return row["Orien_std_1"] - 180
    else:
        return row["Orien_std_1"]
    
def generate_y(x):
    step = np.zeros(199).astype(np.int8)
    step[99 + x:] = 1
    return step

def encode_OP_DP(df, op_dp_columns, column, prefix, mode_="train"):
    temp = df[column].iloc[np.arange(0, len(df), 22)].apply(lambda x: pd.Series(handle_offence_personnel(x)))
    temp.columns = [prefix + c for c in temp.columns]
    if mode_ == "train":
        op_dp_columns += list(temp.columns)
    temp['PlayId'] = df['PlayId'].iloc[np.arange(0, len(df), 22)]
    df = df.merge(temp, on = "PlayId")
    return df, op_dp_columns


# In[ ]:


def compound_1_f(play):
    ball_carrier = play.loc[play['IsBallCarrier'] ==1]
    closeness_seq = np.sqrt(np.power(play.loc[ play['IsOnOffense'] == 0, 'X_std'] - ball_carrier['X_std'].item(), 2) + np.power(play.loc[ play['IsOnOffense'] == 0, 'Y_std'] - ball_carrier['Y_std'].item(), 2))
    sorted_closeness_seq = closeness_seq.sort_values()
    closest_opponent_1_id = sorted_closeness_seq[0:1].index.item()
    closest_opponent_2_id = sorted_closeness_seq[1:2].index.item()
    closest_opponent_1 = play.loc[closest_opponent_1_id]
    closest_opponent_2 = play.loc[closest_opponent_2_id]
        
    dy_1 = np.abs(closest_opponent_1["Y_std"].item() - ball_carrier["Y_std"].item())
    dy_2 = np.abs(closest_opponent_2["Y_std"].item() - ball_carrier["Y_std"].item())
    ave_dy = (dy_1 + dy_2)/2
    dx_1 = np.abs(closest_opponent_1["X_std"].item() - ball_carrier["X_std"].item())
    dx_2 = np.abs(closest_opponent_2["X_std"].item() - ball_carrier["X_std"].item())
    ave_dx = (dx_1 + dx_2)/2
    angle_1 = np.arctan(dy_1/dx_1)
    angle_2 = np.arctan(dy_2/dx_2) 
    delta_angle = np.abs(angle_1 - angle_2)
    dist_opp_1 = np.sqrt(np.power(dy_1 ,2) + np.power(dx_1 ,2)) 
    dist_opp_2 = np.sqrt(np.power(dy_2 ,2) + np.power(dx_2 ,2))
    dist_opp_ave = np.sqrt(np.power(ave_dy ,2) + np.power(ave_dx ,2))
    area = 0.5 * np.sin(delta_angle) * dist_opp_1 * dist_opp_2
    
    is_bc_dir_reverse = 1 if ball_carrier["Dir_std_2"].item() < 0 or ball_carrier["Dir_std_2"].item() > 180 else 0
    is_bc_or_reverse  = 1 if ball_carrier["Orien_std_2"].item() < 0 or ball_carrier["Orien_std_2"].item() > 180 else 0
    diff_bc_dir_or = ball_carrier["Dir_std_2"].item() - ball_carrier["Orien_std_2"].item()

    return area, dist_opp_1, dist_opp_2, dist_opp_ave, max(area, dist_opp_1, dist_opp_2, dist_opp_ave), min(area, dist_opp_1, dist_opp_2, dist_opp_ave),                 is_bc_dir_reverse, is_bc_or_reverse, diff_bc_dir_or

def compound_2_f(play):
    offensive = play.loc[play['IsOnOffense'] == 1]
    defensive = play.loc[play['IsOnOffense'] == 0]
    
    mean_X_offensive = offensive['X_std'].mean()
    mean_X_defensive = defensive['X_std'].mean()
    
    mean_Y_offensive = offensive['Y_std'].mean()
    mean_Y_defensive = defensive['Y_std'].mean()
    
    mean_S_x_offensive = offensive['S_x'].mean()
    mean_S_x_defensive = defensive['S_x'].mean()
        
    mean_S_y_offensive = offensive['S_y'].mean()
    mean_S_y_defensive = defensive['S_y'].mean()
    
    mean_A_x_offensive = offensive['A_x'].mean()
    mean_A_x_defensive = defensive['A_x'].mean()
    
    mean_A_y_offensive = offensive['A_y'].mean()
    mean_A_y_defensive = defensive['A_y'].mean()
    
    dx = np.abs(mean_X_offensive - mean_X_defensive)
    dy = np.abs(mean_Y_offensive - mean_Y_defensive)
    dist_dx_dy = np.sqrt(np.power(dx ,2) + np.power(dy ,2)) 
    
    return  mean_X_offensive, mean_X_defensive,                 mean_Y_offensive, mean_Y_defensive,                 mean_S_x_offensive, mean_S_x_defensive,             mean_S_y_offensive, mean_S_y_defensive,             mean_A_x_offensive, mean_A_x_defensive,             mean_A_y_offensive, mean_A_y_defensive,             dx, dy, dist_dx_dy

def handle_compound(df, column_1, column_2):
    temp = df.groupby(['PlayId']).apply(lambda x: compound_1_f(x)).reset_index()
    temp.columns= ["PlayId", column_1]
    df = df.merge(temp, how="left", on='PlayId')
    del temp
    gc.collect()
    df['trig_area'] = df[column_1].apply(lambda x: x[0])
    df['dist_opp_1'] = df[column_1].apply(lambda x: x[1])
    df['dist_opp_2'] = df[column_1].apply(lambda x: x[2])
    df['dist_opp_ave'] = df[column_1].apply(lambda x: x[3])
    df['max_compound_1'] = df[column_1].apply(lambda x: x[4])
    df['min_compound_1'] = df[column_1].apply(lambda x: x[5])

    df['is_bc_dir_reverse'] = df[column_1].apply(lambda x: x[6])
    df['is_bc_or_reverse'] = df[column_1].apply(lambda x: x[7])
    df['diff_bc_dir_or'] = df[column_1].apply(lambda x: x[8])
    
    df.drop(column_1, axis=1, inplace=True)

    temp = df.groupby(['PlayId']).apply(lambda x: compound_2_f(x)).reset_index()
    temp.columns= ["PlayId", column_2]
    df = df.merge(temp, how="left", on='PlayId')
    del temp
    gc.collect()

    df['mean_X_offensive'] = df[column_2].apply(lambda x: x[0])
    df['mean_X_defensive'] = df[column_2].apply(lambda x: x[1])

    df['mean_Y_offensive'] = df[column_2].apply(lambda x: x[2])
    df['mean_Y_defensive'] = df[column_2].apply(lambda x: x[3])

    df['mean_S_x_offensive'] = df[column_2].apply(lambda x: x[4])
    df['mean_S_x_defensive'] = df[column_2].apply(lambda x: x[5])

    df['mean_S_y_offensive'] = df[column_2].apply(lambda x: x[6])
    df['mean_S_y_defensive'] = df[column_2].apply(lambda x: x[7])

    df['mean_A_x_offensive'] = df[column_2].apply(lambda x: x[8])
    df['mean_A_x_defensive'] = df[column_2].apply(lambda x: x[9])

    df['mean_A_y_offensive'] = df[column_2].apply(lambda x: x[10])
    df['mean_A_y_defensive'] = df[column_2].apply(lambda x: x[11])

    df['field_dx'] = df[column_2].apply(lambda x: x[12])
    df['field_dy'] = df[column_2].apply(lambda x: x[13])
    df['field_dist'] = df[column_2].apply(lambda x: x[14])
    df.drop(column_2, axis=1, inplace=True)
    return df

def impute(df, fill_map, mode_="train"):
    if mode_ == "train":
        for col in df.select_dtypes(include="number").columns:
            if col not in ['GameId', 'PlayId', 'NflId', 'NflIdRusher']:
                fill_value = df[col].median()
                fill_map.update({col: fill_value})
                df[col].fillna(fill_value, inplace=True)

        for col in df.select_dtypes(include="object").columns:
            fill_value = df[col].mode()[0]
            fill_map.update({col: fill_value})
            df[col].fillna(fill_value, inplace=True)
    else:
        df.fillna(fill_map, inplace=True)


# # Preprocessing

# In[ ]:


def major_preprocess(df, mode="train"):
    global poor_fill_rate, fill_map, op_dp_columns, play_cat_classes, player_cat_classes, le_play_cat, le_player_cat, ss, ss_compound

    df.drop(poor_fill_rate, inplace=True, axis=1)    
    impute(df, fill_map, mode_=mode)
    
    df, op_dp_columns = encode_OP_DP(df, op_dp_columns, 'OffensePersonnel', "OP_", mode_=mode)
    df, op_dp_columns = encode_OP_DP(df, op_dp_columns, 'DefensePersonnel', "DP_", mode_=mode)
    df.drop(['DefensePersonnel', 'OffensePersonnel'], axis=1, inplace=True)
    
    df['SnapHandoffDiff'] = ((pd.to_datetime(df['TimeHandoff'], infer_datetime_format=True, utc=True) -                      pd.to_datetime(df['TimeSnap'], infer_datetime_format=True, utc=True))).dt.seconds.astype(np.int8)

    df['WindSpeed'] = df['WindSpeed'].astype(str).apply(handle_windspeed)

    df['Turf'] = df['Turf'].map(Turf)

    df['DiffHomeVisitor'] = df['HomeScoreBeforePlay'] - df['VisitorScoreBeforePlay']

    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: handle_height(x)).astype(np.int8)
    
    df['GameClock'] = 15 - df['GameClock'].apply(lambda x: handle_game_clock(x)).astype(np.int8)

    df['HowTired'] = df['GameClock'] + (df['Quarter'] - 1) * 15

    df['Age'] = ((pd.to_datetime(df['TimeHandoff'], infer_datetime_format=True, utc=True) -                      pd.to_datetime(df['PlayerBirthDate'], infer_datetime_format=True, utc=True)).dt.days/360).astype(np.int8)

    df['ToLeft'] = df['PlayDirection'].apply(lambda x: tag_to_left(x)) # not a feature
    
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].apply(lambda x: fix_team_abbr(x))
    
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].apply(lambda x: fix_team_abbr(x))
    
    df['IsBallCarrier'] = (df['NflId'] == df['NflIdRusher'])

    df['TeamOnOffense'] = df[['PossessionTeam', 'HomeTeamAbbr']].apply(func=tag_team_on_offence, axis=1, result_type='reduce') # not a feature
    
    df['IsOnOffense'] = (df['Team'] == df['TeamOnOffense'])

    df['YardsFromOwnGoal'] = df.apply(func=tag_yards_from_own_goal, axis=1, result_type='reduce')

    df['X_std'] = df.apply(func=tag_X_std, axis=1, result_type='reduce')
    
    df['Y_std'] = df.apply(func=tag_Y_std, axis=1, result_type='reduce')
    
    df['inv_X'] = 1/df['X_std']
    
    df['inv_X'] = df['inv_X'].replace([np.inf, -np.inf], [0, 0])
    
    df['inv_Y'] = 1/df['Y_std']
    
    df['inv_Y'] = df['inv_Y'].replace([np.inf, -np.inf], [0, 0])
    
    df['euc_dist'] = np.sqrt(np.power(df["X_std"], 2) + np.power(df["Y_std"], 2))

    df["Dir_std_1"] = df.apply(func=tag_dir_std_1, axis=1, result_type='reduce')
    
    df["Dir_std_1"] = df.apply(func=tag_dir_to_right, axis=1, result_type='reduce')
    
    df["Dir_std_2"] = df.apply(func=tag_dir_std_2, axis=1, result_type='reduce')

    df["Orien_std_1"] = df.apply(func=tag_orien_std_1, axis=1, result_type='reduce')
    
    df["Orien_std_1"] = df.apply(func=tag_orien_to_right, axis=1, result_type='reduce')
    
    df["Orien_std_2"] = df.apply(func=tag_orien_std_2, axis=1, result_type='reduce')

    df['is_dir_reverse'] = df['Dir_std_2'].apply(lambda x: 1 if x < 0 or x > 180 else 0)
    
    df['is_or_reverse'] = df['Orien_std_2'].apply(lambda x: 1 if x < 0 or x > 180 else 0)
    
    df['diff_dir_or'] = df['Dir_std_2'] - df['Orien_std_2']

    df['IsBallCarrier'] = df['IsBallCarrier'].apply(lambda x: 1 if x is True else 0)
    
    df['IsOnOffense'] = df['IsOnOffense'].apply(lambda x: 1 if x is True else 0)

    dir_sin = df["Dir_std_2"].apply(lambda x: np.sin(x * np.pi/180.0))
    
    dir_cos = df["Dir_std_2"].apply(lambda x: np.cos(x * np.pi/180.0))

    df['S_y'] = dir_sin * df['S']
    
    df['S_x'] = dir_cos * df['S']

    df['A_y'] = dir_sin * df['A']
    
    df['A_x'] = dir_cos * df['A']
    
    df.fillna(-999, inplace=True)
    
    df.sort_values(['GameId', 'PlayId', 'IsOnOffense', 'IsBallCarrier', 'X_std'], ascending=[True, True, True, False, True], inplace=True)
    
    df = handle_compound(df, 'Compound_1', 'Compound_2')
    
    df[total_cat] = df[total_cat].astype(str)

    for col in total_cat:
        df[col] = col + "_" + df[col]

    if mode == "train":
        for col in play_cat:
            play_cat_classes.append(df[col].unique().tolist())

        for col in player_cat:
            player_cat_classes.append(df[col].unique().tolist())

        play_cat_classes = [item for l in play_cat_classes for item in l] + ['unknown_play_cat']
        player_cat_classes = [item for l in player_cat_classes for item in l] + ['unknown_player_cat']

        le_play_cat.fit(play_cat_classes)
        le_player_cat.fit(player_cat_classes)

        for col in play_cat:
            df[col] = le_play_cat.transform(df[col])

        for col in player_cat:
            df[col] = le_player_cat.transform(df[col])
    else:
        for col in play_cat:
            df[col] = df[col].apply(lambda x: le_play_cat.transform([x])[0] if x in le_play_cat.classes_ else le_play_cat.transform(["unknown_play_cat"])[0])

        for col in player_cat:
            df[col] = df[col].apply(lambda x: le_player_cat.transform([x])[0] if x in le_player_cat.classes_ else le_player_cat.transform(["unknown_player_cat"])[0])
    
    if mode == "train":
        df[total_numeric] = ss.fit_transform(df[total_numeric])
        df[compound] = ss_compound.fit_transform(df[compound])
    else:
        df[total_numeric] = ss.transform(df[total_numeric])
        df[compound] = ss_compound.transform(df[compound])
    
    return df


# # Input Data Parcels & Target

# In[ ]:


def data_parcels(df, mode="train"):
    df_play_cat = df[play_cat]
    df_player_cat = df[player_cat]
    df_play_numeric = df[play_numeric]
    df_player_numeric = df[player_numeric]
    df_compound = df[compound]
    
    df_play_cat = df_play_cat.iloc[[i for i in range(0, len(df_play_cat), 22)]].reset_index(drop=True).values
    df_play_numeric = df_play_numeric.iloc[[i for i in range(0, len(df_play_numeric), 22)]].reset_index(drop=True).values
    df_player_cat = np.stack([df_player_cat.iloc[[i for i in range(j, len(df_player_cat), 22)]].reset_index(drop=True).values for j in range(22)]).transpose(1, 0, 2)
    df_player_numeric = np.stack([df_player_numeric.iloc[[i for i in range(j, len(df_player_numeric), 22)]].reset_index(drop=True).values for j in range(22)]).transpose(1, 0, 2)
    df_compound = df_compound.iloc[[i for i in range(0, len(df_compound), 22)]].reset_index(drop=True).values
    
    if mode == "train":
        df_y = df["Yards"]
        df_y = df_y.iloc[[i for i in range(0, len(df_y), 22)]].reset_index(drop=True)
        df_y = np.vstack(df_y.apply(lambda x: generate_y(x)).values)
    else:
        df_y = np.zeros(1)
        
    if mode == "train":
        print([df_play_cat.shape, df_play_numeric.shape, df_player_cat.shape, df_player_numeric.shape, df_compound.shape, df_y.shape])

    return df_play_cat, df_play_numeric, df_player_cat, df_player_numeric, df_compound, df_y


# # Main

# In[ ]:


poor_fill_rate = ['WindDirection', 'Temperature', 'GameWeather', 'StadiumType']
fill_map = dict()
op_dp_columns = list()
play_cat_classes = list()
player_cat_classes = list()
le_play_cat = LabelEncoder()
le_player_cat = LabelEncoder()
ss = StandardScaler()
ss_compound = StandardScaler()


# In[ ]:


Turf = {'Field Turf':0, 'A-Turf Titan':0, 'Grass':1, 'UBU Sports Speed S5-M':0, 
        'Artificial':0, 'DD GrassMaster':0, 'Natural Grass':1, 'UBU Speed Series-S5-M':0, 
        'FieldTurf':0, 'FieldTurf 360':0, 'Natural grass':1, 'grass':1, 'Natural':1, 
        'Artifical':0, 'FieldTurf360':0, 'Naturall Grass':1, 'Field turf':0, 'SISGrass':0, 
        'Twenty-Four/Seven Turf':0, 'natural grass':1}


# In[ ]:


play_cat = ['Team', 'Season', 'PossessionTeam', 'FieldPosition', 'OffenseFormation', 'PlayDirection', 'HomeTeamAbbr', 
        'VisitorTeamAbbr', 'Week', 'Stadium', 'StadiumType', 'Location',  'GameWeather', 'WindDirection']

play_cat = feature_set(play_cat, poor_fill_rate, "sub")

play_numeric = ['GameClock', 'YardLine', 'Quarter', 'Down', 'Distance', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 
                'DefendersInTheBox', 'Temperature', 'Humidity', 'WindSpeed', 'Turf', 'SnapHandoffDiff', 'DiffHomeVisitor', 
                'IsOnOffense', 'YardsFromOwnGoal'] + op_dp_columns

play_numeric = feature_set(play_numeric, poor_fill_rate, "sub")

player_cat = ['PlayerCollegeName', 'Position']

player_numeric = ['S', 'A', 'Dis', 'PlayerHeight', 'PlayerWeight', 'S_y', 'S_x', 'A_y', 'A_x', 'Age', 'HowTired', 
                  'IsBallCarrier', 'X_std', 'Y_std', "Dir_std_2", "Orien_std_2", "is_dir_reverse", "is_or_reverse", 
                  "diff_dir_or", 'euc_dist', 'inv_X', 'inv_Y']

ids = ['GameId', 'PlayId', 'NflId', 'NflIdRusher', 'DisplayName', 'JerseyNumber']

time = ['TimeHandoff', 'TimeSnap', 'PlayerBirthDate']

compound_1 = ['trig_area', 'dist_opp_1', 'dist_opp_2', 'dist_opp_ave', 'max_compound_1', 'min_compound_1', 
              'is_bc_dir_reverse', 'is_bc_or_reverse', 'diff_bc_dir_or']

compound_2 = ['mean_X_offensive', 'mean_X_defensive', 'mean_Y_offensive', 'mean_Y_defensive', 'mean_S_x_offensive', 
              'mean_S_x_defensive', 'mean_S_y_offensive', 'mean_S_y_defensive', 'mean_A_x_offensive', 'mean_A_x_defensive', 
              'mean_A_y_offensive', 'mean_A_y_defensive', 'field_dx', 'field_dy', 'field_dist']

total_cat = play_cat + player_cat

total_numeric = play_numeric + player_numeric

compound = compound_1 + compound_2


# In[ ]:


os.system("echo 'Preprocessing Start'")
train = major_preprocess(train, mode="train")
train_play_cat, train_play_numeric, train_player_cat, train_player_numeric, train_compound, train_y = data_parcels(train, mode="train")


# # Modeling

# In[ ]:


def crps(y_true, y_pred):
    loss = K.mean((K.clip(K.cumsum(y_pred, axis=1), 0, 1) - y_true)**2)
    return loss

class BaseLogger(Callback):
    def __init__(self):
        super(BaseLogger, self).__init__()
    
    def on_train_begin(self, logs={}):
        pass
    
    def on_epoch_begin(self, epoch, logs={}):
        pass
    
    def on_epoch_end(self, epoch, logs={}):
        improve_in_train_crps = np.nan
        improve_in_val_crps = np.nan
        
        try:
            improve_in_train_crps = self.model.history.history['crps'][-2] - self.model.history.history['crps'][-1]
            improve_in_val_crps = self.model.history.history['val_crps'][-2] - self.model.history.history['val_crps'][-1]
        except Exception:
            improve_in_train_crps = np.nan
            improve_in_val_crps = np.nan
        
        print(f"Model: -- train_crps: {logs['crps']:.6f}, -- val_crps: {logs['val_crps']:.6f}")
        print(f"Improve: -- train_crps: {improve_in_train_crps:.6f}, -- val_crps: {improve_in_val_crps:.6f}")
    
    def on_train_end(self, logs={}):
        pass

def get_top_models(models, num=2):
    val_scores = list()
    for model in models:
        val_score = np.mean(model.history.history['val_crps'][-3:])
        val_scores.append(val_score)
    
    sorted_models = sorted(list(zip(val_scores, models)), key=lambda x: x[0], reverse=False)[0:num]
    model_chosen = [m[1] for m in sorted_models]
    return model_chosen

def print_scores(models):
    train_crps = 0
    val_crps = 0

    for m in models:
        train_crps += np.mean(m.history.history['crps'][-3:])
        val_crps += np.mean(m.history.history['val_crps'][-3:])

    mean_train_crps = train_crps/len(models)
    mean_val_crps = val_crps/len(models)

    print("train crps: {0}".format(mean_train_crps))
    print("val crps: {0}".format(mean_val_crps))
    
    try:
        os.system("echo 'train: {0}'".format(mean_train_crps))
        os.system("echo 'val: {0}'".format(mean_val_crps))
    except Exception:
        pass


# In[ ]:


os.system("echo 'Modeling Start'")

KFolds = 5
kf = KFold(n_splits=KFolds, random_state=RANDOM_STATE)

models = list()

for train_idx, val_idx in kf.split(train_y):
    X_train_play_cat, X_val_play_cat = train_play_cat[train_idx], train_play_cat[val_idx]
    X_train_player_cat, X_val_player_cat = train_player_cat[train_idx], train_player_cat[val_idx]
    X_train_play_numeric, X_val_play_numeric = train_play_numeric[train_idx], train_play_numeric[val_idx]

    X_train_compound, X_val_compound = train_compound[train_idx], train_compound[val_idx]
    X_train_player_numeric, X_val_player_numeric = train_player_numeric[train_idx], train_player_numeric[val_idx]
    y_train, y_val = train_y[train_idx], train_y[val_idx]
    
    input_play_cat = Input(shape=(X_train_play_cat.shape[1],), name= "input_play_cat")
    input_play_numeric = Input(shape=(X_train_play_numeric.shape[1],), name= "input_play_numeric")
    input_compound = Input(shape=(X_train_compound.shape[1],), name= "input_compound")

    input_player_cat = Input(shape=(X_train_player_cat.shape[1], X_train_player_cat.shape[2]), name= "input_player_cat")
    input_player_numeric = Input(shape=(X_train_player_numeric.shape[1], X_train_player_numeric.shape[2]), name= "input_player_numeric")
    # ^ injection point for GlobalAveragePooling input_player_numeric (2D)
    rsh_player_numeric = Reshape((input_player_numeric.shape[1] * input_player_numeric.shape[2],), name= "reshape_player_numeric")(input_player_numeric)

    embedding_play_cat = Embedding(len(le_play_cat.classes_) + 1, 8, embeddings_regularizer=regularizers.l2(1e-4), name="embedding_play_cat")
    embedding_player_cat = Embedding(len(le_player_cat.classes_) + 1, 12, embeddings_regularizer=regularizers.l2(1e-4), name="embedding_player_cat")
    
    emb_play_cat = embedding_play_cat(input_play_cat)
    
    emb_player_cat = embedding_player_cat(input_player_cat)
    emb_player_cat = Reshape((int(emb_player_cat.shape[1]), int(emb_player_cat.shape[2]) * int(emb_player_cat.shape[3])), name= "reshape_emb_player_cat")(emb_player_cat)
    # ^ injection point for GlobalAveragePooling emb_player_cat (2D)

    flat_emb_play_cat = Flatten(name= "flat_emb_play_cat")(emb_play_cat)
    flat_emb_player_cat = Flatten(name= "flat_rsh_emb_player_cat")(emb_player_cat)
    
    # GlobalAveragePooling
    concat_pooling_feat = Concatenate(name = "player_features_GA")([input_player_numeric, emb_player_cat])
    global_averages = list()
    pooling_units = int(64)
    for d in range(3):
        concat_pooling_feat = Dense(pooling_units, activation=None, kernel_initializer=glorot_normal(seed=RANDOM_STATE), name=f"dense_GA_{d}")(concat_pooling_feat)
        global_averages.append(GlobalAveragePooling1D(name=f"GA_{d}")(concat_pooling_feat))
        concat_pooling_feat = BatchNormalization(name=f"batchN_GA_{d}")(concat_pooling_feat)
        concat_pooling_feat = Activation('relu')(concat_pooling_feat)
        pooling_units =int(pooling_units/2)
        
    encoded_features = Concatenate(name = "encoded_GA_features")(global_averages)
    encoded_features = BatchNormalization(name="batchN_GA_concat")(encoded_features)
    encoded_features = Activation('relu')(encoded_features)
    encoded_features = Dropout(0.2)(encoded_features)
    
    residual_input = Concatenate(name = "all_features")([flat_emb_play_cat, flat_emb_player_cat, input_play_numeric, input_compound, rsh_player_numeric, encoded_features])
    residual_input_skip = residual_input
    
    # Residual Network
    for r in range(3):
        # Full-Block
        hidden_layer_fb = Dense(256, activation=None, kernel_initializer=glorot_normal(seed=RANDOM_STATE), name=f"resnet_fb_{r}")(residual_input)
        hidden_layer_fb = BatchNormalization(name=f"batchN_resnet_{r}")(hidden_layer_fb)
        hidden_layer_fb = Activation('relu')(hidden_layer_fb)
        hidden_layer_fb = Dropout(0.2)(hidden_layer_fb)

        # Block_1_of_2
        hidden_layer_b_1_2 = Dense(128, activation=None, kernel_initializer=glorot_normal(seed=RANDOM_STATE), name=f"resnet_b_{r}_1_2")(hidden_layer_fb)
        hidden_layer_b_1_2 = BatchNormalization()(hidden_layer_b_1_2)

        # Skip Connection
        hidden_layer_skip = Dense(128, activation=None, kernel_initializer=glorot_normal(seed=RANDOM_STATE), name=f"skip_connection_{r}")(residual_input_skip)
        hidden_layer_skip = BatchNormalization(name=f"batchN_skip_{r}")(hidden_layer_skip)

        # Block_2_of_2
        hidden_layer_b_2_2 = Add()([hidden_layer_b_1_2, hidden_layer_skip])
        hidden_layer_b_2_2 = Activation('relu')(hidden_layer_b_2_2)
        hidden_layer_b_2_2 = Dropout(0.1)(hidden_layer_b_2_2)

        residual_input = hidden_layer_b_2_2
        residual_input_skip = hidden_layer_b_2_2

    # output layer
    out_ = Dense(199, activation="softmax", name = "softmax")(hidden_layer_b_2_2)

    model = Model(inputs=[input_play_cat, input_play_numeric, input_compound, input_player_cat, input_player_numeric], outputs=[out_])
    model.compile(loss=[crps], optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=[crps])

    earlyStop = keras.callbacks.callbacks.EarlyStopping(monitor='val_crps', patience=20, verbose=1, mode='min', restore_best_weights=True)
    reduceLR = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_crps', factor=0.7, patience=10, verbose=1, min_lr=0.00001, mode='min')
    modelCheck = ModelCheckpoint('best_model.h5', monitor='val_crps', mode='min', save_best_only=True, verbose=1, save_weights_only=True)
    
    x_train = [X_train_play_cat, X_train_play_numeric, X_train_compound, X_train_player_cat, X_train_player_numeric]
    y_train = y_train
    
    x_val = [X_val_play_cat, X_val_play_numeric, X_val_compound, X_val_player_cat, X_val_player_numeric]
    y_val = y_val
    
    verbose = 1
    if commit == 1:
        verbose = 0
        
    model.fit(
        x_train,
        y_train,
        batch_size=len(y_train),
        epochs=400,
        verbose=verbose,
        validation_data=(x_val, y_val),
        callbacks=[BaseLogger(), earlyStop, reduceLR, modelCheck],
        use_multiprocessing=True
    )
    
    model.load_weights("best_model.h5")
    
    models.append(model)


# # Scores, Model Summary and Visuals

# In[ ]:


models = get_top_models(models, num=2)
print_scores(models)


# In[ ]:


if commit == 0:
    print(models[0].summary())


# In[ ]:


plot_model(models[0], to_file='model.png')


# # Prediction & Submission

# In[ ]:


os.system("echo 'Prediction Start'")
env = nflrush.make_env()

for (test_df, sample_prediction_df) in env.iter_test():
    test_df = major_preprocess(test_df, mode="test")
    test_df_play_cat, test_df_play_numeric, test_df_player_cat, test_df_player_numeric, test_df_compound, dummy = data_parcels(test_df, mode="test")

    y_pred = 0
    for m in models:
        pred = m.predict([test_df_play_cat, test_df_play_numeric, test_df_compound, test_df_player_cat, test_df_player_numeric])
        pred =  np.cumsum(pred)
        y_pred += pred
    y_pred /= len(models)
    y_pred = np.clip(y_pred, 0, 1)
    
    env.predict(pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns))
    
env.write_submission_file()


# In[ ]:


finish = tm.time()
print(f"Build took: {(finish - start)/60.0} minutes ... good bye!")

