#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# Mostly copied from bestpredict's kernel [Location EDA 8eb410][1]. I have applied the random forest (RF) regressor proposed in Dimension's kernel [[NFL] - [001]- [Feature selection]][2]. Let's see whether the reduced feature dimension will result in LB score improvement (V1-V3) and whether blending RF and neural network (NN) will improve the LB score (V4+). Please correct me if you have found any mistake. Thank you very much.
# 
# ## Changelog
# * **V10+**: Try blending with LGBM classifier from [[NFL] - [001]- [LightGBM]][5].
# * **V9**: Drop highly correlated features based on correlation heatmap. (37->30)
# * **V8**: Plot feature importance from random forest.
# * **V7**: Add Rusher features applied in [Location EDA with rusher features][4].
# * **V6**: Try Bayesian Hyperparameter Tuning in my another kernel [Bayesian Hyperparameter Tuning (NN&RF)][3]. Apply results here.
# * **V5**: Try larger NN and elu activation.
# * **V4**: Since there is no improvement on LB score by discarding the original features, I try blending random forest and NN.
# * **V3**: Try threshold 0.2* mean
# * **V2**: Try threshold 0.5* mean
# * **V1**: Add feature selection based on the feature importance from a prefitted random forest regressor
# 
# [1]: https://www.kaggle.com/bestpredict/location-eda-8eb410
# [2]: https://www.kaggle.com/coolcoder22/nfl-001-feature-selection
# [3]: https://www.kaggle.com/gogo827jz/bayesian-hyperparameter-tuning-nn-rf?scriptVersionId=23677874
# [4]: https://www.kaggle.com/dandrocec/location-eda-with-rusher-features
# [5]: https://www.kaggle.com/enzoamp/nfl-lightgbm

# In[ ]:


import os
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import time
import lightgbm as lgb

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization,LeakyReLU,PReLU,ELU,ThresholdedReLU,Concatenate
from keras.models import Model
import keras.backend as K
from  keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
import re
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import codecs
from keras.utils import to_categorical
from sklearn.metrics import f1_score

import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

TRAIN_OFFLINE = False

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
if TRAIN_OFFLINE:
    train = pd.read_csv('../input/train.csv', dtype={'WindSpeed': 'object'})
else:
    train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})


# In[ ]:


outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()


# In[ ]:


train.head()


# In[ ]:


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

def OffensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0, 'QB' : 0, 'RB' : 0, 'TE' : 0, 'WR' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def DefensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x/15))
    except:
        return "nan"
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


    ## Orientation and Dir
    train["Orientation_ob"] = train["Orientation"].apply(lambda x : orientation_to_cat(x)).astype("object")
    train["Dir_ob"] = train["Dir"].apply(lambda x : orientation_to_cat(x)).astype("object")

    train["Orientation_sin"] = train["Orientation"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    train["Orientation_cos"] = train["Orientation"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
    train["Dir_sin"] = train["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    train["Dir_cos"] = train["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))

    ## diff Score
    train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]
    train["diffScoreBeforePlay_binary_ob"] = (train["HomeScoreBeforePlay"] > train["VisitorScoreBeforePlay"]).astype("object")

    ## Turf
    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 
    train['Turf'] = train['Turf'].map(Turf)

    ## OffensePersonnel
    temp = train["OffensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(OffensePersonnelSplit(x)))
    temp.columns = ["Offense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on = "PlayId")

    ## DefensePersonnel
    temp = train["DefensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(DefensePersonnelSplit(x)))
    temp.columns = ["Defense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on = "PlayId")

    ## sort
#     train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index(drop = True)
    train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop = True)
    return train


# ## Functions for anchoring offense moving left from {0,0}

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
    
    def rusher_features(df):
        
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Dir', 'S', 'A', 'X', 'Y']]
        rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS', 'RusherA', 'RusherX', 'RusherY']
        
       
        radian_angle = (90 - rusher['RusherDir']) * np.pi / 180.0
        v_horizontal = np.abs(rusher['RusherS'] * np.cos(radian_angle))
        v_vertical = np.abs(rusher['RusherS'] * np.sin(radian_angle)) 
        
       
        rusher['v_horizontal'] = v_horizontal
        rusher['v_vertical'] = v_vertical
        
        
        rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS','RusherA','RusherX', 'RusherY','v_horizontal', 'v_vertical']
        
        
        return rusher
    
    def static_features(df):
        
        
        add_new_feas = []

        ## Height
        df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
        
        add_new_feas.append('PlayerHeight_dense')

        ## Time
        df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

        df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        df['PlayerBirthDate'] =df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

        ## Age
        seconds_in_year = 60*60*24*365.25
        df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
        add_new_feas.append('PlayerAge')

        ## WindSpeed
        df['WindSpeed_ob'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
        df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
        df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
        df['WindSpeed_dense'] = df['WindSpeed_ob'].apply(strtofloat)
        add_new_feas.append('WindSpeed_dense')

        ## Weather
        df['GameWeather_process'] = df['GameWeather'].str.lower()
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
        df['GameWeather_dense'] = df['GameWeather_process'].apply(map_weather)
        add_new_feas.append('GameWeather_dense')
#         ## Rusher
#         train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])
#         train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")
#         temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})
#         train = train.merge(temp, on = "PlayId")
#         train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

        ## dense -> categorical
#         train["Quarter_ob"] = train["Quarter"].astype("object")
#         train["Down_ob"] = train["Down"].astype("object")
#         train["JerseyNumber_ob"] = train["JerseyNumber"].astype("object")
#         train["YardLine_ob"] = train["YardLine"].astype("object")
        # train["DefendersInTheBox_ob"] = train["DefendersInTheBox"].astype("object")
        # train["Week_ob"] = train["Week"].astype("object")
        # train["TimeDelta_ob"] = train["TimeDelta"].astype("object")

        ## Orientation and Dir
        df["Orientation_ob"] = df["Orientation"].apply(lambda x : orientation_to_cat(x)).astype("object")
        df["Dir_ob"] = df["Dir"].apply(lambda x : orientation_to_cat(x)).astype("object")

        df["Orientation_sin"] = df["Orientation"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["Orientation_cos"] = df["Orientation"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        df["Dir_sin"] = df["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["Dir_cos"] = df["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        add_new_feas.append("Dir_sin")
        add_new_feas.append("Dir_cos")

        ## diff Score
        df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
        add_new_feas.append("diffScoreBeforePlay")
        
        
        static_features = df[df['NflId'] == df['NflIdRusher']][add_new_feas+['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
                                                            'YardLine','Quarter','Down','Distance','DefendersInTheBox']].drop_duplicates()
        static_features.fillna(-999,inplace=True)

        return static_features


    def combine_features(relative_to_back, defense, rushing, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,rushing,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df
    
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    rush_feats = rusher_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats, rush_feats, static_feats, deploy=deploy)
    
    return basetable


# In[ ]:


get_ipython().run_line_magic('time', 'train_basetable = create_features(train, False)')


# In[ ]:


X = train_basetable.copy()
yards = X.Yards

y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1

X.drop(['GameId','PlayId','Yards'], axis=1, inplace=True)


# ## Plot Correlation Heatmap

# In[ ]:


sns.set(rc={'figure.figsize':(30, 30)})
corr = X.corr()
plt.figure() 
ax = sns.heatmap(corr, linewidths=.5, annot=True, cmap="YlGnBu", fmt='.1g')
plt.savefig('corr_heatmap.png')
plt.show()


# In[ ]:


# Drop highly correlated features (37->30)
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.99:
            if columns[j]:
                columns[j] = False

feature_columns = X.columns[columns].values
drop_columns = X.columns[columns == False].values
print(feature_columns)
print('-'*73)
print(drop_columns)


# In[ ]:


# Categorical features for LGBM
cat_feats = X.columns[(X.nunique() <= 50)].values.tolist()
cat_feats = list(set(cat_feats) & set(feature_columns))
print(cat_feats)


# ## Scale Training Data

# In[ ]:


scaler = StandardScaler()
scaled_features = scaler.fit_transform(X[feature_columns].values)
X = pd.DataFrame(scaled_features, columns=feature_columns)
print(X.shape)


# # Let's build NN and Random Forest Regressor

# Below class Metric based entirely on: https://www.kaggle.com/kingychiu/keras-nn-starter-crps-early-stopping
# <br></br>
# Below early stopping entirely based on: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112868#latest-656533

# In[ ]:


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
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s


# In[ ]:


# Calculate CRPS score
def crps_score(y_prediction, y_valid, shape=X.shape[0]):
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_prediction, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * shape)
    crps = np.round(val_s, 6)
    
    return crps


# ## NN

# In[ ]:


def get_nn(x_tr, y_tr, x_val, y_val, shape):
    K.clear_session()
    inp = Input(shape = (x_tr.shape[1],))
    x = Dense(1024, input_dim=X.shape[1], activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    
    es = EarlyStopping(monitor='CRPS_score_val', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=10)

    mc = ModelCheckpoint('best_model.h5',monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=1, save_weights_only=True)
    
    bsz = 1024
    steps = x_tr.shape[0]/bsz

    model.fit(x_tr, y_tr,callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], epochs=100, batch_size=bsz, verbose=1)
    model.load_weights("best_model.h5")
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    crps = crps_score(y_pred, y_valid, shape=shape)

    return model,crps


# ## LGBM

# In[ ]:


metric = "multi_logloss"
param = {'num_leaves': 50,
         'min_data_in_leaf': 30,
         'objective':'multiclass',
         'num_class': 199,
         'max_depth': 6, # -1
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.4, #0.7
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": metric,
         "lambda_l1": 0.1,
         "verbosity": -1,
         'n_jobs': -1,
         "seed":1234}


# In[ ]:


def get_lgbm(x_tr, y_tr, x_val, y_val, shape):
    y_valid = y_val
    y_tr = np.argmax(y_tr, axis=1)
    y_val = np.argmax(y_val, axis=1)
    trn_data = lgb.Dataset(x_tr, label=y_tr, categorical_feature=cat_feats)
    val_data = lgb.Dataset(x_val, label=y_val, categorical_feature=cat_feats)
    model = lgb.train(param, trn_data, 10000, valid_sets = [val_data], verbose_eval = 100, early_stopping_rounds = 200)
    
    y_pred = model.predict(x_val, num_iteration=model.best_iteration)
    crps = crps_score(y_pred, y_valid, shape=shape)
    
    return model, crps


# ## RF

# In[ ]:


def get_rf(x_tr, y_tr, x_val, y_val, shape):
    model = RandomForestRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=15, 
                                  min_samples_split=8, n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(x_tr, y_tr)
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    crps = crps_score(y_pred, y_valid, shape=shape)
    
    return model, crps


# ## Training

# In[ ]:


loop = 2
fold = 5

oof_nn = np.zeros([loop, y.shape[0], y.shape[1]])
oof_lgbm = np.zeros([loop, y.shape[0], y.shape[1]])
oof_rf = np.zeros([loop, y.shape[0], y.shape[1]])

models_nn = []
crps_csv_nn = []

models_lgbm = []
crps_csv_lgbm = []

models_rf = []
crps_csv_rf = []

feature_importance = np.zeros([loop, fold, X.shape[1]])

s_time = time.time()

for k in range(loop):
    kfold = KFold(fold, random_state = 42 + k, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
        print("-----------")
        print(f'Loop {k+1}/{loop}' + f' Fold {k_fold+1}/{fold}')
        print("-----------")
        tr_x, tr_y = X.loc[tr_inds], y[tr_inds]
        val_x, val_y = X.loc[val_inds], y[val_inds]
        
        # Train NN
        nn, crps_nn = get_nn(tr_x, tr_y, val_x, val_y, shape=val_x.shape[0])
        models_nn.append(nn)
        print("the %d fold crps (NN) is %f"%((k_fold+1), crps_nn))
        crps_csv_nn.append(crps_nn)
        
        # Train LGBM
        lgbm, crps_lgbm = get_lgbm(tr_x, tr_y, val_x, val_y, shape=val_x.shape[0])
        models_lgbm.append(lgbm)
        print("the %d fold crps (LGBM) is %f"%((k_fold+1), crps_lgbm))
        crps_csv_lgbm.append(crps_lgbm)
        
#         # Train RF
#         rf, crps_rf = get_rf(tr_x, tr_y, val_x, val_y, shape=val_x.shape[0])
#         models_rf.append(rf)
#         print("the %d fold crps (RF) is %f"%((k_fold+1), crps_rf))
#         crps_csv_rf.append(crps_rf)
        
        # Feature Importance
        feature_importance[k, k_fold, :] = lgbm.feature_importance()
#         feature_importance[k, k_fold, :] = rf.feature_importances_
        
        #Predict OOF
        oof_nn[k, val_inds, :] = nn.predict(val_x)
        oof_lgbm[k, val_inds, :] = lgbm.predict(val_x, num_iteration=lgbm.best_iteration)
#         oof_rf[k, val_inds, :] = rf.predict(val_x)


# In[ ]:


a = lgbm.predict(val_x, num_iteration=lgbm.best_iteration)


# In[ ]:


crps_oof_nn = []
crps_oof_lgbm = []
crps_oof_rf = []

for k in range(loop):
    crps_oof_nn.append(crps_score(oof_nn[k,...], y))
    crps_oof_lgbm.append(crps_score(oof_lgbm[k,...], y))
#     crps_oof_rf.append(crps_score(oof_rf[k,...], y))


# In[ ]:


print("mean crps (NN) is %f"%np.mean(crps_csv_nn))
print("mean crps (LGBM) is %f"%np.mean(crps_csv_lgbm))
# print("mean crps (RF) is %f"%np.mean(crps_csv_rf))


# In[ ]:


print("mean OOF crps (NN) is %f"%np.mean(crps_oof_nn))
print("mean OOF crps (LGBM) is %f"%np.mean(crps_oof_lgbm))
# print("mean OOF crps (RF) is %f"%np.mean(crps_oof_rf))


# ## Plot Feature Importance

# In[ ]:


feature_importances = pd.DataFrame(np.mean(feature_importance, axis=0).T, columns=[[f'fold_{fold_n}' for fold_n in range(fold)]])
feature_importances['feature'] = feature_columns
feature_importances['average'] = feature_importances[[f'fold_{fold_n}' for fold_n in range(fold)]].mean(axis=1)
feature_importances.sort_values(by=('average',), ascending=False).head(10)


# In[ ]:


feature_importance_flatten = pd.DataFrame()
for i in range(len(feature_importances.columns)-2):
    col = ['feature', feature_importances.columns.values[i][0]]
    feature_importance_flatten = pd.concat([feature_importance_flatten, feature_importances[col].rename(columns={f'fold_{i}': 'importance'})], axis=0)

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importance_flatten.sort_values(by=('importance',), ascending=False), x=('importance',), y=('feature',))
plt.title(f'Feature Importances over {loop} loops and {fold} folds')
plt.savefig("feature_importance.png")
plt.show()


# ## Blending Weight Optimisation

# In[ ]:


def weight_opt(oof_nn, oof_rf, y_true):
    weight_nn = np.inf
    best_crps = np.inf
    
    for i in np.arange(0, 1.01, 0.05):
        crps_blend = np.zeros(oof_nn.shape[0])
        for k in range(oof_nn.shape[0]):
            crps_blend[k] = crps_score(i * oof_nn[k,...] + (1-i) * oof_rf[k,...], y_true)
        if np.mean(crps_blend) < best_crps:
            best_crps = np.mean(crps_blend)
            weight_nn = round(i, 2)
            
        print(str(round(i, 2)) + ' : mean crps (Blend) is ', round(np.mean(crps_blend), 6))
        
    print('-'*36)
    print('Best weight for NN: ', weight_nn)
    print('Best weight for LGBM: ', round(1-weight_nn, 2))
#     print('Best weight for RF: ', round(1-weight_nn, 2))
    print('Best mean crps (Blend): ', round(best_crps, 6))
    
    return weight_nn, round(1-weight_nn, 2)


# In[ ]:


weight_nn, weight_lgbm = weight_opt(oof_nn, oof_lgbm, y)
# weight_nn, weight_rf = weight_opt(oof_nn, oof_rf, y)


# # Time for the actual submission

# ## Define Prediction Function

# In[ ]:


def predict(x_te, models_nn, models_rf, weight_nn, weight_rf, iteration=False):
    model_num_nn = len(models_nn)
    model_num_rf = len(models_rf)
    for k,m in enumerate(models_nn):
        if k==0:
            y_pred_nn = m.predict(x_te, batch_size=1024)
            if iteration:
                y_pred_rf = models_rf[k].predict(x_te, num_iteration=models_rf[k].best_iteration)
            else:
                y_pred_rf = models_rf[k].predict(x_te)
        else:
            y_pred_nn += m.predict(x_te, batch_size=1024)
            if iteration:
                y_pred_rf += models_rf[k].predict(x_te, num_iteration=models_rf[k].best_iteration)
            else:
                y_pred_rf += models_rf[k].predict(x_te)
            
    y_pred_nn = y_pred_nn / model_num_nn
    y_pred_rf = y_pred_rf / model_num_rf
    
    return weight_nn * y_pred_nn + weight_rf * y_pred_rf


# In[ ]:


get_ipython().run_cell_magic('time', '', "if  TRAIN_OFFLINE==False:\n    from kaggle.competitions import nflrush\n    env = nflrush.make_env()\n    iter_test = env.iter_test()\n\n    for (test_df, sample_prediction_df) in iter_test:\n        basetable = create_features(test_df, deploy=True)\n        basetable.drop(['GameId','PlayId'], axis=1, inplace=True)\n        \n        scaled_basetable_features = scaler.transform(basetable[feature_columns].values)\n        scaled_basetable = pd.DataFrame(scaled_basetable_features, columns=feature_columns)\n        \n        y_pred = predict(scaled_basetable, models_nn, models_lgbm, weight_nn, weight_lgbm, iteration=True)\n#         y_pred = predict(scaled_basetable, models_nn, models_rf, weight_nn, weight_rf)\n        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]\n\n        preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)\n        env.predict(preds_df)\n\n    env.write_submission_file()")

