#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORTS 
import numpy as np
import pandas as pd
import datetime

import sklearn.metrics as mtr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Input, Concatenate, Reshape, Dropout, merge, Add, Layer, BatchNormalization
from tensorflow.keras.layers import Embedding
from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions

from sklearn.model_selection import KFold,GroupKFold
import warnings
import random as rn
import os

warnings.filterwarnings("ignore")
from kaggle.competitions import nflrush
env = nflrush.make_env()
iter_test = env.iter_test()


# In[ ]:


# evaluation metric
def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0]) 


# author : nlgn
# Link : https://www.kaggle.com/kingychiu/keras-nn-starter-crps-early-stopping
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
    
    
def preprocess(df):
    
    train = df[['PlayId','GameId','WindSpeed','GameWeather','Turf','OffenseFormation','OffensePersonnel','DefensePersonnel']]
    ## WindSpeed
    train['WindSpeed_ob'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

    ## Weather
    train['GameWeather_process'] = train['GameWeather'].str.lower()
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    #train['GameWeather_dense'] = train['GameWeather_process'].apply(map_weather)

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
    train = train.drop_duplicates()

    ## sort
#     train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index(drop = True)
#   train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop = True)
    return train


# In[ ]:


# author : ryancaldwell
# Link : https://www.kaggle.com/ryancaldwell/location-eda
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
    
    def offensive_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        offense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        offense = offense[offense['Team'] == offense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        offense['off_dist_to_back'] = offense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        offense = offense.groupby(['GameId','PlayId'])                         .agg({'off_dist_to_back':['min','max','mean','std']})                         .reset_index()
        offense.columns = ['GameId','PlayId','off_min_dist','off_max_dist','off_mean_dist','off_std_dist']

        return offense

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
    
    def combine_features(relative_to_back, defense, offense, static, personnel, prep, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,offense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,personnel,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,prep,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df
    
    prep = preprocess(df)
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    off_feats = offensive_features(df)
    static_feats = static_features(df)
    personnel = personnel_features(df)
    basetable = combine_features(rel_back, def_feats, off_feats, static_feats, personnel, prep, deploy=deploy)
    
    return basetable


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})\noutcomes = train[['GameId','PlayId','Yards']].drop_duplicates()\ntrain_basetable = create_features(train, False)")


# In[ ]:


X = train_basetable.drop_duplicates().copy()
yards = X.Yards
X = X.drop(['PlayId', 'Yards'], axis = 1)

y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1
    
print(X.shape)
print(y.shape)


# In[ ]:


cat_features = []
dense_features = []
for col in X.columns:
    if X[col].dtype =='object':
        cat_features.append(col)
        print("*cat*", col, len(X[col].unique()))
    else:
        dense_features.append(col)
        print("!dense!", col, len(X[col].unique()))


# In[ ]:


cat_features = [e for e in cat_features if e not in ['JerseyNumber_ob','Quarter_ob','Week_ob','Down_ob','WindSpeed',
                                                     'GameClock_minute','StadiumType','TimeDelta_ob','TeamOnOffense',
                                                     'YardLine_ob', 'PlayerHeight', 'WindSpeed_ob','WindDirection','GameWeather']]

cat_features = cat_features + (['back_oriented_down_field','back_moving_down_field'])
print(cat_features)

dense_features = [e for e in dense_features if e not in ['GameClock_sec', 'back_oriented_down_field','back_from_scrimmage',
                                                         'back_moving_down_field','Position','Team','PlayDirection','Quarter','Down',
                                                         'Turf','GameWeather_process',
                                                         'TimeDelta','WindDirection','OffenseFormation','OffensePersonnel',
                                                         'DefensePersonnel','num_DL', 'num_LB', 'num_DB', 'num_QB', 'num_RB',
                                                         'num_WR', 'num_TE', 'num_OL', 'OL_diff', 'OL_TE_diff','Yards','GameId',
                                                         'OffenseDB','OffenseDL','OffenseLB','OffenseOL','OffenseQB','OffenseRB',
                                                         'OffenseTE','OffenseWR','DefenseDB','DefenseDL','DefenseLB','DefenseOL']]
print(dense_features)


# In[ ]:


#clean cat
categories = []
most_appear_each_categories = {}
for col in cat_features:
    print(col)
    X.loc[:,col] = X[col].fillna("nan")
    X.loc[:,col] = col + "__" + X[col].astype(str)
    most_appear_each_categories[col] = list(X[col].value_counts().index)[0]
    categories.append(X[col].unique())
categories = np.hstack(categories)
print(len(categories))


# In[ ]:


#encoding cat
le = LabelEncoder()
le.fit(categories)
for col in cat_features:
    print(col)
    X.loc[:, col] = le.transform(X[col])
num_classes = len(le.classes_)


# In[ ]:


sss = {}
medians = {}
for col in dense_features:
    print(col)
    X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    medians[col] = np.nanmedian(X[col])
    X.loc[:, col] = X[col].fillna(medians[col])
    ss = StandardScaler()
    X.loc[:, col] = ss.fit_transform(X[col].values[:,None])
    sss[col] = ss


# In[ ]:


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

class GaussianLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GaussianLayer, self).build(input_shape) 

    def call(self, x):
        xx = K.arange(-99, 100, dtype=tf.float32)
        sigma = tf.identity(K.exp(0.5 * tf.reshape(x[:, 1], (-1, 1))), name="sigma")
        pdf = 1/(sigma* K.sqrt(2 * tf.constant(m.pi))) * K.exp( -(tf.subtract(xx, tf.reshape(x[:, 0], (-1, 1))))**2 / (2 * sigma**2) )
        cdf = []
        for i in range(199):
            if i == 0:
                cdf += [tf.reshape(pdf[:, i], (-1, 1))]
            else:
                cdf += [cdf[i-1] + tf.reshape(pdf[:, i], (-1, 1))]
        return tf.concat(cdf, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# In[ ]:


def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))


# In[ ]:


def model_LSTM_1():
    inputs = []
    embeddings = []
    
    input_numeric = Input(shape=(len(dense_features),))
    inputs.append(input_numeric)
    embeddings.append(input_numeric)


    #embedding_numeric = Dense(128, activation='relu')(input_numeric)
    #embedding_numeric = Dropout(0.5)(embedding_numeric)
    
    for col in cat_features:
        
        no_of_unique_cat  = int(np.absolute(X[col]).max())
        embedding_size = min(np.ceil((no_of_unique_cat)/2), 4)
        
        input_cat = Input(shape=(1,))
        inputs.append(input_cat)

        embedding = Embedding(no_of_unique_cat+1, embedding_size, input_length=1, embeddings_regularizer=l1_reg)(input_cat)
        embedding = LSTM(128)(embedding)
        #embedding = tfp.layers.Convolution1DFlipout(embedding_size, kernel_size=4, padding='SAME', activation="elu")(embedding)
        #embedding = Reshape(target_shape=(embedding_size,))(embedding)
        #embedding = Concatenate()([embedding, embedding_numeric])
        #embedding = tfp.layers.DenseFlipout(32, activation='relu')(embedding)
        #embedding = Dropout(0.5)(embedding)

        embeddings.append(embedding)
            
    x = Concatenate()(embeddings)
    x = Dense(1024, input_dim=X.shape[1], activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    
    output = Dense(199, activation='softmax')(x)
    model = Model(inputs, output)
    
    return model


# In[ ]:


def return_step(x):
    temp = np.zeros(199)
    temp[x + 99:] = 1
    return temp

#train_y_raw = train["Yards"].iloc[np.arange(0, len(train), 22)].reset_index(drop = True)
train_y_raw = outcomes['Yards']
train_y = np.vstack(outcomes['Yards'].apply(return_step).values)


# In[ ]:


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
n_splits = 3
kf = GroupKFold(n_splits=n_splits)
models = []
score = []

for k in range(2):
    kfold = KFold(5, random_state = 42 + k, shuffle = True)
    #for i_369, (tdx, vdx) in enumerate(kf.split(X, y, X['GameId'])):
    for i_lstm, (tdx, vdx) in enumerate(kfold.split(train_y_raw)):
        print(f'Fold : {i_lstm}')
        
        K.clear_session()
        X_train, X_val, y_train, y_val = X.iloc[tdx], X.iloc[vdx], y[tdx], y[vdx]
        X_train = [X_train[dense_features]] + [X_train[col] for col in cat_features]
        X_val = [X_val[dense_features]] + [X_val[col] for col in cat_features]

        model = model_LSTM_1()
        model.compile(optimizer=keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999), loss='categorical_crossentropy', metrics=[])

        es = EarlyStopping(monitor='val_CRPS', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=2, 
                       patience=5)
        
        es.set_model(model)
        metric = Metric(model, [es], [(X_train,y_train), (X_val,y_val)])

        model.fit(X_train, y_train, callbacks=[metric], epochs=200, batch_size=1024, verbose=1)
        models.append(model)

        score_ = crps(y_val, model.predict(X_val))

        print(f'keras_LSTM_{i_lstm}.h5')
        print(score_)
        score.append(score_)

print(np.mean(score))


# In[ ]:


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[ ]:


for (test_df, sample_prediction_df) in iter_test:
    basetable = create_features(test_df, deploy=True)
    test = basetable.drop('PlayId', axis = 1)

    ### categorical
    for col in (cat_features):
        test.loc[:,col] = test[col].fillna("nan")
        test.loc[:,col] = col + "__" + test[col].astype(str)
        isnan = ~test.loc[:,col].isin(categories)
        if np.sum(isnan) > 0:
    #             print("------")
    #             print("test have unseen label : col")
            if not ((col + "__nan") in categories):
    #                 print("not nan in train : ", col)
                test.loc[isnan,col] = most_appear_each_categories[col]
            else:
    #                 print("nan seen in train : ", col)
                test.loc[isnan,col] = col + "__nan"
    for col in (cat_features):
        test.loc[:, col] = le.transform(test[col])

    ### dense
    for col in dense_features:
        test[col] = test[col].replace([np.inf, -np.inf], np.nan)
        test.loc[:, col] = test[col].fillna(medians[col])
        test.loc[:, col] = sss[col].transform(test[col].values[:,None])

    ### divide
    #test = [test.iloc[np.arange(k, len(test), 22)].reset_index(drop = True) for k in range(22)]
    test_ = [test[dense_features]] + [test[col] for col in  cat_features]

    y_pred = np.mean([model.predict(test_) for model in models], axis=0)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]

    preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
    env.predict(preds_df)

env.write_submission_file()


# In[ ]:




