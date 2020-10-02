#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime

from kaggle.competitions import nflrush
import tqdm
import re
from sklearn.preprocessing import StandardScaler

import keras
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf

import warnings  
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})


# In[ ]:


# Fix inconsistent abbreviations
# Fix Field Possession has null values when the ball is at the 50 yard line
map_abbr = {'ARI': 'ARZ', 'BAL':'BLT','CLE':'CLV', 'HOU':'HST'}
for abb in df['PossessionTeam'].unique():
    map_abbr[abb] = abb
map_abbr['50YDL'] = '50YDL'

df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)

df['FieldPosition'] = df['FieldPosition'].replace(np.nan, '50YDL')
df['FieldPosition'] = df['FieldPosition'].map(map_abbr)

df['HomePossession'] = df['PossessionTeam'] == df['HomeTeamAbbr']
df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']
df['HomeField'] = df['FieldPosition'] == df['HomeTeamAbbr']


# In[ ]:


df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: int(x.split('-')[0])*12 + int(x.split('-')[1]))
df['PlayerAge'] = df['TimeHandoff'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')) - df['PlayerBirthDate'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
df['PlayerAge'] = df['PlayerAge'].apply(lambda x: int(np.floor(x.days/365.333)))
df['IsRunner'] = (df['NflId'] == df['NflIdRusher'])


# In[ ]:


def map_weather(txt):
    weather_encodings = {'indoors': 3, 'sunny': 2, 'cloudy': 1, 'cold': -1, 'rain': -2, 'snow': -3}
    if (pd.isna(txt) or txt not in list(weather_encodings.keys())):
        return 0
    return weather_encodings[txt]

df['GameWeather'] = df['GameWeather'].str.lower()
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*sun.*$', value='sunny')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*clear.*$', value='sunny')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*fair.*$', value='sunny')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*rain.*$', value='rain')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*shower.*$', value='rain')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*snow.*$', value='snow')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*clou.*$', value='cloudy')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*over.*$', value='cloudy')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*haz.*$', value='cloudy')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*cou.*$', value='cloudy')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*cold.*$', value='cold')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*indoor.*$', value='indoors')
df['GameWeather'] = df['GameWeather'].replace(regex=r'.*control.*$', value='indoors')
df['GameWeather'] = df['GameWeather'].apply(map_weather)


# In[ ]:


def map_wind_speed(speed):
    if(len(str(speed)) == 4):
        return (int(speed[0:2]) + int(speed[2:4]))/2
    elif(len(str(speed)) == 2 or len(str(speed)) == 1):
        return int(speed)
    else:
        return np.nan
    
df['WindSpeed'] = df['WindSpeed'].astype(str).str.lower()
df['WindSpeed'] = df['WindSpeed'].replace(regex=r'[a-z, , -]', value='')
df['WindSpeed'] = df['WindSpeed'].apply(map_wind_speed)


# In[ ]:


def map_wind_direction(txt):
    direction_map = {'e': 0, 'n': 1, 'w': 2, 's': 3, 'ne': 0.5, 'nw': 1.5, 'sw': 2.5, 'se': 3.5, 'ene': 0.25, 'nne': 0.75, 'nnw': 1.25, 'wnw': 1.75, 'wsw': 2.25, 'ssw': 2.75, 'sse': 3.25, 'ese': 3.75}
    if (pd.isna(txt) or txt not in list(direction_map.keys())):
        return np.nan
    return direction_map[txt]

df['WindDirection'] = df['WindDirection'].astype(str)
df['WindDirection'] = df['WindDirection'].str.lower()
df['WindDirection'] = df['WindDirection'].replace(regex=r'[0-9,-, ,-]', value='')
df['WindDirection'] = df['WindDirection'].replace('calm', value='')
df['WindDirection'] = df['WindDirection'].replace('nan', value='')
df['WindDirection'] = df['WindDirection'].replace(regex=r'.*north.*$', value='n')
df['WindDirection'] = df['WindDirection'].replace(regex=r'.*south.*$', value='s')
df['WindDirection'] = df['WindDirection'].replace(regex=r'.*east.*$', value='e')
df['WindDirection'] = df['WindDirection'].replace(regex=r'.*west.*$', value='w')

df[(df['WindDirection'].str.contains('from'))] = df[(df['WindDirection'].str.contains('from'))].replace('n', value='s')
df[(df['WindDirection'].str.contains('from'))] = df[(df['WindDirection'].str.contains('from'))].replace('s', value='n')
df[(df['WindDirection'].str.contains('from'))] = df[(df['WindDirection'].str.contains('from'))].replace('e', value='w')
df[(df['WindDirection'].str.contains('from'))] = df[(df['WindDirection'].str.contains('from'))].replace('w', value='e')
df['WindDirection'] = df['WindDirection'].replace(regex=r'.*from.*$', value='n')

df['WindDirection'] = df['WindDirection'].replace('', value='other')

df['WindSpeed'] = df['WindSpeed'].apply(map_wind_direction)


# In[ ]:


def map_stadium(txt):
    stadium_map = {'outdoor': 0, 'indoor': 1}
    if (pd.isna(txt) or txt not in list(stadium_map.keys())):
        return np.nan
    return stadium_map[txt]

df['StadiumType'] = df['StadiumType'].str.lower()
df['StadiumType'] = df['StadiumType'].replace(regex=r'^ou.*$', value='outdoor')
df['StadiumType'] = df['StadiumType'].replace(regex=r'^in.*$', value='indoor')
df['StadiumType'] = df['StadiumType'].replace(regex=r'^.*open.*$', value='outdoor')
df['StadiumType'] = df['StadiumType'].replace(regex=r'^.*closed.*$', value='indoor')
# Assume that dome means indoor
df['StadiumType'] = df['StadiumType'].replace(regex=r'^.*dome.*$', value='indoor')
# Stadiums stay open most of the time, except during heavy rain. There wasn't any rain in the 
# missing values except during week 17 in Houston
df['StadiumType'] = df['StadiumType'].replace('retractable roof', value='outdoor')
df['StadiumType'].loc[(df['HomeTeamAbbr']=='HOU') & (df['Week']==17)] = 'indoor'
# Change Heinz Field to outdoor because it is
df['StadiumType'] = df['StadiumType'].replace('heinz field', value='outdoor')
# Two Jags games have messed up stadium types. TIAA Bank Field is open, so let's adjust that real quick
df['StadiumType'] = df['StadiumType'].replace('cloudy', value='outdoor')
df['StadiumType'] = df['StadiumType'].replace('bowl', value='outdoor')
    
df['StadiumType'] = df['StadiumType'].apply(map_stadium)


# In[ ]:


temp = pd.to_datetime(df['GameClock'], format='%M:%S:%f').dt.strftime('%H:%M:%S')
temp = temp.apply(lambda x: int(x.split(':')[0])*3600 + int(x.split(':')[1])*60 + int(x.split(':')[2]))
df['ElapsedTime'] = df['Quarter']*15*60 - temp

df['SnapToHandoffTime'] = df['TimeHandoff'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')) - df['TimeSnap'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
df['SnapToHandoffTime'] = df['SnapToHandoffTime'].apply(lambda x: x.seconds)


# In[ ]:


df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x.strip() == 'right')
df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')


# In[ ]:


def new_orientation(angle, play_direction):
    if play_direction == 0:
        new_angle = 360.0 - angle
        if new_angle == 360.0:
            new_angle = 0.0
        return new_angle
    else:
        return angle
    
df['X'] = df.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)
df['Orientation'] = df.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)
df['Dir'] = df.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)

df['YardsLeft'] = df.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
df['YardsLeft'] = df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)


# In[ ]:


df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='OFormation')], axis=1)
#df = pd.concat([df.drop(['OffensePersonnel'], axis=1), pd.get_dummies(df['OffensePersonnel'], prefix='OPersonnel')], axis=1)
#df = pd.concat([df.drop(['DefensePersonnel'], axis=1), pd.get_dummies(df['DefensePersonnel'], prefix='DPersonnel')], axis=1)
df.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)
df.drop(df.index[(df['YardsLeft']<df['Yards']) | (df['YardsLeft']-100>df['Yards'])], inplace=True)


# In[ ]:


df = df.sort_values(by=['PlayId', 'Team', 'IsRunner', 'JerseyNumber']).reset_index()
df.drop(['GameId', 'PlayId', 'index', 'IsRunner', 'Team'], axis=1, inplace=True)


# In[ ]:


dummy_cols = df.columns


# In[ ]:


cat_features = []
for col in df.columns:
    if df[col].dtype =='object':
        cat_features.append(col)

df = df.drop(cat_features, axis=1)
df.fillna(-999, inplace=True)


# In[ ]:


players_col = []
for col in df.columns:
    if df[col][:22].std()!=0:
        players_col.append(col)


# In[ ]:


X_df = np.array(df[players_col]).reshape(-1, len(players_col)*22)
play_col = df.drop(players_col+['Yards'], axis=1).columns
X_play_col = np.zeros(shape=(X_df.shape[0], len(play_col)))
for i, col in enumerate(play_col):
    X_play_col[:, i] = df[col][::22]
X_train = np.concatenate([X_df, X_play_col], axis=1)
y_train = np.zeros(shape=(X_df.shape[0], 199))

for i,yard in enumerate(df['Yards'][::22]):
    y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# In[ ]:


def crps(y_true, y_pred):
    return K.mean(K.square(y_true - K.cumsum(y_pred, axis=1)), axis=1)


# In[ ]:


def get_model():
    
    model = keras.Sequential()
    model.add(keras.layers.Dense(1024, activation=tf.nn.relu, input_dim=X_train.shape[1]))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    
    #model.add(keras.layers.GaussianNoise(0.3))
    model.add(keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.GaussianNoise(0.3))
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.GaussianNoise(0.3))

    model.add(keras.layers.Dense(units=199, activation=tf.nn.softmax))
    return model

batch_size=64
def train_model(X_train, y_train, X_val, y_val):
    model = get_model()
    optim = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optim, loss=crps)
    er = EarlyStopping(patience=10, min_delta=1e-3, restore_best_weights=True, monitor='val_loss')
    model.fit(X_train, y_train, epochs=200, callbacks=[er], validation_data=[X_val, y_val], batch_size=batch_size, verbose=0)
    
    y_pred = model.predict(X_val)
    y_valid = y_val
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_val.shape[0])
    crps_val = np.round(val_s, 6)
    return model, crps_val


# In[ ]:


from sklearn.model_selection import RepeatedKFold

#X_train, y_train, dummy_train, X_cols = make_pred(train_df, True, [], 0, 0, 0)

rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

models = []
crps_csv = []
for tr_idx, vl_idx in rkf.split(X_train, y_train):
    
    x_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    x_vl, y_vl = X_train[vl_idx], y_train[vl_idx]
    
    model, crps_val = train_model(x_tr, y_tr, x_vl, y_vl)
    models.append(model)
    crps_csv.append(crps_val)
print('crps mean: ' , np.mean(crps_csv))


# In[ ]:


def make_pred(df, dummy_cols, sample, env, models):
    # Fix inconsistent abbreviations
    # Fix Field Possession has null values when the ball is at the 50 yard line
    map_abbr = {'ARI': 'ARZ', 'BAL':'BLT','CLE':'CLV', 'HOU':'HST'}
    for abb in df['PossessionTeam'].unique():
        map_abbr[abb] = abb
    map_abbr['50YDL'] = '50YDL'

    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)

    df['FieldPosition'] = df['FieldPosition'].replace(np.nan, '50YDL')
    df['FieldPosition'] = df['FieldPosition'].map(map_abbr)

    df['HomePossession'] = df['PossessionTeam'] == df['HomeTeamAbbr']
    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']
    df['HomeField'] = df['FieldPosition'] == df['HomeTeamAbbr']
    
    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: int(x.split('-')[0])*12 + int(x.split('-')[1]))
    df['PlayerAge'] = df['TimeHandoff'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')) - df['PlayerBirthDate'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
    df['PlayerAge'] = df['PlayerAge'].apply(lambda x: int(np.floor(x.days/365.333)))
    df['IsRunner'] = (df['NflId'] == df['NflIdRusher'])
    
    df['GameWeather'] = df['GameWeather'].str.lower()
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*sun.*$', value='sunny')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*clear.*$', value='sunny')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*fair.*$', value='sunny')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*rain.*$', value='rain')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*shower.*$', value='rain')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*snow.*$', value='snow')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*clou.*$', value='cloudy')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*over.*$', value='cloudy')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*haz.*$', value='cloudy')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*cou.*$', value='cloudy')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*cold.*$', value='cold')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*indoor.*$', value='indoors')
    df['GameWeather'] = df['GameWeather'].replace(regex=r'.*control.*$', value='indoors')
    df['GameWeather'] = df['GameWeather'].apply(map_weather)
    
    df['WindSpeed'] = df['WindSpeed'].astype(str).str.lower()
    df['WindSpeed'] = df['WindSpeed'].replace(regex=r'[a-z, , -]', value='')
    df['WindSpeed'] = df['WindSpeed'].apply(map_wind_speed)
    
    df['WindDirection'] = df['WindDirection'].astype(str)
    df['WindDirection'] = df['WindDirection'].str.lower()
    df['WindDirection'] = df['WindDirection'].replace(regex=r'[0-9,-, ,-]', value='')
    df['WindDirection'] = df['WindDirection'].replace('calm', value='')
    df['WindDirection'] = df['WindDirection'].replace('nan', value='')
    df['WindDirection'] = df['WindDirection'].replace(regex=r'.*north.*$', value='n')
    df['WindDirection'] = df['WindDirection'].replace(regex=r'.*south.*$', value='s')
    df['WindDirection'] = df['WindDirection'].replace(regex=r'.*east.*$', value='e')
    df['WindDirection'] = df['WindDirection'].replace(regex=r'.*west.*$', value='w')

    df[(df['WindDirection'].str.contains('from'))] = df[(df['WindDirection'].str.contains('from'))].replace('n', value='s')
    df[(df['WindDirection'].str.contains('from'))] = df[(df['WindDirection'].str.contains('from'))].replace('s', value='n')
    df[(df['WindDirection'].str.contains('from'))] = df[(df['WindDirection'].str.contains('from'))].replace('e', value='w')
    df[(df['WindDirection'].str.contains('from'))] = df[(df['WindDirection'].str.contains('from'))].replace('w', value='e')
    df['WindDirection'] = df['WindDirection'].replace(regex=r'.*from.*$', value='n')

    df['WindDirection'] = df['WindDirection'].replace('', value='other')

    df['WindSpeed'] = df['WindSpeed'].apply(map_wind_direction)
    
    df['StadiumType'] = df['StadiumType'].str.lower()
    df['StadiumType'] = df['StadiumType'].replace(regex=r'^ou.*$', value='outdoor')
    df['StadiumType'] = df['StadiumType'].replace(regex=r'^in.*$', value='indoor')
    df['StadiumType'] = df['StadiumType'].replace(regex=r'^.*open.*$', value='outdoor')
    df['StadiumType'] = df['StadiumType'].replace(regex=r'^.*closed.*$', value='indoor')
    # Assume that dome means indoor
    df['StadiumType'] = df['StadiumType'].replace(regex=r'^.*dome.*$', value='indoor')
    # Stadiums stay open most of the time, except during heavy rain. There wasn't any rain in the 
    # missing values except during week 17 in Houston
    df['StadiumType'] = df['StadiumType'].replace('retractable roof', value='outdoor')
    df['StadiumType'].loc[(df['HomeTeamAbbr']=='HOU') & (df['Week']==17)] = 'indoor'
    # Change Heinz Field to outdoor because it is
    df['StadiumType'] = df['StadiumType'].replace('heinz field', value='outdoor')
    # Two Jags games have messed up stadium types. TIAA Bank Field is open, so let's adjust that real quick
    df['StadiumType'] = df['StadiumType'].replace('cloudy', value='outdoor')
    df['StadiumType'] = df['StadiumType'].replace('bowl', value='outdoor')

    df['StadiumType'] = df['StadiumType'].apply(map_stadium)
    
    temp = pd.to_datetime(df['GameClock'], format='%M:%S:%f').dt.strftime('%H:%M:%S')
    temp = temp.apply(lambda x: int(x.split(':')[0])*3600 + int(x.split(':')[1])*60 + int(x.split(':')[2]))
    df['ElapsedTime'] = df['Quarter']*15*60 - temp

    df['SnapToHandoffTime'] = df['TimeHandoff'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')) - df['TimeSnap'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    df['SnapToHandoffTime'] = df['SnapToHandoffTime'].apply(lambda x: x.seconds)
    
    df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x.strip() == 'right')
    df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')
    
    df['X'] = df.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)
    df['Orientation'] = df.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)
    df['Dir'] = df.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)

    df['YardsLeft'] = df.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
    df['YardsLeft'] = df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)
    
    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='OFormation')], axis=1)
    #df = pd.concat([df.drop(['OffensePersonnel'], axis=1), pd.get_dummies(df['OffensePersonnel'], prefix='OPersonnel')], axis=1)
    #df = pd.concat([df.drop(['DefensePersonnel'], axis=1), pd.get_dummies(df['DefensePersonnel'], prefix='DPersonnel')], axis=1)
    df.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)
    
    df = df.sort_values(by=['PlayId', 'Team', 'IsRunner', 'JerseyNumber']).reset_index()
    df.drop(['GameId', 'PlayId', 'index', 'IsRunner', 'Team'], axis=1, inplace=True)
    
    missing_cols = set( dummy_cols ) - set( df.columns ) - set('Yards')
    for c in missing_cols:
        df[c] = 0
    df = df[dummy_cols]
    
    df.drop(['Yards'], axis=1, inplace=True)
    
    cat_features = []
    for col in df.columns:
        if df[col].dtype =='object':
            cat_features.append(col)

    df = df.drop(cat_features, axis=1)
    df.fillna(-999, inplace=True)
    
    players_col = []
    for col in df.columns:
        if df[col][:22].std()!=0:
            players_col.append(col)
            
    X_df = np.array(df[players_col]).reshape(-1, len(players_col)*22)
    play_col = df.drop(players_col, axis=1).columns
    X_play_col = np.zeros(shape=(X_df.shape[0], len(play_col)))
    for i, col in enumerate(play_col):
        X_play_col[:, i] = df[col][::22]

    X = np.concatenate([X_df, X_play_col], axis=1)
    X = scaler.transform(X)
    y_pred = np.mean([np.cumsum(model.predict(X), axis=1) for model in models], axis=0)
    yardsleft = np.array(df['YardsLeft'][::22])

    for i in range(len(yardsleft)):
        y_pred[i, :yardsleft[i]-1] = 0
        y_pred[i, yardsleft[i]+100:] = 1
    env.predict(pd.DataFrame(data=y_pred.clip(0, 1), columns=sample.columns))
    return y_pred


# In[ ]:


env = nflrush.make_env()


# In[ ]:


for test, sample in tqdm.tqdm(env.iter_test()):
     make_pred(test, dummy_cols, sample, env, models)


# In[ ]:


env.write_submission_file()

