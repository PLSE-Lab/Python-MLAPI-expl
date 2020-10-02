#!/usr/bin/env python
# coding: utf-8

# ## SARIMAX Model[](http://)

# ### Imports and Options

# In[ ]:


import datetime
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tqdm
import warnings

from kaggle.competitions import nflrush


# In[ ]:


pd.options.display.max_rows = 500
pd.options.display.max_columns = 100
sns.set_style('darkgrid')
mpl.rcParams['figure.figsize'] = [15,10]


# ### Initialize NFL Environment

# In[ ]:


env = nflrush.make_env()


# ### Training Data Exploration

# In[ ]:


train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})


# In[ ]:


train.columns


# In[ ]:


train['GameId'].value_counts()


# In[ ]:


agame = train[train['GameId'] == 2017121000]
agame


# In[ ]:


agame['PossessionTeam'].tail(100)


# In[ ]:


len(agame['WindDirection'].unique())


# In[ ]:


aplay = agame[agame['PlayId'] == 20171210000058]
aplay


# In[ ]:


aplay['is_rusher'] = aplay['NflId'] == aplay['NflIdRusher']


# In[ ]:


aplay


# In[ ]:


rusher_values = aplay.loc[aplay['is_rusher']][['X', 'Y', 'S', 'A']]
rusher_values = pd.DataFrame(np.repeat(rusher_values.values, len(aplay), axis=0))
rusher_values.columns = ['X_rusher', 'Y_rusher', 'S_rusher', 'A_rusher']
rusher_values


# In[ ]:


train.head()


# In[ ]:


train['PlayId'].value_counts()


# In[ ]:


train['Yards'].describe()


# In[ ]:


ax = sns.distplot(train['Yards'])
plt.vlines(train['Yards'].mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.text(train['Yards'].mean()-8, plt.ylim()[1]-0.005, "Mean Yards Gained", size=15, color='r')
plt.xlabel("")
plt.title("Yards Gained Distribution", size=20);


# Train Model per Team (per game, per play)
# 
# * Records: (1870, **22**)
# * PlayId: (85 plays, 1)
# * Team: per **player** (2, 2)
# * X: per **player** (1631, 22)
# * Y: per **player** (1337, 22)
# * S: per **player** (470, 21)
# * A: per **player** (323, 20)
# * Dis: per **player** (60, 16)
# * Orientation: per **player** (1808, 22)
# * Dir: per **player** (1816, 22)
# * NflId: per **player** (71, 22)
# * DisplayName: per **player** (71, 22)
# * JerseyNumber: per **player** (55, 21)
# * Season: (1, 1)
# * YardLine: (36, 1)
# * Quarter: (5, 1)
# * GameClock: (82, 1)
# * PossessionTeam: per play (2, 1)
# * Down: per play (4, 1)
# * Distance: per play (15, 1)
# * FieldPosition: per play (3, 1)
# * HomeScoreBeforePlay: per play (2, 1)
# * VisitorScoreBeforePlay: per play (2, 1)
# * NflIdRusher: per **player** (5, 1)
# * OffenseFormation: (4, 1)
# * OffensePersonnel: (7, 1)
# * DefendersInTheBox: (6, 1)
# * DefensePersonnel: (6, 1)
# * PlayDirection: (2, 1)
# * TimeHandoff: (85 plays, 1)
# * TimeSnap: (85 plays, 1)
# * **Yards**: (18, 1)
# * PlayerHeight: per **player** (12, 9)
# * PlayerWeight: per **player** (52, 21)
# * PlayerBirthDate: per **player** (70, 22)
# * PlayerCollegeName: per **player** (55, 16)
# * Position: per **player** (19, 14)
# * HomeTeamAbbr: (1, 1)
# * VisitorTeamAbbr: (1, 1)
# * Week: (1, 1)
# * Stadium: (1, 1)
# * Location: (1, 1)
# * StadiumType: (1, 1)
# * Turf: (1, 1)
# * GameWeather: (1, 1)
# * Temperature: (1, 1)
# * Humidity: (1, 1)
# * WindSpeed: (1, 1)
# * WindDirection: (1, 1)
# 

# ## Feature Engineering

# In[ ]:


train.columns


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

    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
                                                            'YardLine','Quarter', 'PossessionTeam', 'Down','Distance','DefendersInTheBox']].drop_duplicates()
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

    def combine_features(relative_to_back, defense, static, personnel, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,personnel,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df
    
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    static_feats = static_features(df)
    personnel = personnel_features(df)
    basetable = combine_features(rel_back, def_feats, static_feats, personnel, deploy=deploy)
    return basetable


# In[ ]:


outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()
outcomes


# In[ ]:


train_basetable = create_features(train, False)


# In[ ]:


play_number = False
if play_number:
    train_basetable['play_number'] = train_basetable.groupby('GameId').cumcount()
    train_basetable[['GameId', 'PlayId', 'play_number']].tail(100)


# In[ ]:


play_number_team = False
if play_number_team:
    train_basetable['play_number_team'] = train_basetable.groupby(['GameId', 'PossessionTeam']).cumcount()
    train_basetable[['GameId', 'PlayId', 'PossessionTeam', 'play_number_team']].tail(100)


# In[ ]:


last6 = False
if last6:
    train_basetable['last6'] = train_basetable['PlayId'] % 1000000
    train_basetable['last6_diff'] = train_basetable.groupby('GameId')['last6'].diff().fillna(0).astype(int)
    train_basetable.drop(columns=['last6'], inplace=True)


# ## Model

# ### Cumulative Distribution Function

# In[ ]:


pdf, edges = np.histogram(train['Yards'], bins=199, range=(-99, 100), density=True)
cdf = pdf.cumsum().clip(0, 1)
cdf_df = pd.DataFrame(data=cdf.reshape(-1, 1).T, 
                      columns=['Yards'+str(i) for i in range(-99, 100)])
cdf = cdf_df.values.reshape(-1,)


# In[ ]:


cdf


# In[ ]:


all_teams = train['PossessionTeam'].unique()
all_teams


# In[ ]:


cdf_team = {}
for t in all_teams:
    df_team = train[train['PossessionTeam'] == t]
    pdf, edges = np.histogram(df_team['Yards'], bins=199, range=(-99, 100), density=True)
    cdf = pdf.cumsum().clip(0, 1)
    cdf_df = pd.DataFrame(data=cdf.reshape(-1, 1).T, 
                          columns=['Yards'+str(i) for i in range(-99, 100)])
    cdf_team[t] = cdf_df.values.reshape(-1,)


# In[ ]:


cdf_team['NE']


# ### Evaluation Metric

# In[ ]:


def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0]) 


# ### Training Data (X) and Yards (y)

# In[ ]:


def process_two(t_):
    t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) - np.square(t_.Y.values))))
    t_['fe5'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values  # N
    t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
    t_['fe8'] = t_['S'].values / np.clip(t_['fe1'].values, 0.6, None)
    radian_angle = (90 - t_['Dir']) * np.pi / 180.0
    t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
    t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
    return t_


# In[ ]:


train_basetable.columns


# In[ ]:


X = train_basetable.copy()


# In[ ]:


X = process_two(X)


# In[ ]:


yards = X.Yards
yards.value_counts()


# In[ ]:


X.drop(['GameId', 'PlayId', 'Yards'], axis=1, inplace=True)


# In[ ]:


X.sample(20)


# ### Initialize Distribution (y)

# In[ ]:


y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1


# In[ ]:


len(y)


# ### Model

# In[ ]:


def train_model(endog, exog, p=1, d=0, q=0):
    # optimize p, d, q
    pass
    # train model
    smax = SARIMAX(endog, exog, order=(p, d, q))
    model = smax.fit(method='powell')
    # get feature importances
    features_html = model.summary().tables[1].as_html()
    df_feat = pd.read_html(features_html)[0].iloc[1:, :]
    df_feat.columns = ['feature', 'coef', 'std err', 'Z', 'P>|z|', 'ci_low', 'ci_high']
    # return predictions and feature importances
    return model, df_feat


# In[ ]:


models = {}
for t in all_teams:
    X_team = X[X['PossessionTeam'] == t]
    X_team.drop(columns=['PossessionTeam'], inplace=True)
    y_team = yards[X_team.index].reset_index(drop=True)
    X_team.reset_index(drop=True, inplace=True)
    training_size = len(X_team)
    print("Training %d rows for %s Model" % (training_size, t))
    model, dfi = train_model(y_team, X_team)
    models[t] = [model, training_size, dfi]


# In[ ]:


models


# In[ ]:


models['CAR'][2].sort_values(by='P>|z|')


# ### Predictions

# In[ ]:


def get_score(y_pred, cdf, w, dist_to_end):
    y_pred = int(y_pred)
    if y_pred == w:
        y_pred_array = cdf.copy()
    elif y_pred - w > 0:
        y_pred_array = np.zeros(199)
        y_pred_array[(y_pred-w):] = cdf[:(-(y_pred-w))].copy()
    elif w - y_pred > 0:
        y_pred_array = np.ones(199)
        y_pred_array[:(y_pred-w)] = cdf[(w-y_pred):].copy()
    y_pred_array[-1] = 1
    y_pred_array[(dist_to_end + 99):] = 1
    return y_pred_array


# In[ ]:


def make_prediction(model, exogp, exog_len):
    prediction = model.predict(exog_len, exog_len, exog=exogp)
    return prediction


# In[ ]:


test_team = 'NYG'
X_team = X[X['PossessionTeam'] == test_team]
X_team.drop(columns=['PossessionTeam'], inplace=True)
y_team = yards[X_team.index].reset_index(drop=True)
X_team.reset_index(drop=True, inplace=True)


# In[ ]:


index = 350
y_pred = make_prediction(models[test_team][0], X_team[index:index+1], models[test_team][1])


# In[ ]:


y_pred.tolist()[0], y_team[index]


# In[ ]:


iter_test = env.iter_test()


# In[ ]:


for (test_df, sample_prediction_df) in tqdm.tqdm(iter_test):
    # get possession team for time series
    pteam = test_df['PossessionTeam'].iloc[0]
    sarimax = False
    if sarimax:
        # calculate distance to end for distribution
        test_df['own_field'] = (test_df['FieldPosition'] == test_df['PossessionTeam']).astype(int)
        dist_to_end_test = test_df.apply(lambda x:(100 - x.loc['YardLine']) if x.loc['own_field']==1 else x.loc['YardLine'], axis=1)
        test_df.drop(columns=['own_field'], inplace=True)
        # extract features from test frame
        X_test = create_features(test_df, deploy=True)
        X_test = process_two(X_test)
        X_test.drop(['GameId', 'PlayId', 'PossessionTeam'], axis=1, inplace=True)
        # make prediction
        model = models[pteam][0]
        tsize = models[pteam][1]
        y_pred = make_prediction(model, X_test, tsize)
        print("%s Prediction: %f" % (pteam, y_pred.values[0]))
        pred_data = list(get_score(y_pred, cdf, 4, dist_to_end_test.values[0]))
        pred_data = np.array(pred_data).reshape(1, 199)
        pred_df = pd.DataFrame(index = sample_prediction_df.index,                                columns = sample_prediction_df.columns,                                data = pred_data)
        env.predict(pred_df)
    else:
        sample_prediction_df.iloc[0, :] = cdf_team[pteam]
        env.predict(sample_prediction_df)
    # augment X with X_test and yards with y_pred for next prediction
    pass


# In[ ]:


env.write_submission_file()


# # End of Notebook
