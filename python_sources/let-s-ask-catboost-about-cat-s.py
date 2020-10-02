#!/usr/bin/env python
# coding: utf-8

# 
# ### There are so many useless categorical features!
# 
# There are many categorical variables in this dataset, and I heard that lots of people do not use them because these variables do not seem to contribute to the prediction very much. I wonder if this is due to the way we treat these categorical variables ... there are actually a number of ways to deal with categorical columns such as label encoding, one-hot encoding, target encoding, and so on. Also we might want to seek for important interactions between these categorical variables.
# 
# It is exhaustive to try all kinds of encoding for categorical data, yet what we can try is to use CatBoost to see what this model can do. CatBoost automatically performs target encoding for categorical data and tries to estimate interactions between them. So it may be worthwhile looking into what CatBoost has to say about our categorical variables. 

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


###raw mae
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold,TimeSeriesSplit,KFold,GroupKFold
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error
import tqdm
import sqlite3
import datetime
from scipy.stats import pearsonr
import gc
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from kaggle.competitions import nflrush


# In[ ]:


train_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv',low_memory=False)


# In[ ]:


#from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087
Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 
        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 
        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 
        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 

# from https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in train_df['PossessionTeam'].unique():
    map_abbr[abb] = abb
    
train_df["HomeTeamAbbr"] = train_df["HomeTeamAbbr"].map(map_abbr)
train_df["VisitorTeamAbbr"] = train_df["VisitorTeamAbbr"].map(map_abbr)
train_df["Possession"] = train_df["PossessionTeam"].map(map_abbr)
    
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

def preprocess(df, labelEncoders=None):
    X = df.copy()
    X = X.select_dtypes(include=['number', 'object'])
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
    def transform_time_quarter(str1):
        return int(str1[:2])*60 + int(str1[3:5])
    def transform_time_all(str1,quarter):
        if quarter<=4:
            return 15*60 - (int(str1[:2])*60 + int(str1[3:5])) + (quarter-1)*15*60
        if quarter ==5:
            return 10*60 - (int(str1[:2])*60 + int(str1[3:5])) + (quarter-1)*15*60
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
            return 'outdoor'
        elif stadium in indoor_closed:
            return 'indoor closed'
        elif stadium in indoor_open:
            return 'indoor open'
        elif stadium in dome_closed:
            return 'dome closed'
        elif stadium in dome_open:
            return 'dome open'
        else:
            return 'unknown'

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
            return 'rain'
        elif weather in overcast:
            return 'overcast'
        elif weather in clear:
            return 'clear'
        elif weather in snow:
            return 'snow'
        elif weather in none:
            return 'none'
        else:
            return 'none'

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
        
    X["Dir"] = np.mod(90 - df["Dir"].values, 360)
#     X['Team'] = df['Team'].map({"home": 0, "away": 1})
    X['Turf'] = df['Turf'].map(Turf)
#     X['Turf'] = X['Turf'].map({"Natural": 0,"Artificial": 1})
    X['HomePossesion'] = 1 * (df['PossessionTeam'] == df['HomeTeamAbbr'])
#     X["OffenseFormation"] = df['OffenseFormation'].apply(clean_offenceformation)
#     X['OffenseFormation'] = X['OffenseFormation'].fillna(7)
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
    X['PlayerWeight'] = df['PlayerWeight']
    X['PlayerHeight'] = df['PlayerHeight'].apply(height2inch)
    X['BMI'] = X['PlayerWeight'] / X['PlayerHeight']
    X['time_quarter'] = df["GameClock"].map(lambda x:transform_time_quarter(x)).values
    X['time_end'] = df.apply(lambda x:transform_time_all(x.loc['GameClock'],x.loc['Quarter']),axis=1).values
    X['GameClock'] = df['GameClock'].apply(gameclock2min)
    X['StadiumType'] = df['StadiumType'].apply(group_stadium_types)
    X['GameWeather'] = df['GameWeather'].apply(group_game_weather)
    X['WindSpeed'] = df['WindSpeed'].apply(clean_wind_speed)
#     X['WindDirection'] = df['WindDirection'].apply(clean_wind_direction)
#     X['WindDirection'] = 2 * np.pi * (90 - X['WindDirection'].values) / 360
    X['Humidity'] = df['Humidity'].fillna(df['Humidity'].median())
    X['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    X['DefendersInTheBox'] = df['DefendersInTheBox'].fillna(df['DefendersInTheBox'].median())
        
    # from https://www.kaggle.com/ryches/model-free-benchmark
    X['Field_eq_Possession'] = 1 * (df['FieldPosition'] == df['PossessionTeam'])    
    X['is_rusher'] = 1 * (df['NflId'] == df['NflIdRusher'])
    X['seconds_need_to_first_down'] = (df['Distance']*0.9144) / (df['Dis'].values + 0.01)
    X['seconds_need_to_YardsLine'] = (df['YardLine']*0.9144) / (df['Dis'].values + 0.01)
    X['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']
    
    # based on https://www.kaggle.com/sryo188558/cox-proportional-hazard-model
    playdir = df['PlayDirection'].map({'right': 1, 'left': -1}).values
    X["Start"] = X["YardLine"]
    X.loc[(X["Field_eq_Possession"] == 1) & (playdir == 1), "Start"] = X.loc[(X["Field_eq_Possession"] == 1) & (playdir == 1), 
                                                                                       "YardLine"] + 10
    X.loc[(X["Field_eq_Possession"] == 1) & (playdir == -1), "Start"] = 120 - X.loc[(X["Field_eq_Possession"] == 1) & (playdir == -1), 
                                                                                       "YardLine"] - 10
    X.loc[(X["Field_eq_Possession"] == 0) & (playdir == 1), "Start"] = 120 - X.loc[(X["Field_eq_Possession"] == 0) & (playdir == 1), 
                                                                                       "YardLine"] - 10
    X.loc[(X["Field_eq_Possession"] == 0) & (playdir == -1), "Start"] = X.loc[(X["Field_eq_Possession"] == 0) & (playdir == -1), 
                                                                                       "YardLine"] + 10
    X['Orientation'] = 2 * np.pi * (90 - X['Orientation']) / 360
    X['locX'] = (X['X'].values - X['Start'].values) * playdir
    X['locY'] = X['Y'].values - 53.3 / 2
    X['velX'] = X['S'].values * np.cos(X['Orientation'].values) * playdir
    X['velY'] = X['S'].values * np.sin(X['Orientation'].values)
    X['accX'] = X['A'].values * np.cos(X['Orientation'].values) * playdir
    X['accY'] = X['A'].values * np.sin(X['Orientation'].values)
    
    i_cols = ['VisitorScoreBeforePlay','HomeScoreBeforePlay','YardLine']
    uids = ['DisplayName']
    aggregations = ['mean','std']
    X_agg = uid_aggregation(df, i_cols, uids, aggregations)
    X = pd.concat([X, X_agg], axis=1)

    return X


# In[ ]:


train_single = train_df.copy()
train_single = train_single.loc[train_single.NflId == train_single.NflIdRusher, :]
train_single['own_field'] = 1 * (train_single['FieldPosition'].values == train_single['PossessionTeam'].values)
dist_to_end_train = train_single.apply(lambda x:(100 - x.loc['YardLine']) if x.loc['own_field']==1 else x.loc['YardLine'],axis=1)


# In[ ]:


play = preprocess(train_df)
print(play.shape)
play.head()


# In[ ]:


rm_cols = ['index','GameId','PlayId']
features = [c for c in train_df.columns.values if c not in rm_cols]
print("There are {} features".format(len(features)))
print(features)


# In[ ]:


play = play[features]
print(play.shape)
play.head()


# In[ ]:


play['is_run'] = play.NflId == play.NflIdRusher
play = play[play.is_run==True]


# In[ ]:


y_train = play["Yards"]
X_train = play.drop(['Yards'],axis=1)

y_train = y_train.reset_index(drop=True, inplace=False)
X_train = X_train.reset_index(drop=True, inplace=False)


# ## CatBoost

# In[ ]:


categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
features = X_train.columns.values
for c in categorical_features_indices:
    X_train[features[c]] = X_train[features[c]].fillna("nan")


# In[ ]:


model = CatBoostRegressor(iterations=2000, 
                          learning_rate = 0.01,
                          use_best_model = True,
                          eval_metric = 'RMSE',
                          loss_function = 'RMSE',
                          cat_features = categorical_features_indices,
                          boosting_type = 'Ordered', # or Plain
                          verbose = 0)                                     


# In[ ]:


# train, test split
trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

# model training
model.fit(trainX, trainY, eval_set=[(trainX, trainY)], early_stopping_rounds=100)

# feature importance
importance = model.get_feature_importance()
ranking = np.argsort(-importance)
fig, ax = plt.subplots(figsize=(20, 20))
sns.barplot(x=importance[ranking], y=X_train.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()


# If we split the plot into one with categorical and the other with numeric...

# In[ ]:


# categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
numeric_features_indices = np.where(X_train.dtypes == np.float)[0]

fig, ax = plt.subplots(1, 2, figsize=(20, 20))
ax = ax.flatten()
sns.barplot(x=importance[categorical_features_indices], 
            y=X_train.columns.values[categorical_features_indices], 
            orient='h', ax=ax[0])
ax[0].set_xlabel("feature importance")
ax[0].set_xlim([0, 50])
ax[0].set_title("categorical features")
sns.barplot(x=importance[numeric_features_indices], 
            y=X_train.columns.values[numeric_features_indices], 
            orient='h', ax=ax[1])
ax[1].set_xlabel("feature importance")
ax[1].set_xlim([0, 50])
ax[1].set_title("numeric features")
plt.tight_layout()


# "Season" and "Yardline" are relatively importanct, but overall it seems that categorical variables in this dataset are not very important for the prediction...

# The following submission code is based on https://www.kaggle.com/newbielch/lgbm-regression-view.

# In[ ]:


def get_cdf_df(yards_array):
    pdf, edges = np.histogram(yards_array, bins=199,
                 range=(-99,100), density=True)
    cdf = pdf.cumsum().clip(0, 1)
    cdf_df = pd.DataFrame(data=cdf.reshape(-1, 1).T, 
                            columns=['Yards'+str(i) for i in range(-99,100)])
    return cdf_df
cdf = get_cdf_df(y_train).values.reshape(-1,)

def get_score(y_pred,cdf,w,dist_to_end):
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
    for a,b,c in zip(y_preds, y_trues,dist_to_ends):
        tmp.append(get_score_pingyi1(a,b,cdf,w,c))
    return np.mean(tmp)


# In[ ]:


n_splits = 2
kf=KFold(n_splits = n_splits)
resu1 = 0
impor1 = 0
resu2_cprs = 0
resu3_mae=0
##y_pred = 0
stack_train = np.zeros([X_train.shape[0],])
models = []
for train_index, test_index in kf.split(X_train, y_train):
    # split
    X_train2= X_train.iloc[train_index,:]
    y_train2= y_train.iloc[train_index]
    X_test2= X_train.iloc[test_index,:]
    y_test2= y_train.iloc[test_index]
    
    # catboost
    model = CatBoostRegressor(iterations=2000, 
                          learning_rate = 0.01,
                          use_best_model = True,
                          eval_metric = 'RMSE',
                          loss_function = 'RMSE',
                          cat_features = categorical_features_indices,
                          boosting_type = 'Ordered', # or Plain
                          verbose = 0)   
    model.fit(X_train2,y_train2,eval_set = [(X_train2,y_train2),(X_test2,y_test2)], early_stopping_rounds=100)
    
    # cv
    models.append(model)
    temp_predict = model.predict(X_test2)
    stack_train[test_index] = temp_predict
    ##y_pred += clf.predict(X_test)/5
    mse = mean_squared_error(y_test2, temp_predict)
    crps = CRPS_pingyi1(temp_predict,y_test2,4,cdf,dist_to_end_train.iloc[test_index])
    mae = mean_absolute_error(y_test2, temp_predict)
    print(crps)
    
    resu1 += mse/n_splits
    resu2_cprs += crps/n_splits
    resu3_mae += mae/n_splits 
    impor1 += model.feature_importances_/n_splits
    gc.collect()
print('mean mse:',resu1)
print('oof mse:',mean_squared_error(y_train,stack_train))
print('mean mae:',resu3_mae)
print('oof mae:',mean_absolute_error(y_train,stack_train))
print('mean cprs:',resu2_cprs)
print('oof cprs:',CRPS_pingyi1(stack_train,y_train,4,cdf,dist_to_end_train))


# In[ ]:


env = nflrush.make_env()


# In[ ]:


for (test_df, sample_prediction_df) in env.iter_test():
    
    test_df["HomeTeamAbbr"] = test_df["HomeTeamAbbr"].map(map_abbr)
    test_df["VisitorTeamAbbr"] = test_df["VisitorTeamAbbr"].map(map_abbr)
    test_df["Possession"] = test_df["PossessionTeam"].map(map_abbr)
    test_single = test_df.copy()
    test_single['own_field'] = 1 * (test_single['FieldPosition'] == test_single['PossessionTeam'])
    dist_to_end_test = test_single.apply(lambda x:(100 - x.loc['YardLine']) if x.loc['own_field']==1 else x.loc['YardLine'],axis=1)
    X_test = preprocess(test_df)
    X_test['is_run'] = X_test.NflId == X_test.NflIdRusher
    X_test = X_test[features]    
    for c in categorical_features_indices:
        X_test[features[c]] = X_test[features[c]].fillna("nan")
    pred_value = 0
    for model in models:
        pred_value += model.predict(X_test)[0]/n_splits
    pred_data = list(get_score(pred_value,cdf,4,dist_to_end_test.values[0]))
    pred_data = np.array(pred_data).reshape(1,199)
    pred_target = pd.DataFrame(index = sample_prediction_df.index,                                columns = sample_prediction_df.columns,                                #data = np.array(pred_data))
                               data = pred_data)
    #print(pred_target)
    env.predict(pred_target)
env.write_submission_file()
    


# In[ ]:


# kf=KFold(n_splits = 5)
# resu1 = 0
# impor1 = 0
# ##y_pred = 0
# stack_train = np.zeros([X_train.shape[0],])
# for train_index, test_index in kf.split(X_train, y_train):
#     X_train2= X_train.iloc[train_index,:]
#     y_train2= y_train.iloc[train_index]
#     X_test2= X_train.iloc[test_index,:]
#     y_test2= y_train.iloc[test_index]
#     clf = lgb.LGBMRegressor(n_estimators=10000, random_state=47,subsample=0.7,
#                              colsample_bytree=0.7,learning_rate=0.03,importance_type = 'gain',
#                      max_depth = -1, num_leaves = 256,min_child_samples=20,min_split_gain = 0.001,
#                        bagging_freq=1,reg_alpha = 0,reg_lambda = 0,n_jobs = -1)
#     clf.fit(X_train2,y_train2,eval_set = [(X_train2,y_train2),(X_test2,y_test2)],early_stopping_rounds=100,verbose=50)
#     temp_predict = clf.predict(X_test2)
#     stack_train[test_index] = temp_predict
#     ##y_pred += clf.predict(X_test)/5
#     mse = mean_squared_error(y_test2, temp_predict)
#     print(mse)
#     resu1 += mse/5
#     impor1 += clf.feature_importances_/5
#     gc.collect()

