#!/usr/bin/env python
# coding: utf-8

# # prediction group by match type
# Hello, it's my first public kernel. i will try to make prediction for each match type. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import math
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import gc

from sklearn import preprocessing
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.?


# In[ ]:


train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")


# In[ ]:


train = train.dropna()


# In[ ]:


train['rideDistance'] = (train['rideDistance']/10).round(0)
train['swimDistance'] = (train['swimDistance']/10).round(0)
train['walkDistance'] = (train['walkDistance']/10).round(0)


# In[ ]:


test['rideDistance'] = (test['rideDistance']/10).round(0)
test['swimDistance'] = (test['swimDistance']/10).round(0)
test['walkDistance'] = (test['walkDistance']/10).round(0)


# In[ ]:


#a Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df,display=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    if display:
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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

    if display:
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# # Split by matchType

# In this part, I merge test and train value to create each matchType dataset. To identified test value, I add winPlacePerc value at -1.

# In[ ]:


test["winPlacePerc"] = -1


# In[ ]:


df = pd.concat([train, test])


# In[ ]:


del train
del test


# In[ ]:


df["Id"] = df.index 


# I tried with different combination, second is better. I make dict with name of type and batch size. I serialize each dataset to release RAM.

# In[ ]:


#squad_fpp = df[(df['matchType']=='squad-fpp') | (df['matchType']=='normal-squad-fpp')]
#duo = df[(df['matchType']=='duo') | (df['matchType']=='normal-duo')]
#solo_fpp = df[(df['matchType']=='solo-fpp') | (df['matchType']=='normal-solo-fpp')]
#squad = df[(df['matchType']=='squad') | (df['matchType']=='normal-squad')]
#duo_fpp = df[(df['matchType']=='duo-fpp') | (df['matchType']=='normal-duo-fpp')]
#solo = df[(df['matchType']=='solo') | (df['matchType']=='normal-solo')]
#flare = df[(df['matchType']=='flaretpp')|(df['matchType']=='flarefpp')]
#crash = df[(df['matchType']=='crashtpp') | (df['matchType']=='crashfpp')]


# In[ ]:


#dp_by_type = {'squad_fpp':[squad_fpp,39525],
#              'flare':[flare,62],
#              'crash':[crash,149],
#              'solo_fpp':[solo_fpp,12098],
#              'duo_fpp':[duo_fpp,22586],
#              'squad':[squad,14110],
#              'solo':[solo,4067],
#              'duo':[duo,7105]
#              }


# In[ ]:


duo = df[(df['matchType']=='duo') | (df['matchType']=='normal-duo')|(df['matchType']=='duo-fpp') | (df['matchType']=='normal-duo-fpp')]
squad = df[(df['matchType']=='squad') | (df['matchType']=='normal-squad')|(df['matchType']=='squad-fpp') | (df['matchType']=='normal-squad-fpp')]
solo = df[(df['matchType']=='solo') | (df['matchType']=='normal-solo')|(df['matchType']=='solo-fpp') | (df['matchType']=='normal-solo-fpp')]
flare = df[(df['matchType']=='flaretpp')|(df['matchType']=='flarefpp')]
crash = df[(df['matchType']=='crashtpp') | (df['matchType']=='crashfpp')]


# In[ ]:


dp_by_type = {'flare':[flare,62],
              'crash':[crash,149],
              'squad':[squad,53635],
              'solo':[solo,16165],
              'duo':[duo,29691]
              }


# In[ ]:


for name,ele in dp_by_type.items():
    print(name + " : " + str(len(ele[0])))
    ele[0].to_csv(name+'.csv', index=False)
    ele[0] = 0


# In[ ]:


#del squad_fpp,duo,solo_fpp,squad,duo_fpp,solo,flare,crash


# In[ ]:


del duo,squad,solo,flare,crash


# In[ ]:


del df
gc.collect()


# # FeatureEngineering

# this part is for feature engineering. there are two part, one for make new feartures and the second to coralate data together.

# In[ ]:


def featureEngineering(df):
    return featureEngineeringSecond(reduce_mem_usage(featureEngineeringFirst(df)))


# In[ ]:


def items(df):
    df['items'] = df['heals'] + df['boosts']
    return df

def survival(df):
    df["survival"] = df["revives"] + df["boosts"] + df["heals"]
    return df

def players_in_team(df):
    agg = df.groupby(['groupId']).size().to_frame('players_in_team')
    return df.merge(agg, how='left', on=['groupId'])

def total_distance(df):
    df['total_distance'] = df['rideDistance'] + df['swimDistance'] + df['walkDistance']
    return df

def total_time_by_distance(df):
    df["total_time_by_distance"] = df["rideDistance"]/5+df["walkDistance"]+df["swimDistance"]*5
    return df

def headshotKills_over_kills(df):
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['headshotKills_over_kills'].fillna(0, inplace=True)
    return df

def teamwork(df):
    df['teamwork'] = df['assists'] + df['revives']
    return df

def total_items_acquired(df):
    df['total_items_acquired'] = df["boosts"] + df["heals"] + df["weaponsAcquired"]
    return df

def killPlace_over_maxPlace(df):
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['killPlace_over_maxPlace'].fillna(0, inplace=True)
    df['killPlace_over_maxPlace'].replace(np.inf, 0, inplace=True)
    return df

def distance_over_heals(df):
    df['walkDistance_over_heals'] = df['total_distance'] / df['heals']
    df['walkDistance_over_heals'].fillna(0, inplace=True)
    df['walkDistance_over_heals'].replace(np.inf, 0, inplace=True)
    return df

def distance_over_kills(df):
    df['walkDistance_over_kills'] = df['total_distance'] / df['kills']
    df['walkDistance_over_kills'].fillna(0, inplace=True)
    df['walkDistance_over_kills'].replace(np.inf, 0, inplace=True)
    return df

def headshot_kill_rate(df):
    df['headshot_kill_rate'] = (df['headshotKills']+1)/(df['kills']+1)
    return df


# In[ ]:


def featureEngineeringFirst(df):
    print("        Feature Engineering First started...")
        
    df = items(df)
    gc.collect()
        
    df = survival(df)
    gc.collect()
    
    df = players_in_team(df)
    gc.collect()
    
    df = total_distance(df)
    gc.collect()
    
    df = total_time_by_distance(df)
    gc.collect()
    
    #df = headshotKills_over_kills(df)
    gc.collect()

    df = teamwork(df)
    gc.collect()
    
    df = total_items_acquired(df)
    gc.collect()
    
    #df = killPlace_over_maxPlace(df)
    gc.collect()
    
    #df = distance_over_heals(df)
    gc.collect()
    
    #df = distance_over_kills(df)
    gc.collect()
    
    #df = headshot_kill_rate(df)
    gc.collect()
    
    df = reduce_mem_usage(df)
    
    print("        Feature Engineering First fished ")
    
    return df


# In[ ]:


def min_by_team(df):
    features = list(df.columns)
    features.remove('Id')
    features.remove('groupId')
    features.remove('matchId')
    agg = df.groupby(['matchId','groupId'])[features].min()
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    return agg, agg_rank

def max_by_team(df):
    features = list(df.columns)
    features.remove('Id')
    features.remove('groupId')
    features.remove('matchId')
    agg = df.groupby(['matchId', 'groupId'])[features].max()
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    return agg, agg_rank

def sum_by_team(df):
    features = list(df.columns)
    features.remove('Id')
    features.remove('groupId')
    features.remove('matchId')
    agg = df.groupby(['matchId', 'groupId'])[features].sum()
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    return agg, agg_rank

def median_by_team(df):
    features = list(df.columns)
    features.remove('Id')
    features.remove('groupId')
    features.remove('matchId')
    agg = df.groupby(['matchId', 'groupId'])[features].median()
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    return agg, agg_rank

def mean_by_team(df):
    features = list(df.columns)
    features.remove('Id')
    features.remove('groupId')
    features.remove('matchId')
    agg = df.groupby(['matchId', 'groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    return agg, agg_rank


# In[ ]:


def mergeWithAgg(df,agg,agg_rank,name):
    print("            Merge "+name)
    df = df.merge(agg, suffixes=["", "_"+name], how='left', on=['matchId', 'groupId'])
    df = df.merge(agg_rank, suffixes=["", "_"+name+"_rank"], how='left', on=['matchId', 'groupId'])
    return reduce_mem_usage(df)


# In[ ]:


def featureEngineeringSecond(df):
    print("        Feature Engineering Second started...")
    
    print("            Min")
    min_, min_rank = min_by_team(df)
    gc.collect()
    
    print("            Max")
    max_, max_rank = max_by_team(df)
    gc.collect()
    
    print("            Sum")
    sum_, sum_rank = sum_by_team(df)
    gc.collect()
    
    print("            Median")
    median_, median_rank = median_by_team(df)
    gc.collect()
    
    print("            Mean")
    mean_, mean_rank = mean_by_team(df)
    gc.collect()
    
    df = mergeWithAgg(df,min_, min_rank,"min")
    del min_, min_rank
    df = mergeWithAgg(df,max_, max_rank,"max")
    del max_, max_rank
    df = mergeWithAgg(df,sum_, sum_rank,"sum")
    del sum_, sum_rank
    df = mergeWithAgg(df,median_, median_rank,"median")
    del median_, median_rank
    df = mergeWithAgg(df,mean_, mean_rank,"mean")
    del mean_, mean_rank
        
    print("        Feature Engineering Second finished")
    
    return df


# # Learning part

# This part is for training and predicting. I use Sequential model from keras. 

# In[ ]:


def baseline_model(input_dim):
    model = Sequential()
    # create model
    model.add(Dense(32, kernel_initializer='he_normal',input_dim=input_dim , activation='selu'))
    model.add(Dense(64, kernel_initializer='he_normal', activation='selu'))
    model.add(Dense(128, kernel_initializer='he_normal', activation='selu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, kernel_initializer='he_normal', activation='selu'))
    model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer='he_normal', activation='selu'))
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='he_normal', activation='selu'))
    model.add(Dense(8, kernel_initializer='he_normal', activation='selu'))
    model.add(BatchNormalization())
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# In[ ]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if df[feature_name].dtype != object:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[ ]:


def learningPart(df,batch_size):
    print("    Data processing started...")
    df_winPlacePerc = df.winPlacePerc
    df = df.drop('winPlacePerc', axis=1)
    df = df.drop('matchType', axis=1)
    df = featureEngineering(df)
    df = df.drop('matchId', axis=1)
    df = df.drop('groupId', axis=1)
    gc.collect()
    df['winPlacePerc'] =  df_winPlacePerc.values
    
    del df_winPlacePerc 
    gc.collect()
    
    print("    Data scaling started...")
    df = normalize(df)
    print("    Data scaling finised")
    
    train_df = df[df["winPlacePerc"] != -1]
    train_df = train_df.dropna()
    gc.collect()
    test_df = df[df["winPlacePerc"] == -1]
    test_df = test_df.drop('winPlacePerc', axis=1)
    del df
    gc.collect()
    
    train_df_Y = train_df.winPlacePerc
    train_df_X = train_df.drop('winPlacePerc', axis=1)
    del train_df
    print("    Data processing finished")
    

    
    print("    Data spliting started...")
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(train_df_X, train_df_Y, test_size=0.2)
    del train_df_Y
    del train_df_X
    print("    Data spliting finished")
    
    print("    Model training started...")
    epochs = 80
    
    callbacks = [ModelCheckpoint('best_model_df.h5', verbose=0, monitor='val_loss',save_best_only=True, mode='auto')]
    
    model_df = baseline_model(X_train_df.shape[1])
    history_df = model_df.fit(X_train_df, y_train_df, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_test_df, y_test_df),callbacks=callbacks)
    gc.collect()    
    print("    Model training finished")
    
    print("    Prediction started...")
    best_model_df = load_model('best_model_df.h5')
    
    predict_y_df = best_model_df.predict(t_test_df)
    predict_y_df = pd.DataFrame({'Id':test_df['Id'].values,'winPlacePerc':predict_y_df.flatten()})
    print("    Prediction finished")
    
    return predict_y_df, history_df


# In[ ]:


def minmax(df):
    df_minmax = pd.DataFrame()
    for feature_name in df.columns:
        v_min = df[feature_name].min()
        v_max = df[feature_name].max()
        df_minmax[feature_name] = pd.Series([v_min,v_max])
    
    return df_minmax


# In[ ]:


def learningPart(df,batch_size):
    print("    Data processing started...")
    df = df.drop('matchType', axis=1)
    train_df = df[df["winPlacePerc"] != -1]
    gc.collect()
    test_df = df[df["winPlacePerc"] == -1]
    test_df = test_df.drop('winPlacePerc', axis=1)
    del df
    gc.collect()
    
    train_df_Y = train_df.winPlacePerc
    train_df_X = train_df.drop('winPlacePerc', axis=1)
    del train_df
    
    train_df_X = featureEngineering(train_df_X)
    train_df_X = train_df_X.drop('matchId', axis=1)
    train_df_X = train_df_X.drop('groupId', axis=1)
       
    print("    Data processing finished")
    
    
    print("    Data spliting started...")
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(train_df_X, train_df_Y, test_size=0.2)
    del train_df_Y
    del train_df_X
    gc.collect()
    print("    Data spliting finished")
    

    print("    Data test processing started...")
    test_df = featureEngineering(test_df)
    test_df = test_df.drop('matchId', axis=1)
    test_df = test_df.drop('groupId', axis=1) 
    
    print("    Data test processing finished")
    
    
    print("    Data scaling started...")
    minmaxv = pd.concat([minmax(test_df),minmax(X_train_df),minmax(X_test_df)])
    scaler_df = preprocessing.MinMaxScaler(feature_range=(-1, 1),copy=False).fit(minmaxv)
    del minmaxv
    gc.collect()
    X_train_df = scaler_df.transform(X_train_df)
    gc.collect()
    X_test_df = scaler_df.transform(X_test_df)
    gc.collect()
    t_test_df = scaler_df.transform(test_df)
    print("    Data scaling finised")
    
    print("    Model training started...")
    epochs = 80
    
    callbacks = [ModelCheckpoint('best_model_df.h5', verbose=0, monitor='val_loss',save_best_only=True, mode='auto')]
    
    model_df = baseline_model(X_train_df.shape[1])
    history_df = model_df.fit(X_train_df, y_train_df, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_test_df, y_test_df),callbacks=callbacks)
    gc.collect()    
    print("    Model training finished")
    
    print("    Prediction started...")
    best_model_df = load_model('best_model_df.h5')
    
    predict_y_df = best_model_df.predict(t_test_df)
    predict_y_df = pd.DataFrame({'Id':test_df['Id'].values,'winPlacePerc':predict_y_df.flatten()})
    print("    Prediction finished")
    
    return predict_y_df, history_df


# In[ ]:


result = {}
for name,typeGame in dp_by_type.items():
    print("Start "+ name)
    predict, history = learningPart(pd.read_csv(name+".csv"),typeGame[1])
    result[name] = [predict,history]
    print("End "+ name)
    dp_by_type[name] = 0


# # Result

# This part is for visualied training evolution.

# In[ ]:


def displayHistory(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation mae values
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean Abosulte Error')
    plt.ylabel('Mean absolute error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# In[ ]:


for key,ele in result.items():
    print("History for " + str(key))
    displayHistory(ele[1])


# # PART concat

# This part is to merge result and make submission.

# In[ ]:


data = []
for key,ele in result.items():
    data.append(ele[0])
    


# In[ ]:


df = pd.concat(data, ignore_index=True)
df = df.sort_values(by='Id', ascending=[True])
df = df.reset_index(drop=True)


# In[ ]:


submission = pd.read_csv("../input/sample_submission_V2.csv")


# In[ ]:


submission["winPlacePerc"] = df.winPlacePerc


# In[ ]:


submission.to_csv('submission.csv', index=False)


# Thank you for reading. If you have any questions/remarks, ask me.
