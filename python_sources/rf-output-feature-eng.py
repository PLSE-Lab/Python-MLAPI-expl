#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
np.random.seed(1527)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import os
import gc, sys
gc.enable()

# Any results you write to the current directory are saved as output.

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

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

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def preProcess(is_train=True,debug=True):\n    test_idx = None\n    if is_train: \n        print("processing train.csv")\n        if debug == True:\n            df = pd.read_csv(\'../input/train_V2.csv\', nrows=10000)\n        else:\n            df = pd.read_csv(\'../input/train_V2.csv\')           \n\n        df = df[df[\'maxPlace\'] > 1]\n    else:\n        print("processing test.csv")\n        df = pd.read_csv(\'../input/test_V2.csv\')\n        test_idx = df.Id\n        \n    df[\'totalDistance\'] = df[\'rideDistance\'] + df[\'walkDistance\'] + df[\'swimDistance\']\n    df[\'kills_assists\'] = (df[\'kills\'] + df[\'assists\'])    \n    df[\'killsWithoutMoving\'] = ((df[\'kills\'] > 0) & (df[\'totalDistance\'] == 0))\n    df[\'healthitems\'] = df[\'heals\'] + df[\'boosts\']\n    \n    df = reduce_mem_usage(df)\n\n    print("remove some columns")\n    target = \'winPlacePerc\'\n    features = list(df.columns)\n    features.remove("Id")\n    features.remove("matchId")\n    features.remove("groupId")\n    \n    features.remove("matchType")\n        \n    y = None\n    \n    \n    if is_train: \n        print("get target")\n        y = np.array(df[target])\n        features.remove(target)\n\n    print("get group mean feature")\n    agg = df.groupby([\'matchId\',\'groupId\'])[features].agg(\'mean\')\n    agg_rank = agg.groupby(\'matchId\')[features].rank(pct=True).reset_index()\n    \n    df_out = df[[\'matchId\',\'groupId\']]\n\n    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how=\'left\', on=[\'matchId\', \'groupId\'])\n    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how=\'left\', on=[\'matchId\', \'groupId\'])\n        \n    print("get group max feature")\n    agg = df.groupby([\'matchId\',\'groupId\'])[features].agg(\'max\')\n    agg_rank = agg.groupby(\'matchId\')[features].rank(pct=True).reset_index()\n    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how=\'left\', on=[\'matchId\', \'groupId\'])\n    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how=\'left\', on=[\'matchId\', \'groupId\'])\n    \n    print("get group min feature")\n    agg = df.groupby([\'matchId\',\'groupId\'])[features].agg(\'min\')\n    agg_rank = agg.groupby(\'matchId\')[features].rank(pct=True).reset_index()\n    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how=\'left\', on=[\'matchId\', \'groupId\'])\n    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how=\'left\', on=[\'matchId\', \'groupId\'])\n\n    \n    print("get group size feature")\n    agg = df.groupby([\'matchId\',\'groupId\']).size().reset_index(name=\'group_size\')\n    df_out = df_out.merge(agg, how=\'left\', on=[\'matchId\', \'groupId\'])\n    \n    print("get match mean feature")\n    agg = df.groupby([\'matchId\'])[features].agg(\'mean\').reset_index()\n    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how=\'left\', on=[\'matchId\'])\n        \n    print("get match size feature")\n    agg = df.groupby([\'matchId\']).size().reset_index(name=\'match_size\')\n    df_out = df_out.merge(agg, how=\'left\', on=[\'matchId\'])\n    df_out= reduce_mem_usage(df_out)\n    print("get match Type feature")\n    \'\'\'\n    agg=df[[\'matchId\',\'matchType\']]\n    agg = agg.drop_duplicates()\n    agg[\'matchType\'] = agg[\'matchType\'].astype(\'category\')\n    agg[\'matchType\'] = agg[\'matchType\'].cat.codes\n    df_out = df_out.merge(agg, how=\'left\', on=[\'matchId\'])\n    \'\'\'\n\n    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)\n    \n    X = df_out\n    \n    feature_names = list(df_out.columns)\n\n    del df, df_out, agg, agg_rank\n    gc.collect()\n\n    return X, y, feature_names, test_idx\n\nx_train, y_train, train_columns, _ = preProcess(True,False)\nx_test, _, _ , test_idx = preProcess(False,True)\n\n')


# In[ ]:


# Split the train and the validation set for the fitting
random_seed=1
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)


# In[ ]:


# Random Forest Model definition
model = RandomForestRegressor(n_estimators=20, min_samples_leaf=5, max_features=0.5,
                          n_jobs=-1,verbose=2)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(x_train, y_train)\n')


# In[ ]:


# train set and validation set predictions
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
print('mae train: ', mean_absolute_error(y_train_pred, y_train))
print('mae validation: ', mean_absolute_error(y_val_pred, y_val))


# In[ ]:


# Actual vs Prediction distribution histogram
plt.subplot(2, 1, 1)
plt.hist(y_val_pred)

plt.subplot(2, 1, 2)
plt.hist(y_val)


# In[ ]:


df_test = pd.read_csv('../input/' + 'test_V2.csv')

pred = model.predict(x_test)

df_test['winPlacePerc'] = pred

submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission_rf_feat1.csv', index=False)


# In[ ]:





# In[ ]:




