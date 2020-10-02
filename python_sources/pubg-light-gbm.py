#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_Train = pd.read_csv("../input/train_V2.csv")

df_Test = pd.read_csv("../input/test_V2.csv")

print("Train data set :\n",df_Train.head())
print("Test data set :\n", df_Test.head())


# In[ ]:


# Copied from another kernel
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
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

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[ ]:


df_Train = reduce_mem_usage(df_Train)
df_Test = reduce_mem_usage(df_Test)


# In[ ]:


df_Train.info()


# In[ ]:


total = df_Train.isnull().sum().sort_values(ascending= False)
percent_1= df_Train.isnull().sum()/df_Train.isnull().count()*100
percent_2= round(percent_1,1).sort_values(ascending= False)
missing_data = pd.concat([total,percent_2], axis= 1, keys = ['Total', '%']) 
print(missing_data)


# In[ ]:


df_Trial = df_Train[df_Train['winPlacePerc'].isna()==True]
print(df_Trial)


# In[ ]:


df_Train.loc[df_Train['winPlacePerc'].isna()==True, 'winPlacePerc']= 0.5


# In[ ]:


ColumnList = ['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints',
       'winPlacePerc']


# In[ ]:


# Align with maxPlace
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
subset = df_Train.loc[df_Train.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_Train.loc[df_Train.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case
df_Train.loc[(df_Train.maxPlace > 1) & (df_Train.numGroups == 1), "winPlacePerc"] = 0
assert df_Train["winPlacePerc"].isnull().sum() == 0

#df_sub[["Id", "winPlacePerc"]].to_csv("submission_adjusted.csv", index=False)


# In[ ]:


df_Train= df_Train.set_index(['Id'])
df_Test= df_Test.set_index(['Id'])

df_Train = df_Train[ColumnList]



# In[ ]:


ColumnList.remove('winPlacePerc')


# In[ ]:


df_Test = df_Test[ColumnList]


# In[ ]:


X_Column = ColumnList
Y_Column = 'winPlacePerc'
X_Train = df_Train[X_Column]
Y_Train = df_Train[Y_Column]
X_test = df_Test[X_Column]


# In[ ]:


# Courtesy Koon-Hi-Koom
X_Train.loc[X_Train['headshotKills'].abs()> 0,'headshotrate']= X_Train['kills']/X_Train['headshotKills'] 
X_Train.loc[X_Train['headshotKills'].abs()== 0,'headshotrate']= 0

X_test.loc[X_test['headshotKills'].abs()> 0,'headshotrate']= X_test['kills']/X_test['headshotKills'] 
X_test.loc[X_test['headshotKills'].abs()== 0,'headshotrate']= 0

X_Train.loc[X_Train['kills'].abs()> 0,'killStreakrate']= X_Train['killStreaks']/X_Train['kills']
X_Train.loc[X_Train['kills'].abs()== 0,'killStreakrate']= 0

X_test.loc[X_test['kills'].abs()> 0,'killStreakrate']= X_test['killStreaks']/X_test['kills']
X_test.loc[X_test['kills'].abs()== 0,'killStreakrate']= 0

X_Train['healthitems'] = X_Train['heals'] + X_Train['boosts']
X_test['healthitems'] = X_test['heals'] + X_test['boosts']


# In[ ]:


X_Train.columns


# In[ ]:


X_Train = X_Train**0.5
X_test= X_test**0.5


# In[ ]:


ColumnList = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills',
       'killPlace', 'killPoints', 'kills', 'killStreaks', 'longestKill',
       'maxPlace', 'numGroups', 'revives', 'rideDistance', 'roadKills',
       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
       'weaponsAcquired', 'winPoints', 'headshotrate',
       'killStreakrate', 'healthitems']
X_Train = X_Train[ColumnList]
X_test = X_test[ColumnList]


# In[ ]:


#from sklearn.preprocessing import StandardScaler
#sc_X= StandardScaler()
#X_Train = sc_X.fit_transform(X_Train)
#X_test= sc_X.transform(X_test)


# In[ ]:


import lightgbm as lgb


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


import gc
folds = KFold(n_splits=3,random_state=6)
oof_preds = np.zeros(X_Train.shape[0])
sub_preds = np.zeros(X_test.shape[0])


valid_score = 0

feature_importance_df = pd.DataFrame()

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_Train, Y_Train)):
    trn_x, trn_y = X_Train.iloc[trn_idx], Y_Train[trn_idx]
    val_x, val_y = X_Train.iloc[val_idx], Y_Train[val_idx]    
    
    train_data = lgb.Dataset(data=trn_x, label=trn_y)
    valid_data = lgb.Dataset(data=val_x, label=val_y)   
    
    params = {"objective" : "regression", "metric" : "mae", 'n_estimators':15000, 'early_stopping_rounds':200,
              "num_leaves" : 31, "learning_rate" : 0.05, "bagging_fraction" : 0.9,
               "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7
             }
    
    lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000) 
    
    oof_preds[val_idx] = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
    oof_preds[oof_preds>1] = 1
    oof_preds[oof_preds<0] = 0
    sub_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) 
    sub_pred[sub_pred>1] = 1 # should be greater or equal to 1
    sub_pred[sub_pred<0] = 0 
    sub_preds += sub_pred/ folds.n_splits
    print('Fold %2d MAE : %.6f' % (n_fold + 1, mean_absolute_error(val_y, oof_preds[val_idx])))
    valid_score += mean_absolute_error(val_y, oof_preds[val_idx])
    
    #fold_importance_df = pd.DataFrame()
    #fold_importance_df["feature"] = train_columns
    #fold_importance_df["importance"] = lgb_model.feature_importance()
    #fold_importance_df["fold"] = n_fold + 1
    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    gc.collect()


# In[ ]:


#%%time
#d_train = lgb.Dataset(X_Train, label=Y_Train)
#params = {}
#params['learning_rate'] = 0.05
#params['boosting_type'] = 'gbdt'
#params['objective'] = 'regression'
#params['metric'] = 'mae'
#params['sub_feature'] = 0.9
#params['num_leaves'] = 500
#params['min_data'] = 1
#params['max_depth'] = 30
#params['min_gain_to_split']= 0.00001
#clf = lgb.train(params, d_train, 1000)
#print('Fold %2d MAE : %.6f' % (n_fold + 1, mean_absolute_error(val_y, oof_preds[val_idx])))


# #y_pred=lgb_model.predict(X_test)
# for i in range(len(df_Test)):
#     winPlacePerc = pred[i]
#     maxPlace = int(df_Test.iloc[i]['maxPlace'])
#     if maxPlace == 0:
#         winPlacePerc = 0.0
#     elif maxPlace == 1:
#         winPlacePerc = 1.0
#     else:
#         gap = 1.0 / (maxPlace - 1)
#         winPlacePerc = round(winPlacePerc / gap) * gap
#     
#     if winPlacePerc < 0: winPlacePerc = 0.0
#     if winPlacePerc > 1: winPlacePerc = 1.0    
#     pred[i] = winPlacePerc
# 
#     if (i + 1) % 100000 == 0:
#         print(i, flush=True, end=" ")

# In[ ]:


df_output = df_Test
pred = sub_preds
print("fix winPlacePerc")
df_output['winPlacePerc'] = pred
df_output.loc[df_output['maxPlace']==0, 'winPlacePerc']= 0.0
df_output.loc[df_output['maxPlace']==1, 'winPlacePerc']= 1.0
df_output.loc[df_output['winPlacePerc']<= 0.0, 'winPlacePerc']= 0.0
df_output.loc[df_output['winPlacePerc']>= 1.0, 'winPlacePerc']= 1.0
df_output.loc[(df_output['winPlacePerc']< 1.0) & (df_output['winPlacePerc']> 1.0), 'winPlacePerc']= round(df_output['winPlacePerc']/(1.0/(df_output['maxPlace']-1)))* (1.0/(df_output['maxPlace']-1))

df_output= df_output[[Y_Column]]
print(df_output.head())


# In[ ]:





# In[ ]:


df_output= df_output.reset_index()

df_output.to_csv('submission.csv', index= False)

