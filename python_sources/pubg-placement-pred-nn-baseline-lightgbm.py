#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import optimizers

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb


# # Load train data, and reduce memory

# In[ ]:


INPUT_DIR = "../input/"
LABEL = 'winPlacePerc'


# In[ ]:


df_train = pd.read_csv(INPUT_DIR+'train_V2.csv')


# In[ ]:


# credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
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

    return df


# In[ ]:


df_train = reduce_mem_usage(df_train)
gc.collect()


# # EDA

# ## Data type, and bad value 

# In[ ]:


df_train.dtypes


# In[ ]:


df_train.isnull().any()


# There are missing values. 

# In[ ]:


df_train = df_train.dropna()


# In[ ]:


# there are matches with no players or just one player, which is abnormal
df_train = df_train[df_train['maxPlace'] > 1]


# # Feature Engineering

# ### The final placement depends on the performance of the group and the match dynamics, not individuals.  So first, get all the group and match features.
# 
# from group:
# - group size. More people, more chance to rank high
# - the summary of group performace: the max,mean,min of each variant, except of id, groupId, matchId, numGroup, match duration, martch type, maxPlace
# 
# from match:
# - **mean, max, and min of group features except group size**
# - numGroup. The smaller, the better
# - match duration
# - match type
# - maxPlace

# In[ ]:


def FE(df,train=True):
    LABEL = 'winPlacePerc'
    
    
    # get label data
    if train:
        df_y = df.groupby(['matchId','groupId'])[LABEL].agg('mean')
        ## now we can delete label and 'id' column to save GPU
        df = df.drop([LABEL],axis=1)
    else:
        df_ids = df[['Id','matchId','groupId']]
        
    df = df.drop('Id',axis=1)
    
    # define group and match features
    MATCH_FEATURE_part = ['numGroups','matchDuration','matchType','maxPlace']
    
    GROUP_FEATURE = df.columns.tolist()
    GROUP_FEATURE.remove('groupId')
    GROUP_FEATURE.remove('matchId')
    for fe in MATCH_FEATURE_part:
        GROUP_FEATURE.remove(fe)
    # ATTENTION: here group feature doesn't include 'groupSize'
    MATCH_FEATURE = MATCH_FEATURE_part + GROUP_FEATURE
    MATCH_FEATURE.remove('matchType') 
    
    # get group features
    ## group size
    dm = df.groupby(['matchId','groupId'])
    df_X = dm.size().to_frame(name='groupSize') #df_X has indices: matchid, groupid
    
    ## other group features
    gp = dm[GROUP_FEATURE].agg('max')
    gp_rank = gp.reset_index().groupby(['matchId'])[GROUP_FEATURE].rank(pct=True).set_index(df_X.index)
    df_X = df_X.join(gp).join(gp_rank,rsuffix='_max_rank') # join by indices
    
    gp = dm[GROUP_FEATURE].agg('min')
    gp_rank = gp.reset_index().groupby(['matchId'])[GROUP_FEATURE].rank(pct=True).set_index(df_X.index)
    df_X = df_X.join(gp,rsuffix='_min').join(gp_rank,rsuffix='_min_rank') # join by indices
    
    gp = dm[GROUP_FEATURE].agg('mean')
    gp_rank = gp.reset_index().groupby(['matchId'])[GROUP_FEATURE].rank(pct=True).set_index(df_X.index)
    df_X = df_X.join(gp,rsuffix='_mean').join(gp_rank,rsuffix='_mean_rank') # join by indices

    #a variable called killPlace, it's already the rank in the match, so the _rank are all duplicates, so need to delete
    df_X = df_X.drop(columns = ['killPlace_min_rank','killPlace_max_rank','killPlace_mean_rank'])
    
    # get match features except mactchType
    dm = df.groupby(['matchId'])
    df_X = df_X.join(dm[MATCH_FEATURE].agg('mean'),lsuffix='_max',rsuffix='_mean_match')
    
    # get matchType
    df_X = df_X.join(pd.concat([pd.get_dummies(df.matchType),df[['matchId','groupId']]],axis=1).groupby(['matchId','groupId']).agg('min'))
    
    # prepare for output
    if not train:
        df_X_index = df_X.reset_index()[['matchId','groupId']]
    
    lst_features = list(df_X.columns)
    
    del df,dm,gp,gp_rank
    gc.collect()
    
    if train:
        return df_X,df_y,lst_features
    else:
        return df_X, df_X_index, df_ids
    


# In[ ]:


X_train,y_train,lst_features = FE(df_train)


# In[ ]:


list(X_train.columns)


# In[ ]:


X_train.shape, y_train.shape


# 

# # NN using Keras

# ### Split train data to train and validation

# In[ ]:


# X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)


# ### Feature-wise normalization

# In[ ]:


# mean = X_train.mean(axis=0)
# X_train -= mean
# std = X_train.std(axis=0)
# X_train /= std

# X_vali -= mean
# X_vali /= std


# ### Build the model

# In[ ]:


# nn_model = Sequential()
# nn_model.add(Dense(512,input_dim= X_train.shape[1], activation='relu'))
# nn_model.add(Dropout(0.1))
# nn_model.add(Dense(256, activation='relu'))
# nn_model.add(Dropout(0.1))
# nn_model.add(Dense(128, activation='relu'))
# nn_model.add(Dropout(0.1))
# nn_model.add(Dense(1,activation='linear')) 

# nn_model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae'])
# nn_model.summary()


# In[ ]:


# history = nn_model.fit(X_train, y_train, 
#                  validation_data=(X_vali,y_vali),
#                  epochs=10,
#                  batch_size=10000,
#                  verbose=1)
# del X_train, y_train
# gc.collect()


# In[ ]:


# from keras.models import load_model

# nn_model.save('NN_model.h5')  # creates a HDF5 file


# 
# ### Visualization

# In[ ]:


# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # Plot training & validation mae values
# plt.plot(history.history['mean_absolute_error'])
# plt.plot(history.history['val_mean_absolute_error'])
# plt.title('Mean Abosulte Error')
# plt.ylabel('Mean absolute error')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()


# ### Make prediction Using NN

# In[ ]:


# df_test = pd.read_csv(INPUT_DIR+'test_V2.csv')
# df_test = reduce_mem_usage(df_test)
# gc.collect()
# X_test, X_test_index,df_test_ids = FE(df_test,train=False)
# X_test.shape


# In[ ]:


# del df_test
# gc.collect()


# In[ ]:


# # need to normailize for NN
# X_test -= mean
# X_test /= std


# In[ ]:


# pred = nn_model.predict(X_test)
# pred = np.clip(pred, a_min=0, a_max=1)
# df_pred = X_test_index.assign(winPlacePerc=pred)
# result = pd.merge(df_test_ids, df_pred, how='left', on=['matchId', 'groupId'])


# In[ ]:


# submission = result[['Id', LABEL]]
# submission.to_csv('submission.csv', index=False)
# submission.head()


# ## LightGBM

# ### prep

# In[ ]:


# lgb_model = lgb.Booster(model_file='lgb_model.txt')  #init model


# In[ ]:


del df_train
gc.collect()


# In[ ]:


df_test = pd.read_csv(INPUT_DIR+'test_V2.csv')
df_test = reduce_mem_usage(df_test)
gc.collect()


# In[ ]:


X_test, X_test_index,df_test_ids = FE(df_test,train=False)


# In[ ]:


# prep for interatoion on every fold; initialize the value
folds = KFold(n_splits=3,random_state=3)
vali_pred = np.zeros(X_train.shape[0])
pred = np.zeros(X_test.shape[0]) #final pred
df_feature_importance = pd.DataFrame()
valid_score = 0


# ### define model

# In[ ]:


params = {"objective" : "regression", "metric" : "mae", 'n_estimators':20000, 'early_stopping_rounds':100,
          "num_leaves" : 25, "learning_rate" : 0.05, "bagging_fraction" : 0.9, "feature_fraction":0.7,
           "bagging_seed" : 0, "num_threads" : 4
         }


# In[ ]:


for n_fold, (train_idx, vali_idx) in enumerate(folds.split(X_train, y_train)): # for each fold
    # split train data
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train[train_idx]
    X_vali, y_vali = X_train.iloc[vali_idx], y_train[vali_idx]    
    
    #build model, fit
    train_data = lgb.Dataset(data=X_train_fold, label=y_train_fold)
    valid_data = lgb.Dataset(data=X_vali, label=y_vali)   
    
    lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000) 
    
    #predict
    pred_fold = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) 
    pred_fold[pred_fold>1] = 1 
    pred_fold[pred_fold<0] = 0 
    pred += pred_fold/ folds.n_splits
    
    #evaluate 1: check on the validation data
    vali_pred[vali_idx] = lgb_model.predict(X_vali, num_iteration=lgb_model.best_iteration)
    vali_pred[vali_pred>1] = 1
    vali_pred[vali_pred<0] = 0
    print('Fold %2d MAE : %.6f' % (n_fold + 1, mean_absolute_error(y_vali, vali_pred[vali_idx])))

    #evaluation 2: check the importance of each feature
    df_fold_importance = pd.DataFrame()
    df_fold_importance = df_fold_importance.assign(feature= lst_features)
    df_fold_importance = df_fold_importance.assign(importance = lgb_model.feature_importance())
    df_fold_importance = df_fold_importance.assign(fold = n_fold + 1)
    df_feature_importance = pd.concat([df_feature_importance, df_fold_importance])
    
    gc.collect()
    


# In[ ]:


print('Full mae score %.6f' % mean_absolute_error(y_train, vali_pred))


# In[ ]:


lgb_model.save_model('lgb_model.txt')


# In[ ]:


top_features = df_feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].reset_index()

plt.figure(figsize=(14,10))
sns.barplot(x="importance", y="feature", data=top_features)
plt.title('LightGBM_Features (avg over folds)')
plt.tight_layout()
plt.savefig('LightGBM_Importances.png')


# In[ ]:





# ### Make Prediction

# In[ ]:


pred = np.clip(pred, a_min=0, a_max=1)
df_pred = X_test_index.assign(winPlacePerc=pred)
result = pd.merge(df_test_ids, df_pred, how='left', on=['matchId', 'groupId'])


# In[ ]:


submission = result[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:





# In[ ]:





# In[ ]:




