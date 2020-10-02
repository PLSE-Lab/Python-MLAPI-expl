#!/usr/bin/env python
# coding: utf-8

# For more details :
# http://fastml.com/adversarial-validation-part-one/
# https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms
# https://www.kaggle.com/ogrellier/adversarial-validation-and-lb-shakeup
# 
# Basic data processing from:
# https://www.kaggle.com/shahules/xgboost-feature-selection-dsbowl
# 
# **Some code snippet below is from other past competition kernels , but now I don't remember the owner, if you are the one please mention in comment and I will add your credit here :)**

# In[ ]:


# Load libraries
import numpy as np
import pandas as pd
import gc
import datetime
from scipy.stats import mode

from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn import preprocessing
# Params
NFOLD = 5
DATA_PATH = '../input/'


# In[ ]:


get_ipython().run_cell_magic('time', '', "keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count',\n             'event_code','title' ,'game_time', 'type', 'world','timestamp']\ntrain=pd.read_csv('../input/data-science-bowl-2019/train.csv',usecols=keep_cols)\ntrain_labels=pd.read_csv('../input/data-science-bowl-2019/train_labels.csv',\n                         usecols=['installation_id','game_session','accuracy_group'])\ntest=pd.read_csv('../input/data-science-bowl-2019/test.csv',usecols=keep_cols)\nsubmission=pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')")


# In[ ]:


not_req=(set(train.installation_id.unique()) - set(train_labels.installation_id.unique()))


# In[ ]:


train_new=~train['installation_id'].isin(not_req)
train.where(train_new,inplace=True)
train.dropna(inplace=True)
train['event_code']=train.event_code.astype(int)


# In[ ]:


def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekofyear'] = df['timestamp'].dt.weekofyear
    return df


# In[ ]:


time_features=['month','hour','year','dayofweek','weekofyear']
def prepare_data(df):
    df=extract_time_features(df)
    
    df=df.drop('timestamp',axis=1)
    #df['timestamp']=pd.to_datetime(df['timestamp'])
    #df['hour_of_day']=df['timestamp'].map(lambda x : int(x.hour))
    

    join_one=pd.get_dummies(df[['event_code','installation_id','game_session']],
                            columns=['event_code']).groupby(['installation_id','game_session'],
                                                            as_index=False,sort=False).agg(sum)

    agg={'event_count':sum,'game_time':['sum','mean'],'event_id':'count'}

    join_two=df.drop(time_features,axis=1).groupby(['installation_id','game_session']
                                                   ,as_index=False,sort=False).agg(agg)
    
    join_two.columns= [' '.join(col).strip() for col in join_two.columns.values]
    

    join_three=df[['installation_id','game_session','type','world','title']].groupby(
                ['installation_id','game_session'],as_index=False,sort=False).first()
    
    join_four=df[time_features+['installation_id','game_session']].groupby(['installation_id',
                'game_session'],as_index=False,sort=False).agg(mode)[time_features].applymap(lambda x: x.mode[0])
    
    join_one=join_one.join(join_four)
    
    join_five=(join_one.join(join_two.drop(['installation_id','game_session'],axis=1))).                         join(join_three.drop(['installation_id','game_session'],axis=1))
    
    return join_five


# In[ ]:


join_train=prepare_data(train)
cols=list(join_train.columns)[2:-3]
join_train[cols]=join_train[cols].astype('int16')


# In[ ]:


join_test=prepare_data(test)
cols=list(join_test.columns)[2:-3]
join_test[cols]=join_test[cols].astype('int16')


# In[ ]:


cols=list(join_test.columns[2:-12])
cols.append('event_id count')
cols.append('installation_id')


# In[ ]:


df=join_test[['event_count sum','game_time mean','game_time sum',
    'installation_id']].groupby('installation_id',as_index=False,sort=False).agg('mean')

df_two=join_test[cols].groupby('installation_id',as_index=False,
                               sort=False).agg('sum').drop('installation_id',axis=1)

df_three=join_test[['title','type','world','installation_id']].groupby('installation_id',
         as_index=False,sort=False).last().drop('installation_id',axis=1)

df_four=join_test[time_features+['installation_id']].groupby('installation_id',as_index=False,sort=False).         agg(mode)[time_features].applymap(lambda x : x.mode[0])


# In[ ]:


final_train=pd.merge(train_labels,join_train,on=['installation_id','game_session'],
                                         how='left').drop(['game_session'],axis=1)

#final_test=join_test.groupby('installation_id',as_index=False,sort=False).last().drop(['game_session','installation_id'],axis=1)
final_test=(df.join(df_two)).join(df_three.join(df_four)).drop('installation_id',axis=1)


# In[ ]:


df=final_train[['event_count sum','game_time mean','game_time sum','installation_id']].     groupby('installation_id',as_index=False,sort=False).agg('mean')

df_two=final_train[cols].groupby('installation_id',as_index=False,
                                 sort=False).agg('sum').drop('installation_id',axis=1)

df_three=final_train[['accuracy_group','title','type','world','installation_id']].         groupby('installation_id',as_index=False,sort=False).         last().drop('installation_id',axis=1)

df_four=join_train[time_features+['installation_id']].groupby('installation_id',as_index=False,sort=False).         agg(mode)[time_features].applymap(lambda x : x.mode[0])



final_train=(df.join(df_two)).join(df_three.join(df_four)).drop('installation_id',axis=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
final=pd.concat([final_train,final_test])
encoding=['type','world','title']
for col in encoding:
    lb=LabelEncoder()
    lb.fit(final[col])
    final[col]=lb.transform(final[col])
    
final_train=final[:len(final_train)]
final_test=final[len(final_train):]


# In[ ]:


final_train.shape,final_test.shape


# In[ ]:


set(final_train.columns) - set(final_test.columns)


# In[ ]:


predictors = list(final_train.columns)


# In[ ]:


#train.drop('accuracy_group', axis=1,inplace=True)
# Mark train as 1, test as 0
final_train['target'] = 1
final_test['target'] = 0

# Concat dataframes
n_train = final_train.shape[0]
df = pd.concat([final_train, final_test], axis = 0)
del final_train, final_test
gc.collect()


# In[ ]:


df.drop('accuracy_group',axis=1,inplace=True)


# In[ ]:


predictors.remove('accuracy_group')


# In[ ]:


# Prepare for training

# Shuffle dataset
df = df.iloc[np.random.permutation(len(df))]
df.reset_index(drop = True, inplace = True)

# Get target column name
target = 'target'

# lgb params
lgb_params = {
        'boosting': 'gbdt',
        'application': 'binary',
        'metric': 'auc', 
        'learning_rate': 0.1,
        'num_leaves': 32,
        'max_depth': 8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'feature_fraction': 0.7,
}

# Get folds for k-fold CV
folds = KFold(n_splits = NFOLD, shuffle = True, random_state = 0)
fold = folds.split(df)
    
eval_score = 0
n_estimators = 0
eval_preds = np.zeros(df.shape[0])


# In[ ]:


# Run LightGBM for each fold
for i, (train_index, test_index) in enumerate(fold):
    print( "\n[{}] Fold {} of {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i+1, NFOLD))
    train_X, valid_X = df[predictors].values[train_index], df[predictors].values[test_index]
    train_y, valid_y = df[target].values[train_index], df[target].values[test_index]

    dtrain = lgb.Dataset(train_X, label = train_y,
                          feature_name = list(predictors)
                          )
    dvalid = lgb.Dataset(valid_X, label = valid_y,
                          feature_name = list(predictors)
                          )
        
    eval_results = {}
    
    bst = lgb.train(lgb_params, 
                         dtrain, 
                         valid_sets = [dtrain, dvalid], 
                         valid_names = ['train', 'valid'], 
                         evals_result = eval_results, 
                         num_boost_round = 5000,
                         early_stopping_rounds = 100,
                         verbose_eval = 100)
    
    print("\nRounds:", bst.best_iteration)
    print("AUC: ", eval_results['valid']['auc'][bst.best_iteration-1])

    n_estimators += bst.best_iteration
    eval_score += eval_results['valid']['auc'][bst.best_iteration-1]
   
    eval_preds[test_index] += bst.predict(valid_X, num_iteration = bst.best_iteration)
    
n_estimators = int(round(n_estimators/NFOLD,0))
eval_score = round(eval_score/NFOLD,6)

print("\nModel Report")
print("Rounds: ", n_estimators)
print("AUC: ", eval_score)    


# In[ ]:


# Feature importance
lgb.plot_importance(bst, max_num_features = 20)


# As we can see, the separation is almost perfect - which strongly suggests that the train / test rows are very easy to distinguish. **Meaning the distribution of Train and Test is not the same**
