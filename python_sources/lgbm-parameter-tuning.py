#!/usr/bin/env python
# coding: utf-8

# ## This kernel base on [Alexander Teplyuk](https://www.kaggle.com/ateplyuk/lgbm-str-w) here I tuned min_data_in_leaf to 120, add random state and shuffle = True

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
import lightgbm as lgb


# In[ ]:


t_res = pd.read_csv('../input/wdatafiles/WNCAATourneyCompactResults.csv')
t_ds = pd.read_csv('../input/wdatafiles/WNCAATourneySeeds.csv')
sub = pd.read_csv('../input/WSampleSubmissionStage1.csv')


# In[ ]:


t_ds['seed_int'] = t_ds.Seed.apply(lambda a : int(a[1:3]))


# In[ ]:


drop_lbls = ['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT']
t_ds.drop(labels=['Seed'], inplace=True, axis=1)
t_res.drop(labels=drop_lbls, inplace=True, axis=1)


# In[ ]:


ren1 = {'TeamID':'WTeamID', 'seed_int':'WS'}
ren2 = {'TeamID':'LTeamID', 'seed_int':'LS'}


# In[ ]:


df1 = pd.merge(left=t_res, right=t_ds.rename(columns=ren1), how='left', on=['Season', 'WTeamID'])
df2 = pd.merge(left=df1, right=t_ds.rename(columns=ren2), on=['Season', 'LTeamID'])

df_w = pd.DataFrame()
df_w['dff'] = df2.WS - df2.LS
df_w['rsl'] = 1

df_l = pd.DataFrame()
df_l['dff'] = -df_w['dff']
df_l['rsl'] = 0

df_prd = pd.concat((df_w, df_l))


# In[ ]:


X = df_prd.dff.values.reshape(-1,1)
y = df_prd.rsl.values


# In[ ]:


X_test = np.zeros(shape=(len(sub), 1))


# In[ ]:


for ind, row in sub.iterrows():
    yr, o, t = [int(x) for x in row.ID.split('_')]  
    X_test[ind, 0] = t_ds[(t_ds.TeamID == o) & (t_ds.Season == yr)].seed_int.values[0] - t_ds[(t_ds.TeamID == t) & (t_ds.Season == yr)].seed_int.values[0]


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1,shuffle = True, random_state=2019)


# In[ ]:


from sklearn.metrics import f1_score
params = {'num_leaves': 70,
          "boosting": "gbdt",
          'min_data_in_leaf': 120,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.01,
          "metric": 'auc',
          "verbosity": -1,
          }

f1s_valid = []
f1s_train = []
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
bst = lgb.train(params, dtrain, 1000, valid_sets=dvalid, verbose_eval=200,
   early_stopping_rounds=200)

#clf = lgb.LGBMClassifier(n_estimators=50, silent=True).fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=200, verbose=100)
#rg = lgb.LGBMRegressor(params,n_estimators=50, silent=True).fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=100)
y_pred_valid = np.where(bst.predict(X_val, num_iteration= bst.best_iteration) > 0.5, 1, 0)
y_pred_train = np.where(bst.predict(X_train, num_iteration= bst.best_iteration) > 0.5, 1, 0)

f1s_valid.append(f1_score(y_val, y_pred_valid))
f1s_train.append(f1_score(y_train, y_pred_train))
#model = AdaBoostClassifier(n_estimators=200, learning_rate=1.4)
#model = sklearn.ensemble.GradientBoostingClassifier(learning_rate=0.01,max_leaf_nodes = 120)
#model.fit(X_train, y_train)


# In[ ]:


print('CV mean train score: {0:.4f}, std: {1:.4f}.'.format(np.mean(f1s_train), np.std(f1s_train)))


# In[ ]:


test_pred = bst.predict(
    X_test)
print('Log Loss:', test_pred)


# In[ ]:


sub.Pred = test_pred   
sub.to_csv('submission.csv', index=False)

