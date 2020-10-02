#!/usr/bin/env python
# coding: utf-8

# **The purpose of this kernel is to show how to get the mean(yards)/Rusher.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import io
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
import multiprocessing
import datetime
import seaborn as sns


# In[ ]:


train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# In[ ]:


train_df['ToLeft'] = 0
train_df.loc[train_df.PlayDirection == "left",'ToLeft'] = 1
train_df.loc[train_df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
train_df.loc[train_df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

train_df.loc[train_df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
train_df.loc[train_df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

train_df.loc[train_df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
train_df.loc[train_df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

train_df.loc[train_df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
train_df.loc[train_df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

train_df['TeamOnOffense'] = "home"
train_df.loc[train_df.PossessionTeam != train_df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
train_df['IsOnOffense'] = train_df.Team == train_df.TeamOnOffense # Is player on offense?
train_df['YardLine_std'] = 100 - train_df.YardLine
train_df.loc[train_df.FieldPosition.fillna('') == train_df.PossessionTeam,  
          'YardLine_std'
         ] = train_df.loc[train_df.FieldPosition.fillna('') == train_df.PossessionTeam,  
          'YardLine']


# In[ ]:


train_df['Teamball'] = 0


# In[ ]:


train_df.loc[(train_df.TeamOnOffense == "home"),'Teamball'] = train_df.loc[(train_df.TeamOnOffense == "home"),'HomeTeamAbbr']
train_df.loc[(train_df.TeamOnOffense == "away"),'Teamball'] = train_df.loc[(train_df.TeamOnOffense == "away"),'VisitorTeamAbbr']


# In[ ]:


train_df['Year'] = train_df['TimeHandoff'].apply(lambda x : int(x[0:4]))


# In[ ]:


train_df = train_df[train_df.Year == 2018]


# In[ ]:


train_df['TeamOnOffense'] = "home"
train_df.loc[train_df.PossessionTeam != train_df.HomeTeamAbbr, 'TeamOnOffense'] = "away"


# In[ ]:


train_df = train_df.loc[train_df.Team == train_df.TeamOnOffense ] 


# In[ ]:


merge1 = train_df.groupby(['DisplayName'])[['Yards']].agg(['sum','mean','std','count'])


# In[ ]:


merge1.columns = ['sum','mean_Yards','std_Yards','count_play']


# In[ ]:


merge2 = train_df.groupby(['DisplayName'])[['GameId']].agg(['nunique'])


# In[ ]:


merge2.columns = ['G']


# In[ ]:


merge1 = merge2.merge(merge1, how='left', on = ['DisplayName'],left_index=True)


# In[ ]:


from scipy.ndimage.interpolation import shift


# In[ ]:


from scipy.ndimage.interpolation import shift
for i in range(0, 100,5) :
    merge1['shift_Yards_'+str(i)] = train_df.groupby(['DisplayName'])['Yards'].apply(lambda x : shift(x,i).mean())
    merge1['quantile_Yards_'+str(i)] = train_df.groupby(['DisplayName'])['Yards'].apply(lambda x : np.quantile(x, i/100))
    print(i,'%')


# In[ ]:


train_df = merge1.merge(train_df, how='left', on = ['DisplayName'],left_index=True)
#train_df = train_df.sort_values(['GameId','PlayId','Team'], ascending = True).reset_index(drop = True)


# In[ ]:


train_df = train_df[list(merge1.columns)+ ['DisplayName','Position','Teamball']]


# In[ ]:


IDP = pd.read_csv('/kaggle/input/rusherid/Rusher.csv', low_memory=False)


# In[ ]:


IDP['DisplayName']= IDP['Player']
IDP.drop('Player',axis = 1 ,inplace = True)


# In[ ]:


IDP['DisplayName'] = IDP['DisplayName'].str.split("\\").apply(lambda x :x[0])


# In[ ]:


IDP['DisplayName'] = IDP['DisplayName'].str.replace("*",'')
IDP['DisplayName'] = IDP['DisplayName'].str.replace("+",'')


# In[ ]:


IDP # G represents the number of match, Yds = Sum of Yards by rusher in 2018


# In[ ]:


train_df = train_df.drop_duplicates()


# In[ ]:


train_df.head()


# In[ ]:


train_df = IDP.merge(train_df, how='inner', on = ['DisplayName','G'],left_index=True)


# In[ ]:


train_df['Yds'] = train_df['Yds'].apply(lambda x : 0 if (x <0) else x)


# In[ ]:


train_df['mean_Yds'] = train_df['Yds']/train_df['count_play']


# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


train_df = train_df.dropna()


# In[ ]:


from sklearn import preprocessing
le_Teamball = preprocessing.LabelEncoder()

train_df['Teamball'] = le_Teamball.fit_transform(train_df['Teamball'])


# In[ ]:


le_Position = preprocessing.LabelEncoder()

train_df['Position'] = le_Position.fit_transform(train_df['Position'])


# In[ ]:


tmp = train_df


# In[ ]:


features = list(train_df.columns)
features =  [col for col in features if col not in  ['DisplayName','Yds','sum','mean_Yards','mean_Yds','Teamball','G','count_play']]


# In[ ]:


train_df


# In[ ]:


features


# In[ ]:


y_tr = train_df['mean_Yds'].values
import matplotlib.pylab as plt


# In[ ]:


from sklearn.model_selection import KFold
nfold = 5
folds = KFold(n_splits=nfold, shuffle=True, random_state=42)

print('-'*20)
print(str(nfold) + ' Folds training...')
print('-'*20)


# # 1) First Solution

# In[ ]:


best_params_lgb = {'lambda_l1': 0.13413394854686794, 
'lambda_l2': 0.0009122197743451751, 
'num_leaves': 10, 
'feature_fraction': 1, 
'bagging_fraction': 0.9999128827046064, 

"learning_rate": 0.001,

'objective': 'regression', 
'metric': 'mae', 
'verbosity': -1, 
'boosting_type': 'gbdt', 
"boost_from_average" : True,
'random_state': 42}


# In[ ]:


import lightgbm as lgb
oof = np.zeros(len(train_df))
#y_valid_pred = np.zeros(X_train.shape[0])
feature_importance_df = pd.DataFrame()

tr_mae = []
val_mae = []
models1 = []

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,y_tr)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    

    

    X_tr, X_val = train_df.iloc[trn_idx][features], train_df.iloc[val_idx][features]
    train_y, y_val = y_tr[trn_idx], y_tr[val_idx]

    model = lgb.LGBMRegressor(**best_params_lgb, n_estimators = 5000, n_jobs = -1,early_stopping_rounds = 100)
    model.fit(X_tr, 
              train_y, 
              eval_set=[(X_tr, train_y), (X_val, y_val)], 
              eval_metric='mae',
              verbose=100
              
             )
    oof[val_idx] = model.predict(X_val)
    val_score = mean_absolute_error(y_val, oof[val_idx])
    val_mae.append(val_score)
    tr_score = mean_absolute_error(train_y, model.predict(X_tr))
    tr_mae.append(tr_score)
    models1.append(model)
    
    
    # Feature importance
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = X_tr.columns
    fold_importance_df["importance"] = model.feature_importances_[:len(X_tr.columns)]
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


# In[ ]:


plt.style.use('dark_background')
cols = (feature_importance_df[["Feature", "importance"]]
    .groupby("Feature")
    .mean()
    .sort_values(by="importance", ascending=False)[:40].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(10,10))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False),
        edgecolor=('white'), linewidth=2, palette="rocket")
plt.title('LGB Features importance (averaged/folds)', fontsize=18)
plt.tight_layout()


# In[ ]:


mean_mae_tr = np.mean(tr_mae)
std_mae_tr =  np.std(tr_mae)

mean_mae_val =  np.mean(val_mae)
std_mae_val =  np.std(val_mae)

all_mae = mean_absolute_error(oof,y_tr)

print('-'*20)
print("Train's Score")
print('-'*20,'\n')
print("Mean mae: %.5f, std: %.5f." % (mean_mae_tr, std_mae_tr),'\n')

print('-'*20)
print("Validation's Score")
print('-'*20,'\n')
print("Mean mae: %.5f, std: %.5f." % (mean_mae_val, std_mae_val),'\n')

print("All mae: %.5f." % (all_mae))


# ## Before

# In[ ]:


mean_absolute_error(train_df['mean_Yds'],train_df['mean_Yards'])


# In[ ]:


plt.figure(figsize=(15,10))
plt.plot(y_tr)
plt.plot(train_df['mean_Yards'].values)


# ## After

# In[ ]:


mean_absolute_error(y_tr,oof)


# In[ ]:


plt.figure(figsize=(15,10))
plt.plot(y_tr)
plt.plot(oof) 


# # 2) Fast & Compact Solution 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


features = ['Position','shift_Yards_0']


# In[ ]:



oof = np.zeros(len(train_df))
#y_valid_pred = np.zeros(X_train.shape[0])
feature_importance_df = pd.DataFrame()

tr_mae = []
val_mae = []
models2 = []

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,y_tr)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    

    

    X_tr, X_val = train_df.iloc[trn_idx][features], train_df.iloc[val_idx][features]
    train_y, y_val = y_tr[trn_idx], y_tr[val_idx]

    model = DecisionTreeRegressor(criterion='mse',max_leaf_nodes=4,random_state=0)
    model.fit(X_tr, train_y)
    oof[val_idx] = model.predict(X_val)
    val_score = mean_absolute_error(y_val, oof[val_idx])
    val_mae.append(val_score)
    tr_score = mean_absolute_error(train_y, model.predict(X_tr))
    tr_mae.append(tr_score)
    models2.append(model)
    
    
    # Feature importance
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = X_tr.columns
    fold_importance_df["importance"] = model.feature_importances_[:len(X_tr.columns)]
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


# In[ ]:




mean_mae_tr = np.mean(tr_mae)
std_mae_tr =  np.std(tr_mae)

mean_mae_val =  np.mean(val_mae)
std_mae_val =  np.std(val_mae)

all_mae = mean_absolute_error(oof,y_tr)

print('-'*20)
print("Train's Score")
print('-'*20,'\n')
print("Mean mae: %.5f, std: %.5f." % (mean_mae_tr, std_mae_tr),'\n')

print('-'*20)
print("Validation's Score")
print('-'*20,'\n')
print("Mean mae: %.5f, std: %.5f." % (mean_mae_val, std_mae_val),'\n')

print("All mae: %.5f." % (all_mae))


# In[ ]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz


# In[ ]:


dot_data = StringIO()

export_graphviz(models2[0], out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)


# In[ ]:


from sklearn import tree
import graphviz


# In[ ]:


tree_graph = tree.export_graphviz(models2[0], out_file=None, max_depth = 5,
    impurity = False, feature_names = features,
    rounded = True, filled= True )
graphviz.Source(tree_graph)  


# In[ ]:


plt.figure(figsize=(15,10))
plt.plot(y_tr)
plt.plot(oof)


# In[ ]:


import pickle
    # save the model to disk
filename = 'LGBM-NFL.pkl'
pickle.dump(models1, open(filename, 'wb'))


# In[ ]:


filename = 'DecisionTreeRegressorRusher.pkl'
pickle.dump(models2, open(filename, 'wb'))


# I'll let you do the rest.

# # **Upvote my kernel and my dataset if you think it will be useful for you**
