#!/usr/bin/env python
# coding: utf-8

# ## Basketball Prediciton

# **Thanks to:
# FE - https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg   
# Model tunung - https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models#FE,-tuning-and-comparison-of-the-20-popular-models

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
#  
#  1. [Import libraries](#1)
#  1. [Download dataset](#2)
#  1. [Preparing to analysis](#3)
#  1. [FE](#4)
#  1. [Model tuning](#5)
#      -  [KNeighborsClassifier](#5.1)
#      -  [RandomForestClassifier](#5.2)
#      -  [GradientBoostingClassifier](#5.3)
#  1. [Prediction](#6)

# ## 1. Import libraries <a class="anchor" id="1"></a>
#  
# ### [Back to Table of Contents](#0.1)

# In[ ]:


#imports
import numpy as np
import pandas as pd
import lightgbm as lgbm
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


# options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500) 
np.set_printoptions(threshold=100)


# ## 2. Download dataset <a class="anchor" id="2"></a>
#  
# ### [Back to Table of Contents](#0.1)

# In[ ]:


train_data = pd.read_csv('../input/nba-enhanced-stats/2016-17_teamBoxScore.csv')
test_data = pd.read_csv('../input/nba-enhanced-stats/2017-18_teamBoxScore.csv')

base_train_data = train_data.copy()

train_data.head()


# In[ ]:


train_data.describe()


# ## 3. Preparing to analysis <a class="anchor" id="3"></a>
#  
# ### [Back to Table of Contents](#0.1)

# In[ ]:


# Date processing 
date_value = pd.to_datetime(train_data['gmDate'], errors='coerce')
time_value = pd.to_datetime(train_data['gmTime'], errors='coerce')


train_data['year'] = date_value.dt.year 
train_data['month'] = date_value.dt.month 
train_data['day'] = date_value.dt.day 
train_data['hour'] = time_value.dt.hour 
train_data['minute'] = time_value.dt.minute

del train_data['gmDate']
del train_data['gmTime']


# In[ ]:


train_data.head()


# In[ ]:


# Mapping of teamRslt column
mapping = {'Loss': 2, 'Win': 1}

train_data = train_data.replace({'teamRslt': mapping})
test_data = test_data.replace({'teamRslt': mapping})


# In[ ]:


# Drop columns with missing values
cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()] 
train_data.drop(cols_with_missing, axis=1, inplace=True)
test_data.drop(cols_with_missing, axis=1, inplace=True)


# In[ ]:


# Categorical data processing
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = train_data.columns.values.tolist()
for col in features:
    if train_data[col].dtype in numerics: continue
    categorical_columns.append(col)
indexer = {}
for col in categorical_columns:
    if train_data[col].dtype in numerics: continue
    _, indexer[col] = pd.factorize(train_data[col])
    
for col in categorical_columns:
    if train_data[col].dtype in numerics: continue
    train_data[col] = indexer[col].get_indexer(train_data[col])


# In[ ]:


train_data.head()


# In[ ]:


corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(20,18))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[ ]:


k = 12
cols = corrmat.nlargest(k, 'teamRslt')['teamRslt'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


k = 12
cols = corrmat.nlargest(k, 'teamPTS')['teamPTS'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


k = 12
cols = corrmat.nlargest(k, 'opptPTS')['opptPTS'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


k = 12
cols = corrmat.nlargest(k, 'teamDayOff')['teamDayOff'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


k = 12
cols = corrmat.nlargest(k, 'teamLoc')['teamLoc'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


y = train_data['teamRslt']

columns_to_delete = ['teamRslt', 'opptRslt', 'teamEDiff', 'teamFIC', 'opptEDiff', 'opptFIC']

train_data.drop(columns_to_delete, axis=1, inplace=True)

X = train_data;


# In[ ]:


# data split for train
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# ## 4. FE<a class="anchor" id="4"></a>
#  
# ### [Back to Table of Contents](#0.1)

# In[ ]:


train_set = lgbm.Dataset(X_train, y_train, silent=False)
valid_set = lgbm.Dataset(X_valid, y_valid, silent=False)

params = {
        'boosting_type':'gbdt', 'objective': 'regression', 'num_leaves': 31,
        'learning_rate': 0.05, 'max_depth': -1, 'subsample': 0.8,
        'bagging_fraction' : 1, 'max_bin' : 5000 , 'bagging_freq': 20,
        'colsample_bytree': 0.6, 'metric': 'rmse', 'min_split_gain': 0.5,
        'min_child_weight': 1, 'min_child_samples': 10, 'scale_pos_weight':1,
        'zero_as_missing': True, 'seed':0,        
    }

modelL = lgbm.train(params, train_set = train_set, num_boost_round=1000,
                   early_stopping_rounds=50,verbose_eval=10, valid_sets=valid_set)

fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgbm.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();


# In[ ]:


data_tr  = xgb.DMatrix(X_train, label=y_train)
data_cv  = xgb.DMatrix(X_valid   , label=y_valid)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]

parms = {'max_depth':8, #maximum depth of a tree
         'objective':'reg:squarederror',
         'eta'      :0.3,
         'subsample':0.8,#SGD will use this percentage of data
         'lambda '  :4, #L2 regularization term,>1 more conservative 
         'colsample_bytree ':0.9,
         'colsample_bylevel':1,
         'min_child_weight': 10}
modelx = xgb.train(parms, data_tr, num_boost_round=200, evals = evallist, early_stopping_rounds=30, maximize=False, verbose_eval=10)

fig =  plt.figure(figsize = (15,25))
axes = fig.add_subplot(111)
xgb.plot_importance(modelx,ax = axes,height = 1)
plt.show();plt.close()


# In[ ]:


feature_columns = ['opptPTS', 'teamDrtg', 'teamTO', 'teamORB', 'teamFGM']
X = X[feature_columns];

X.head()


# In[ ]:


feature_columns = ['opptPTS', 'teamDrtg', 'teamTO', 'teamORB', 'teamFGM']
sns.pairplot(train_data[feature_columns], height=2.5)
plt.show()


# ## 5. Model tuning<a class="anchor" id="5"></a>
#  
# ### [Back to Table of Contents](#0.1)

# In[ ]:


def parseResult(data):
    def parse(n): 
        left = n[0] 
        rigth = n[1]
        return 1 if left > rigth else 2
    
    return list(map(parse, data))


# In[ ]:


# data split for model tuning
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# ### 5.1 KNeighborsClassifier <a class="anchor" id="5.1"></a>
#  
# ### [Back to Table of Contents](#0.1)

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_valid)
results = []
result = accuracy_score(y_valid, pred) * 100
results.append(result)
print(result)


# In[ ]:


pred


# In[ ]:


y_valid


# ### 5.2 RandomForestClassifier <a class="anchor" id="5.2"></a>
# 
# ### [Back to Table of Contents](#0.1)

# In[ ]:


clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
pred = clf.predict(X_valid)
result = accuracy_score(y_valid, pred) * 100
results.append(result)
print(result)


# ### 5.3 GradientBoostingClassifier <a class="anchor" id="5.3"></a>
#  
# ### [Back to Table of Contents](#0.1)

# In[ ]:


clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
                                    max_depth=1, random_state=0).fit(X_train, y_train)

result = clfgtb.score(X_valid, y_valid) * 100
results.append(result)
print(result)


# In[ ]:


x = np.arange(3)

fig, ax = plt.subplots()
plt.bar(x, results)
ax.set_ylim(bottom=75)
plt.xticks(x, ('KNeighbors', 'RandomForest', 'GradientBoosting'))
plt.show()


# ## 6. Prediction<a class="anchor" id="6"></a>
# 
# ### [Back to Table of Contents](#0.1)

# In[ ]:


test_data.describe()


# In[ ]:


x_new = test_data[feature_columns]
y_new = test_data['teamRslt']
x_new.head()


# In[ ]:


clss = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
clss.score(x_new, y_new)


# In[ ]:


matches = [
    {'home_team': 'MIN', 'away_team': 'BOS'},
    {'home_team': 'CLE', 'away_team': 'LAL'},
    {'home_team': 'CHA', 'away_team': 'MIN'},
    {'home_team': 'ORL', 'away_team': 'HOU'},
    {'home_team': 'DET', 'away_team': 'UTA'},
]

for match in matches:
    home_team = match['home_team']    
    away_team = match['away_team']

    prev_matches = base_train_data.loc[(base_train_data['teamAbbr'] == home_team) & (base_train_data['opptAbbr'] == away_team)][feature_columns]
    avg = prev_matches.mean()

    avg_prev = [prev_matches.mean().values.tolist()]

    pred = clss.predict(avg_prev)
    prob = clss.predict_proba(avg_prev)

    print(home_team + ' vs ' + away_team)
    print(pred)
    print(prob)
    print('-------------------------------\n')

