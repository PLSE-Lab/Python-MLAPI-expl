#!/usr/bin/env python
# coding: utf-8

# NFL play by play data is from NFL Savant: http://nflsavant.com/about.php

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


# The goal of this notebook is to experiment with feature engineering with NFL Savant's play-by-play data. I had previously tried to preform some elementary data analysis on this data; however, the data being largely categorical (or logistical) I found it difficult to study. Now, with various encoders I'll attempt to label the data and explore some simple statistics using Machine Learning.  
# 
# The problem I chose to tackle was: how can we use the data provided to predict whether or not a team will get a first down on a given set of downs?

# In[ ]:


import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn import metrics
from sklearn.model_selection import train_test_split
import category_encoders as ce
import pandas as pd
import numpy as np
NFL1 = pd.read_csv("../input/nfl-pbp/pbp-2019.csv")
NFL2 = pd.read_csv("../input/nfl-pbp/pbp-2018.csv")
NFL3 = pd.read_csv("../input/nfl-pbp/pbp-2017.csv")
NFL = pd.concat([NFL2, NFL3, NFL1], axis=0)
NFL


# The goal is to measure the probability of a first down (a binary variable) based on several features from the dataset. 

# In[ ]:


NFL = NFL.assign(outcome=(NFL['SeriesFirstDown'] == 'successful').astype(int))


# First we Label Encode (I chose to use the count encoder-which preforms better- after having some issues with label) 

# In[ ]:


cat_features = ['Formation','PassType', 'PlayType', 'RushDirection'] 
encoder = ce.CountEncoder()

count_encoded = encoder.fit_transform(NFL[cat_features])

#count_encoded.head()
data = NFL[['SeriesFirstDown','ToGo', 'Down', 'YardLine', 'IsPass', 'IsRush', 'Yards', 'IsIncomplete']].join(count_encoded)

data.head(10)


# Adding interaction

# In[ ]:


interaction1 = NFL1['Formation']+ "_" + NFL1['PlayType'] 
#interaction2 = NFL['Formation']+ "_" + NFL['PassType']
#interaction3 = NFL['Formation']+ "_" + NFL['RushDirection']

encoder = ce.CountEncoder()
data_interaction1 = data.assign(interact1=encoder.fit_transform(interaction1))

data_interaction1.head()


# Split Data into train, test, valid

# In[ ]:


valid_fraction = 0.1
valid_size = int(len(data)*valid_fraction)

train = data_interaction1[:-2 * valid_size]
valid = data_interaction1[-2 * valid_size:-valid_size]
test = data_interaction1[-valid_size:]

#train = data[:-2 * valid_size]
#valid = data[-2 * valid_size:-valid_size]
#test = data[-valid_size:]


# In[ ]:


def get_data_splits(dataframe, valid_fraction=0.1):
    valid_fraction = 0.1
    valid_size = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_size * 2]
    # valid size == test size, last two sections of the data
    valid = dataframe[-valid_size * 2:-valid_size]
    test = dataframe[-valid_size:]
    
    return train, valid, test


def train_model(train, valid):
    feature_cols = train.columns.drop('SeriesFirstDown')

    dtrain = lgb.Dataset(train[feature_cols], label=train['SeriesFirstDown'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['SeriesFirstDown'])

    param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'auc', 'seed': 7}
   # print("Training model!")
    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 
                    early_stopping_rounds=10, verbose_eval=False)

    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['SeriesFirstDown'], valid_pred)
    print(f"AUC: {valid_score:.4f}")
    return bst

def train_model2(train, valid):
    feature_cols = train.columns.drop('SeriesFirstDown')

    dtrain = lgb.Dataset(train[feature_cols], label=train['SeriesFirstDown'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['SeriesFirstDown'])

    param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'mae', 'seed': 7}
   # print("Training model!")
    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 
                    early_stopping_rounds=10, verbose_eval=False)

    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.mean_absolute_error(valid['SeriesFirstDown'], valid_pred)
    print(f"MAE: {valid_score:.4f}")
    return bst


# Next we build the model

# In[ ]:


feature_cols = train.columns.drop('SeriesFirstDown')

dtrain = lgb.Dataset(train[feature_cols], label=train['SeriesFirstDown'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['SeriesFirstDown'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'mae'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)


# And finally; make predictions:

# In[ ]:


ypred = bst.predict(test[feature_cols])

score = metrics.roc_auc_score(test['SeriesFirstDown'], ypred)
score2 = metrics.mean_absolute_error(test['SeriesFirstDown'], ypred)

print(f"AUC: {score}")
print(f"MAE: {score2}")


# We can also try Target Encoding:

# In[ ]:


target_enc = ce.TargetEncoder(cols=cat_features)

target_enc.fit(train[cat_features], train['SeriesFirstDown'])
train, valid, _ = get_data_splits(data_interaction1)

train = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'), how = 'left')
valid = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'), how = 'left')

#train.head()
bst = train_model(train, valid)
bst2 = train_model2(train, valid)


# Which we see does slightly worse (but is comparable) to count encoding. I also wanted to compare how this model does with a Decsion Tree model (as opposed to lgb). The initial assumption would be that this new model should do worse. 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

X = NFL[['ToGo', 'Down', 'YardLine', 'IsPass', 'IsRush', 'Yards', 'IsIncomplete']] #.join(count_encoded)
y = NFL['SeriesFirstDown']

NFL_model = DecisionTreeRegressor(random_state=1)
NFL_model.fit(X, y)

print(X.head(10))
print("Predictions:")
print(NFL_model.predict(X.head(10)))
print("Actual Results:")
print(y.head(10))


# That was fun; here you can 'imagine' several plays and their outcomes (i.e did the offense get a first down?). For the error I used MAE.

# In[ ]:


predicted = NFL_model.predict(X)
mean_absolute_error(y, predicted)


# Next, let's implement RandomForrests and see if we can do better

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

NFL_model_2 = RandomForestRegressor(random_state=1)
NFL_model_2.fit(X,y)
predicted_2 = NFL_model_2.predict(X)

print("Actual Results:")
print(y.head(10))
print("Predictions:")
print(predicted_2[:10])


# The error:

# In[ ]:


mean_absolute_error(y, predicted_2)


# The MAE for the Decision tree is actually lower than the MAE for LGB; even though the tree model doesn't use any of the encoded atributes. Strange?! Why this is, is a difficult question to answer; it may be that for newer datasets (eg. 2020 pbp) the LGB will do better than the tree model. 
# 
# The random forrests model is truly random (see the plot below); so at first I was confused at why the Decision tree would do better (than random forrests), but the visualization of the predictions makes the answet to this question quite clear. 
# 
# Finally, some visulaizations of the data:

# Decision tree predictions v.s Data
# 
# Below: interesting plots of the data; in particular, my next notebook will involve fitting a normal curve to the 4th plot (bottom right). 

# In[ ]:


import matplotlib.pyplot as plt
x1 = np.linspace(0, 1, 132495)

plt.figure()
plt.subplot(321)
plt.scatter(x1, y)
plt.subplot(322)
plt.scatter(x1, NFL_model.predict(X))
plt.subplot(325)
plt.scatter(NFL['Yards'], NFL['Down'])
plt.subplot(326)
plt.scatter(NFL['Yards'], NFL['ToGo'])
plt.show()


# LGB model predictions v.s Data

# In[ ]:


x2 = np.linspace(0, 1, 39127)

plt.figure()
plt.subplot(321)
plt.scatter(x1, y)
plt.subplot(322)
plt.scatter(x2, ypred)
plt.show()


# Random Forrests predictions v.s Data

# In[ ]:


plt.figure()
plt.subplot(321)
plt.scatter(x1, y)
plt.subplot(322)
plt.scatter(x1, NFL_model_2.predict(X))
plt.show()

