#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# Hi, I have  implemented a lightGBM model in a previous [kernel](https://www.kaggle.com/teemingyi/pubg-my-first-lightgbm-submission). Now, I will be trying to do my prediction with multiple models based on the each match's maxPlace.

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


#read the data
data = pd.read_csv('../input/train_V2.csv')


# In[ ]:


data.shape


# In[ ]:


data = data.dropna()
data.shape


# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


# I have decided to group a few of the matchType into others.
# 'normal-squad-fpp', 'crashfpp', 'crashtpp', 'normal-duo-fpp', 'flarefpp', 'normal-solo-fpp', 'flaretpp', 'normal-duo', 'normal-squad', 'normal-solo' are converted into others.

# In[ ]:


data.matchType.unique()


# In[ ]:


def merge_matchType(x):
    if x in {'normal-squad-fpp', 'crashfpp', 'crashtpp', 'normal-duo-fpp',
       'flarefpp', 'normal-solo-fpp', 'flaretpp', 'normal-duo',
       'normal-squad', 'normal-solo'}:
        return 'others'
    else:
        return x


# In[ ]:


data['matchType'] = data.matchType.apply(merge_matchType)

data.matchType.unique()


# I will convert the categorical variable into numerical. I have thus generated dummy variables for the matchType variable. matchType_others was dropped from the dataframe.

# In[ ]:


data_dumm = pd.get_dummies(data, columns=['matchType'])
data_dumm.head()


# In[ ]:


data_dumm = data_dumm.drop('matchType_others', axis=1)


# In[ ]:


data_dumm.columns


# In[ ]:


data = data_dumm.loc[:,['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace', 'numGroups',
       'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance',
       'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired',
       'winPoints', 'matchType_duo', 'matchType_duo-fpp',
       'matchType_solo', 'matchType_solo-fpp', 'matchType_squad',
       'matchType_squad-fpp', 'winPlacePerc']]


# In[ ]:


print(data.shape)
data.head()


# I will look at the distribution of maxPlace variable.

# In[ ]:


data.maxPlace.plot(kind='hist')


# In[ ]:


print('There are {} unique maxPlace.'.format(len(data.maxPlace.unique())))


# I will split the data into their different maxPlace.

# In[ ]:


data_store_by_maxPlace = {}
for x in data.maxPlace.unique():
    data_store_by_maxPlace[x] = data.loc[data.maxPlace==x]


# In[ ]:


# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4

def adjust_pred(x, maxPlace):
    space = 1/(maxPlace-1)
    return np.round(x / space) * space


# In[ ]:


def generate_lgb_model(data, for_eval):
    # split data into X and y
    X = data.iloc[:,3:33]
    
    maxPlace = X.maxPlace.unique()
    X = X.drop('maxPlace', axis=1)
    Y = data.iloc[:,33]
    
    if for_eval == True:
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    else:
        X_train, y_train = X, Y
        
    d_train = lgb.Dataset(X_train, label=y_train)

    params = {}
    params['objective'] = 'regression'
    params['metric'] = 'mae'

    model = lgb.train(params, d_train)
    
    if for_eval == True:
        #Prediction
        y_pred=model.predict(X_test)
        
        y_pred = adjust_pred(y_pred, maxPlace)
        mae = mean_absolute_error(y_test, y_pred)
        print("MAE for maxPlace = {}: {}".format(maxPlace, mae))
    
        return [model, (list(y_test), y_pred)]
    
    else:
        return model


# In[ ]:


model_store = {}
maxPlace_set = set(data_store_by_maxPlace.keys())
print('There are {} unique maxPlace.'.format(len(maxPlace_set)))

for key, value in data_store_by_maxPlace.items():
    model_store[key] = generate_lgb_model(value,for_eval=True)
    maxPlace_set = maxPlace_set - {key}
    print('There are {} more models to go.'.format(len(maxPlace_set)))


# In[ ]:


y_test_overall = []
y_pred_overall = []
for value in model_store.values():
    y_test_overall.append(value[1][0])
    y_pred_overall.append(value[1][1])


# In[ ]:


print('The overall mae is {}.'.format(mean_absolute_error([y for x in y_test_overall for y in x],[y for x in y_pred_overall for y in x])))


# Now, I will train models using the full dataset.

# In[ ]:


model_store_full = {}
maxPlace_set = set(data_store_by_maxPlace.keys())
print('There are {} unique maxPlace.'.format(len(maxPlace_set)))

for key, value in data_store_by_maxPlace.items():
    model_store_full[key] = generate_lgb_model(value,for_eval=False)
    maxPlace_set = maxPlace_set - {key}
    print('There are {} more models to go.'.format(len(maxPlace_set)))


# In[ ]:


X_submit = pd.read_csv('../input/test_V2.csv')
print(X_submit.shape)
X_submit.head()


# In[ ]:


X_submit['matchType'] = X_submit.matchType.apply(merge_matchType)

X_submit.matchType.unique()

X_submit_dumm = pd.get_dummies(X_submit, columns=['matchType'])
X_submit_dumm.head()

X_submit_dumm = X_submit_dumm.drop('matchType_others', axis=1)

X_submit = X_submit_dumm.loc[:,['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace', 'numGroups',
       'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance',
       'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired',
       'winPoints', 'matchType_duo', 'matchType_duo-fpp',
       'matchType_solo', 'matchType_solo-fpp', 'matchType_squad',
       'matchType_squad-fpp']]

X_submit.head()


# In[ ]:


data_store_by_maxPlace_submit = {}
for x in X_submit.maxPlace.unique():
    data_store_by_maxPlace_submit[x] = X_submit.loc[X_submit.maxPlace==x]


# In[ ]:


#Prediction
prediction = {}
for key, value in data_store_by_maxPlace_submit.items():
    maxPlace = key
    train_data = value.iloc[:,3:33]

    train_data = train_data.drop('maxPlace', axis=1)
    
    pred_submit = model_store_full[key].predict(train_data)
    
    pred_submit = adjust_pred(pred_submit, maxPlace)
    prediction[key] = pd.concat([value.Id.reset_index(drop=True), pd.Series(pred_submit, name='winPlacePerc')], axis=1)


# In[ ]:


#Submission file
submission = pd.concat([x for x in prediction.values()])

submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# **Conclusion**
# 
# There was some improvement in the Mean Absolute Error. However, it is still pretty high as compared to the other submissions in the competition.
