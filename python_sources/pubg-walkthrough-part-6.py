#!/usr/bin/env python
# coding: utf-8

# In the last two parts we saw how three different models (lightgbm, xgboost, and catboost) worked on the individual and group datasets. Now for this part I want to do some parameter tuning. I anticipate that the best model will be some sort of combination between individual and group features, which I plan to test in a later part, but for now I will tune the paramters for the best model--lightgbm on the group data

# In[ ]:


# Import necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import warnings

warnings.filterwarnings("ignore")

# For plot sizes
plt.rcParams["figure.figsize"] = (18,8)
sns.set(rc={'figure.figsize':(18,8)})


# In[ ]:


os.listdir('../input')


# In[ ]:


# Load Part 2 data
data_group = pd.read_csv('../input/pubg-walkthrough-part-2/Training_Data_New_Groups.csv')
print("Done loading group data")


# In[ ]:


data_group.drop(data_group.index[0], inplace=True)
data_group.drop(['Unnamed: 0', 'index', 'groupId'], axis=1, inplace=True)


# In[ ]:


data_group = data_group.astype(float)


# In[ ]:


from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
import tqdm


# In[ ]:


y = data_group['winPlacePerc']


# In[ ]:


X = data_group
X.drop('winPlacePerc', axis=1, inplace=True)


# In[ ]:


print("y data shape")
print(y.shape)
print('X data shape')
print(X.shape)


# In[ ]:


X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(X, y, test_size=0.15, random_state=12)


# In[ ]:


from lightgbm import LGBMRegressor
import datetime


# In[ ]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


# In[ ]:


def objective(space):
    print(space)
    clf = LGBMRegressor(objective='mae', n_jobs=-1, random_state=12,
                       learning_rate = 0.3,
                       max_depth = int(space['max_depth']),
                       min_data_in_leaf = int(space['min_data_in_leaf']),
                       alpha = space['alpha'],
                       num_leaves = int(space['num_leaves']),
                       )
    
    eval_set  = [( X_train_group, y_train_group), ( X_test_group, y_test_group)]

    clf.fit(X_train_group, y_train_group,
            eval_set=eval_set, eval_metric="mae",
            early_stopping_rounds=10,verbose=False)

    pred = clf.predict(X_test_group)
    mae_scr = mae(y_test_group, pred)
    print("SCORE:", mae_scr)
    #change the metric if you like
    return {'loss':mae_scr, 'status': STATUS_OK }


# In[ ]:


space ={'max_depth': hp.quniform("x_max_depth", 4, 16, 1),
        'min_data_in_leaf': hp.quniform ('x_min_data_in_leaf', 10, 30, 1),
        'num_leaves': hp.quniform ('x_subsample', 2, 100, 1),
        'alpha' : hp.uniform ('x_alpha', 0.1,0.5),
        }


# In[ ]:


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials)


# In[ ]:


print(best)


# In[ ]:




