#!/usr/bin/env python
# coding: utf-8

# # Machine Learning: RF and XGBoost

# Version 18 is another look at this competition. I am using data engineered on the fork *Maps and EDA*.
# 
# This is only the Machine Learning part. I already studied and engineered the data on an other Kernel (forked).
# 
# Basically, I created a column with the distances in a straight line between the pickup and the dropoff (in km). The other feature, called `zones` is an indication of where the pickups are: inside Manhattan, near John F. Kennedy International Airport, near LaGuardia Airport, around Newark, on the rest of the city area, or outside.
# 
# Without that enginneering, I was at 0.43. Here, I will try to reach 0.40 or under

# # Table Of Contents
# 
# ----------
# 
# **[I. Features engineering](#one)**
# - [a. Filters](#one-a)
# - [b. Features Selection & Extraction](#one-b)
# 
# **[II. Machine Learning](#two)**
# - [a. Model Selection](#two-a)
#     - [1.  Random Forest](#two-a-1)
#     - [2.  SGBoost](#two-a-2)
# 
# - [b. Model Training](#two-b)
# - [c. Predictions](#two-c)
# 
# **[III. Submission](#three)**
# 
# --------------------

#  # **<a id="one"> I. Features engineering</a>**

# In[ ]:


import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


TRAINPATH = '../input/ny-taxis/train_w_zones2.csv'
TESTPATH = '../input/ny-taxis/test_w_zones2.csv'
df_train = pd.read_csv(TRAINPATH)
df_test = pd.read_csv(TESTPATH)
print('df_train:', df_train.shape, '\ndf_test:', df_test.shape)
l=df_train.shape[0]
df_train.head()


#  ## **<a id="one-a">I.a Filters</a>**

# In[ ]:


# remove trip of less than 10m (#8665)
#print('There is ', df_train[df_train['distances']<=0.01].shape[0], 'travels of less than 100m before filtering')
#df_train = df_train[df_train['distances']>0.01]
#print('There is ', df_train[df_train['distances']<=0.01].shape[0], 'travels of less than 100m after filtering')


# In[ ]:


#remove trips that lasted less than 1 min (#4933 left after previous filtering)
#print('There is', df_train[df_train['trip_duration']<=1*60].shape[0], 'travels of less than 1min before filtering')
#df_train = df_train[df_train['trip_duration']>1*60]
#print('There is', df_train[df_train['trip_duration']<=1*60].shape[0], 'travels of less than 1min after filtering')


# In[ ]:


#remove trip with an average speed greater than 200 km/h (distances are in straigth lines, I could probably choose a smaller number) (#22 after the two filters)
#print('There is', df_train[df_train['distances']/(df_train['trip_duration'])>=200/3600].shape[0], 'travels with an average speed faster than 200km/h before filtering')
#df_train = df_train[df_train['distances']/(df_train['trip_duration'])<200/3600]
#print('There is', df_train[df_train['distances']/(df_train['trip_duration'])>=200/3600].shape[0], 'travels with an average speed faster than 200km/h after filtering')


# In[ ]:


# remove trips that took longer that 3 hours (Who does that ??) (#2101 after filtering)
#print('There is', df_train[df_train['trip_duration']>=3*3600].shape[0], 'travels that took longer than 3 hours before filtering')
#df_train = df_train[df_train['trip_duration']<3*3600]
#print('There is', df_train[df_train['trip_duration']>=3*3600].shape[0], 'travels that took longer than 3 hours after filtering')


# In[ ]:


#remove trip with an average speed less than 1 km/h (I could probably choose a bigger number) (#3233 after the filters)
#print('There is', df_train[df_train['distances']/(df_train['trip_duration'])<=1/3600].shape[0], 'travels with an average speed slower than 1km/h before filtering (you walk at ~5km/h)')
#df_train = df_train[df_train['distances']/(df_train['trip_duration'])>1/3600]
#print('There is', df_train[df_train['distances']/(df_train['trip_duration'])<=1/3600].shape[0], 'travels with an average speed slower than 1km/h after filtering')


# In[ ]:


#print('We filtered','{:.3}'.format((l-df_train.shape[0])/l*100), '% of the dataset' )


# I thougth of a few filters we can apply to the training data. Like delete the small distances, the fast trips... These filters didn't seemed to be helping.
# It turns out that the models works better without it.

#  ## **<a id="one-b">I.b Features Selection & Extraction</a>**

# In[ ]:


def featuresSelection(df_in):
    VARS_CAT = [ 'store_and_fwd_flag' ]
    VARS_NUM = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'zone', 'distances', 'pickup_Month', 'pickup_Hour', 'pickup_Weekend', 'passenger_count', 'vendor_id' ]
    vars_cat = VARS_CAT
    vars_num = VARS_NUM

    X=df_in.loc[:, vars_cat + vars_num]

    for cat in vars_cat:
        X[cat] = X[cat].astype('category').cat.codes

    return X


# In[ ]:


X_train = featuresSelection(df_train)
target = 'trip_duration'
y_train = df_train.loc[:, target]
print(X_train.shape, y_train.shape)
y_train = np.log1p( y_train )
X_train.head()


#   # **<a id="two"> II. Machine Learning</a>**

# In[ ]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_log_error as MSLE
import xgboost


# ## **<a id="two-a"> II.a Model Selection</a>**

# In[ ]:


X_train_sample, X_validation, y_train_sample, y_validation = train_test_split(X_train, y_train, test_size=.2, random_state=42 )
print(X_train_sample.shape, y_train_sample.shape , X_validation.shape, y_validation.shape)
X_train_sample.head(5)


# ### **<a id="two-a-1"> II.a.1 Random Forest</a>**

# There is a few hyper-parameters to optimize for the RandomForestRegressor:
# 1. `n_estimators` or the number of trees in the forest. The bigger, the better, but the longer it takes. 10 is 30 sec, 100 is <10 min. 
# 2. `min_samples_leaf` or the number of samples in the final leaf. Best at 1, to get all the small variations.
# 3. `max_features` or the number of features used for each trees. Best at 0.4 (40% of total, or 4 features used)
# 
# This model work best without any filters on the training data.

# In[ ]:


#min_samples_leaf = {  1: 0.14335025261894946,  2: 0.13981831370645642,   3: 0.13852060557356807,  4: 0.1374604137021863, 5: 0.13701190316428685, 6: 0.13719592541154788,  7: 0.13647552678899308,   8: 0.13668619429239404,  9: 0.13678934918189598, 10: 0.13720206662667936, 15: 0.1378838545097919,   20: 0.13858468007164235,  25: 0.1397767624826059, 30: 0.14040835836429333, 35: 0.14162848146663448,  40: 0.14219905657487034,  45: 0.14265841548835242, 50: 0.14374664124817566, 100: 0.14924626267746,   150: 0.15302159678464494, 200: 0.15600849362124466, 250: 0.1578977545855252,  300: 0.16053779676581148, }
#plt.plot(min_samples_leaf.keys(), min_samples_leaf.values());
#plt.title('min_samples_leaf optimization');
#plt.legend(" with hyperparameters: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features=0.4, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=9, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1, oob_score=False, random_state=50, verbose=0, warm_start=False)") #plt.legend(" with features: pickup_latitude	pickup_longitude	dropoff_latitude	dropoff_longitude	zone	distances	pickup_Month	pickup_Hour	pickup_Weekend	passenger_count ")
#plt.xlabel('min_samples_leaf');
#plt.ylabel('MSLE score');
#min(min_samples_leaf, key=min_samples_leaf.get)


# Without filters on the training data (as it is better) and after optimizings the hyperparameters of the random forest, my score is: 0.407 on the scoreboard. Not bad, but can I do better with XGBoost ?

# ### **<a id="two-a-2"> II.a.2 XGBoost</a>**

# I did a lot of hyperparameters optimizations for this model. Here is the parameters that worked best for me:

# |parameters | value||parameters | value||parameters | value||parameters | value||parameters | value|
# |--------------|----------||--------------|----------||--------------|----------||--------------|----------||--------------|----------|
# |'booster'|'gbtree'| |'verbosity'|1| |'max_depth'|15| |'subsample'|1| |'objective'|'reg:linear'| 
#  | 'lamda'|0| | 'max_delta_step'|3| |'colsample_bytree'|0.9| |'colsample_bylevel'|0.9||'learning_rate'|0.08|
# 

# In[ ]:


#%%time
#modelparams = { 'booster':'gbtree', 'verbosity':1, 'max_depth':15, 'subsample': 1, 'lamda':0, 'max_delta_step':3, 'objective':'reg:linear', 'learning_rate':0.08, 'colsample_bytree':0.9, 'colsample_bylevel':0.9}
#data_train = xgboost.DMatrix(X_train_sample,y_train_sample)
#model = xgboost.train(modelparams, data_train, num_boost_round=200)


# In[ ]:


#real = list(np.expm1(y_validation))
#predicted = list(np.expm1(model.predict(xgboost.DMatrix(X_validation))))
#print('\nMean Square Log Error score:', MSLE(real, predicted))


# With XGBoost, I managed to get 0.3994 on the public score with the filtering. Pretty much the same, too bad. Let's use the XGBoost model since it's technically under 0.40.

# ## **<a id="two-b"> II.b Model training</a>**

# In[ ]:


get_ipython().run_cell_magic('time', '', "#rf = RandomForestRegressor( n_estimators=100, min_samples_leaf=1, max_depth=None, max_features=.4, oob_score=False, bootstrap=True, n_jobs=-1 )\n\nmodelparams = { 'booster':'gbtree', 'verbosity':1, 'max_depth':15, 'subsample': 1, 'lamda':0, 'max_delta_step':3, 'objective':'reg:linear', 'learning_rate':0.08, 'colsample_bytree':0.9, 'colsample_bylevel':0.9}\ndata_train = xgboost.DMatrix(X_train,y_train)\nxg = xgboost.train(modelparams, data_train, num_boost_round=200)")


# In[ ]:


#%%time
#rf.fit( X_train, y_train );
#rf.feature_importances_


# In[ ]:


#rf1_scores=-cross_val_score( rf, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error' )
#rf1_scores, np.mean(rf1_scores)


# ## **<a id="two-c"> II.c Predictions</a>**

# In[ ]:


X_test = featuresSelection(df_test)
X_test.head()


# In[ ]:


#y_test_predict = model_final.predict(X_test)
#y_test_predict = np.expm1(y_test_predict)
#y_test_predict[:5]


# In[ ]:


y_test_predict = np.expm1(xg.predict(xgboost.DMatrix(X_test)))
y_test_predict[:5]


# # **<a id="three">Submission</a>**

# In[ ]:


submission = pd.DataFrame(df_test.loc[:, 'id'])
submission['trip_duration']=y_test_predict
print(submission.shape)
submission.head()


# In[ ]:


submission.to_csv("submit_file.csv", index=False)

