#!/usr/bin/env python
# coding: utf-8

# Hey guys,
# 
# This is a work in progress Kernel.  Happy Learning :D
# 
# Me being a big fan of PUBG. Let's find out if we can predict the ways to kill them all and the best stratergy for Winner Winner Chicken Dinner :D

# **Importing the Dataset**

# In[ ]:


# Importing the library
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# training set
train = pd.read_csv("../input/train.csv")
train.head()


# In[ ]:


# test dataset
test = pd.read_csv("../input/test.csv")
test.head()


# In[ ]:


train.shape


# In[ ]:


train.info()


# **Visualisations**
# 
# There are alot of ways to visualize but lets use the simple ones.

# In[ ]:


import seaborn as sns
correlations = train.corr()
sns.heatmap(correlations)


# In[ ]:


# Lets define a custom function to get a better view
# custom function to set the style for heatmap
import numpy as np
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    corr = df.corr()
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    
plot_correlation_heatmap(train)


# In[ ]:


# Calling the dataframe.pivot_table() function for assists
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
pclass_pivot = train.pivot_table(index = "assists", values = "winPlacePerc")
pclass_pivot.plot.bar()
plt.show()


# In[ ]:


# Calling the dataframe.pivot_table() function for boosts
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
pclass_pivot = train.pivot_table(index = "boosts", values = "winPlacePerc")
pclass_pivot.plot.bar()
plt.show()


# In[ ]:


# Calling the dataframe.pivot_table() function for DBNOs
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
pclass_pivot = train.pivot_table(index = "DBNOs", values = "winPlacePerc")
pclass_pivot.plot.bar()
plt.show()


# In[ ]:


# Calling the dataframe.pivot_table() function for kills
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
pclass_pivot = train.pivot_table(index = "kills", values = "winPlacePerc")
pclass_pivot.plot.bar()
plt.show()


# In[ ]:


# predictors  = [ "assists",
#                 "boosts",
#                 "damageDealt",        
#                 "DBNOs",              
#                 "headshotKills",      
#                 "heals",     
#                 "killPlace",          
#                 "killPoints",         
#                 "kills",              
#                 "killStreaks",
#                 "longestKill",        
#                 "maxPlace",           
#                 "numGroups",          
#                 "revives",            
#                 "rideDistance",       
#                 "roadKills",          
#                 "swimDistance",       
#                 "teamKills",          
#                 "vehicleDestroys",    
#                 "walkDistance",       
#                 "weaponsAcquired",    
#                 "winPoints"]

# x_train = train[predictors]
# x_train.head()


# In[ ]:


# Some Feature Engineering
train["distance"] = train["rideDistance"]+train["walkDistance"]+train["swimDistance"]
# train["healthpack"] = train["boosts"] + train["heals"]
train["skill"] = train["headshotKills"]+train["roadKills"]
train.head()


# In[ ]:


test["distance"] = test["rideDistance"]+test["walkDistance"]+test["swimDistance"]
# test["healthpack"] = test["boosts"] + test["heals"]
test["skill"] = test["headshotKills"]+test["roadKills"]
test["distance"].head()


# In[ ]:


predictors  = [ "kills",
                "maxPlace",
                "numGroups",
                "distance",
                "boosts",
                "heals",
                "revives",
                "killStreaks",
                "weaponsAcquired",
                "winPoints",
                "skill",
                "assists",
                "damageDealt",
                "DBNOs",
                "killPlace",
                "killPoints",
                "vehicleDestroys",
                "longestKill"
               ]
x_train = train[predictors]
x_train.head()


# In[ ]:


y_train = train["winPlacePerc"]
y_train.head()


# In[ ]:


# # Using Random Forest Regressor
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
# regressor.fit(x_train, y_train)


# In[ ]:


# # Finding the cross validation score with 10 folds
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(regressor, x_train, y_train, cv=10)
# print(scores)


# In[ ]:


# accuracy = scores.mean()
# print(accuracy)


# We will use scikit-learn's inbuilt feature selection classes. We will be using the feature_selection.RFECV class which performs recursive feature elimination with cross-validation.
# 
# The RFECV class starts by training a model using all of your features and scores it using cross validation. It then uses the logit coefficients to eliminate the least important feature, and trains and scores a new model. At the end, the class looks at all the scores, and selects the set of features which scored highest.

# In[ ]:


# from sklearn.feature_selection import RFECV
# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
# selector = RFECV(regressor, cv = 10)
# selector.fit(x_train, y_train)

# optimized_predictors = x_train.columns[selector.support_]
# print(optimized_predictors)


# In[ ]:


# predictors = train[optimized_predictors]
# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
# regressor.fit(predictors, y_train)


# In[ ]:


# # Finding the cross validation score with 10 folds
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(regressor, x_train, y_train, cv=10)
# print(scores)


# In[ ]:


# accuracy = scores.mean()
# print(accuracy)


# In[ ]:


# import lightgbm as lgb
# from sklearn.feature_selection import RFECV
# regressor = lgb.LGBMRegressor(objective='regression',num_leaves=5,
#                               learning_rate=0.05, n_estimators=720,
#                               max_bin = 55, bagging_fraction = 0.8,
#                               bagging_freq = 5)

# selector = RFECV(regressor, cv = 10)
# selector.fit(x_train, y_train)

# optimized_predictors = x_train.columns[selector.support_]
# print(optimized_predictors)


# In[ ]:


import lightgbm as lgb
regressor = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 20, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.8)
regressor.fit(x_train, y_train)


# In[ ]:


# Finding the cross validation score with 10 folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regressor, x_train, y_train, cv=10)
print(scores)


# In[ ]:


accuracy = scores.mean()
print(accuracy)


# In[ ]:


x_test = test[predictors]
x_test.head()


# In[ ]:


y_predict = regressor.predict(x_test)
print(y_predict)


# In[ ]:


y_predict[y_predict > 1] = 1
test['winPlacePercPredictions'] = y_predict

aux = test.groupby(['matchId','groupId'])['winPlacePercPredictions'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
    
submission = test[['Id','winPlacePerc']]


# In[ ]:


submission.to_csv("kill_them_all.csv", index=False)


# **Feature Importance**
# Lets find the importance of features.

# In[ ]:


lgb.plot_importance(regressor, max_num_features=20, figsize=(10, 8));
plt.title('Feature importance');


# In[ ]:


# Lets take the top 10 features
important_predictors  = [ "kills",
                "maxPlace",
                "numGroups",
                "distance",
                "killStreaks",
                "weaponsAcquired",
                "winPoints",
                "killPlace",
                "killPoints",
                "longestKill"
               ]
x_train = train[important_predictors]
x_train.head()


# In[ ]:


regressor = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 20, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.8)
regressor.fit(x_train, y_train)


# In[ ]:


x_test = test[important_predictors]
x_test.head()


# In[ ]:


y_predict = regressor.predict(x_test)
print(y_predict)


# In[ ]:


y_predict[y_predict > 1] = 1
test['winPlacePercPredictions'] = y_predict

aux = test.groupby(['matchId','groupId'])['winPlacePercPredictions'].agg('mean').groupby('matchId').rank(pct=True).reset_index()


# In[ ]:


aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
test.head()


# In[ ]:


# Finding the cross validation score with 10 folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regressor, x_train, y_train, cv=10)
print(scores)


# In[ ]:


accuracy = scores.mean()
print(accuracy)

