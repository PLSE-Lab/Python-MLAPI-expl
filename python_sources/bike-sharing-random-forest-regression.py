#!/usr/bin/env python
# coding: utf-8

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
print(os.listdir("../input/bike-sharing-dataset"))
# Any results you write to the current directory are saved as output.


# In[ ]:


raw = pd.read_csv('../input/bike-sharing-dataset/hour.csv')


# In[ ]:


raw.head()


# ## Data Exploration

# This dataset contains the hourly and daily count of rental bikes between years 2011 and 2012 in Capital bikeshare system in Washington, DC with the corresponding weather and seasonal information.
# 
# ### Content
# Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv
# 
# * instant: Record index
# * dteday: Date
# * season: Season (1:springer, 2:summer, 3:fall, 4:winter)
# * yr: Year (0: 2011, 1:2012)
# * mnth: Month (1 to 12)
# * hr: Hour (0 to 23)
# * holiday: weather day is holiday or not (extracted from Holiday Schedule)
# * weekday: Day of the week
# * workingday: If day is neither weekend nor holiday is 1, otherwise is 0.
# * weathersit: (extracted from Freemeteo)
#   1.  Clear, Few clouds, Partly cloudy, Partly cloudy
#   2.  Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#   3.  Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#   4.  Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# * temp: Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
# * atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# * hum: Normalized humidity. The values are divided to 100 (max)
# * windspeed: Normalized wind speed. The values are divided to 67 (max)
# * casual: count of casual users
# * registered: count of registered users
# * cnt: count of total rental bikes including both casual and registered

# In[ ]:


raw.info()


# In[ ]:


raw.describe()


# ## Data Visualization

# In[ ]:


raw.hist(figsize=(12,10))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(18,13))
sns.heatmap(raw.corr(), annot=True)


# ## Categorical variables to dummy variables

# In[ ]:


def generate_dummies(df, dummy_column):
    dummies = pd.get_dummies(df[dummy_column], prefix=dummy_column)
    df = pd.concat([df, dummies], axis=1)
    return df

X = pd.DataFrame.copy(raw)
dummy_columns = ["season", "yr", "mnth", "hr", "weekday", "weathersit"]
for dummy_column in dummy_columns:
    X = generate_dummies(X, dummy_column)


# In[ ]:


X.head()


# In[ ]:


#remove the original categorical variables: "season", "yr", "mnth", "hr", "weekday", "weathersit"

for dummy_column in dummy_columns:
    del X[dummy_column]


# In[ ]:


X.head()


# In[ ]:


first_5_weeks = 5*7*24 # 3 weeks (7 days), 24 hours each day
X[:first_5_weeks].plot(x='dteday', y='cnt', figsize=(18, 5))


# In[ ]:


## We use 'cnt' as the response variable. We drop 'casual' and 'registered'

y = X['cnt']
del X['cnt']
del X['casual']
del X['registered']


# In[ ]:


## drop also the variables 'instant' and 'dteday' since they are irrelevant

del X['instant']
del X['dteday']


# In[ ]:


X.head()


# ## Split the data set into training set and testing set, with 80% as training set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 22, test_size = 0.2)


# ## Random Forest Regression model

# In[ ]:


# Grid Search
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

regressor = RandomForestRegressor()
parameters = [{'n_estimators' : [150,200,250,300], 'max_features' : ['auto','sqrt','log2']}]
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# In[ ]:


best_parameters


# In[ ]:


# Random Forest Regression model
# Use the best parameters found from above to build the model

regressor = RandomForestRegressor(n_estimators = 300, max_features = 'auto') 
regressor.fit(X_train,y_train)

# Predicting the values 

y_pred = regressor.predict(X_test) 


# In[ ]:


# Comparing predicted values with true values in testing set

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)


# In[ ]:


# Using k-fold cross validation to evaluate the performance of the model

from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv =10)
accuracy.mean()


# In[ ]:


# Relative importance of features 

feature_importance = regressor.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(12,10))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[ ]:




