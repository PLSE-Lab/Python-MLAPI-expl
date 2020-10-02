#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


# import modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
from datetime import datetime as dt
import seaborn as sns
# set graphics dark mode
plt.style.use('dark_background')
# import dataset
trainset = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
testset = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
# dataset quick view
trainset.head()


# The datetime field contains the information about the date and the time in a single feature. It might be interesting to have this info in two separate variables, one for the date and other for the time. The reason is that it is more usual to rent a bike in the weekdays than in the weekend and in the morning than in the night, so having this two kind of information in two separate variables, might let the model to learn this trends.

# In[ ]:


# create date feature from datetime
trainset.insert(1, 'date', pd.DataFrame([x[:10] for x in trainset.datetime]))
# create time feature from datetime
trainset.insert(2, 'time', pd.DataFrame([x[11:] for x in trainset.datetime]))
# convert datetime from string to datetime
trainset.date = [dt.strptime(x, '%Y-%m-%d').weekday() for x in trainset.date]
# drop datetime column since we created two variables and casual and registered since their value is contained in count
trainset.drop(['datetime'], axis = 1, inplace = True)
# trainset quick view
trainset.head()


# We must convert time feature to numerical values. A simple approach is to create factors. For example, if we had the list: '01:00', '02:00', '03:00', '04:00' ... we can convert it to an int factor list where we use the index of each element in the original list: 0, 1, 2, 3...

# In[ ]:


# get index of the time elements in unique list
_, idx = np.unique(trainset.time, return_inverse = True)
# replace time feature with the index just computed
trainset.time = idx
# trainset quick view
trainset.head()


# Check if there is a significative difference in the rentals for the different weekdays

# In[ ]:


# date - count boxplot
plt.figure(), sns.boxplot(x = trainset['date'], y = trainset['count'])


# So, actually, the rentals are higher in the weekdays than in the weekends, with no difference between the members of each group. Knowing this, we can change the date feature to a binary feature named weekday which is 1 from monday to friday and 0 from saturday to sunday.

# In[ ]:


# replace date with weekday
trainset.date = [1 if x >= 0 and x < 6 else 0 for x in trainset.date]
# replace feature name
trainset.rename(columns = {'date':'weekday'}, inplace = True)
# trainset quick view
trainset.head()


# Check if there are empty values in any of the samples

# In[ ]:


# check sum of nulls
trainset.isnull().sum()


# Check the distrbution and relationships of the variables.

# In[ ]:


# draw the pairplot of the variables
plt.figure(), sns.pairplot(trainset)
# check target boxplot to see outliers
plt.figure(), sns.boxplot(trainset['count'])


# In[ ]:


# apply log transform to remove the number of outliers
trainset['count'] = np.log(trainset['count'])
# repeat pairplot
plt.figure(), sns.pairplot(trainset)
# repeat boxplot
plt.figure(), sns.boxplot(trainset['count'])


# Now that all the features are numerical and we checked there are no null values, let's check the correlation between variables.

# In[ ]:


# variables correlation heatmap
plt.figure(figsize = (10,10)), sns.heatmap(trainset.corr())


# In[ ]:


# remove features highly correlated
trainset.drop(['casual','registered','temp'], axis = 1, inplace = True)
# graph heatmap again
plt.figure(figsize = (10,10)), sns.heatmap(trainset.corr())


# Let's group the time values (0 to 23) to "day segments" which we can define as 0 (dawn: 0 to 5), 1 (morning: 6 to 12), 2 (afternoon: 13 to 18) and 3 (evening: 19 to 23). This way we will have less sparse time information.

# In[ ]:


# group time values into day segments
trainset.time = [0 if x >= 0 and x < 6 else(1 if x > 5 and x < 13 else (2 if x > 12 and x < 19 else 3)) for x in trainset.time]
# trainset quick view
trainset.head()


# Prepare the test set applying the same transformations we applied for the training set.

# In[ ]:


# get original datetime column for submission
testdates = testset.datetime
# create date feature from datetime
testset.insert(1, 'date', pd.DataFrame([x[:10] for x in testset.datetime]))
# create time feature from datetime
testset.insert(2, 'time', pd.DataFrame([x[11:] for x in testset.datetime]))
# convert datetime from string to datetime
testset.date = [dt.strptime(x, '%Y-%m-%d').weekday() for x in testset.date]
# drop datetime column since we created two variables and casual and registered since their value is contained in count
testset.drop(['datetime'], axis = 1, inplace = True)
# get index of the time elements in unique list
_, idx = np.unique(testset.time, return_inverse = True)
# replace time feature with the index just computed
testset.time = idx
# replace date with weekday
testset.date = [1 if x >= 0 and x < 6 else 0 for x in testset.date]
# replace feature name
testset.rename(columns = {'date':'weekday'}, inplace = True)
# remove features highly correlated
testset.drop(['temp'], axis = 1, inplace = True)
# group time values into day segments
testset.time = [0 if x >= 0 and x < 6 else(1 if x > 5 and x < 13 else (2 if x > 12 and x < 19 else 3)) for x in testset.time]
# testset quick view
testset.head()


# Model development and predictions

# In[ ]:


# features
Xtrain = trainset.iloc[:,:-1]
Xtest = testset.iloc[:,:]
# target
ytrain = trainset.iloc[:,-1]
# standard scaler
sca = StandardScaler().fit(Xtrain)
# standarize features
Xtrain = sca.transform(Xtrain)
Xtest = sca.transform(Xtest)
# classifier
clf = RandomForestRegressor(random_state = 0)
# regularization parameter range
param_grid = {'n_estimators': [25, 50, 100], 'max_features': [3, 6]}
# grid search
grid = GridSearchCV(estimator = clf, scoring = 'neg_mean_squared_log_error', param_grid = param_grid)
# training
clf.fit(Xtrain, ytrain)
# predictions
preds = np.round(np.exp(clf.predict(Xtest)))
# clip negatives in case there are
preds[preds < 0] = 0
# submission
pd.DataFrame({'datetime': testdates, 'count': preds}).to_csv('my_submission.csv', index = False)

