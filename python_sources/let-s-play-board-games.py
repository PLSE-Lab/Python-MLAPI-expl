#!/usr/bin/env python
# coding: utf-8

# ![](https://images.pexels.com/photos/776654/pexels-photo-776654.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1650)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


games = pd.read_csv('../input/games.csv')
games.head()


# In[ ]:


print (games.shape)


# In[ ]:


# Target variable is Average Rating. So let us explore its variation.
plt.hist(games['average_rating'])
plt.xlabel('Average Rating')
plt.ylabel('Frequency')


# In[ ]:


games.describe()


# **Data Cleanup**
# 
# From description of data it can be noted that there is data with negative years and very large play time. Let us filter out the data with negative year published, large playtime, and games with no ratings. 

# In[ ]:


# For a game with zero average rating, there are no users rated. 
games[games['average_rating']==0].iloc[0]


# In[ ]:


# For a game with average rating >0, user have rated game. 
games[games['average_rating']>0].iloc[0]


# In[ ]:


games_cleaned = games[games['users_rated']>0]
games_cleaned = games_cleaned[games_cleaned['yearpublished']>1900]
games_cleaned = games_cleaned[games_cleaned['maxplaytime']<500]

# Drop any row with no data
games_cleaned = games_cleaned.dropna(axis=0)


# In[ ]:


games_cleaned.describe()


# In[ ]:


print ("Percentage data filtered {:.2f}".format((games.shape[0]-games_cleaned.shape[0])/games.shape[0]*100)) # This is quite significant. 


# In[ ]:


games_cleaned.corr()["average_rating"]


# **Some interesting observations**
# 
# * Age : Very few people older than 50, play board games
# * Average rating: Right skewed distribution
# * Total number of comments and total number of owners are proportional which is expected.
# * More and more games are published yearly. Nearly with exponential rize. 

# In[ ]:


sns.pairplot(games_cleaned[['total_comments','yearpublished','playingtime','total_owners','minage','average_rating']])


# In[ ]:


df_year_mean  = games_cleaned.groupby(['yearpublished']).mean()
df_year_mean.reset_index(inplace=True)


# In[ ]:


df_year_mean.head()


# In[ ]:


df_year_mean.plot.scatter('yearpublished','average_rating',alpha=0.6,sizes=(10, 100))
plt.xlabel('Year Published')
plt.ylabel('Mean Rating')


# 1. Average rating is closely correlated average weight and ids. Id probably due to age of the id created in the database of the games. Columns 'Bayes Average Rating' (~average rating), board game type and name are dropped from the features.

# In[ ]:


y = games_cleaned['average_rating']
X = games_cleaned.drop(['average_rating','type','name','id','bayes_average_rating'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


print("Training set has {} samples.".format(X_train.shape[0]))
print("Validation set has {} samples.".format(X_val.shape[0]))


# In[ ]:


# Linear Regression
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

LinModel = LinearRegression()
LinModel.fit(X_train,y_train)
y_pred = LinModel.predict(X_val)

error = math.sqrt(mean_squared_error(y_pred,y_val))
print (f'Root mean squared error is {error}')

print ('Coefficients of the linear model are',LinModel.coef_)
print('Intercept of the model is',LinModel.intercept_)


# In[ ]:


def report_coef(names,coef,intercept):
    r = pd.DataFrame( { 'coef': coef, 'positive': coef>=0  }, index = names )
    r = r.sort_values(by=['coef'])
    display(r)
    print("Intercept: {}".format(intercept))
    r['coef'].plot(kind='barh', color=r['positive'].map({True: 'b', False: 'r'}))
    
column_names = X_train.columns.tolist()
report_coef(
  column_names,
  LinModel.coef_,
  LinModel.intercept_)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RF_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
RF_model.fit(X_train, y_train)
RF_predictions = RF_model.predict(X_val)
mean_squared_error(RF_predictions, y_val)


# In[ ]:


import lightgbm as lgb
train_data = lgb.Dataset(X_train,label=y_train)
test_data = lgb.Dataset(X_val,label=y_val)
param = {'num_leaves':131, 'num_trees':100, 'objective':'regression','metric':'rmse'}
num_round = 1000
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])

