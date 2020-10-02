#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets, svm
import sklearn.ensemble as ske
from sklearn.model_selection import train_test_split


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()


# ## Data Visualization

# In[ ]:


#Import the visual libs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


# ###Visual over time
# In this section we will discover the relationship between time and the demand for bike rentals

# In[ ]:


# Convert the datetime column to datetime dtype
train_df.datetime = pd.to_datetime(train_df.datetime)


# ## Demand over year
#  What do we discover
#   - Number of bike rentals by registered users are always more the number of casual users
#   - Demand of bike rental is growing over the years
#   - Demand of bike rental is peaked at the middle of the year

# In[ ]:


date_df = train_df[['datetime','casual','registered','count']]
date_df.is_copy = False
date_df['datetime'] = pd.to_datetime(date_df['datetime'])
date_df = date_df.groupby(pd.TimeGrouper(key='datetime',freq='M'))
date_df.sum().plot()


# ### Demand across season

# In[ ]:


season_df = train_df[['datetime','season','casual','registered','count']]
season_df.is_copy = False
season_df['year'] = season_df.datetime.dt.year
season_df = season_df.drop(['datetime'],axis=1)
# Change season from number to string for better understanding
def change_season(number):
    season_dict = {1:'Spring',2:'Summer',3:'Fall',4:'Winter'}
    return season_dict[number]
season_df.season.applymap(change_season)
season_df.groupby(['year','season']).mean().plot(kind='bar',stacked=True,figsize=(12,6))


# # Data processing
# - Convert the date time column to date time data type in pandas
# - Get rid of noise data in the data set

# In[ ]:


train_df['datetime'] = pd.to_datetime(train_df['datetime'])
train_df['year'] = train_df['datetime'].dt.year
train_df['month'] = train_df['datetime'].dt.month
train_df['dayofweek'] = train_df['datetime'].dt.dayofweek
train_df['hour'] = train_df['datetime'].dt.hour
# Now datetime column is redundant
train_df = train_df.drop(['datetime','season','casual','registered','atemp'],axis=1)
train_df.info()


# In[ ]:


features = train_df.drop(['count'],axis=1).values
target = train_df['count'].values


# In[ ]:


features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.2, random_state = 0)
print(features_train.shape, features_test.shape)


# In[ ]:


target_train.shape, target_test.shape


# # Different Models
# - Gradient Boosting
# - Random Forest

# In[ ]:


def model_score(model):
    model.fit(features_train,target_train)
    acc = model.score(features_test, target_test)
    print('Model accuracy is: ', acc)


# In[ ]:


method_rf = ske.RandomForestRegressor(n_estimators=20)
model_score(method_rf)


# In[ ]:


## Gradient boosting for regression
- Learning rate: this shrinks the contribution of each tree by leanrning rate.
- n_estimators: the number of boosting stages to perform. 
  Gradient boosting is fairly robust to over-fittingso a large number usually
  results in better performance


# In[ ]:


method_gb = ske.GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 500)
model_score(mrandom forest
            ethod_gb)


# In[ ]:


# Read the test data
test_df.head()


# In[ ]:


# Convert the date time
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
test_df['year'] = test_df['datetime'].dt.year
test_df['']

