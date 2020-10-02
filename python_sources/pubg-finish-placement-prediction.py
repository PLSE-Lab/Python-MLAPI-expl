#!/usr/bin/env python
# coding: utf-8

# # PUBG Finish Placement Prediction
# 
# Autor: Daniel Martinez Bielostotzky

# ## Table of contents
# * **Imports: Dataset, Libraries and Usefull Functions**
# * **Preprocessing: Missing Values**
# * **Feature Engenieering: Team and Match Features**
# * **Feature Selection and Outliers**
# * **LightGBM Model**
# * **Test Data Prediction and Submit**

# ## Imports: Dataset, Libraries and Usefull Functions
# 
# For this notebook, I'll use two functions that are from a kind of EDA framework that I always use and that its open to contributions on [GitHub](https://github.com/Bielos/EDA-Framework).

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def get_null_observations(dataframe, column):
    return dataframe[pd.isnull(dataframe[column])]

def delete_null_observations(dataframe, column):
    fixed_df = dataframe.drop(get_null_observations(dataframe,column).index)
    return fixed_df
    
def get_missing_data_table(dataframe):
    total = dataframe.isnull().sum()
    percentage = dataframe.isnull().sum() / dataframe.isnull().count()
    
    missing_data = pd.concat([total, percentage], axis='columns', keys=['TOTAL','PERCENTAGE'])
    return missing_data.sort_index(ascending=True)

df = pd.read_csv('../input/train_V2.csv')
df.head()


# ## Preprocessing: Missing Values
# 
# To check the integrity of the data, the missing values and data types are displayed.
# ### Missing Vales
# Using *get_missing_data_table* we can see that the training dataset has only one record with a missing value in the 'winPlacePerc' column since it is the target column no completion method can be applied.

# In[ ]:


get_missing_data_table(df)


# In[ ]:


df = delete_null_observations(dataframe=df, column='winPlacePerc')
get_missing_data_table(df)


# ## Feature Engenieering: Team and Match Features
# Grouping records by *groupId* and *matchId* the features *teamKills* (Sum of kills in the team), *teamSize* (Total number of players in the team), *matchKills* and *matchSize* are created. 

# In[ ]:


# Adding team features
df_team_dict = (df.groupby('groupId', as_index = True)
          .agg({'Id':'count', 'kills':'sum'})
          .rename(columns={'Id':'teamSize', 'kills':'teamKills'})).to_dict()

teamKills = []
teamSize = []

for teamId in df['groupId']:
    teamKills.append(df_team_dict['teamKills'][teamId])
    teamSize.append(df_team_dict['teamSize'][teamId])

df['teamKills'] = teamKills
df['teamSize'] = teamSize
df.head()


# In[ ]:


# Adding match features
df_team = (df.groupby('groupId', as_index = False)
          .agg({'Id':'count', 'matchId':lambda x: x.unique()[0], 'kills':'sum'})
          .rename(columns={'Id':'teamSize', 'kills':'teamKills'})).reset_index()

df_match = (df_team.groupby('matchId', as_index = True)
           .agg({'teamSize':'sum', 'teamKills':'sum'})
           .rename(columns={'teamSize':'matchSize', 'teamKills':'matchKills'})).to_dict()
matchSize = []
matchKills = []

for matchId in df['matchId']:
    matchSize.append(df_match['matchSize'][matchId])
    matchKills.append(df_match['matchKills'][matchId])

df['matchSize'] = matchSize
df['matchKills'] = matchKills
df.head()


# ## Feature Selection and Outliers
# 
# Features that represent IDs are meaningless for any model so they are dropped out. 

# In[ ]:


#Drop insignificant features
df.drop(['Id'], axis='columns', inplace=True)
df.drop(['groupId'], axis='columns', inplace=True)
df.drop(['matchId'], axis='columns', inplace=True)
df.head()


# ### Outliers
# Some records may be rare cases and may affect the generalization power of the model because they are just noise.
# 
# The outliers to be deleted are:
# 1.  Records with low *matchDuration* (According to box plot) 
# 1. Players with 0 *rideDistance* and *roadKills* greater than 0

# In[ ]:


# matchDuration boxplot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sn.boxplot(data=df['matchDuration'], ax= ax)
ax.set(title='Match Duration Box Plot')
plt.show()


# In[ ]:


# Delete Outliers according to matchDuration
previous_record_size = df.shape[0]

h_spread = df['matchDuration'].quantile(.75) - df['matchDuration'].quantile(.25)
limit = df['matchDuration'].quantile(.25) - 2 * h_spread
df.drop(df[df['matchDuration'] < limit].index, inplace=True)

new_record_size = df.shape[0]
print('Total records deleted: {} ({:.7%} of previous record size)'.format(previous_record_size - new_record_size, 1 - new_record_size / previous_record_size))


# In[ ]:


# Delete Outliers according to rideDistance and roadKills
previous_record_size = df.shape[0]

df.drop(df.query('rideDistance == 0 and roadKills > 0').index, inplace=True)

new_record_size = df.shape[0]
print('Total records deleted: {} ({:.7%} of previous record size)'.format(previous_record_size - new_record_size, 1 - new_record_size / previous_record_size))


# ## LightGBM Model
# 
# A LightGBM model is used to predict the target *winPlacePerc*,  the model use 15000 iterations, 70% of features and 90% of training data per tree

# In[ ]:


# Label encode matchType

from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
df['matchType'] = encoder.fit_transform(df['matchType'])

df.head()


# In[ ]:


# X and y split
y = df['winPlacePerc'].values
X = df.drop(['winPlacePerc'], axis='columns').values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#LightGBM
import lightgbm as lgb

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=[12])
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# set matchType

params = {
        "objective" : "regression",
        "metric" : "mae",
        "n_estimators":15000,
        "early_stopping_rounds":100,
        "num_leaves" : 31, 
        "learning_rate" : 0.05, 
        "bagging_fraction" : 0.9,
        "bagging_seed" : 0, 
        "num_threads" : 4,
        "colsample_bytree" : 0.7
        }

model = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5,
                verbose_eval=1000)


# ## Test Data Prediction and Submit

# In[ ]:


df_test = pd.read_csv('../input/test_V2.csv')
df_test['matchType'] = encoder.transform(df_test['matchType'])
df_test_team_dict = (df_test.groupby('groupId', as_index = True)
          .agg({'Id':'count', 'kills':'sum'})
          .rename(columns={'Id':'teamSize', 'kills':'teamKills'})).to_dict()

teamKills_test = []
teamSize_test = []

for teamId in df_test['groupId']:
    teamKills_test.append(df_test_team_dict['teamKills'][teamId])
    teamSize_test.append(df_test_team_dict['teamSize'][teamId])

df_test['teamKills'] = teamKills_test
df_test['teamSize'] = teamSize_test

df_team_test = (df_test.groupby('groupId', as_index = False)
          .agg({'Id':'count', 'matchId':lambda x: x.unique()[0], 'kills':'sum'})
          .rename(columns={'Id':'teamSize', 'kills':'teamKills'})).reset_index()

df_match_test = (df_team_test.groupby('matchId', as_index = True)
           .agg({'teamSize':'sum', 'teamKills':'sum'})
           .rename(columns={'teamSize':'matchSize', 'teamKills':'matchKills'})).to_dict()
matchSize_test = []
matchKills_test = []

for matchId in df_test['matchId']:
    matchSize_test.append(df_match_test['matchSize'][matchId])
    matchKills_test.append(df_match_test['matchKills'][matchId])

df_test['matchSize'] = matchSize_test
df_test['matchKills'] = matchKills_test

X_testdata = df_test.drop(['Id','groupId','matchId'], axis='columns').values

df_test['winPlacePerc'] = model.predict(X_testdata, num_iteration=model.best_iteration)
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)
print('Done!')

