#!/usr/bin/env python
# coding: utf-8

# # BigQuery-Geotab Intersection Congestion

# * Forked from : https://www.kaggle.com/bgmello/how-one-percentile-affect-the-others
# 
# * Todo: submission code is very klunky.
# * Todo: intersection ID is non unique, it's only unique per coty. make a new one and (if not using LGBM/embedders), OHE it. 

# # Table of contents
# - [Imports and initial exploration](#imports)
# - [Exploratory Data Analysis](#eda)
#     - [Exploring street features](#streetfeatures)
# - [Preprocessing](#prepro)
# 
# - [Baseline model](#baseline)

# ## Imports and initial exploration
# <a id='imports'></a>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

sns.set_style('darkgrid')


# In[ ]:


# train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv',nrows=22345) # read in sample of data for fast experimentation

train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')


# In[ ]:


test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')
sample = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv')
with open('../input/bigquery-geotab-intersection-congestion/submission_metric_map.json') as f:
    submission_metric_map = json.load(f)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# ## Exploratory Data Analysis
# <a id='eda'></a>

# ### Exploring street features
# <a id='streetfeatures'></a>

# In[ ]:


street_features = ['EntryStreetName', 'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Path']
train[street_features].head()


# We can see path is just a concatenation of the other features, so we could drop it

# In[ ]:


train.drop('Path', axis=1, inplace=True)
test.drop('Path', axis=1, inplace=True)


# The cardinal directions can be expressed using the equation: $$ \frac{\theta}{\pi} $$
# 
# Where $\theta$ is the angle between the direction we want to encode and the north compass direction, measured clockwise.
# 
# * This is an **important** feature, as shown by janlauge here : https://www.kaggle.com/janlauge/intersection-congestion-eda
# 
# * We can fill in this code in python (e.g. based on: https://www.analytics-link.com/single-post/2018/08/21/Calculating-the-compass-direction-between-two-points-in-Python , https://rosettacode.org/wiki/Angle_difference_between_two_bearings#Python , https://gist.github.com/RobertSudwarts/acf8df23a16afdb5837f ) 
# 
# * TODO: circularize / use angles

# In[ ]:


directions = {
    'N': 0,
    'NE': 1/4,
    'E': 1/2,
    'SE': 3/4,
    'S': 1,
    'SW': 5/4,
    'W': 3/2,
    'NW': 7/4
}


# In[ ]:


train['EntryHeading'] = train['EntryHeading'].map(directions)
train['ExitHeading'] = train['ExitHeading'].map(directions)

test['EntryHeading'] = test['EntryHeading'].map(directions)
test['ExitHeading'] = test['ExitHeading'].map(directions)


# In[ ]:


train['diffHeading'] = train['EntryHeading']-train['ExitHeading']  # TODO - check if this is right. For now, it's a silly approximation without the angles being taken into consideration

test['diffHeading'] = test['EntryHeading']-test['ExitHeading']  # TODO - check if this is right. For now, it's a silly approximation without the angles being taken into consideration

train[['ExitHeading','EntryHeading','diffHeading']].drop_duplicates().head(10)


# In[ ]:





# ## Preprocessing
# <a id='prepro'></a>

# Let's create a new dataframe with the new following features: TotaTimeStopped, DistanceToFirstStop and Percentile.
# 
# Creating a dataframe in the following way can enable us to use the percentile as a feature and can help us boost the model.
# 
# * This step seems very klunky and could be improved. TODO. 

# In[ ]:


new_train_columns = ['IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName',
       'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend', 'DistanceToFirstStop',
       'Month', 'TotalTimeStopped', 'Percentile', 'City'
                    ,'diffHeading'
                    ]


# In[ ]:


new_test_columns = ['IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName',
       'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend',
       'Month', 'Percentile', 'City'
                   ,'diffHeading'
                   ]


# In[ ]:


new_train = pd.DataFrame(columns=new_train_columns)


# In[ ]:


new_test = pd.DataFrame(columns=new_test_columns)


# In[ ]:


for per in [20, 40, 50, 60, 80]:
    new_df = train.copy()
    new_df['TotalTimeStopped'] = new_df['TotalTimeStopped_p'+str(per)]
    new_df['DistanceToFirstStop'] = new_df['DistanceToFirstStop_p'+str(per)]
    new_df['Percentile'] = pd.Series([per for _ in range(len(new_df))])
    new_df.drop(['TotalTimeStopped_p20', 'TotalTimeStopped_p40',
       'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80',
       'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',
       'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',
       'TimeFromFirstStop_p80', 'DistanceToFirstStop_p20',
       'DistanceToFirstStop_p40', 'DistanceToFirstStop_p50',
       'DistanceToFirstStop_p60', 'DistanceToFirstStop_p80', 'RowId'], axis=1,inplace=True)
    new_train = pd.concat([new_train, new_df], sort=True)


# In[ ]:


for per in [20, 50, 80]:
    new_df = test.copy()
    new_df['Percentile'] = pd.Series([per for _ in range(len(new_df))])
    new_test = pd.concat([new_test, new_df], sort=True)


# In[ ]:


new_train = pd.concat([new_train.drop('City', axis=1), pd.get_dummies(new_train['City'])], axis=1)


# In[ ]:


new_test = pd.concat([new_test.drop('City', axis=1), pd.get_dummies(new_test['City'])], axis=1)


# In[ ]:


new_train = new_train.reindex(sorted(new_train.columns), axis=1)
new_test = new_test.reindex(sorted(new_test.columns), axis=1)


# In[ ]:


new_test = new_test.sort_values(by=['RowId', 'Percentile'])


# In[ ]:


X_train = np.array(new_train.drop(['EntryStreetName', 'ExitStreetName', 'IntersectionId', 'TotalTimeStopped', 'DistanceToFirstStop'], axis=1))
X_test = np.array(new_test.drop(['EntryStreetName', 'ExitStreetName', 'IntersectionId', 'RowId'], axis=1))


# In[ ]:


y_train = np.array(new_train[['TotalTimeStopped', 'DistanceToFirstStop']])


# ## Baseline model
# <a id='baseline'></a>

# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


clf = RandomForestRegressor(n_jobs=4, n_estimators=12)

clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


sample['Target'] = y_pred.reshape(-1)


# In[ ]:


l = []
for i in range(1920335):
    for j in [0,3,1,4,2,5]:
        l.append(str(i)+'_'+str(j))


# In[ ]:


sample['TargetId'] = l


# In[ ]:


sample = sample.sort_values(by='TargetId')


# In[ ]:


sample.to_csv('sample_submission.csv', index=False)

