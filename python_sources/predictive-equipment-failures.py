#!/usr/bin/env python
# coding: utf-8

# #### The Challenge:
# A data set has been provided that has documented failure events that occurred on surface equipment and down-hole equipment. For each failure event, data has been collected from over 107 sensors that collect a variety of physical information both on the surface and below the ground.
# 
# Using this data, can we predict failures that occur both on the surface and below the ground? Using this information, how can we minimize costs associated with failures?
# 
# The goal of this challenge will be to predict surface and down-hole failures using the data set provided. This information can be used to send crews out to a well location to fix equipment on the surface or send a workover rig to the well to pull down-hole equipment and address the failure.

# In[ ]:


#importing the required libraries

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from collections import Counter
warnings.filterwarnings('ignore')


# In[ ]:


#loading the given training dataset

train_set = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv')


# In[ ]:


#exploring first few rows of training data

train_set.head()


# In[ ]:


#reading the given test data

test_set = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')


# In[ ]:


#exploring first few rows of test data

test_set.head()


# ## Exploratory Data Analysis

# In[ ]:


# Counting the occurence of each target value - 0 and 1
g = sns.countplot(x='target', data = train_set)
plt.xlabel('Target')
plt.ylabel('Number of Records')


# We have an imabalanced class situation in the train data. The number of observations having downhole failures seems to be quite less. Hence it is important to evaluate the model using the F1 score instead of a common metric like accuracy.

# In[ ]:


# Concatenating train_set and test_set to clean both the sets together

data = pd.concat(objs=[train_set, test_set], axis=0).reset_index(drop=True)


# In[ ]:


# Length of training data so that later we can split our training and test data

train_len = len(train_set)


# In[ ]:


# Replacing string 'na' with NaN values

data = data.replace('na', np.NaN)


# In[ ]:


# Converting all measures to numerical data type
for col in data.columns:
    if col not in ['id', 'target']:
        data[col] = data[col].astype(np.float)


# In[ ]:


# Checking for unique values in each column in full dataset

unique_data = data.nunique().reset_index()
unique_data.columns = ['Name','Unique_Count']


# In[ ]:


# Checking the columns which have less than 2 unique values across the training and test data sets because columns with 
# constant values do not support our model's prediction

unique_data[unique_data.Unique_Count < 2]


# Here sensor54_measure seems to have a constant value across all observations. Hence it is of no use to our model. We will remove this feature. 

# In[ ]:


# Checking for null values across each column in dataset

null_df = data.isna().sum().reset_index()
null_df.columns = ['Name', 'Unique_Count']


# In[ ]:


# sorting the columns having null values

null_df.sort_values('Unique_Count', ascending=False).head(5)


# Here, all columns with more than 75% missing value have been removed.

# In[ ]:


#Outlier detection - Finding rows with more than two outlier values in columns

def outliers(df, n, features):
    
    outlier_indices = []
    
    for col in features:
        
        #1st quartile
        Q1 = np.percentile(df[col], 25)
        
        #3rd quartile
        Q3 = np.percentile(df[col], 75)
        
        #inter quartile range
        IQR = Q3 - Q1
        
        #identify index of outlier rows
        outlierlist = df[(df[col] < (Q1 - (1.5 * IQR))) | (df[col] > (Q3 + (1.5 * IQR)))].index
        
        outlier_indices.extend(outlierlist)
        
    #selecting rows with more than two outliers
    outlier_indices = Counter(outlier_indices)
    multipleoutliers = list(k for k,v in outlier_indices.items() if v > n)
    
    return multipleoutliers

#detect outliers from Age, SibSp, Fare and Parch
finaloutliers = outliers(data, 2, data.columns)


# In[ ]:


#outlier detection
data.iloc[finaloutliers]


# We do not have any rows having two or more outlier values. Hence we are not removing any observations from our dataset.

# In[ ]:


# Dropping sensor54_measure as it has constant values + nulls

data.drop(columns=['sensor54_measure'], axis=1, inplace=True)


# In[ ]:


# Removing columns which mostly have null values - more than 75%

data.drop(columns=['sensor43_measure', 'sensor42_measure', 'sensor41_measure', 'sensor40_measure', 'sensor2_measure',                   'sensor39_measure', 'sensor38_measure', 'sensor68_measure'], axis=1, inplace = True)


# Since most of the obeservations in each sensor measure is close to 0 and rest of the observations have an extremely high value, imputing missing values using mean would result in incorrect high values. Hence we choose to impute the missing values using the median of each column

# In[ ]:


# Replacing NaN with median values

for col in data.columns:
    if col not in ['id','target']:
        data[col] = data[col].fillna(data[col].median())


# In[ ]:


data.isnull().sum().sort_values(ascending=False).head(5)


# In[ ]:


# Correlation matrix between highly correlated varaibles and target value
fig, ax = plt.subplots(figsize = (18, 18))
g = sns.heatmap(data[['sensor104_measure','sensor103_measure','sensor10_measure','sensor11_measure','sensor12_measure','sensor13_measure',
     'sensor14_measure','sensor15_measure','sensor46_measure','sensor27_measure','sensor31_measure',
     'sensor32_measure','sensor33_measure',
      'sensor44_measure','sensor48_measure','sensor49_measure','sensor59_measure',
     'sensor53_measure', 'sensor78_measure', 'sensor72_measure','sensor87_measure','sensor88_measure','sensor89_measure',
     'sensor8_measure', 'sensor90_measure', 'sensor91_measure', 'sensor94_measure', 'sensor95_measure', 'target']].corr(), annot=True, ax=ax)


# Since there are multiple features that are highly correlated within themselves and with the target feature as well, we will add a few interaction terms to make the relationship more explicit

# In[ ]:


data['sensor_1415'] = data['sensor14_measure'] - data['sensor15_measure']
data['sensor_7872'] = data['sensor78_measure'] - data['sensor72_measure']
data['sensor3214'] = data['sensor32_measure'] - data['sensor14_measure']
data['sensor148'] = data['sensor14_measure'] - data['sensor8_measure']
data['sensor4615'] = data['sensor46_measure'] - data['sensor15_measure']
data['sensor815'] = data['sensor15_measure'] - data['sensor8_measure']
data['sensor468'] = data['sensor46_measure'] - data['sensor8_measure']
data['sensor278'] = data['sensor27_measure'] - data['sensor8_measure']
data['sensor8933'] = data['sensor89_measure'] - data['sensor33_measure']
data['sensor9495'] = data['sensor94_measure'] - data['sensor95_measure']
data['sensor1427'] = data['sensor14_measure'] - data['sensor27_measure']


# In[ ]:


# Dropping sensor32_measure as it mostly duplicates sensor8_measure
data.drop(columns=['sensor32_measure'], axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


# Dropping id feature as it does not affect model performance
data.drop(columns=['id'], axis=1, inplace=True)


# ## Modeling

# Below algorithms are being used in our modeling process.
# 
# - Random Forest
# - Gradient Boosting Decision Trees
# - Ada Boost
# 
# We will be comparing the performance of the models on our validation set and use the better performing one for our test set predictions. 

# In[ ]:


# train_set and test_set split
train_set = data[:train_len]
test_set = data[train_len:]


# In[ ]:


# X and y split
X = train_set.drop(labels=['target'], axis=1)
y = train_set['target']


# In[ ]:


# Train and val split

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=0)


# In[ ]:


# RF classifier

rfr = RandomForestClassifier(n_estimators = 100, random_state=0, n_jobs=4, class_weight={0:1,1:2}, verbose=1)
rfr.fit(X_train,y_train)


# In[ ]:


#Predicting validation set results
y_pred = rfr.predict(X_val)


# In[ ]:


#Checking f1 score based on validation set results
from sklearn.metrics import f1_score
f1_score(y_val, y_pred)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, loss='deviance', verbose=1)


# In[ ]:


# Fitting with train values and prediciting for validation set
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_val)


# In[ ]:


# Checking f1 score for GBDT model using validation set results
from sklearn.metrics import f1_score
f1_score(y_val, y_pred_gb)


# In[ ]:


# Ada Boost classifier
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)


# In[ ]:


# Predicting validation set results
y_pred_ada = ada.predict(X_val)

#Checking f1 score
f1_score(y_val, y_pred_ada)


# From all three results, we see that Random Forest classifier performs the best on our validation set. Hence we will be using this model to predict our test set results

# In[ ]:


test_set.drop(columns=['target'], axis=1, inplace=True)


# In[ ]:


def finalpred(testset):
    finalpred = rfr.predict(testset)
    
    prediction = pd.Series(finalpred, name = 'target')
    test_id = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')
    submission = pd.concat([test_id['id'], prediction], axis = 1)
    submission['target'] = submission['target'].astype(np.int)
    submission.to_csv('finalsub.csv', index=False)


# In[ ]:


finalpred(test_set)

