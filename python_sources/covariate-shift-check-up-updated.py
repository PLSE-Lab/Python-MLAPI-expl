#!/usr/bin/env python
# coding: utf-8

# Let's find if any variable displays a distribution change between the train and the test set! It is a common problem in real-world data sets and a very important step in the feature selection process.
# 
# We will do this by trying to predict to which set each variable belongs to (train or test).

# In[ ]:


#Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import os
import json
from pandas.io.json import json_normalize


# In[ ]:


#Define function to load data (Kudos to Julian for that)
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# In[ ]:


print('Loading Train set...')
train = load_df()
print('Loading Test set...')
test = load_df("../input/test.csv")


# In[ ]:


#Impute missing values (mean for numeric, mode for categorical)
print('Imputing missing values...')
for i in train.columns:
    if train[i].dtype == 'object':
      train[i] = train[i].fillna(train[i].mode().iloc[0])
    elif (train[i].dtype != 'object'):
      train[i] = train[i].fillna(np.mean(train[i]))


for i in test.columns:
    if test[i].dtype == 'object':
      test[i] = test[i].fillna(test[i].mode().iloc[0])
    elif (test[i].dtype != 'object'):
      test[i] = test[i].fillna(np.mean(test[i]))
    

print('Nulls in train set:', train.isnull().sum().sum())
print('Nulls in test set:', test.isnull().sum().sum())


# In[ ]:


## label encode categorical variables
print('Label Encoding categorical variables...')
for col in train.columns:
    if train[col].dtype == 'object':
      train[col] = train[col].astype('category')
      train[col] = train[col].cat.codes

for col in test.columns:
    if test[col].dtype == 'object':
      test[col] = test[col].astype('category')
      test[col] = test[col].cat.codes


# In[ ]:


## Creating a dummy y label and drop the target variable
train['set'] = 0
test['set'] = 1
train = train.drop(['totals.transactionRevenue', 'trafficSource.campaignCode'],axis=1)


# In[ ]:


## Use a sample set from both train and test and concatenate into a single dataframe
train_df = train.sample(10000, random_state=697)
test_df = test.sample(10000, random_state=466)

all_data = train_df.append(test_df)
y_label = all_data['set']
all_data = all_data.drop('set',axis=1)

#Make sure the new dataframe contains all the initial features
print('New dataframe shape:', all_data.shape)


# In[ ]:


## Find all the features with covariate shift. Print during the procedure and then save in array
model = RandomForestClassifier(n_estimators = 50, max_depth = 5, min_samples_leaf = 5)
feat_to_drop = []
for col in all_data.columns:
    score = cross_val_score(model,pd.DataFrame(all_data[col]),y_label,cv=4,scoring='roc_auc')
    if np.mean(score) > 0.8:
        feat_to_drop.append(col)
    print(col,np.mean(score))


# In[ ]:


#Print number of features with covariate shift
print('Number of features with covariate shift:', len(feat_to_drop))


# 14 of our independent variables have exceeded the 0.8 AUC threshold that we set. 
# 
# As far as I know there are two main approaches to tackle shift distribution: The first one is to exclude those features with covariate shift which have been deemed unimportant in our model and the second is to produce a set of weights for those features with covariate shift which 'need' to be included in the model. The first case is simple. The second one can be performed with a Density Ratio Estimation. I will try both and will extend this kernel at a later stage. 
# 
# Some initial thoughts: 
# 'Device Browser' seems to present the greatest value but this could be due to its very high cardinality. 'adContent ' also presents high predictability of the test set but is ranked low in my model so it could be deleted. 'trafficSource.keyword' is also high so in covariate shift but has a fair impact on my model so it should be taken care of. The same goes for 'trafficSource.source'. 'geoNetwork.city' is really important in my LightGBM model but also displays covariate shift so definitely some sort of action is needed here. 'totals.pageviews' and 'totals.hits' are marginally above and below threshold repsectively but are very import explanatory variables so they will probably need some attention.
# 
# Feel free to come up with your own strategies!
# 
# ##Edit 1: 
# I excluded "device.browser" from my model and it slightly improved my PL score. My local CV score actually worsened a bit (from 1.5986 it went to 1.5992) but PL went from 1.4419 to 1.4415. This is not my current best model, I will try excluding more variables with covariate shift and then apply it to my best performing model to check the performance.
# 
# ##Edit 2: 
# Excluded "trafficSource.keyword" in addition to "device.browser" and I see some minor improvement again. Local CV down to 1.5990, PL down to 1.4409
# 
# ##Edit 3: 
# Excluded 'trafficSource.adContent'. Results seem to have worsened slightly: Local CV increased to 1.5993, PL increased to 1.4413

# 

# 
