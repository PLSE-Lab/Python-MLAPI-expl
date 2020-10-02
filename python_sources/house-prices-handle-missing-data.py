#!/usr/bin/env python
# coding: utf-8

# I already made a kernel on the House Sales dataset, you can find it [here](https://www.kaggle.com/mukul1904/house-prices-random-forest-and-xgboost). In that, I got around 0.14 score and couldn't get more (less) than that, and I tried everythign from PCA to different models. The trick in this competetion is handling missing data.
# 
# So, in this notebook I'll focus on handling the missing values for each column, by analyzing each column separately.

# # Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import os
import sys
import tqdm
from multiprocessing import  Pool
import warnings
warnings.filterwarnings("ignore")
from math import sqrt
train_on_gpu = False

# Visualisation libs
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor


# In[ ]:


print('In input directory:')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load the data

# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

train.shape, test.shape, sample_submission.shape


# # Helper functions

# In[ ]:


def score(y_actual, y_predicted):
    # because competetion uses RMSLE
    return sqrt(mean_squared_log_error(y_actual, y_predicted))
    
def fillNaNInfinity(df):
    df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    return df

def fillInfinity(df):
    df.replace([np.inf, -np.inf], np.nan)
    return df


# # EDA

# In[ ]:


data = pd.concat([
    train.loc[:, train.columns != 'SalePrice'], test
])

target = np.log(train['SalePrice'] + 1)

data.shape, target.shape


# Check the number of null values in each column

# In[ ]:


# From https://www.kaggle.com/miguelangelnieto/pca-and-regression#Simple-Neural-Network, loved it
nans = pd.isnull(data).sum()
nans[ nans > 0 ]


# See columns which have more than 500 cells null

# In[ ]:


columns_to_remove = nans[ nans > 500 ].reset_index()['index'].tolist()
columns_to_remove


# In[ ]:


data.drop(labels=columns_to_remove, axis=1, inplace=True)
data.shape


# In[ ]:


nans = pd.isnull(data).sum()
nans[ nans > 5 ]


# Now we will look into each of above fields individually. You can see from above, most of the missing data is of basement and garage.
# Before that, let's try to see if we just remove all of them, what will happen

# In[ ]:


df = data.copy() # don't want to modify orginal data

more_columns_to_remove = nans[ nans > 5 ].reset_index()['index'].tolist()
df.drop(labels=more_columns_to_remove, axis=1, inplace=True)
print(df.shape)
df = pd.get_dummies(df)
print(df.shape)
xgbr = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7)
X = df[:1460]
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

xgbr.fit(X_train, y_train)

score(np.exp(y_train) - 1, np.exp(xgbr.predict(X_train)) - 1), score(np.exp(y_test) - 1, np.exp(xgbr.predict(X_test)) - 1)


# Not a great score, so let's continue with our handling missing data route

# # Handle missing data

# ## LotFrontage

# In[ ]:


data['LotFrontage'].describe()


# In[ ]:


sns.distplot(data['LotFrontage'].fillna(0));


# In[ ]:


data['Neighborhood'].describe()


# In[ ]:


data.groupby('Neighborhood')['LotFrontage'].agg(['count', 'mean', 'median'])


# In[ ]:


data[ data['LotFrontage'].isnull() ]['Neighborhood'].reset_index()['Neighborhood'].isnull().sum()


# This means, all the records which have LotFrontage has null, but they have Neighborhood filled. From that we can find the LotFrontage, as it shouldn't be too different from other properties.

# In[ ]:


# https://stackoverflow.com/questions/39480997/how-to-use-a-user-function-to-fillna-in-pandas
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda group: group.fillna(group.median()))

data['LotFrontage'].isnull().sum()


# ## MasVnrType and MasVnrArea

# In[ ]:


data['MasVnrType'].describe()


# In[ ]:


data['MasVnrType'].value_counts()


# In[ ]:


data['MasVnrArea'].describe()


# In[ ]:


sns.distplot(data['MasVnrArea'].fillna(0));


# In[ ]:


data.groupby('MasVnrType')['MasVnrArea'].agg(['mean', 'median', 'count'])


# In[ ]:


# data.drop(labels=['MasVnrArea'], axis=1, inplace=True)

# I checked that model works . better if we keep these 2 columns


# ## Basement

# In[ ]:


data[['BsmtCond', 'BsmtQual']].describe()


# In[ ]:


data[['BsmtCond', 'BsmtQual']].isnull().sum()


# In[ ]:


data['BsmtCond'].value_counts()


# In[ ]:


data['BsmtQual'].value_counts()


# In[ ]:


# NA is not available
data[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].fillna('NA', inplace=True)


# ## Garage

# In[ ]:


data[[
    'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual' 
]].describe()


# Coming soon, garage missing data handling

# # Training model

# One hot-encoding

# In[ ]:


df = data.copy()
df = pd.get_dummies(df)
df.shape


# In[ ]:


xgbr = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7)
X = df[:1460]
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

xgbr.fit(X_train, y_train)


# In[ ]:


score(np.exp(y_train) - 1, np.exp(xgbr.predict(X_train)) - 1), score(np.exp(y_test) - 1, np.exp(xgbr.predict(X_test)) - 1)


# The score is little better than when we just removed all null value columns, and far better than [when we just replaced NaNs with 0](https://www.kaggle.com/mukul1904/house-prices-random-forest-and-xgboost).

# In[ ]:


test = df[1460:]
sample_submission['SalePrice'] = xgbr.predict(test)
sample_submission['SalePrice'] = np.exp(sample_submission['SalePrice']) - 1
sample_submission.head()


# In[ ]:


sample_submission.to_csv('submission.csv', index=False)

