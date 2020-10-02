#!/usr/bin/env python
# coding: utf-8

# # Welcome to the House Prices competition
# This notebook is a starter code for all beginners and easy to understand. We will give an introduction to analysis and feature engineering.
# 
# Therefore we focus on
# 
# * a simple analysis of the data,
# * create new features,
# * encoding and
# * scale data. 
# 
# We use categorical feature encoding techniques, compare
# https://www.kaggle.com/drcapa/categorical-feature-encoding-challenge-xgb
# 
# The data set includes a lot of features. The data description ist here: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# 
# We label the **necessary** operations and the operations for **advanced** feature engeneering. So for the first run you can skip the advanced feature engeneering.
# 
# For prediction we use a simple XGB Regressor.

# # Load Libraries

# In[ ]:


import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import os


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop,Adam
import keras.backend as K


# # Input path

# In[ ]:


path_in = '../input/house-prices-advanced-regression-techniques/'
os.listdir(path_in)


# # Load data

# In[ ]:


train_data = pd.read_csv(path_in+'train.csv', index_col=0)
test_data = pd.read_csv(path_in+'test.csv', index_col=0)
samp_subm = pd.read_csv(path_in+'sample_submission.csv', index_col=0)


# 
# # Overview

# In[ ]:


print('number train samples: ', len(train_data.index))
print('number test samples: ', len(test_data.index))


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


samp_subm.head()


# # Concatenate train and test data
# We recomment to concatenate the train and test data for a better preparation. We have to deal with missing values and encode features. So we don't want to do all operations twice. First we extract the target feature *SalePrice* and then we concatenate. After the preparation we split the data to X_train and X_test based on the *Id*.

# In[ ]:


y_train = train_data['SalePrice'].copy().to_frame()
del train_data['SalePrice']
y_train.head()


# In[ ]:


data = pd.concat([train_data, test_data])


# In[ ]:


data


# # Handle missing values (necessary)
# There are a lot of missing values we have to deal with. First we need to know which feature has missing values and which datetype it is. 

# In[ ]:


cols_with_missing_values = [col for col in data.columns if data[col].isnull().any()]


# In[ ]:


print('# of cols with missing values: ', len(cols_with_missing_values))
print('columns with missing data:', cols_with_missing_values)


# Show the number of different datatypes in the data set.

# In[ ]:


data.dtypes.value_counts()


# Show the number of different datatypes in the data set with missing values.

# In[ ]:


data[cols_with_missing_values].dtypes.value_counts()


# There are many opportunities to deal with missing values. Additionally we could create new features and label the rows with missing values. This we have to do before handling missing values. This information could be helpful later.

# In[ ]:


for col in cols_with_missing_values:
    data[col + '_was_missing'] = data[col].isnull()


# The easiest way to hanlde missing values is to say take the most frequent values. So you can deal with both numerical and categorical features.

# In[ ]:


for col in cols_with_missing_values:
    data[col] = data[col].fillna(data[col].value_counts().index[0])


# For more options look here: https://www.kaggle.com/drcapa/titanic-eda-feature-enginneering-xgb <br>
# This is also a starter code.

# ## Encoding data (necessary)
# The easiest way to encode non numerical values is to use the one-hot-encoding. <br>
# 
# For a more goal-oriented encoding we recommend another competition: https://www.kaggle.com/drcapa/categorical-feature-encoding-challenge-xgb <br>
# There are some classes of features we have to encode with different techniques: <br>
# 1) categorical features, <br>
# 2) binary features, <br>
# 3) ordinal features. <br>

# First we extract als features for the one-hot-encoding and write the names into a list. And we wirte the cols for scaling into another list.

# In[ ]:


cols_for_one_hot_encdoing = []
cols_for_scale = []
for col in data.columns:
    if data[col].dtypes == 'object':
        cols_for_one_hot_encdoing.append(col)
    elif data[col].dtypes != 'bool':
        cols_for_scale.append(col)


# In[ ]:


data = pd.get_dummies(data, columns = cols_for_one_hot_encdoing, prefix = cols_for_one_hot_encdoing)


# # Scale data (advanced)

# In[ ]:


for col in cols_for_scale:
    data[col] = data[col].astype('float32')
    mean = data[col].mean(axis=0)
    data[col] -= data[col].mean(axis=0)
    std = data[col].std(axis=0)
    data[col] /= data[col].std(axis=0)


# # Define X_train and X_test

# In[ ]:


X_train = data.loc[train_data.index]
X_test = data.loc[test_data.index]


# # Split train and val data

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.001, random_state=2019)


# # Define Model
# We use a simple XGB Regressor.

# In[ ]:


model_XGB = XGBRegressor(n_estimators=1000)
model_XGB.fit(X_train, y_train)
y_val_pred = model_XGB.predict(X_val)
np.sqrt(mean_squared_log_error(y_val, y_val_pred))


# In[ ]:


y_test = model_XGB.predict(X_test)


# # Write output for submission

# In[ ]:


output = pd.DataFrame({'Id': samp_subm.index,
                       'SalePrice': y_test})
output.to_csv('submission.csv', index=False)


# In[ ]:


output.head()

