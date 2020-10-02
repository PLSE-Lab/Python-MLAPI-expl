#!/usr/bin/env python
# coding: utf-8

#  I will be using Catboost to create a model and compare it with the xgboost model
#  
#   By using Xgboost got a score of 94 and 91 for train and test data sets respectively. With got a score of 88.2 
# and Score 83.5 this is without imputing nan values,  replaced both numerical and categorical variables with -999.
#     But when numerical features were imputed got a train and test score of 88.8 and 86
#   Can I equal or better the xgboost model or is it expected to be slightly lower because of the categorical features
#   
# 
#      
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#loading Data

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


print(df.shape)
print(df_test.shape)


# In[ ]:


#pd.options.display.max_columns = None
df.head(5)


# In[ ]:


df.describe()


# In[ ]:


# get only null columns
nullcol = df.columns[df.isna().any()]


# In[ ]:


df[nullcol].isnull().sum()


# missing values as a percentage 

# In[ ]:


df[nullcol].isnull().sum() * 100 / len(df)


# In[ ]:


#dropping the columns where the missing values are over 40%

df.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)


# In[ ]:


#checking the data types of missing value columns
nullcol = df.columns[df.isna().any()]
df[nullcol].dtypes


# In[ ]:



#df.fillna(-999, inplace=True)
#df_test.fillna(-999, inplace=True)


# In[ ]:


# selecting columns where the type is strings with missing values

objcols= df[nullcol].select_dtypes(['object']).columns
objcols


# In[ ]:


#replacing the missing values of the strings with the -999

df[objcols] = df[objcols].replace(np.nan, -999)


# In[ ]:


df[objcols].isnull().sum()


# In[ ]:


#imputing numeric values

#get numerical features by dropping categorical features from the list
num_null=(nullcol.drop(objcols))

df[num_null] = df[num_null].fillna(df.mean().iloc[0])


# In[ ]:



df.columns[df.isna().any()]


# In[ ]:


#numerical data
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num_cols = df.select_dtypes(include=numerics)

#categorical data
string_cols = df.select_dtypes(exclude =numerics)


# In[ ]:



print(num_cols.shape)
print(string_cols.shape)


# In[ ]:





# # checking the trainnig data set and the test data 
# 

# In[ ]:


test_nullcol = df_test.columns[df_test.isna().any()]
df_test[test_nullcol].isnull().sum() * 100 / len(df_test)


# In[ ]:


test_nullcol = df_test.columns[df_test.isna().any()]

#string columns in test 
test_objcols= df_test[test_nullcol].select_dtypes(['object']).columns


df_test[test_objcols] = df_test[test_objcols].replace(np.nan, -999)

#get numerical features by dropping categorical features from the list
test_num_null=(test_nullcol.drop(test_objcols))


#replacing with the mean
df_test[test_num_null] = df[test_num_null].fillna(df_test.mean().iloc[0])


# In[ ]:


df_test.columns[df_test.isna().any()]


# In[ ]:


#drop features that were removed from trainig set
df_test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)


# In[ ]:


df_test.shape


# In[ ]:





# In[ ]:


#numerical data
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#numerical features in the test set
test_num_cols = df_test.select_dtypes(include=numerics)

#categorical data features
test_string_cols = df_test.select_dtypes(exclude =numerics)


# In[ ]:


print(df.shape)
print(df_test.shape)


# # lets build the the model on the training data set

# In[ ]:


#assiging X and target label y
y = df['SalePrice']
X = df.drop('SalePrice',axis=1)


# In[ ]:


print(y.shape)
print(X.shape)


# In[ ]:


categorical_features_indices = np.where(X.dtypes != np.float)[0]

#splitting data to training and testing
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=40)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


from catboost import CatBoostRegressor

model=CatBoostRegressor(iterations=1200, depth=7, learning_rate=0.1, loss_function='RMSE')


# In[ ]:



model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test))


# In[ ]:


print("Train Score", model.score(X_train,y_train))
print("Test Score", model.score(X_test,y_test))


# In[ ]:





# In[ ]:




