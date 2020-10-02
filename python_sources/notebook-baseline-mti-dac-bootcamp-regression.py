#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/mti-bootcamp-day-3/train_dataset.csv')
test = pd.read_csv('../input/mti-bootcamp-day-3/test_dataset.csv')
sample = pd.DataFrame(columns = ['Id', 'SalePrice'])
sample['Id'] = test['Id']


# In[ ]:


train.head()


# # EDA

# In[ ]:


train.describe()


# In[ ]:


train.columns


# In[ ]:


train.dtypes


# In[ ]:


train.isnull().sum().sort_values(ascending = False).head(10)


# In[ ]:


plt.figure(figsize= [15,15])
sns.heatmap(train.corr().abs(), annot = True, square = True)


# In[ ]:


train.corr()['SalePrice'].abs().sort_values(ascending = False)


# In[ ]:


sns.distplot(train['SalePrice'])


# # Data Preprocessing

# In[ ]:


df_all = pd.concat([train, test], axis = 0)


# ## Missing Values

# In[ ]:


categorical_col = ['Alley', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'FireplaceQu', 'GarageQual', 'HouseStyle', 'LotShape', 'MSZoning', 'MiscFeature', 
                   'PoolQC', 'Street', 'Utilities']

for col in categorical_col:
  df_all[col] = df_all[col].fillna('Tidak Ada')


# In[ ]:


# Missing Values imputation
numerical_col = ['BsmtFullBath', 'BsmtHalfBath', 'LotFrontage', 'TotalBsmtSF']
for col in numerical_col:
  df_all[col] = df_all[col].fillna(df_all[col].median())  


# In[ ]:


df_all.shape


# ## One-hot Encoding

# In[ ]:


df_all = pd.get_dummies(df_all, columns = categorical_col, drop_first = False)
df_all.shape


# # Feature Engineering

# In[ ]:


df_all['Total_Bath'] = df_all['FullBath'] + (0.5*df_all['HalfBath']) + df_all['BsmtFullBath'] + (0.5*df_all['BsmtHalfBath'])
df_all = df_all.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis = 1)


# #Feature Selection

# In[ ]:


df_all.head()


# In[ ]:


df_all = df_all.drop(['Id', 'MiscVal'], axis = 1)


# # Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler

train_preprocessed = df_all[:train.shape[0]]
test_preprocessed = df_all[train.shape[0]:]

X = train_preprocessed.drop(['SalePrice'], axis = 1)
y = train['SalePrice']
X_subm = test_preprocessed.drop(['SalePrice'], axis = 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_subm = scaler.transform(X_subm)


# # Modelling

# In[ ]:


# Split Train - Validation Set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.1, shuffle = True, random_state = 101)


# In[ ]:


# Model Training
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

model = LinearRegression()
model.fit(X_train, y_train)

print('MSLE train : ', mean_squared_log_error(y_train, abs(model.predict(X_train))))


# In[ ]:


# Validation
y_val_pred = model.predict(X_val)

print('MSLE : ', mean_squared_log_error(y_val, y_val_pred))


# In[ ]:


print('Intercept :' , model.intercept_)
print('Coef : ', model.coef_)


# # Submission

# In[ ]:


y_subm = model.predict(X_subm)


# In[ ]:


sample['SalePrice'] = y_subm


# In[ ]:


sample.to_csv('submission.csv', index = False)

