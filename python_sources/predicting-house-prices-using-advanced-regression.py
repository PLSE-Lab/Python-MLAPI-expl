#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score


# # Importing The The Train And Test Dataset

# In[ ]:


dataset_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
dataset_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


dataset_train.head()


# In[ ]:


dataset_train.shape


# In[ ]:


dataset_test.head()


# In[ ]:


dataset_test.shape


# # Combining the Test and Train Dataset
# 
# **I am doing so because it will make some easy doing and we will have less data to be treated as both test and train are merged together**

# In[ ]:


dataset = pd.concat([dataset_train, dataset_test])


# # Checking Null values in dataset
# **First of all we will check is there any null or nan value in our dataset.For This I will use two methods.**
# 
# By using inbuilt method of our data ,i.e., isnull() method
# By using heatmap function of seaborn library

# In[ ]:


sns.heatmap(dataset.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


null_value_train = dict(dataset.isnull().sum())
for i,j in null_value_train.items():
    print(i,"==>",j)


# # Droping Columns Having High Amount of Nan Value For Dataset
# 
# **For Train Dataset As We can See PoolQC,Alley,Fence,MiscFeature Have High Amount of Nan Value so Drop Them.**
# **Also I will Drop Id Column As it has no value in determining the Price of Houses.**

# In[ ]:


dataset.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Id'], axis=1, inplace=True)


# # Inserting new Values at the place of missing data in dataset

# In[ ]:


dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode())
dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())
dataset['Utilities'] = dataset['Utilities'].fillna(dataset['Utilities'].mode())
dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].mode())
dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].mode())
dataset['MasVnrType'] = dataset['MasVnrType'].fillna(dataset['MasVnrType'].mode())
dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean())
dataset['BsmtQual'] = dataset['BsmtQual'].fillna(dataset['BsmtQual'].mode())
dataset['BsmtCond'] = dataset['BsmtCond'].fillna(dataset['BsmtCond'].mode())
dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna(dataset['BsmtExposure'].mode())
dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(dataset['BsmtFinType1'].mode())
dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(dataset['BsmtFinSF1'].mean())
dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna(dataset['BsmtFinType2'].mode())
dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(dataset['BsmtFinSF2'].mean())
dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(dataset['BsmtUnfSF'].mean())
dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].mean())
dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode())
dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(dataset['BsmtFullBath'].median())
dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(dataset['BsmtHalfBath'].median())
dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].mode())
dataset['Functional'] = dataset['Functional'].fillna(dataset['Functional'].mode())        
dataset['GarageType'] = dataset['GarageType'].fillna(dataset['GarageType'].mode())
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].median())
dataset['GarageFinish'] = dataset['GarageFinish'].fillna(dataset['GarageFinish'].mode())
dataset['GarageCars'] = dataset['GarageCars'].fillna(dataset['GarageCars'].median())
dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean())
dataset['GarageQual'] = dataset['GarageQual'].fillna(dataset['GarageQual'].mode())        
dataset['GarageCond'] = dataset['GarageCond'].fillna(dataset['GarageCond'].mode())
dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna(dataset['FireplaceQu'].mode())
dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].mode())


# # Encoding categorical Columns using Pandas Library

# In[ ]:


dataset = pd.get_dummies(dataset, drop_first=True)


# # Making our dependent and independent features
# 
# **Now making our dependent and independent features to test our model and predict for future values.**

# In[ ]:


dataset_train_1 = dataset.iloc[:1460, :]
dataset_test_1 = dataset.iloc[1460:, :]


# In[ ]:


y_train = dataset_train_1['SalePrice'].values
dataset_train_1 = dataset_train_1.drop('SalePrice', axis=1)
dataset_test_1 = dataset_test_1.drop('SalePrice', axis=1)


# In[ ]:


X_train = dataset_train_1.iloc[:, :].values
X_test = dataset_test_1.iloc[:, :].values


# # Traing the train dataset 

# In[ ]:


from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)


# # Calculating the R^2 for our training set
# **I am calculating the R^2 for my training dataset to see how well my model is adapted to the train dataset to predict housing price.**

# In[ ]:


y_pred_train = regressor.predict(X_train)
print(r2_score(y_train,y_pred_train))


# # Prediction for test Set

# In[ ]:


y_pred_test = regressor.predict(X_test)


# In[ ]:


output = pd.DataFrame({'Id': dataset_test.Id, 'SalePrice': y_pred_test})
output.to_csv('my_submission_house_prediction_3.csv', index=False)
print("Your submission was successfully saved!")

