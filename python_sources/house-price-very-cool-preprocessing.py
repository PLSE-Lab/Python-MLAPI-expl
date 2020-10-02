#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
y_test = pd.read_csv('../input/sample_submission.csv')
test.shape, train.shape


# In[ ]:


train.head()


# In[ ]:


y_train = train.iloc[:,[-1]]
train = train.iloc[:,:-1]
y_test = y_test.iloc[:,[1]]
y_train.shape, y_test.shape


# In[ ]:


df_train = pd.DataFrame(train)
df_test = pd.DataFrame(test)
frames = [df_train, df_test]
dataset = pd.concat(frames)


# In[ ]:


dataset.shape


# In[ ]:


dataset


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.drop(['Id','LotFrontage','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


dataset.isnull().sum()


# In[ ]:


n_dataset = dataset.loc[:,['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','BsmtFullBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MiscVal','MoSold','MoSold']]
c_dataset = dataset.loc[:,['Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']]
n_dataset.shape, c_dataset.shape


# In[ ]:


c_df_dataset = pd.DataFrame(c_dataset)
c_df_dataset.fillna(method='bfill',inplace=True)
n_df_dataset = pd.DataFrame(n_dataset)
n_df_dataset.fillna(method='bfill',inplace=True)


# In[ ]:


c_df_dataset.isnull().sum()


# In[ ]:


n_df_dataset.isnull().sum()


# In[ ]:


c_df_dataset = pd.get_dummies(c_df_dataset,drop_first=True)


# In[ ]:


c_df_dataset.head()


# In[ ]:


n_df_dataset


# In[ ]:


# Feature Scaling
n_df_dataset_sc = pd.DataFrame(n_df_dataset)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
n_df_dataset_sc = sc_X.fit_transform(n_df_dataset_sc)
n_df_dataset_sc = pd.DataFrame(n_df_dataset_sc, columns = n_df_dataset.columns)
n_df_dataset_sc


# In[ ]:


#c_df_dataset = c_df_dataset.loc[~c_df_dataset.index.duplicated(keep='first')]


# In[ ]:


c_df_dataset = n_df_dataset_sc
c_df_dataset.shape


# In[ ]:


n_df_dataset.isnull().sum()


# In[ ]:


dataset_new = pd.concat([c_df_dataset, c_df_dataset], axis=1)
dataset_new.head()


# In[ ]:


dataset_new.isnull().sum()


# In[ ]:


dataset_new.shape


# In[ ]:


X_train = dataset_new
X_test = dataset_new
X_train = X_train.iloc[:2335,:]
X_test = X_test.iloc[2335:,:]


# In[ ]:


y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_dataset = y_train.append(y_test)

y_train = y_dataset.iloc[:2335,:]
y_test = y_dataset.iloc[2335:,:]
y_dataset.isnull().sum()


# In[ ]:


y_test


# In[ ]:


X_test


# In[ ]:


dataset_new.isnull().sum()


# In[ ]:


X_train


# In[ ]:


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)


# In[ ]:


X_test.shape, y_test.shape


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


print("Training Accuracy = ", regressor.score(X_train, y_train))
print("Test Accuracy = ", regressor.score(X_test, y_test))


# In[ ]:




