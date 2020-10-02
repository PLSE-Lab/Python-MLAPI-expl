#!/usr/bin/env python
# coding: utf-8

# #  <font color='green'> House Prices : Data cleaning, visualization and modeling  </font>

# In[ ]:


# Import packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math


# <font color='blue'>  Importing **train** and **test** datasets </font>

# In[ ]:


sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
#Creating a copy of the train and test datasets
c_test  = test.copy()
c_train  = train.copy()


# * <font color='blue'>  Getting information about train dataset </font>

# In[ ]:


c_train.head()


# 
# * <font color='blue'>  Getting information about test dataset </font>
# 

# In[ ]:


c_test.head()


#  <font color='blue'> 1. We have 81 columns.
# 2. Our target variable is SalePrice.
# 3. Id is just an index that we can drop but we will need it in the final submission.
# 1. We have many missing values </font>
# 
# 
#  <font color='red'>   * * * * we have 79 features in our dataset. </font>
# 
# 

# 
# * <font color='blue'>  Concat Train and Test datasets </font>
# 

# In[ ]:


c_train['train']  = 1
c_test['train']  = 0
df = pd.concat([c_train, c_test], axis=0,sort=False)


# #  <font color='green'> Data preprocessing </font>

# 
# * <font color='blue'>  Calculating the percentage of missing values of each feature </font>
# 

# In[ ]:


#Percentage of NAN Values 
NAN = [(c, df[c].isna().mean()*100) for c in df]
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])


# * <font color='blue'>  Features with more than 50% of missing values. </font>

# In[ ]:


NAN = NAN[NAN.percentage > 50]
NAN.sort_values("percentage", ascending=False)


# * <font color='blue'>  We can drop PoolQC, MiscFeature, Alley and Fence features because they have more than 80% of missing values. <font>

# In[ ]:


#Drop PoolQC, MiscFeature, Alley and Fence features
df = df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)


# * <font color='blue'>  Now we will select numerical and categorical features  <font>

# In[ ]:


object_columns_df = df.select_dtypes(include=['object'])
numerical_columns_df =df.select_dtypes(exclude=['object'])


# 
# * <font color='blue'>  **Categorical Features** :  <font>

# In[ ]:


object_columns_df.dtypes


# * <font color='blue'>  **Numerical Features** :  <font>

# In[ ]:


numerical_columns_df.dtypes


# 
# * <font color='blue'>  Deeling with **categorical** feature  <font>

# In[ ]:


#Number of null values in each feature
null_counts = object_columns_df.isnull().sum()
print("Number of null values in each column:\n{}".format(null_counts))


# 
# * <font color='blue'>   We will fill -- **BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, GarageType, GarageFinish, GarageQual, FireplaceQu, GarageCond** -- with "None" (Take a look in the data description). </font>
# * <font color='blue'>    We will fill the rest of features with th most frequent value (using its own most frequent value). </font>

# In[ ]:


columns_None = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']
object_columns_df[columns_None]= object_columns_df[columns_None].fillna('None')


# In[ ]:


columns_with_lowNA = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType']
#fill missing values for each column (using its own most frequent value)
object_columns_df[columns_with_lowNA] = object_columns_df[columns_with_lowNA].fillna(object_columns_df.mode().iloc[0])


# * <font color='blue'>   Now we have a clean categorical features </font>
# * <font color='blue'>   In the next step we will deal with the **numerical** features </font>

# In[ ]:


#Number of null values in each feature
null_counts = numerical_columns_df.isnull().sum()
print("Number of null values in each column:\n{}".format(null_counts))


# 1. <font color='blue'>  Fill GarageYrBlt and LotFrontage </font>
# 1. <font color='blue'>  Fill the rest of columns with 0 </font>

# In[ ]:


print((numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt']).median())
print(numerical_columns_df["LotFrontage"].median())


# * <font color='blue'>  So we will fill the year with 1979 and the Lot frontage with 68 </font>
# 

# In[ ]:


numerical_columns_df['GarageYrBlt'] = numerical_columns_df['GarageYrBlt'].fillna(numerical_columns_df['YrSold']-35)
numerical_columns_df['LotFrontage'] = numerical_columns_df['LotFrontage'].fillna(68)


# * <font color='blue'> Fill the rest of columns with 0  <font>
# 

# In[ ]:


numerical_columns_df= numerical_columns_df.fillna(0)


# * <font color='blue'>  We finally end up with a clean dataset  <font>

# 
# * <font color='blue'> After making some plots we found that we have some colums with low variance so we decide to delete them  <font>
# 

# In[ ]:


object_columns_df['Utilities'].value_counts().plot(kind='bar',figsize=[10,3])
object_columns_df['Utilities'].value_counts() 


# In[ ]:


object_columns_df['Street'].value_counts().plot(kind='bar',figsize=[10,3])
object_columns_df['Street'].value_counts() 


# In[ ]:


object_columns_df['Condition2'].value_counts().plot(kind='bar',figsize=[10,3])
object_columns_df['Condition2'].value_counts() 


# In[ ]:


object_columns_df['RoofMatl'].value_counts().plot(kind='bar',figsize=[10,3])
object_columns_df['RoofMatl'].value_counts() 


# In[ ]:


object_columns_df['Heating'].value_counts().plot(kind='bar',figsize=[10,3])
object_columns_df['Heating'].value_counts() #======> Drop feature one Type


# In[ ]:


object_columns_df = object_columns_df.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)


# * <font color='blue'> **Now we will create some new features**  <font>

# In[ ]:


numerical_columns_df['Age_House']= (numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt'])
numerical_columns_df['Age_House'].describe()


# In[ ]:


Negatif = numerical_columns_df[numerical_columns_df['Age_House'] < 0]
Negatif


# 
# * <font color='blue'> Like we see here tha the minimun is -1 ??? <font>
# * <font color='blue'>It is strange to find that the house was sold in 2007 before the YearRemodAdd 2009.
# 
#     So we decide to change the year of sold to 2009 <font>

# In[ ]:


numerical_columns_df.loc[numerical_columns_df['YrSold'] < numerical_columns_df['YearBuilt'],'YrSold' ] = 2009
numerical_columns_df['Age_House']= (numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt'])
numerical_columns_df['Age_House'].describe()


#  <font color='blue'> 
# * TotalBsmtBath : Sum of :
# BsmtFullBath and  1/2 BsmtHalfBath
# 
# * TotalBath : Sum of :
# FullBath and 1/2 HalfBath
# 
# * TotalSA : Sum of : 
# 1stFlrSF and 2ndFlrSF and basement area
# </font>
# 
# 
# 
# 

# In[ ]:


numerical_columns_df['TotalBsmtBath'] = numerical_columns_df['BsmtFullBath'] + numerical_columns_df['BsmtFullBath']*0.5
numerical_columns_df['TotalBath'] = numerical_columns_df['FullBath'] + numerical_columns_df['HalfBath']*0.5 
numerical_columns_df['TotalSA']=numerical_columns_df['TotalBsmtSF'] + numerical_columns_df['1stFlrSF'] + numerical_columns_df['2ndFlrSF']


# In[ ]:


numerical_columns_df.head()


# 
# * <font color='blue'>  Now the next step is to encode categorical features  <font>
# 

# 
# * <font color='blue'>  **Ordinal categories features** - Mapping from 0 to N  <font>

# In[ ]:


bin_map  = {'TA':2,'Gd':3, 'Fa':1,'Ex':4,'Po':1,'None':0,'Y':1,'N':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,"None" : 0,
            "No" : 2, "Mn" : 2, "Av": 3,"Gd" : 4,"Unf" : 1, "LwQ": 2, "Rec" : 3,"BLQ" : 4, "ALQ" : 5, "GLQ" : 6
            }
object_columns_df['ExterQual'] = object_columns_df['ExterQual'].map(bin_map)
object_columns_df['ExterCond'] = object_columns_df['ExterCond'].map(bin_map)
object_columns_df['BsmtCond'] = object_columns_df['BsmtCond'].map(bin_map)
object_columns_df['BsmtQual'] = object_columns_df['BsmtQual'].map(bin_map)
object_columns_df['HeatingQC'] = object_columns_df['HeatingQC'].map(bin_map)
object_columns_df['KitchenQual'] = object_columns_df['KitchenQual'].map(bin_map)
object_columns_df['FireplaceQu'] = object_columns_df['FireplaceQu'].map(bin_map)
object_columns_df['GarageQual'] = object_columns_df['GarageQual'].map(bin_map)
object_columns_df['GarageCond'] = object_columns_df['GarageCond'].map(bin_map)
object_columns_df['CentralAir'] = object_columns_df['CentralAir'].map(bin_map)
object_columns_df['LotShape'] = object_columns_df['LotShape'].map(bin_map)
object_columns_df['BsmtExposure'] = object_columns_df['BsmtExposure'].map(bin_map)
object_columns_df['BsmtFinType1'] = object_columns_df['BsmtFinType1'].map(bin_map)
object_columns_df['BsmtFinType2'] = object_columns_df['BsmtFinType2'].map(bin_map)

PavedDrive =   {"N" : 0, "P" : 1, "Y" : 2}
object_columns_df['PavedDrive'] = object_columns_df['PavedDrive'].map(PavedDrive)


# 
# * <font color='blue'>  Will we use One hot encoder to encode the rest of categorical features  <font>

# In[ ]:


#Select categorical features
rest_object_columns = object_columns_df.select_dtypes(include=['object'])
#Using One hot encoder
object_columns_df = pd.get_dummies(object_columns_df, columns=rest_object_columns.columns) 


# In[ ]:


object_columns_df.head()


# 
# * <font color='blue'>  Concat Categorical (after encoding) and numerical features  <font>
# 

# In[ ]:


df_final = pd.concat([object_columns_df, numerical_columns_df], axis=1,sort=False)
df_final.head()


# In[ ]:


df_final = df_final.drop(['Id',],axis=1)

df_train = df_final[df_final['train'] == 1]
df_train = df_train.drop(['train',],axis=1)


df_test = df_final[df_final['train'] == 0]
df_test = df_test.drop(['SalePrice'],axis=1)
df_test = df_test.drop(['train',],axis=1)


# 
# * <font color='blue'>  Separate Train and Targets  <font>

# In[ ]:


target= df_train['SalePrice']
df_train = df_train.drop(['SalePrice'],axis=1)


# #  <font color='green'> Modeling  </font>

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(df_train,target,test_size=0.33,random_state=0)


# In[ ]:



xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=4, min_child_weight=1.5, n_estimators=2400,
             n_jobs=1, nthread=None, objective='reg:linear',
             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 
             silent=None, subsample=0.8, verbosity=1)


lgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=12000, 
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.4, 
                                       )


# In[ ]:


#Fitting
xgb.fit(x_train, y_train)
lgbm.fit(x_train, y_train,eval_metric='rmse')


# In[ ]:


predict1 = xgb.predict(x_test)
predict = lgbm.predict(x_test)


# In[ ]:


print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict1))))
print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict))))


# 
# * <font color='blue'> Fitting With all the dataset <font>

# In[ ]:


xgb.fit(df_train, target)
lgbm.fit(df_train, target,eval_metric='rmse')


# In[ ]:


predict4 = lgbm.predict(df_test)
predict3 = xgb.predict(df_test)
predict_y = ( predict3*0.45 + predict4 * 0.55)


# In[ ]:


submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": predict_y
    })
submission.to_csv('submission.csv', index=False)

