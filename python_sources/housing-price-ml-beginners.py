#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


df_train.head()


# In[ ]:


train_Id = df_train["Id"]
test_Id = df_test["Id"]


# In[ ]:


df_train.describe()


# In[ ]:


df_test.head()


# In[ ]:


df_test.describe()


# In[ ]:


sns.heatmap(df_train.isnull())


# In[ ]:


sns.heatmap(df_test.isnull())


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


df_train.isnull().sum().sort_values(ascending=False)[0:20]


# In[ ]:


df_test.isnull().sum().sort_values(ascending=False)[0:35]


# # deleting columns which have very high frequency of Na

# In[ ]:


#deleting those columns which have more than 50% Nan values
#as those columns are same for both test and train datas
list_drop=["PoolQC","MiscFeature","Alley","Fence","GarageYrBlt"]

for col in list_drop:
    del df_train[col]
    del df_test[col]


# In[ ]:


df_train.isnull().sum().sort_values(ascending=False)[0:15]


# In[ ]:


df_test.isnull().sum().sort_values(ascending=False)[0:30]


# # replacing Na values

# - ## for LotFrontage

# In[ ]:


df_train.LotFrontage.value_counts(dropna=False)


# In[ ]:


df_train.LotFrontage.fillna(df_train.LotFrontage.mean(),inplace=True)
df_test.LotFrontage.fillna(df_test.LotFrontage.mean(),inplace=True)


# - BsmtCond, BsmtQual, FirePlaceQu, GarageType, GarageCond, GarageFinish, GarageQual

# In[ ]:


#print(df_train.BsmtCond.value_counts(dropna=False))
#print(df_test.BsmtCond.value_counts(dropna=False))


# In[ ]:


#print(df_train.BsmtQual.value_counts(dropna=False))
#print(df_test.BsmtQual.value_counts(dropna=False))


# In[ ]:


#print(df_train.GarageType.value_counts(dropna=False))
#print(df_test.GarageType.value_counts(dropna=False))


# In[ ]:


#print(df_train.GarageCond.value_counts(dropna=False))
#print(df_test.GarageCond.value_counts(dropna=False))


# In[ ]:


#print(df_train.GarageFinish.value_counts(dropna=False))
#print(df_test.GarageFinish.value_counts(dropna=False))


# In[ ]:


#print(df_train.GarageQual.value_counts(dropna=False))
#print(df_test.GarageQual.value_counts(dropna=False))


# In[ ]:


list_fill_train=["BsmtCond", "BsmtQual", "GarageType", "GarageCond", "GarageFinish",
                 "GarageQual","MasVnrType","BsmtFinType2","BsmtExposure","FireplaceQu","MasVnrArea"]

for j in list_fill_train:
    #df_train[j].fillna(df_train[j].mode(),inplace=True)
    # wrong way to do it.
    df_train[j] = df_train[j].fillna(df_train[j].mode()[0])
    df_test[j] = df_test[j].fillna(df_train[j].mode()[0])


# In[ ]:


print(df_train.isnull().sum().sort_values(ascending=False)[0:5])
print(df_test.isnull().sum().sort_values(ascending=False)[0:20])


# In[ ]:


df_train.dropna(inplace=True)


# In[ ]:


df_train.shape


# In[ ]:


list_test_str = ['BsmtFinType1', 'Utilities','BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 
           'Exterior1st', 'KitchenQual','MSZoning']
list_test_num= ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',  'TotalBsmtSF', 'GarageArea',]

for item in list_test_str:
    df_test[item] = df_test[item].fillna(df_test[item].mode()[0])
for item in list_test_num:
    df_test[item] = df_test[item].fillna(df_test[item].mean())


# In[ ]:


print(df_train.isnull().sum().sort_values(ascending=False)[0:5])
print(df_test.isnull().sum().sort_values(ascending=False)[0:5])


# In[ ]:


df_test.shape


# In[ ]:


#df_train.Electrical.value_counts(dropna=False)


# In[ ]:


#df_train.MasVnrType.value_counts(dropna=False)


# In[ ]:


#df_train.BsmtFinType2.value_counts(dropna=False)


# In[ ]:


#df_train.BsmtExposure.value_counts(dropna=False)


# In[ ]:


#df_train.BsmtFinType1.value_counts(dropna=False)


# In[ ]:


#df_train.MasVnrArea.value_counts(dropna=False)


# In[ ]:


#df_train.GarageYrBlt.describe()


# In[ ]:


del df_train["Id"]
del df_test["Id"]


# In[ ]:


print(df_train.shape)
print(df_test.shape)


# ## checking if there is any missing value lest.

# In[ ]:


print(df_train.isnull().any().any())
print(df_test.isnull().any().any())


# ## Feature Engineering
#   - coverting all the categorical variables
#   - we have to combine both train and test data to convert categorical variables so that same no.s are assigned to particular category in train and test data after that we will split that again.

# In[ ]:


#joining data sets
df_final=pd.concat([df_train,df_test],axis=0)


# In[ ]:


df_final.shape


# In[ ]:


columns = ['MSZoning', 'Street','LotShape', 'LandContour', 'Utilities',
           'LotConfig', 'LandSlope','Neighborhood', 'Condition1', 'Condition2',
           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
           'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation',
           'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
           'Heating', 'HeatingQC', 'CentralAir', 'Electrical','KitchenQual',
           'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
           'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']


# In[ ]:


def One_hot_encoding(columns):
    final_df=df_final
    i=0
    for fields in columns:
        df1=pd.get_dummies(df_final[fields],drop_first=True)
        
        df_final.drop([fields],axis=1,inplace=True)
        if i==0:
            final_df=df1.copy()
        else:           
            final_df=pd.concat([final_df,df1],axis=1)
        i=i+1
       
        
    final_df=pd.concat([df_final,final_df],axis=1)
        
    return final_df


# In[ ]:


main_df=df_train.copy()


# In[ ]:


df_final.head()


# In[ ]:


df_final = One_hot_encoding(columns)


# In[ ]:


df_final.head()


# In[ ]:


df_final.shape 


# In[ ]:


df_final =df_final.loc[:,~df_final.columns.duplicated()]


# In[ ]:


df_final.shape


# - Separate the datasets again.

# In[ ]:


df_train_m=df_final.iloc[:1422,:]
df_test_m=df_final.iloc[1422:,:]
df_test_m.shape


# In[ ]:


df_test_m.drop(["SalePrice"],axis=1,inplace=True)


# In[ ]:


df_test_m.shape


# In[ ]:


x_train_final=df_train_m.drop(["SalePrice"],axis=1)
y_train_final=df_train_m["SalePrice"]


# # Applying Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(x_train_final, y_train_final)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)


# In[ ]:


##model building
linear_reg=LinearRegression()
linear_reg.fit(X_train,Y_train)


# In[ ]:


#Y_pred_linear = linear_reg.predict(X_test)


# In[ ]:


print("R-Squared Value for Training Set: {:.3f}".format(linear_reg.score(X_train,Y_train)))
print("R-Squared Value for Test Set: {:.3f}".format(linear_reg.score(X_test,Y_test)))


# In[ ]:


#print(r2_score(Y_test, Y_pred_linear))


# In[ ]:


#y_pred_linear_test=linear_reg.predict(df_test_m)


# In[ ]:


#pred_df = pd.DataFrame(y_pred_linear_test, columns=['SalePrice'])
test_id_df = pd.DataFrame(test_Id, columns=['Id'])


# # Applying RandomForest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


R_forest=RandomForestRegressor()
R_forest.fit(X_train,Y_train)


# In[ ]:


print("R-Squared Value for Training Set: {:.3f}".format(R_forest.score(X_train,Y_train)))
print("R-Squared Value for Test Set: {:.3f}".format(R_forest.score(X_test,Y_test)))


# In[ ]:


y_pred_rforest_test=R_forest.predict(df_test_m)


# In[ ]:


pred_rforest_df = pd.DataFrame(y_pred_rforest_test, columns=['SalePrice'])


# In[ ]:


submission = pd.concat([test_id_df, pred_rforest_df], axis=1)
submission.head()


# # Save the predictions

# In[ ]:


submission.to_csv(r'submission.csv', index=False)

