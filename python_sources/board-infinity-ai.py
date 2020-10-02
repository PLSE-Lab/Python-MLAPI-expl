#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


housing=pd.read_csv('downloads/train.csv')


# In[ ]:


housing.head()


# In[ ]:


housing.info()


# In[ ]:


housing.describe()


# In[ ]:


housing.isnull().sum()


# In[ ]:


housing.columns


# In[ ]:


null_value=housing.columns[housing.isnull().any()]


# In[ ]:


null_value


# In[ ]:


housing[null_value].isnull().sum()


# now we are going to drop the columns which are having high null values

# In[ ]:


housing.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


housing.head()


# In[ ]:





# In[ ]:


housing['LotFrontage']=housing['LotFrontage'].fillna(housing['LotFrontage'].mode()[0])
housing['MasVnrType']=housing['MasVnrType'].fillna(housing['MasVnrType'].mode()[0])
housing['MasVnrArea']=housing['MasVnrArea'].fillna(housing['MasVnrArea'].mode()[0])
housing['BsmtQual']=housing['BsmtQual'].fillna(housing['BsmtQual'].mode()[0])
housing['BsmtCond']=housing['BsmtCond'].fillna(housing['BsmtCond'].mode()[0])
housing['BsmtExposure']=housing['BsmtExposure'].fillna(housing['BsmtExposure'].mode()[0])
housing['BsmtFinType1']=housing['BsmtFinType1'].fillna(housing['BsmtFinType1'].mode()[0])
housing['BsmtFinType2']=housing['BsmtFinType2'].fillna(housing['BsmtFinType2'].mode()[0])
housing['GarageType']=housing['GarageType'].fillna(housing['GarageType'].mode()[0])
housing['GarageYrBlt']=housing['GarageYrBlt'].fillna(housing['GarageYrBlt'].mode()[0])
housing[ 'GarageFinish']=housing[ 'GarageFinish'].fillna(housing[ 'GarageFinish'].mode()[0])
housing['GarageQual']=housing['GarageQual'].fillna(housing['GarageQual'].mode()[0])
housing['GarageCond']=housing['GarageCond'].fillna(housing['GarageCond'].mode()[0])


# In[ ]:


housing.isnull().sum()


# In[ ]:


obj_col=[]
num_col=[]
for col in housing.columns:
    if housing[col].dtype=='O':
        obj_col.append(col)
    else:
        num_col.append(col)


# In[ ]:


print(obj_col)
print(num_col)


# In[ ]:


for col in obj_col:
    plt.figure(figsize=(10,8))
    sns.violinplot(housing[col],housing['SalePrice'])
    plt.xlabel(col)
    plt.ylabel('price')


# In[ ]:


for col in num_col:
    plt.figure(figsize=(10,8))
    sns.jointplot(x=housing[col],y=housing['SalePrice'],kind='reg')
    plt.xlabel(col)
    plt.ylabel('price')


# In[ ]:


plt.figure(figsize=(25,22))
sns.heatmap(housing.corr(),annot=True,cmap='Greens')


# In[ ]:


housing_test=pd.read_csv('downloads/test.csv')


# In[ ]:


housing_test.head()


# In[ ]:


housing_test.shape


# In[ ]:


housing_test.isnull().sum()


# In[ ]:


housing_test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


housing_test['LotFrontage']=housing_test['LotFrontage'].fillna(housing_test['LotFrontage'].mode()[0])
housing_test['MasVnrType']=housing_test['MasVnrType'].fillna(housing_test['MasVnrType'].mode()[0])
housing_test['MasVnrArea']=housing_test['MasVnrArea'].fillna(housing_test['MasVnrArea'].mode()[0])
housing_test['BsmtQual']=housing_test['BsmtQual'].fillna(housing_test['BsmtQual'].mode()[0])
housing_test['BsmtCond']=housing_test['BsmtCond'].fillna(housing_test['BsmtCond'].mode()[0])
housing_test['BsmtExposure']=housing_test['BsmtExposure'].fillna(housing_test['BsmtExposure'].mode()[0])
housing_test['BsmtFinType1']=housing_test['BsmtFinType1'].fillna(housing_test['BsmtFinType1'].mode()[0])
housing_test['BsmtFinType2']=housing_test['BsmtFinType2'].fillna(housing_test['BsmtFinType2'].mode()[0])
housing_test['GarageType']=housing_test['GarageType'].fillna(housing_test['GarageType'].mode()[0])
housing_test['GarageYrBlt']=housing_test['GarageYrBlt'].fillna(housing_test['GarageYrBlt'].mode()[0])
housing_test[ 'GarageFinish']=housing_test[ 'GarageFinish'].fillna(housing_test[ 'GarageFinish'].mode()[0])
housing_test['GarageQual']=housing_test['GarageQual'].fillna(housing_test['GarageQual'].mode()[0])
housing_test['GarageCond']=housing_test['GarageCond'].fillna(housing_test['GarageCond'].mode()[0])


# In[ ]:


housing_test.columns[housing_test.isnull().any()]


# In[ ]:


housing_test['Electrical']=housing_test['Electrical'].fillna(housing_test['Electrical'].mode()[0])


# In[ ]:


housing_test.isnull().sum().sum()


# In[ ]:


obj_col=[]
num_col=[]
for col in housing_test.columns:
    if housing_test[col].dtype=='O':
        obj_col.append(col)
    else:
        num_col.append(col)


# In[ ]:


#one hot encoding
housing1=housing.copy()
housing1=pd.get_dummies(housing1, columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'], drop_first=True)


# In[ ]:


housing1.head()


# In[ ]:


housing1.columns


# In[ ]:


housing_test1=housing_test.copy()
housing_test1=pd.get_dummies(housing_test1, columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'], drop_first=True)


# In[ ]:


housing_test1.head()


# In[ ]:


x_train=housing1.drop('SalePrice',axis=1)
y_train=housing1[['SalePrice']]


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr=LinearRegression()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


lr.intercept_,lr.coef_


# In[ ]:


predictions=lr.predict(x_train)


# In[ ]:


a=predictions[:439]


# In[ ]:


a


# In[ ]:





# In[ ]:


b=housing_test['Id']


# In[ ]:


b


# In[ ]:


pred=pd.Series(list(a))


# In[ ]:


idd=pd.Series(list(b))


# In[ ]:


df=pd.DataFrame([idd,pred])


# In[ ]:





# In[ ]:


df.to_csv('akshay.csv')


# In[ ]:




