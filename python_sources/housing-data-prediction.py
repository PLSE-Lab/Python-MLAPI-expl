#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train=pd.read_csv(r'../input/house-prices-advanced-regression-techniques/train.csv')
train.tail()


# In[ ]:


test=pd.read_csv(r'../input/house-prices-advanced-regression-techniques/test.csv')
test.head()


# In[ ]:


train[train.isnull()==True]
#Train df showing all nan values


# In[ ]:


sns.heatmap(data=train.isnull())
#Heatmap for visualizing null values


# In[ ]:


train.columns


# In[ ]:


train.drop(columns=['Alley','PoolQC','Fence','MiscFeature','Fireplaces','FireplaceQu','GarageType',
       'GarageYrBlt', 'GarageFinish','GarageQual','GarageCond'],inplace=True,axis=1)


# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


train[train['LotFrontage'].isnull()==True]


# **1. 1. Boxplot to measure the inter-quartile range,outliers & mean for missing data imputation**

# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(data=train['LotFrontage'])


# In[ ]:


train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())


# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(data=train.isnull())


# In[ ]:


train.columns


# In[ ]:


train[['BsmtQual', 'BsmtCond',
         'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF']]


# In[ ]:


sns.heatmap(data=train.isnull())


# In[ ]:


train[train['BsmtFinType2'].isnull()==True].iloc[:,30:40]


# In[ ]:


train.drop(columns=['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'],axis=1,inplace=True)


# In[ ]:


plt.figure(figsize=(14,4))
sns.heatmap(train.isnull())


# In[ ]:


train.isnull()==True


# In[ ]:


train.dropna(axis=1,inplace=True)


# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


len(train.columns)


# In[ ]:


train.head(10)


# In[ ]:


test.head(10)


# In[ ]:


test.drop(columns=['Alley','PoolQC','Fence','MiscFeature','Fireplaces','FireplaceQu','GarageType',
       'GarageYrBlt', 'GarageFinish','GarageQual',
                   'GarageCond','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'],axis=1,inplace=True)


# In[ ]:


sns.heatmap(test.isnull())


# In[ ]:


test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())


# In[ ]:


plt.figure(figsize=(14,4))
sns.heatmap(test.isnull())


# In[ ]:


print(test.columns)


# In[ ]:


test.drop(columns=['MasVnrType', 'MasVnrArea', 'BsmtQual', 'Electrical'],axis=1,inplace=True)


# In[ ]:


plt.figure(figsize=(14,4))
sns.heatmap(test.isnull())


# In[ ]:


test[test['Utilities'].isnull()==True]


# In[ ]:


test['Utilities'].fillna('AllPub',inplace=True)


# In[ ]:


test['MSZoning'].fillna('RH',inplace=True)


# In[ ]:


test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean(),inplace=True)


# In[ ]:


test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean(),inplace=True)


# In[ ]:


test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean(),inplace=True)


# In[ ]:


test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean(),inplace=True)


# In[ ]:


test['SaleType'].fillna('WD',inplace=True)


# In[ ]:


test['BsmtFullBath'].fillna(0,inplace=True)


# In[ ]:


test['BsmtHalfBath'].fillna(0,inplace=True)


# In[ ]:


test['Functional'].fillna('Typ',inplace=True)


# In[ ]:


test['GarageCars'].fillna(test['GarageCars'].median(),inplace=True)


# In[ ]:


test['GarageArea'].fillna(test['GarageArea'].mean(),inplace=True)


# In[ ]:


plt.figure(figsize=(20,4))
sns.heatmap(test.isnull())


# In[ ]:


test


# In[ ]:


cat_train=train.select_dtypes(include='object')
cat_train.head(20)


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder


# Encode categorical features as an integer array.
# 
# The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. The features are converted to ordinal integers. This results in a single column of integers (0 to n_categories - 1) per feature.

# In[ ]:


oe=OrdinalEncoder()


# In[ ]:


encat_train=oe.fit_transform(cat_train)
encat_train
#encoded_categorical_train_array


# In[ ]:


encat_train2=pd.DataFrame(data=encat_train,columns=cat_train.columns)
encat_train2
#encoded_categorical_train_dataframe


# In[ ]:


num_train=train.select_dtypes(include='int')
num_train


# In[ ]:


float_train=train.select_dtypes(include='float')
float_train


# In[ ]:


final_train=pd.concat([encat_train2,num_train,float_train],axis=1)
final_train.head(10)


# In[ ]:


cat_test=test.select_dtypes(include='object')
cat_test


# In[ ]:


cat_test=cat_test.fillna('dummy')


# In[ ]:


encat_test=oe.fit_transform(cat_test)
encat_test


# In[ ]:


encat_test2=pd.DataFrame(data=encat_test,columns=cat_test.columns)
encat_test2
#encoded_categorical_test_dataframe


# In[ ]:


num_test=test.select_dtypes(include='int')
num_test


# In[ ]:


float_test=test.select_dtypes(include='float')
float_test


# In[ ]:


final_test=pd.concat([encat_test2,num_test,float_test],axis=1)
final_test


# Cleaning of Final dataset on which we need to work.

# In[ ]:


final_train=final_train.fillna(0)
final_test=final_test.fillna(0)


# In[ ]:


final_train.columns


# Setting up the format & font size for matplotlib graphs

# In[ ]:


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

plt.rc('font', **font)


# Graph shows lot area of all 1460 houses

# In[ ]:


plt.figure(figsize=(16,4))
plt.xlabel('No of Houses')
plt.ylabel('Lot_Area of houses')
final_train['LotArea'].plot(color='red')
plt.tight_layout()


# Graph describing all statistics of sale price of houses

# In[ ]:


plt.figure(figsize=(15,4))
plt.xlabel('Different parameters of SalePrice')
plt.ylabel('Values')
final_train.describe()['SalePrice'].plot(kind='line',color='green')
plt.tight_layout()


# In[ ]:


final_train.iloc[:,25:40]


# In[ ]:


Yr_stay=final_train['YrSold']-final_train['YearBuilt']
Yr_stay
#years stayed


# In[ ]:


plt.figure(figsize=(18,5))
plt.subplot(1,2,1)
plt.plot(Yr_stay)
plt.title('Year stayed in Houses')
plt.xlabel('No of Houses')
plt.ylabel('Yr_sold - Yr_built')
plt.subplot(1,2,2)
plt.plot(Yr_stay.describe(),color='purple')
plt.title('Statistics of Year stayed feature')
plt.xlabel('Statistics')
plt.ylabel('Values')
plt.tight_layout(w_pad=3)


# In[ ]:


sns.jointplot(data=final_train,x='LotArea',y='SalePrice',color='g',kind='reg',height=7,space=0)
plt.tight_layout(w_pad=15)


# In[ ]:


X_train=final_train.drop('SalePrice',axis=1).values
final_train['SalePrice'] = final_train['SalePrice'].astype(float)
y_train=final_train['SalePrice'].values
X_test=final_test.values


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler=MinMaxScaler()


# In[ ]:


scaler.fit(X_train)


# In[ ]:


X_train=scaler.transform(X_train)


# In[ ]:


X_test=scaler.transform(X_test)


# In[ ]:


X_train.min()


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation


# In[ ]:


X_train.shape


# In[ ]:


model=Sequential()

model.add(Dense(61,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(15,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='relu'))

model.compile(optimizer='rmsprop',loss='mse')


# In[ ]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=10000,
          batch_size=256 
          )


# In[ ]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# In[ ]:


prediction=model.predict(X_test)
prediction


# In[ ]:


#from sklearn.tree import DecisionTreeRegressor


# In[ ]:


#clf=DecisionTreeRegressor(max_depth=5)
#clf=clf.fit(X_train,Y_train)


# In[ ]:


#prediction=clf.predict(X_test)


# In[ ]:


ypred=pd.DataFrame(data=prediction,columns=['SalePrice'])
ypred


# In[ ]:


idcol=pd.DataFrame(data=final_test['Id'],columns=['Id'])
idcol


# In[ ]:


result=pd.concat([idcol,ypred],axis=1)
result


# In[ ]:


result.to_csv('submis.csv')


# # **The End**
