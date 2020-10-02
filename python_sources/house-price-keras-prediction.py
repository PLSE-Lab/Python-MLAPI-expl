#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# The Data Cleaning Part was based on the idea of https://www.kaggle.com/meikegw/filling-up-missing-values 
# Special thanks to meikegw for a wonderful notebook on missing value and data preprocessing. 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[3]:


train = pd.read_csv("../input/train.csv")


# In[4]:


train.head()


# In[5]:


train.describe()


# In[6]:


train.columns


# In[7]:


import matplotlib.pyplot as plt 
import seaborn as sns


# In[9]:


var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice')


# In[8]:


train.plot.scatter(x = 'TotalBsmtSF',y='SalePrice')


# In[10]:


train.plot.scatter(x ='YearBuilt',y = 'SalePrice' )


# In[11]:


ab = train.corr()


# In[13]:


ab


# In[14]:


f,ax = plt.subplots(figsize=(12, 9))
sns.heatmap(ab , vmax = .75)


# In[15]:


print(train.isnull().sum())


# In[16]:


miss=train.columns[train.isnull().any()].tolist()


# In[17]:


miss


# In[18]:


train[miss].isnull().sum()


# In[19]:


train['SqrtLotArea']=np.sqrt(train['LotArea'])
sns.pairplot(train[['LotFrontage','SqrtLotArea']].dropna())


# In[20]:


cond = train['LotFrontage'].isnull()
train.LotFrontage[cond]=train.SqrtLotArea[cond]


# In[21]:


del train['SqrtLotArea']


# In[22]:


train[['MasVnrType','MasVnrArea']][train['MasVnrType'].isnull()==True]


# In[23]:


def cat_exploration(column):
    return train[column].value_counts()


# In[24]:


# Imputing the missing values
def cat_imputation(column, value):
    train.loc[train[column].isnull(),column] = value


# In[25]:


cat_exploration('Alley')


# In[26]:


cat_imputation('MasVnrType', 'None')
cat_imputation('MasVnrArea', 0.0)


# In[27]:


basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']
train[basement_cols][train['BsmtQual'].isnull()==True]


# In[28]:


for cols in basement_cols:
    if 'FinSF'not in cols:
        cat_imputation(cols,'None')


# In[29]:


# Impute most frequent value
cat_imputation('Electrical','SBrkr')


# In[30]:


train.head()


# In[31]:


cat_exploration('FireplaceQu')


# In[32]:


train['Fireplaces'][train['FireplaceQu'].isnull()==True].describe()


# In[33]:


cat_imputation('FireplaceQu','None')
pd.crosstab(train.Fireplaces,train.FireplaceQu)


# In[34]:


garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
train[garage_cols][train['GarageType'].isnull()==True]


# In[35]:


for cols in garage_cols:
    if train[cols].dtype==np.object:
        cat_imputation(cols,'None')
    else:
        cat_imputation(cols, 0)


# In[36]:


cat_exploration('PoolQC')
train['PoolArea'][train['PoolQC'].isnull()==True].describe()


# In[37]:


cat_imputation('PoolQC', 'None')
cat_imputation('Fence', 'None')
cat_imputation('MiscFeature', 'None')


# In[38]:


def show_missing():
    missing = train.columns[train.isnull().any()].tolist()
    return missing
train[show_missing()].isnull().sum()


# In[39]:


cat_imputation('Alley','None')


# In[40]:


train[show_missing()].isnull().sum()


# In[41]:


train.head()


# In[42]:


train.columns


# In[43]:


from keras.models import Sequential
from keras.optimizers import SGD,RMSprop
from keras.layers import Dense,Dropout,Activation
from keras.layers.normalization import BatchNormalization


# In[44]:


y = np.log1p(train[['SalePrice']])


# In[45]:


y.mean()


# In[46]:


test = pd.read_csv("../input/test.csv")


# In[47]:


test.head()


# In[48]:


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                    test.loc[:,'MSSubClass':'SaleCondition']))


# In[49]:


all_data = pd.get_dummies(all_data)


# In[50]:


all_data = all_data.fillna(all_data.mean())


# In[51]:


x_train = np.array(all_data[:train.shape[0]])
x_test = np.array(all_data[test.shape[0]+1:])


# In[52]:


x_train


# In[53]:


from sklearn.model_selection import train_test_split 


# In[54]:


X_train, X_valid, y_train, y_valid = train_test_split(x_train, y)


# In[55]:


from keras.activations import relu


# In[56]:


X_train.shape


# In[57]:


model = Sequential()
model.add(Dense(1024,input_dim = 302,kernel_initializer='uniform'))
model.add(Activation(relu))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(512,input_dim=1028,activation='relu',kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.8))
model.add(Dense(256))
model.add(Dropout(0.8))
model.add(Dense(128))
model.add(Dense(1))
model.compile( optimizer='adam',loss='mse',metrics=['mean_squared_error'])


# In[58]:


model.summary()


# In[70]:


model.fit(X_train,y_train,validation_data=(X_valid,y_valid),nb_epoch=30,batch_size=128)


# In[72]:


np.sqrt(model.evaluate(X_valid,y_valid))


# In[73]:


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[74]:


preds = model.predict(np.array(x_test))


# In[75]:


x_test


# In[76]:


subm = pd.read_csv("../input/sample_submission.csv")


# In[77]:


subm.shape


# In[78]:


subm.iloc[:,1] = np.array(model.predict(np.array(x_test)))


# In[79]:


print(subm[['SalePrice']].mean())


# In[80]:


subm['SalePrice'] = np.expm1(subm[['SalePrice']])
print(subm[['SalePrice']].mean())


# In[81]:


subm.to_csv('sub1.csv', index=None)


# In[ ]:




