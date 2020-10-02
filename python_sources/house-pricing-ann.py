#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Training data

# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()


# ### Columns with missing values

# In[ ]:


train.columns[train.isna().any()]


# In[ ]:


train.loc[:, train.isna().any()]


# we will be dropping Alley,FireplaceQu,PoolQc,Fence,MiscFeature as they have too many missing values.

# ### After Dropping columns

# In[ ]:


train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True);
test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True);
train.loc[:, train.isna().any()]


# ### Handling missing values

# Both training set and test set have missing values lets handle those.
# 1. Missing values in numeric we will handle them by replacing Nan with number which is in proper range of min and max of tht column.(We will also consider average for this)
# 2. Missing values in categorical type we will look for any none type(eg: None,No etc.) in the column, if yes replace with them else replace randomly with the unique values in that column.

# #### After handling training set

# In[ ]:


missing = train.columns[train.isna().any()].to_list()
for col in missing:
    if(train[col].dtypes =='float64'):
        mini= int(train[col].quantile(0.25))
        maxi= int(train[col].quantile(0.75))
        listind=train[train[col].isnull()].index.tolist()
        for i in listind:
                train.loc[i,col]=random.randint(mini,maxi)
        train[col]=pd.to_numeric(train[col])
   

    elif(train[col].dtypes == 'object'):
        if('True' in str(train[col].str.contains('No').unique().tolist())):
            train[col].fillna('No',inplace=True)
        elif('True' in str(train[col].str.contains('None').unique().tolist())):
            train[col].fillna('None',inplace=True)
        elif('True' in str(train[col].str.contains('Unf').unique().tolist())):
            train[col].fillna('Unf',inplace=True)
        else:
            listind=train[train[col].isnull()].index.tolist()
            unique = train[col].unique().tolist()
            unique=pd.Series(unique).dropna().tolist()
            for i in listind:
                train.loc[i,col]=random.choice(unique)


# In[ ]:


train.columns[train.isna().any()]


# #### After handling test set

# In[ ]:


missing = test.columns[test.isna().any()].to_list()
for col in missing:
    if(test[col].dtypes=='float64'):
        mini= int(test[col].quantile(0.25))
        maxi= int(test[col].quantile(0.75))
        listind=test[test[col].isnull()].index.tolist()
        for i in listind:
                test.loc[i,col]=float(random.randint(mini,maxi))
        test[col]=pd.to_numeric(test[col]) 
        
    if(test[col].dtypes=='object'):
        if('True' in str(test[col].str.contains('No').unique().tolist())):
            test[col].fillna('No',inplace=True)
        elif('True' in str(test[col].str.contains('None').unique().tolist())):
            test[col].fillna('None',inplace=True)
        elif('True' in str(test[col].str.contains('Unf').unique().tolist())):
            test[col].fillna('Unf',inplace=True)
        else:
            listind=test[test[col].isnull()].index.tolist()
            unique = test[col].unique().tolist()
            unique=pd.Series(unique).dropna().tolist()
            for i in listind:
                test.loc[i,col]=random.choice(unique)


# In[ ]:


test.columns[test.isna().any()]


# ### train x,y split

# In[ ]:


train_x = train.iloc[:,:-1]
train_y= train.iloc[:,-1]
print('Sale price as y')
print('----------------')
print(train_y.head())


# ### remove outliers and unnecessary columns

# In[ ]:


train.drop('Id',axis=1,inplace=True)
train.head()


# ### Feature encoding

# In[ ]:


objects = train.columns[train.dtypes == 'object'].to_list()
train_x=pd.get_dummies(train_x,columns=objects)
for i in objects:
    cols = train_x.filter(like=i).columns
    train_x.drop(cols[0],axis=1,inplace=True)
    
objects = test.columns[test.dtypes == 'object'].to_list()
test=pd.get_dummies(test,columns=objects)
for i in objects:
    cols = test.filter(like=i).columns
    test.drop(cols[0],axis=1,inplace=True)

missing = (list(set(train_x.columns) - set(test.columns)))
train_x.drop(columns = missing,axis = 1,inplace=True)
train_x.head()


# In[ ]:


test.head()


# In[ ]:


train_x.drop('Id',axis=1,inplace = True)
test.drop('Id',axis=1,inplace = True)


#  ### standardizing

# In[ ]:


scaler = MinMaxScaler()

train_x = scaler.fit_transform(train_x)
test = scaler.fit_transform(test)

train_x


# In[ ]:


y = scaler.fit_transform(train_y.values.reshape(-1,1))

y_new =scaler.inverse_transform(y.reshape(-1,1))


# ### Prediction

# In[ ]:


classifier=Sequential()
classifier.add(Dense(output_dim=512,init='uniform',activation='relu',input_dim=217))

classifier.add(Dense(output_dim=128,init='uniform',activation='relu'))

classifier.add(Dense(output_dim=128,init='uniform',activation='relu'))

classifier.add(Dense(output_dim=64,init='uniform',activation='relu'))

classifier.add(Dense(output_dim=1,init='uniform',activation='relu'))
classifier.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])


# In[ ]:


classifier.fit(train_x,y,batch_size=15,epochs=500)


# **Preidcting test set**

# In[ ]:


y_pred = classifier.predict(test) 


# In[ ]:


pred = []
y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
for i in y_pred:
    pred.append(i.tolist()[0])

pred


# In[ ]:


new_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
output = pd.DataFrame({'Id': new_test.Id,'SalePrice': pred})
output.to_csv('submission.csv', index=False)

