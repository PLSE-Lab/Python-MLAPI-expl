#!/usr/bin/env python
# coding: utf-8

# # Getting the data

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


train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
print(train.shape)
print(test.shape)


# In[ ]:


train.columns


# # Handling missing values

# In[ ]:


train_null = train.isnull().sum().sort_values(ascending = False)


# In[ ]:


train_null = train_null[train_null>0]
train_null


# In[ ]:


a = ['PoolQC','MiscFeature','Alley','Fence']
train.drop(a,axis = 1,inplace = True)


# In[ ]:


train.columns


# In[ ]:


train.info()


# In[ ]:


t_cor = train.corr()
t_cor['SalePrice']


# # Getting the high correlated numerical columns

# In[ ]:


high_num_cor_cols = ['LotFrontage','LotArea','OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1',
                 'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','HalfBath','TotRmsAbvGrd',
                'Fireplaces','GarageYrBlt','GarageCars','GarageArea']


# In[ ]:


train_red = train[high_num_cor_cols]


# In[ ]:


train_red.shape


# In[ ]:


train_label = train['SalePrice']


# In[ ]:


train_label.shape


# In[ ]:


train_red.head()


# In[ ]:


train_red.isnull().sum().sort_values(ascending = False)


# In[ ]:


train_red['LotFrontage'].fillna(train_red['LotFrontage'].mean(),inplace = True)
train_red['GarageYrBlt'].fillna(train_red['GarageYrBlt'].mean(),inplace = True)
train_red['MasVnrArea'].fillna(train_red['MasVnrArea'].mean(),inplace = True)
train_red.isnull().sum().sort_values(ascending=False)


# # Splitting the training and validation data

# In[ ]:


from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y = train_test_split(train_red,train_label,test_size = 0.2,shuffle = True)
print(train_X.shape)
print(val_X.shape)


# # RandomForest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rfg_model = RandomForestRegressor(max_depth = 5)
rfg_model.fit(train_red,train_label)


# In[ ]:


pre = rfg_model.predict(val_X)


# In[ ]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(pre,val_y)


# In[ ]:


test_red = test[high_num_cor_cols]


# In[ ]:


test_red.isnull().sum().sort_values(ascending = False)


# In[ ]:


a = ['LotFrontage','GarageYrBlt','MasVnrArea','GarageCars','BsmtFinSF1','TotalBsmtSF','GarageArea','BsmtUnfSF']
test_red.fillna(test_red[a].mean(),inplace = True)
test_red.isnull().sum().sort_values(ascending = False)


# In[ ]:


test_red.info()


# In[ ]:


predict = rfg_model.predict(test_red)


# In[ ]:


predict


# In[ ]:


output = pd.DataFrame({'Id': test['Id'],
                       'SalePrice': predict})

output.to_csv('submission.csv', index=False)

