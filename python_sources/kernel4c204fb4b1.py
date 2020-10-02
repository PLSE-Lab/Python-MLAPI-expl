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


df_tr = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# In[ ]:


df_tr.head()


# In[ ]:


df = df_tr.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
numfeature = df.select_dtypes(include=['int64','float64'])
high_cor = numfeature.corr()['SalePrice']

high_name = high_cor[(high_cor>=0.5) | (high_cor <= -0.5)]

vars_feature = high_name.index[:-1]
print(vars_feature)


# In[ ]:


train_feature = df[vars_feature]


# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


# In[ ]:


split_index = 1000

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

ENet.fit(train_feature[:split_index],df['SalePrice'][:split_index])


# In[ ]:


import matplotlib.pyplot as plt
ypre = ENet.predict(train_feature[split_index:])
ytest = df['SalePrice'][split_index:].to_numpy()

ax = plt.subplot()
x = np.arange(0,800000,50000)
ax.plot(ypre,ytest,'ro',x,x,'b-')


# In[ ]:


df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()


# In[ ]:


dft = df_test[vars_feature]
dft.info()


# In[ ]:


values = {"TotalBsmtSF":dft['TotalBsmtSF'].mean(),
         "GarageCars":dft['GarageCars'].mean(),
          "GarageArea":dft['GarageArea'].mean()
         }
dft = dft.fillna(values)


# In[ ]:


dft.isnull().sum()


# In[ ]:


ytpre = ENet.predict(dft)
plt.boxplot(ytpre)


# In[ ]:


sub = pd.DataFrame()
sub['Id'] = df_test['Id']
sub['SalePrice'] = ytpre
sub.head()


# In[ ]:


os.getcwd()


# In[ ]:


sub.to_csv("./submission.csv",index=False)


# In[ ]:




