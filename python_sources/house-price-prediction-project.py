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


#import the datasets
import pandas as pd
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# # Analyzing

# In[ ]:


# randomly display 10 lines.

train.sample(10)


# In[ ]:


#display the 5 first lines.
train.head()


# In[ ]:


#number of houses in the train dataset.
len(train)


# In[ ]:


#the price of houses.
train.SalePrice 
# or we use train['SalePrice']


# In[ ]:


# draw a histogram.(SKEW)
train.SalePrice.plot.hist(bins=100)


# In[ ]:


#to show the correlation between two features
corr_mat = train.corr()
corr_mat


# In[ ]:


import seaborn as sns
#To view the correlation matrix
sns.heatmap(corr_mat)


# In[ ]:


#we will take an interest only for the SalePrice.
price_corr = corr_mat['SalePrice']
price_corr


# In[ ]:


# we will take only the correlation > 0,4 with the SalePrice.
price_corr = train.corr()['SalePrice']
top_corr_features = price_corr[price_corr.abs() > 0.4].index
top_corr_features


# In[ ]:


sns.heatmap(corr_mat.loc[top_corr_features,top_corr_features])


# In[ ]:


#sorting out features
price_corr[price_corr.abs() > 0.4].sort_values(ascending=False)


# In[ ]:


import matplotlib as plt
plt.rcParams['figure.figsize'] = (12,8)
sns.violinplot(x="OverallQual",y="SalePrice",data=train)
#correlation is not causation !


# In[ ]:


train.GrLivArea.plot.hist(bins=100)


# In[ ]:


train.plot.scatter(x='GrLivArea',y='SalePrice')


# In[ ]:


#other visualization
import seaborn as sns
sns.jointplot(train.GrLivArea,train.SalePrice)


# In[ ]:


#to view the correlation density
sns.jointplot(train.GrLivArea,train.SalePrice,kind='hex')


# In[ ]:


#to view the price of houses that have a 10/10  in quality.
train[train.OverallQual == 10].GrLivArea.plot.hist()


# # Pre-processing

# In[ ]:


#to avoid the problem of distant values.
import numpy as np
train['SalePriceLog'] =np.log(train.SalePrice)
train.SalePriceLog.plot.hist(bins=100)


# In[ ]:


#for categorical features.
train.Heating.value_counts()


# In[ ]:


cat_variables = train.select_dtypes(include=['object']).columns
cat_train = pd.get_dummies(train[cat_variables])
cat_train.head()


# In[ ]:


num_train = train[[c for c in train if c not in cat_variables]]
num_train.head()


# In[ ]:


joined_train = pd.concat([cat_train,num_train],axis=1)
joined_train.head(5).T


# In[ ]:


#for missing values
train.isna().sum(axis=0).sort_values(ascending=False)
#we ignore them.


# # Modeling.

# In[ ]:


joined_train.sample(5)


# In[ ]:


features_train = joined_train[['OverallQual','GrLivArea']]
target = joined_train.SalePriceLog
features_train.sample(5)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test = train_test_split(features_train,target,test_size=0.25)


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model.fit(x_train,y_train)


# In[ ]:


y_predict = model.predict(x_test)


# In[ ]:


from sklearn.dummy import DummyRegressor
baseline = DummyRegressor('mean')
baseline.fit(x_train,y_train)
mean_squared_error(y_test,baseline.predict(x_test))


# In[ ]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_predict))


# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(y_test,y_predict)


# In[ ]:




