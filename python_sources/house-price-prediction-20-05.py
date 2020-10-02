#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv').head()


# In[ ]:


homedata = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
iowa_file_path = '../input/train.csv'
homedata.head()


# In[ ]:


features = (['LotArea','YearBuilt','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd'])
X_train = homedata[features]


# In[ ]:


X_train


# In[ ]:


predict = (['Id','SalePrice'])
y = homedata[predict]
y.head()


# In[ ]:


test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
test_data.head()


# In[ ]:


test_features = (['LotArea','YearBuilt','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd'])
X_test = test_data[test_features]
X_test


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(X_train,y)


# In[ ]:


rsquare = model.score(X_train,y)
print('coefficient of determinants: ',model.score(X_train,y))


# In[ ]:


print('intercept: ',model.intercept_)
print("coefficients: ",model.coef_)


# In[ ]:


y_pred = model.predict(X_train)


# In[ ]:


print(y_pred)


# In[ ]:


y_pred1 = model.predict(X_test)


# In[ ]:


print(y_pred1)


# In[ ]:




