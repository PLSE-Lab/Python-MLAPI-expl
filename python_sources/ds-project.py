#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
from sklearn import datasets
from sklearn.metrics import r2_score,mean_squared_error,roc_curve,precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")


# In[ ]:


items = pd.DataFrame(items)
items.head()


# In[ ]:


shops = pd.DataFrame(shops)
shops.head()


# In[ ]:


sample_submission = pd.DataFrame(sample_submission)
sample_submission.head()


# In[ ]:


sales_train = pd.DataFrame(sales_train)
sales_train.head()


# In[ ]:


test = pd.DataFrame(test)
test.head()


# In[ ]:


print(sales_train.shape)
print(test.shape)


# In[ ]:


x = sales_train.iloc[:,1:-1].values
y = sales_train.iloc[:,5].values
x[:5]


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)


# In[ ]:


x_train[:5]


# In[ ]:


regrassion = LinearRegression()
regrassion.fit(x_train, y_train)


# In[ ]:


y_predictions = regrassion.predict(x_test)
y_predictions[:5]


# In[ ]:


r2_score = r2_score(y_test, y_predictions)
print("R2 score is : ",r2_score)


# In[ ]:


rmse = mean_squared_error(y_test, y_predictions)
print("Root mean squared error is : ",rmse)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y)


# In[ ]:


rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)


# In[ ]:


predictions = rfc.predict(x_test)
predictions


# In[ ]:


y_test


# In[ ]:


rfc.score(x_test, y_test)


# In[ ]:


ridge = Ridge(alpha=0.1)
ridge.fit(x_train, y_train)


# In[ ]:


predictions = ridge.predict(x_test)
predictions[:5]


# In[ ]:


y_test[0]


# In[ ]:


r2_score(y_test, predictions)

