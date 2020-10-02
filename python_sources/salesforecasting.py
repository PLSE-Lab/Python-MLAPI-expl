#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math


# In[ ]:


sample_submission = pd.read_csv("../input/sample_submission.csv")
items = pd.read_csv("../input/items.csv")
item_categories = pd.read_csv("../input/item_categories.csv")
shops = pd.read_csv("../input/shops.csv")
sales_train = pd.read_csv("../input/sales_train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


sample_submission.head()


# In[ ]:


sales_train.head()


# In[ ]:


test.head()


# In[ ]:


shops.head()


# In[ ]:


items.head()


# In[ ]:


item_categories.head()


# In[ ]:


X_train = sales_train[["shop_id", "item_id"]]
y_train = sales_train["item_cnt_day"]
X_test = test.drop("ID", axis=1)

X_train.head()


# In[ ]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_train)


# In[ ]:


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))
print("Train RMSE:", rmse(y_train, y_pred))


# In[ ]:


y_pred = model.predict(X_test)
result = pd.DataFrame({"ID": test["ID"], "item_cnt_month": y_pred})
result.head()


# In[ ]:


result.to_csv("our.csv", index=False)


# In[ ]:




