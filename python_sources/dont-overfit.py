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


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


y_train = train['target']
print(y_train)


# In[ ]:


x_train  = train.drop(['id' ,'target'] ,axis = 1)


# In[ ]:


x_train.describe()


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


test.describe()


# In[ ]:


test = test.drop('id',axis = 1)


# In[ ]:


from sklearn.linear_model import LassoCV , LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

scaler  = StandardScaler()
x_train  = scaler.fit_transform(x_train)
test = scaler.fit_transform(test)


# In[ ]:


model_lasso = LassoCV(alphas =  [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]).fit(x_train, y_train)


# In[ ]:


lasso_preds = model_lasso.predict(test)


# In[ ]:


print(lasso_preds)


# In[ ]:


lasso_preds=lasso_preds.T


# In[ ]:


lasso_preds


# In[ ]:


lasso_preds
test = pd.read_csv('../input/test.csv')


# In[ ]:


solution = pd.DataFrame({"id":test.id, "target":lasso_preds})
solution.to_csv("lasso_sol.csv", index = False)

