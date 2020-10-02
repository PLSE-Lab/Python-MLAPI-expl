#!/usr/bin/env python
# coding: utf-8

# #Using Stochastic Gradient Descent to achieve the low mean squared errors in the values.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


central_df = pd.read_csv("../input/ca-central-1.csv");
east_df = pd.read_csv("../input/us-east-1.csv");
west_df = pd.read_csv("../input/us-west-1.csv");
south_df = pd.read_csv("../input/ap-south-1.csv");


# In[ ]:


central_df.head()


# In[ ]:


#east_df.head()
central_df.dropna(inplace=True)


# In[ ]:


X1 = central_df.drop(['price','datetime'],axis=1)
central_df2  = pd.get_dummies(X1)
X1 = central_df2.values
y1 = central_df['price'].values


# In[ ]:


east_df.dropna(inplace=True)
X2 = east_df.drop(['price','datetime'],axis=1)
east_df2  = pd.get_dummies(X2)
X2 = east_df2.values
y2 = east_df['price'].values


# In[ ]:


west_df.dropna(inplace=True)
X3 = west_df.drop(['price','datetime'],axis=1)
west_df2  = pd.get_dummies(X3)
X3 = west_df2.values
y3 = west_df['price'].values


# In[ ]:


south_df.dropna(inplace=True)
X4 = south_df.drop(['price','datetime'],axis=1)
south_df2  = pd.get_dummies(X4)
X4 = south_df2.values
y4 = south_df['price'].values


# In[ ]:


from sklearn.model_selection import train_test_split

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2,random_state=42 )
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2,random_state=42 )
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2,random_state=42 )
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2,random_state=42 )


# In[ ]:


from sklearn.linear_model import SGDRegressor
clf1 = SGDRegressor()
clf1.fit(X1_train,y1_train)

y1_rbf = clf1.predict(X1_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y1_test, y1_rbf))


# In[ ]:


from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y1_test, y1_rbf))


# In[ ]:


from sklearn.linear_model import SGDRegressor
clf2 = SGDRegressor()
clf2.fit(X2_train,y2_train)

y2_rbf = clf2.predict(X2_test)

print(mean_squared_error(y2_test, y2_rbf))
print(mean_absolute_error(y2_test, y2_rbf))


# In[ ]:


from sklearn.linear_model import SGDRegressor
clf3 = SGDRegressor()
clf3.fit(X3_train,y3_train)

y3_rbf = clf3.predict(X3_test)

print(mean_squared_error(y3_test, y3_rbf))
print(mean_absolute_error(y3_test, y3_rbf))


# In[ ]:


from sklearn.linear_model import SGDRegressor
clf4 = SGDRegressor()
clf4.fit(X4_train,y4_train)

y4_rbf = clf4.predict(X4_test)

print(mean_squared_error(y4_test, y4_rbf))
print(mean_absolute_error(y4_test, y4_rbf))

