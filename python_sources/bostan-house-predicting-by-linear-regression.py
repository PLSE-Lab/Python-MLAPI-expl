#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_boston

boston = load_boston()


# In[ ]:


df = pd.DataFrame(boston.data, columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
                              'RAD','TAX','PTRATIO','LSTAT','MEDV'])
#df.head()
y = pd.DataFrame(boston.target,columns=['Target'])
#y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)


# In[ ]:


X_train.shape


# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)


# In[ ]:


y_pred = reg.predict(X_test)


# In[ ]:


reg.score(X_test,y_pred)


# In[ ]:


from sklearn.metrics import mean_squared_error
accu = mean_squared_error(y_test, y_pred)
print(accu)


# In[ ]:


print(y_pred)


# In[ ]:


from sklearn import linear_model
from sklearn.linear_model import Ridge
reg1 = linear_model.Ridge(alpha=.5)
reg1.fit(X_train, y_train) 
Ridge()


# In[ ]:


y_pred1 = reg1.predict(X_test)


# In[ ]:


reg1.score(X_test,y_pred1)


# In[ ]:


print(y_pred1)


# In[ ]:


accu1 = mean_squared_error(y_test, y_pred1)
print(accu1)


# In[ ]:


from sklearn import linear_model
from sklearn.linear_model import Lasso
reg2 = linear_model.Lasso(alpha=0.1)
reg2.fit(X_train, y_train)
Lasso()
y_pred2 = reg2.predict(X_test)


# In[ ]:


reg1.score(X_test,y_pred2)


# In[ ]:


print(y_pred2)


# In[ ]:


from sklearn.svm import SVR
clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf.fit(X_train, y_train) 
SVR()
y_pred3 = clf.predict(X_test)


# In[ ]:


clf.score(X_test,y_pred3)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, y_train) 
#KNeighborsRegressor()
y_pred4 = neigh.predict(X_test)
print (y_pred4)


# In[ ]:


neigh.score(X_test,y_pred)


# In[ ]:


accu4 = mean_squared_error(y_test, y_pred4)
print(accu4)


# In[ ]:


#RidgeCV implements ridge regression with built-in cross-validation of the alpha parameter
from sklearn import linear_model
from sklearn.linear_model import  RidgeCV
reg5 = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
reg5.fit(X_train, y_train)       
RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3, fit_intercept=True, scoring=None, normalize=False)


# In[ ]:


y_pred5 = reg5.predict(X_test)
reg5.score(X_test,y_pred5)


# In[ ]:


accu5 = mean_squared_error(y_test, y_pred5)
print(accu5)

