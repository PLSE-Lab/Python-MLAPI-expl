#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Assignment</h1>
# <h2 align="center">Faisal Akhtar</h2>
# <h2 align="center">Roll No.: 17/1409</h2>
# <p>Machine Learning - B.Sc. Hons Computer Science - Vth Semester</p>
# <p>Perform linear regression on any dataset (say Boston house prices). Apply Lasso regularization on the same and compare the performance before and after applying L1 regularization. Also compare the results with Ridge regularization performed earlier.</p>

# ### Libraries Imported

# In[ ]:


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
from sklearn.metrics import r2_score
from numpy import sqrt

from sklearn.linear_model import Lasso


# ### Reading data from CSV
# <p>Removing unnecessary columns</p>
# <p>Renaming column-names to something meaningful</p>

# In[ ]:


column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "PRICE"]
data = pd.read_csv("../input/bostonhousing/boston-housing.csv", header=None, delimiter=r"\s+", names=column_names)

print(data.head())

from sklearn.datasets import load_boston
boston_data = load_boston()
data = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
data.head()


# In[ ]:


data['PRICE'] = boston_data.target
data.head()


# In[ ]:


X = data.drop('PRICE',axis=1)
Y = data['PRICE']
print(X.head())
print(Y.head())


# ### Test/Train Split
# <p>Dividing data into test-train sets, 30% and 70%</p>

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
print(X_test.shape, Y_test.shape)


# ### Linear Regression
# <p>Fit the model according to "data" variable obtained from CSV.</p>

# In[ ]:


lr = LinearRegression() 
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)


# **Linear Regression Model metrics**

# In[ ]:


print('Mean absolute error : ', metrics.mean_absolute_error(Y_test,Y_pred))
print('Mean square error : ', metrics.mean_squared_error(Y_test,Y_pred))
print('R squared error', r2_score(Y_test,Y_pred))
print('RMSE', sqrt(metrics.mean_squared_error(Y_test,Y_pred)))


# ### Lasso Regression

# In[ ]:


lr = Lasso(alpha=0.01)
lr.fit(X_train, Y_train)

Y_predRR = lr.predict(X_test)


# **Lasso Regression model metrics**

# In[ ]:


print('Mean absolute error : ', metrics.mean_absolute_error(Y_test,Y_predRR))
print('Mean square error : ', metrics.mean_squared_error(Y_test,Y_predRR))
print('R squared error', r2_score(Y_test,Y_predRR))
print('RMSE', sqrt(metrics.mean_squared_error(Y_test,Y_predRR)))


# ### Before and after metrics
# Comparing the performance before and after applying L2 regularization.

# In[ ]:


train_score=lr.score(X_train, Y_train)
test_score=lr.score(X_test, Y_test)

Lasso_train_score = lr.score(X_train, Y_train)
Lasso_test_score = lr.score(X_test, Y_test)


# In[ ]:


print("Linear regression train score:", train_score)
print("Linear regression test score:", test_score)
print("Lasso regression train score:", Lasso_train_score)
print("Lasso regression test score:", Lasso_test_score)

