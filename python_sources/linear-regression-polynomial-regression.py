#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# From the book Hands on machine learning in sklearn and tensor flow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/polynomial-regression-position-salaries/Position_Salaries.csv')


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.fillna(0,inplace=True)


# In[ ]:


data.head()


# In[ ]:


x=data.iloc[:,1:2]
y=data.iloc[:,-1:]
y.head()


# In[ ]:


plt.scatter(x.Level.values,y.Salary.values)
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# In[ ]:


from sklearn.linear_model import LinearRegression
lin_reg1=LinearRegression()
lin_reg1.fit(x,y)
plt.scatter(x.Level.values,y.Salary.values)
plt.plot(x.values,lin_reg1.predict(x),color="green")
plt.show()


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
y_pred = lin_reg.predict(X_poly)
plt.scatter(x.Level.values,y.Salary.values)
plt.plot(x.values,lin_reg.predict(X_poly),color ="blue")
plt.show()


# In[ ]:


poly_features = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly_features.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
y_pred = lin_reg.predict(X_poly)
plt.scatter(x.Level.values,y.Salary.values)
plt.plot(x.values,lin_reg.predict(X_poly),color ="blue")
plt.show()

