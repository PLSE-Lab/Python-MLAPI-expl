#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv("../input/frogs-data/frogs.csv")


# In[ ]:


df.head()


# In[ ]:


df.drop(["Unnamed: 0"],axis=1,inplace=True)
df.head()


# In[ ]:


y=df.altitude.values.reshape(-1,1)
x=df.distance.values.reshape(-1,1)


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures

polynomial_reg=PolynomialFeatures(degree=2)
x_polynomial=polynomial_reg.fit_transform(x)


# In[ ]:


from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x_polynomial,y)


# In[ ]:


y_head=linear_reg.predict(x_polynomial)


# In[ ]:


plt.plot(x,y_head,color="green",label="polynomial")
plt.legend()
plt.show()


# In[ ]:


y=df.northing.values.reshape(-1,1)
x=df.easting.values.reshape(-1,1)
from sklearn.preprocessing import PolynomialFeatures

polynomial_reg=PolynomialFeatures(degree=2)
x_polynomial=polynomial_reg.fit_transform(x)

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x_polynomial,y)

y_head=linear_reg.predict(x_polynomial)

plt.plot(x,y_head,color="green",label="polynomial")
plt.legend()
plt.show()


# In[ ]:


y=df.northing.values.reshape(-1,1)
x=df.altitude.values.reshape(-1,1)
from sklearn.preprocessing import PolynomialFeatures

polynomial_reg=PolynomialFeatures(degree=2)
x_polynomial=polynomial_reg.fit_transform(x)

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x_polynomial,y)

y_head=linear_reg.predict(x_polynomial)

plt.plot(x,y_head,color="green",label="polynomial")
plt.legend()
plt.show()


# In[ ]:


y=df.easting.values.reshape(-1,1)
x=df.altitude.values.reshape(-1,1)
from sklearn.preprocessing import PolynomialFeatures

polynomial_reg=PolynomialFeatures(degree=2)
x_polynomial=polynomial_reg.fit_transform(x)

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x_polynomial,y)

y_head=linear_reg.predict(x_polynomial)

plt.plot(x,y_head,color="green",label="polynomial")
plt.legend()
plt.show()


# In[ ]:


y=df.northing.values.reshape(-1,1)
x=df.distance.values.reshape(-1,1)
from sklearn.preprocessing import PolynomialFeatures

polynomial_reg=PolynomialFeatures(degree=2)
x_polynomial=polynomial_reg.fit_transform(x)

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x_polynomial,y)

y_head=linear_reg.predict(x_polynomial)

plt.plot(x,y_head,color="green",label="polynomial")
plt.legend()
plt.show()


# In[ ]:


y=df.easting.values.reshape(-1,1)
x=df.distance.values.reshape(-1,1)
from sklearn.preprocessing import PolynomialFeatures

polynomial_reg=PolynomialFeatures(degree=2)
x_polynomial=polynomial_reg.fit_transform(x)

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x_polynomial,y)

y_head=linear_reg.predict(x_polynomial)

plt.plot(x,y_head,color="green",label="polynomial")
plt.legend()
plt.show()

