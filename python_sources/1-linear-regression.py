#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib. pyplot as plt


# In[ ]:


x = np.random.randn(10,1)
y = 0.5*x+0.785 + 0.1*np.random.randn(10,1)


# In[ ]:


plt.scatter(x,y)


# In[ ]:


model = LinearRegression()
model.fit(x, y)


# In[ ]:


coefficient  = model.coef_
y_intercept = model.intercept_
print(coefficient)
print(y_intercept)


# In[ ]:


model.predict(np.array([[3]]))


# In[ ]:


0.5*3+0.785


# In[ ]:


x_test = np.linspace(-4,4)
y_pred = model.predict(x_test[:,None])
plt.scatter(x,y)
plt.plot(x_test,y_pred,"r")
plt.legend(['Regression Line', 'data points'])


# In[ ]:




