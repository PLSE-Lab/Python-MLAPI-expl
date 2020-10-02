#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
mean = [0, 100]
cov = [[1, 0.5], 
       [0.5, 1]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x, y, 'x')
plt.axis('equal')


# In[ ]:


from sklearn import datasets, linear_model


# In[ ]:


regr = linear_model.LinearRegression()


# In[ ]:


x = x.reshape(-1,1)
regr.fit(x, y)


# In[ ]:


y_hat = regr.predict(x)


# In[ ]:


plt.scatter(x, y,  color='black')
plt.plot(x, y_hat, color='blue', linewidth=3)


# In[ ]:


e = y - y_hat


# In[ ]:


plt.scatter(x, e)


# In[ ]:


plt.scatter(y, e)


# In[ ]:




