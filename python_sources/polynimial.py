#!/usr/bin/env python
# coding: utf-8

# # Task 1

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset = pd.read_csv('Position_Salaries.csv')


# In[ ]:


print(dataset)


# In[ ]:


X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures (degree=4)
X_poly = poly_reg.fit_transform(X)


# In[ ]:


X
y


# In[ ]:


plt.scatter(X,y)
plt.plot(X,y)


# In[ ]:


poly_reg.fit(X_poly,y)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


regressor = LinearRegression()


# In[ ]:


regressor.fit(X_poly, y)
regressor.coef_


# In[ ]:


X_poly


# In[ ]:


lin_reg2 = LinearRegression ()
lin_reg2.fit(X_poly,y)


# In[ ]:


lin_reg2_pred = lin_reg2.predict(poly_reg.fit_transform(X))


# In[ ]:


plt.scatter(X,y,color = 'red')
plt.plot(X, lin_reg2_pred, color = 'blue')
plt.title('PolynomialRegression')
plt.xlabel('Position')
plt.ylabel("Salary")


# In[ ]:


a = np.array(6.5)
a


# In[ ]:


a = a.reshape(-1,1)


# In[ ]:


lin_reg2.predict(poly_reg.fit_transform(a))


# In[ ]:





# # Task 2

# In[ ]:




