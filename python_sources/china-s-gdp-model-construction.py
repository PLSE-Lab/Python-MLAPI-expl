#!/usr/bin/env python
# coding: utf-8

# **Importing required libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Loading the Dataset**

# In[ ]:


df = pd.read_csv("../input/china_gdp.csv")
df.head(10)


# **Plotting the Dataset**

# In[ ]:


plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


# **Choosing a model**
# 
# From an initial look at the plot, we determine that the logistic function could be a good approximation, since it has the property of starting with a slow growth, increasing growth in the middle, and then decreasing again at the end; as illustrated below:

# In[ ]:


X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# **Building The Model**
# 
#  Let's build our regression model and initialize its parameters

# In[ ]:


def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y


# Lets look at a sample sigmoid line that might fit with the data:

# In[ ]:


beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')


# Our task here is to find the best parameters for our model. Lets first normalize our x and y:

# In[ ]:


# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)


# **How we find the best parameters for our fit line?**
# 
# We can use curve_fit which uses non-linear least squares to fit our sigmoid function, to data

# In[ ]:


from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))


# Now we plot our resulting regresssion model.

# In[ ]:


x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(12,6))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


# **What is the accuracy of our model?**

# In[ ]:


# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]


# In[ ]:


# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)


# In[ ]:


# predict using test set
y_hat = sigmoid(test_x, *popt)


# In[ ]:


print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )


# 

# 
