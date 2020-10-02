#!/usr/bin/env python
# coding: utf-8

# ## Basic Math

# In[ ]:


1 + 2


# In[ ]:


a = 1
b = 2.5
a + b


# ## Strings

# In[ ]:


a = 'Hello World'
a


# ## Arrays

# In[ ]:


a = ['1', 2, b, 1+2]
a


# In[ ]:


a[1:]


# ## Import Modules

# In[ ]:


import numpy as np                     # array goodnes
from pandas import DataFrame, read_csv # excel for python
from matplotlib import pyplot as plt   # plotting library

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')       # nice colors
plt.xkcd()
plt.rc('font',family='DejaVu Sans')


# ## Numpy

# In[ ]:


a = np.array([2, b, 1+2])
print('mean:', np.mean(a), 'sd:', np.std(a), 'median:', np.median(a))
print('min:', np.min(a), 'max:', np.max(a), 'sum:', np.sum(a))


# ## Read Data
# The Iris dataset
# 
# ![Iris dataset](https://raw.githubusercontent.com/ritchieng/machine-learning-dataschool/master/images/03_iris.png)
# 
# * 50 samples of 3 different species (150 samples total)
# * Measurements: sepal length, sepal width, petal length, petal width

# In[ ]:


df_iris = read_csv('../input/Iris.csv')
df_iris.head()


# In[ ]:


df_iris.describe()


# In[ ]:


sns.pairplot(data=df_iris, hue='Species', diag_kind='kde')


# ## Linear Model

# In[ ]:


X_iris = df_iris.drop('Species', axis=1)
Y_iris = df_iris['Species']


# In[ ]:


X = X_iris['SepalLengthCm']
Y = X_iris['PetalLengthCm']

mask = Y_iris == 'Iris-virginica'
X = X[mask]
Y = Y[mask]

plt.scatter(X, Y, label='Iris-virginica')
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.legend()


# # HANDS ON: Simple Linear Regression
# 
# ![](https://www.wired.com/images_blogs/wiredscience/2011/01/Untitled3.jpg)
# 
# Given a model function 
# $$
# y = \alpha + \beta x~,
# $$
# where $\alpha$ is defined as
# $$
# \alpha = \frac{\sum\nolimits_{i=1}^n(x_i - \bar{x})(y_i - \bar{y})}
#                      {\sum\nolimits_{i=1}^n(x_i - \bar{x})^2}
# $$
# and $\beta$ is defined as
# $$
# \beta = \bar{y} - \alpha \bar{x}~.
# $$
# $\alpha$ and $\beta$ are also known as slope and intercept.

# # Task
# Model a linear regression to fit sepal length and pepal length of the virginica species. The required data points are represented by the variables X (sepal length) and Y (pepal length).

# In[ ]:




