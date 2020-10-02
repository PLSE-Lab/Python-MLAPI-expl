#!/usr/bin/env python
# coding: utf-8

# > **
# Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.**

# ### Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression


# ### Load the data

# In[ ]:


#data = pd.read_csv(r'S:\Udemy\Data Science Bootcamp\Data samples for practice csv_files\1.01. Simple linear regression.csv')
data = pd.read_csv('/kaggle/input/gpa-prediction-using-sat-scores/1.01. Simple linear regression.csv')
data.head()


# ## Create the Regression

# ### Create the dependent and  independent variable

# In[ ]:


y = data['GPA']     # -> Dependent variable (TO BE PREDICTED)
x = data['SAT']     # -> Independent variable


# In[ ]:


x.shape, y.shape


# ### Regression itself

# In[ ]:


reg = LinearRegression()


# In[ ]:


#reg.fit(x,y)  
'''This will give error as the shape shown above is 1-D, Now we have to re shape it to 2-D'''


# In[ ]:


x_matrix = x.values.reshape(-1,1)
x_matrix.shape  #Turned to 2-D


# In[ ]:


reg.fit(x_matrix,y)


# ### R-Squared

# In[ ]:


reg.score(x_matrix,y)


# #### You can check the score here https://www.kaggle.com/dataislife8/simple-linear-regression-simple when we haven't used sklearn

# ### Coefficients

# In[ ]:


reg.coef_


# ### Intercept

# In[ ]:


reg.intercept_


# ### Plotting

# In[ ]:


plt.scatter(x,y)
yhat = reg.coef_*x_matrix + reg.intercept_
plt.plot(x,yhat, lw=3, color='red')
plt.xlabel('SAT', fontsize=15)
plt.ylabel('GPA', fontsize=15)

