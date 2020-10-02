#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Simple Linear Regression in Python using Salary data.
# findout the Salary based on YearExperience.
# YearsExperience- X(Independed Variable)
# Salary - y(Depended Variable)
# import library numpy, pandas,sklearn, matplotlib for prediction.


# In[ ]:


# import the library file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm


# In[ ]:


# import the dataset
dataset = pd.read_csv('../input/Salary_Data.csv')


# In[ ]:


# Seprate the data x and y.
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


# In[ ]:


# Splitting the dataset into the Training set and Test set.
from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


# In[ ]:


# Fitting Simple Linear Regression to the Training set.
from sklearn.linear_model import LinearRegression


# In[ ]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results


y_pred = regressor.predict(X_test)
model1=sm.OLS(y_train,X_train)
result = model1.fit()
result.summary()


# In[ ]:





# In[ ]:


# intercept value 
regressor.intercept_ 


# In[ ]:


# coefficients value Beta
regressor.coef_ 


# In[ ]:


#y_pred = regressor.predict(X_test)
#model1=sm.OLS(y_train,X_train)
#result = model1.fit()
#result.summary()


# In[ ]:


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')


# In[ ]:


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:




