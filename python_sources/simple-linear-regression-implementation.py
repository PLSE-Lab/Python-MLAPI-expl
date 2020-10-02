#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression Implementation
# **Purpose**
# - The purpose of this notebook is to implement Simple Linear Regression with python code and to compare results with sklearn's Linear Regression algorithm.
# 
# **Roadmap**
# 
# - Simple Linear Regression Class Definition
# - Using our Simple Linear Regression
# - Using sklearn Linear Regression
# - Conclusion

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# **Simple Linear Regression Class Definition**
# 
# Simple Linear Regression is basicly can be formulised as;
# y = ax + b
# 
# In our Simple Linear Regression class, we have there class variables;
# 1. coef : this is the coefficient of x,   a  variable in above formula
# 1. intercept: this is the y intercept, b variable in above formula
# 1. rsquared: this is the calculation of rsquare result ( known as statistical measure of how close the data are to the fitted regression line )
# 
# And we have two methods
# 1. fit: Data will be trained with fit method to produce above class variables
# 2. predict: To test new datas.

# In[ ]:


class SimpleLinearRegression:
    coef = 0
    intercept = 0
    rsquared = 0
    def fit(self, x_train, y_train):
        sum_of_x = sum(x_train)
        sum_of_y = sum(y_train)
        sum_of_x2 = np.sum(np.square(x_train))
        sum_of_y2 = np.sum(np.square(y_train))
        dotproduct = np.dot(x_train,y_train)
        length = len(x_train)
        dif_x = sum_of_x2 - sum_of_x * sum_of_x/length
        dif_y = sum_of_y2 - sum_of_y * sum_of_y/length
        numerator = length * dotproduct - sum_of_x * sum_of_y
        denom = (length * sum_of_x2 - sum_of_x * sum_of_x) * (length * sum_of_y2 - (sum_of_y * sum_of_y))
        co = dotproduct - sum_of_x * sum_of_y / length
        self.rsquared = np.square(numerator / np.sqrt(denom))
        self.intercept = sum_of_y / length - ((co / dif_x) * sum_of_x/length)
        self.coef = co / dif_x
    def predict(self,x_test):
        return x_test * self.coef + self.intercept
        


# **Using our Simple Linear Regression**
# - A basic x_train and y_train initialized.
# - Call our fit method
# - See results

# In[ ]:


x_train = np.array([ 1, 2, 3, 4])
y_train = np.array([ 2, 3, 4, 4])


# In[ ]:


slr = SimpleLinearRegression()
slr.fit(x_train,y_train)


# In[ ]:


print("Coefficient:", slr.coef)
print('Y-Intercept:',slr.intercept)
print('R-Squared:',slr.rsquared)


# **Using sklearn Linear Regression**
# - Using standart sklearn Linear Regression
# - Print results

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))
print(lr.coef_)
print(lr.intercept_)


# **Conclusion**
# 
# Both algorithms gives same coef and intercept result.
# 
# Our algorithm seems work
