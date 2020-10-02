#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


rng = np.random.RandomState(1) #random state name


# data.shape #print daata set record amount

# In[ ]:


frame = pd.DataFrame([],columns = ['X1','X2','X3','Y'])
frame


# In[ ]:


frame.X1 = [rng.randint(500,2000) #populate frame column X1
for x in rng.rand(50)]
frame


# In[ ]:


frame.X2 = [rng.randint(100,500) #populate frame column X2
for p in rng.rand(50)]
frame


# In[ ]:


x = rng.rand(50) + 2 #a vector is a 1D array
x


# In[ ]:


frame.X3 = 3 * frame.X1 + x
frame


# In[ ]:


frame.Y = frame.X3 - frame.X2
frame


# Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. Any NA values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.
# takes an optional method parameter, specifying which algorithm to use. The default is pearson. To use Spearman correlation, for example, use
# 

# In[ ]:


frame.corr()


# Default algorithm - Pearson correlation 
# the covariance of two variables divided by the product of their standard deviations. It evaluates the linear relationship between two variables. 
# Pearson correlation coefficient has a value between +1 and -1.
# 
# The value 1 indicates that there is a linear correlation between variable x and y. 
# The value 0 indicates that the variables x and y are not related. 
# The value -1 indicates that there is an inverse correlation between variable x and y.
# __________________________________________________________________________________________________________
# Spearman correlation 
# a nonparametric evaluation that finds the strength and direction of the monotonic relationship between two variables.
# This method is used when the data is not normally distributed or when the sample size is small (less than 30).

# In[ ]:


#correlation between X1 and Y
print('correlation between X1 and Y usine Pearson algorithm')
frame[['X1','Y']].corr()


# In[ ]:


#correlation between X2 and Y
print('correlation between X2 and Y')
frame[['X2','Y']].corr()


# In[ ]:


#correlation between X3 and Y
print('correlation between X3 and Y using spearman algorithm')
frame[['X3','Y']].corr(method="spearman")


# In[ ]:


plt.scatter(frame.X1, frame.Y)
plt.ylabel("Y Axis")
plt.xlabel("X Axis")
plt.suptitle("Correlation between X1 and Y")


# In[ ]:


plt.scatter(frame.X2, frame.Y)
plt.ylabel("Y Axis")
plt.xlabel("X Axis")
plt.suptitle("Correlation between X2 and Y")


# impact is coefficient
# train test split to isolate variables in order to test them
# fit data to model else model wont have data to work with
# 
# 70% of the data will be randomly chosen to train the model and 30% will be used to evaluate the model
# test size

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import linear_model #linear model package


# In[ ]:


x_info = frame[['X1','X2']] #independent
y_info = frame['Y'] #dependent


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_info, y_info, test_size = 0.30)


# In[ ]:


linreg = linear_model.LinearRegression() #linear regression instance


# In[ ]:


linreg.fit(x_train,y_train)


# In[ ]:


linreg.coef_


# #output gave an array of array so put [0] to get one value

# In[ ]:


x_train.columns


# In[ ]:


print("..Regression Coefficients..")
pd.DataFrame(linreg.coef_[0],index=x_train.columns, columns=["Coefficient"])


#  a graph that shows the residuals on the vertical axis and the independent variable on the horizontal axis. If the points in a residual plot are randomly dispersed around the horizontal axis, a linear regression model is appropriate for the data; otherwise, a non-linear model is more appropriate.

# In[ ]:


import seaborn as sns


# In[ ]:


plt.title('Residual Plot with Training data (yellow) and test data (blue)  ')
plt.ylabel('Residuals')               
plt.scatter(linreg.predict(x_train), linreg.predict(x_train)-y_train,c='y',s=20) #s -> size of circles
plt.scatter(linreg.predict(x_test),linreg.predict(x_test)-y_test,c='b',s=20)
plt.hlines(y=0,xmin=np.min(linreg.predict(x_test)),xmax=np.max(linreg.predict(x_test)),color='red',linewidth=2) #np - library instance(linear algebra)


# In[ ]:


from sklearn.metrics import  r2_score


# In[ ]:


print('Coefficient of Determination: %.2f' % r2_score(y_test, linreg.predict(x_test))) 
#x is predicted value
#y is actual value


# result equal one so it is a perfect prediction

# can be negative because the model can be arbitrarily worse
# A constant model that always predicts the expected value of y, disregarding the input features,would get a R^2 score of 0.0.

# In[ ]:


linreg.intercept_


# In[ ]:


#y - dependent variable
#c = intercept
#x = independent
# 2.99992419, -0.99999342 coefficient training data of X1 and X2 respectively


# # 8)Regression Formula
# Y =  2.64814644 +  2.99992419 X1 + -0.99999342 X2

# In[ ]:


prediction = linreg.predict(x_test)


# In[ ]:


prediction


# y is a series

# In[ ]:


outcome = pd.DataFrame({'Original': y_test, 'Prediction' : prediction})
outcome


# In[ ]:


k = 700
c = 300
values = {'x' : [k], 'y' : [c]}
newdata= pd.DataFrame(values)
linreg.predict(newdata)


# #assumption made for previous output: only one set of values so it output as an array and not the datafram i requested

# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error


# In[ ]:


print('Mean squared error: ') 
metrics.mean_absolute_error(y_test, prediction)


# In[ ]:


print('Mean squared error:')
metrics.mean_squared_error(y_test , prediction)


# In[ ]:


print('Root mean squared error:')
np.sqrt(metrics.mean_absolute_error(y_test , prediction))

