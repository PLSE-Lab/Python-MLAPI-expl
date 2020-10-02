#!/usr/bin/env python
# coding: utf-8

# **Linear Regression:** is a method used to model a relationship between a dependent variable (y), and an independent variable (x). Mathematically, the regression is based on the elementary equation of the line as follows:
# 
# **y = a + bx** 
# 
# Where:
# * y = dependent variable or the value to predict
# * b = slope of the line
# * x = independet variable
# * a = the y-intercept
# 
# the main goal is to find the best-fitting line that minimizes the sum of the squared error between the actual value of a stock price (y) and a predicted stock price in all the points of the dataset. 
# 
# There is a lot of different types of lineal regresion, the most simple will only be one independet variable (x) and if there is more than one independet variables, its falls under the category of multiple linear regresion. In this exercise will only have one independent variable wich is the "open" of a stock price and we try to predict the close price. The main idea of this exercise is uses linear regression to predict the price of a stock so we are not looking use all the dataset but only information of one company.

# In[58]:


# Import the libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import seaborn as sns    # plot tools

# Take a look in the Stocks directory to select one dataset from a company

import os
print(os.listdir("../input/Data/"))

# Any results you write to the current directory are saved as output.


# In this case we choose the information of the stock price of google, and we are going to predict the close price starting from the open price. Other aproach could be, to choose the date as independet variable but probably is more useful (to bussines decisions) to know wich could be the "close" price knowing the actual price (open price). 

# In[ ]:


# Directory of the dataset 
filename = '../input/Data/Stocks/googl.us.txt'

# Read the file
Prgoo = pd.read_csv(filename,sep=',',index_col='Date')

# Prices is the predict value and initial the independet variable (y)
prices = Prgoo['Close'].tolist()
initial = (Prgoo['Open']).tolist()
 
#Convert to 1d Vector
prices = np.reshape(prices, (len(prices), 1))
initial = np.reshape(initial, (len(initial), 1))

Prgoo.head(5)


# First, let's check the correlation between the features of the dataset of google.

# In[59]:


Prgoo[['Open']].plot()
plt.title('Google Open Price')
plt.show()

Prgoo[['Close']].plot()
plt.title('Google Close Price')
plt.show()


# It can be observed that all both "open" and "close" has the same trend. with naked eye can be say that they have a positive correlation. Let's check the value of the correlation.

# In[60]:


plt.subplots(figsize=(8,6))
sns.heatmap(Prgoo.corr(),annot=True, linewidth=.5,)


# In[61]:


sns.distplot(Prgoo['Open'], hist = False, kde = True, kde_kws = {'linewidth': 5},label='Open',) 
sns.distplot(Prgoo['Close'], hist = False, kde = True, kde_kws = {'linewidth': 3},label='Close') 

plt.legend(prop={'size': 10}, title = 'Types',loc= 'best')
plt.title('Density Plot the Open and Close of the stock prices')
plt.xlabel('Prices')
plt.ylabel('Density')


# It's seem like there is a high correlation between the dependent and independet variable, so, its means that it more easy to our model to stimate a value. But this fact by general it doesnt happen and we need to make some pre procesing to the data to get a better prediction. But, in this case, due to the strong correlation we are going directly to train and test the model. 

# Now we create the model of linear regression and also a train and test dataset to evaluate the model.Also we use the R^2 (coefficient of determination) regression score function, that give us information about how well is adapted the regression to the data.

# In[63]:


#Splitting the dataset into the Training set and Test set
xtrain, xtest, ytrain, ytest = train_test_split(initial, prices, test_size=0.33, random_state=42)
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
 
#Train Set Graph
print('Train-set /','R2 score:',r2_score(ytrain,regressor.predict(xtrain)))
plt.scatter(xtrain, ytrain, color='red', label= 'Actual Price') #plotting the initial datapoints
plt.plot(xtrain, regressor.predict(xtrain), color='blue', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
plt.title('Linear Regression price| Open vs. Close')
plt.legend()
plt.xlabel('Prices')
plt.show()
 
#Test Set Graph
print('Test-set/','R2 score:',r2_score(ytest,regressor.predict(xtest)))
plt.scatter(xtest, ytest, color='red', label= 'Actual Price') #plotting the initial datapoints
plt.plot(xtest, regressor.predict(xtest), color='blue', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
plt.title('Linear Regression price| Open vs. Close')
plt.legend()
plt.xlabel('Prices')
plt.show()


# In this example we use information a Google, but this company is pretty stable in the stock prices so the simple correlation is going to get a good performance, but with other companies that could have more random behavior the simple linear regression may have issues. Though, in this case we could use other types of regression models for example the polinomial regresion that is not that is not as powerful as more complex models, but have less computacional cost and could get good enough results. 
# 

# In[ ]:




