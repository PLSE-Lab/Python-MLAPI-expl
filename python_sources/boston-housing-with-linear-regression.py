#!/usr/bin/env python
# coding: utf-8

# # Boston Housing with Linear Regression
# 
# ** With this data our objective is create a model using linear regression to predict the houses price  **
# 
# The data contains the following columns:
# * 'crim': per capita crime rate by town.
# * 'zn': proportion of residential land zoned for lots over 25,000 sq.ft.
# * 'indus': proportion of non-retail business acres per town.
# * 'chas':Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# * 'nox': nitrogen oxides concentration (parts per 10 million).
# * 'rm': average number of rooms per dwelling.
# * 'age': proportion of owner-occupied units built prior to 1940.
# * 'dis': weighted mean of distances to five Boston employment centres.
# * 'rad': index of accessibility to radial highways.
# * 'tax': full-value property-tax rate per $10,000.
# * 'ptratio': pupil-teacher ratio by town
# * 'black': 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
# * 'lstat': lower status of the population (percent).
# * 'medv': median value of owner-occupied homes in $$1000s
# 
# Ps: this is my first analysis, i'm learning how to interpret the plots.

# **Lets Start**
# 
# First we need to prepare our enviroment importing some librarys

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


# Importing DataSet and take a look at Data
BostonTrain = pd.read_csv("../input/boston_train.csv")


# ** Here we can look at the BostonTrain data **

# In[51]:


BostonTrain.head()


# In[52]:


BostonTrain.info()
BostonTrain.describe()


# ** Now, or goal is think about the columns, and discovery which columns is relevant to build our model, because if we consider to put columns with not relevant  with our objective "medv" the model may be not efficient **

# In[53]:


#ID columns does not relevant for our analysis.
BostonTrain.drop('ID', axis = 1, inplace=True)


# In[54]:


BostonTrain.plot.scatter('rm', 'medv')


# In this plot its clearly to see a linear pattern. Wheter more average number of rooms per dwelling, more expensive the median value is.

# ** Now lets take a loot how the all variables relate to each other. **

# In[55]:


plt.subplots(figsize=(12,8))
sns.heatmap(BostonTrain.corr(), cmap = 'RdGy')


# At this heatmap plot, we can do our analysis better than the pairplot.
# 
# Lets focus ate the last line, where y = medv:
# 
# When shades of Red/Orange: the more red the color is on X axis, smaller the medv. Negative correlation                           
# When light colors: those variables at axis x and y, they dont have any relation. Zero correlation                               
# When shades of Gray/Black : the more black the color is on X axis, more higher the value med is. Positive correlation

# **Lets plot the paiplot, for all different correlations**

# Negative Correlation. 
# 
# When x is high y is low and vice versa.
# 
# To the right less negative correlation.

# In[56]:


sns.pairplot(BostonTrain, vars = ['lstat', 'ptratio', 'indus', 'tax', 'crim', 'nox', 'rad', 'age', 'medv'])


# Zero Correlation. When x and y are completely independent
# 
# Positive Correlation. When x and y go together
# 
# to the right more independent.

# In[57]:


sns.pairplot(BostonTrain, vars = ['rm', 'zn', 'black', 'dis', 'chas','medv'])


# # Trainning Linear Regression Model
# **Define X and Y**
# 
# X: Varibles named as predictors, independent variables, features.                                                               
# Y: Variable named as response or dependent variable

# In[58]:


X = BostonTrain[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'black', 'lstat']]
y = BostonTrain['medv']


# **Import sklearn librarys:**    
# train_test_split, to split our data in two DF, one for build a model and other to validate.                                     
# LinearRegression, to apply the linear regression.

# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# In[61]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[62]:


predictions = lm.predict(X_test)


# In[63]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[64]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# Considering the RMSE: we can conclude that  this model average error is RMSE at medv, which means RMSE *1000  in money

# In[65]:


sns.distplot((y_test-predictions),bins=50);


# As more normal distribution, better it is.

# In[66]:


coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['coefficients']
coefficients


# How to interpret those coefficients:
#     they are in function of Medv, so 
#     
#     for one unit that nox increase, the house value decrease 'nox'*1000 (Negative correlation) money unit.
#     for one unit that rm increase, the house value increase 'rm'*1000 (Positive correlation) money unit.
# 
# *1000 because the medv is in 1000
# and this apply to the other variables/coefficients.
#     

# As i said, this is my first analysis at my first machine learning method and i'm sure its gonna be better in a near future and my english too =)
# 
# thanks

# In[ ]:




