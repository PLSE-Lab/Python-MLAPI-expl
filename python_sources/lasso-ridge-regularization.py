#!/usr/bin/env python
# coding: utf-8

# # Featurization, Model Selection & Tuning - Linear Regression

# **Why is regularization required ?**
# 
# We are well aware of the issue of 'Curse of dimensionality', where the no. of columns are so huge that the no. of rows does not cover all the permutation and combinations that is applicable for this dataset.
# For eg: Data having 10 columns should have 10! rows but it has only 1000 rows
# 
# Therefore,when we depict this graphically there would be lot of white spaces as the datapoints for those regions may not be covered in the dataset.
# 
# If a  linear regression model is tested over such a data, the model will tend to overfit this data by having sharp peaks & slopes. Such a model would have 100% training accuracy but would definitely fail in the test environment.
# 
# Thus arose the need of introducing slight errors in the form of giving smooth bends instead of sharp peaks (thereby reducing overfit).This is achieved by tweaking the model parameters (coefficients) and the hyperparameters (penalty factor). 

# ## Agenda
# 
# * Perform basic EDA
# * Scale data and apply Linear, Ridge & Lasso Regression with Regularization 
# * Compare the r^2 score to determine which of the above regression methods gives the highest score
# * Compute Root mean squared error (RMSE) which inturn gives a better score than r^2
# * Finally use a scatter plot to graphically depict the correlation between actual and predicted mpg values

# # 1. Import packages and observe dataset

# In[ ]:


#Import numerical libraries
import pandas as pd
import numpy as np

#Import graphical plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Import Linear Regression Machine Learning Libraries
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score


# In[ ]:


data = pd.read_csv('../input/carmpg/car-mpg (1).csv')
data.head()


# In[ ]:


#Drop car name
#Replace origin into 1,2,3.. dont forget get_dummies
#Replace ? with nan
#Replace all nan with median

data = data.drop(['car_name'], axis = 1)
data['origin'] = data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
data = pd.get_dummies(data,columns = ['origin'])
data = data.replace('?', np.nan)
data = data.apply(lambda x: x.fillna(x.median()), axis = 0)


# In[ ]:


data.head()


# We have to predict the mpg column given the features.

# # 2. Model building

# Here we would like to scale the data as the columns are varied which would result in 1 column dominating the others.
# 
# First we divide the data into independent (X) and dependent data (y) then we scale it. 
# 
# #### Tip!: ####
# 
# *The reason we don't scale the entire data before and then divide it into train(X) & test(y) is because once you scale the data, the type(data_s) would be numpy.ndarray. It's impossible to divide this data when it's an array. 
# *
# 
# Hence we divide type(data) pandas.DataFrame, then proceed to scaling it.

# In[ ]:


X = data.drop(['mpg'], axis = 1) # independent variable
y = data[['mpg']] #dependent variable


# In[ ]:


#Scaling the data

X_s = preprocessing.scale(X)
X_s = pd.DataFrame(X_s, columns = X.columns) #converting scaled data into dataframe

y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s, columns = y.columns) #ideally train, test data should be in columns


# In[ ]:


#Split into train, test set

X_train, X_test, y_train,y_test = train_test_split(X_s, y_s, test_size = 0.30, random_state = 1)
X_train.shape


# ## 2.a Simple Linear Model

# In[ ]:


#Fit simple linear model and find coefficients
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

for idx, col_name in enumerate(X_train.columns):
    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))
    
intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))


# ## 2.b Regularized Ridge Regression

# In[ ]:


#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff

ridge_model = Ridge(alpha = 0.3)
ridge_model.fit(X_train, y_train)

print('Ridge model coef: {}'.format(ridge_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here    


# ## 2.c Regularized Lasso Regression

# In[ ]:


#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff

lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(X_train, y_train)

print('Lasso model coef: {}'.format(lasso_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here   


# Here we notice many coefficients are turned to 0 indicating drop of those dimensions from the model

# # 3. Score Comparison

# In[ ]:


#Model score - r^2 or coeff of determinant
#r^2 = 1-(RSS/TSS) = Regression error/TSS 


#Simple Linear Model
print(regression_model.score(X_train, y_train))
print(regression_model.score(X_test, y_test))

print('*************************')
#Ridge
print(ridge_model.score(X_train, y_train))
print(ridge_model.score(X_test, y_test))

print('*************************')
#Lasso
print(lasso_model.score(X_train, y_train))
print(lasso_model.score(X_test, y_test))


# ## Polynomial Features

# If you wish to further compute polynomial features, you can use the below code.

# In[ ]:


#poly = PolynomialFeatures(degree = 2, interaction_only = True)

#Fit calculates u and std dev while transform applies the transformation to a particular set of examples
#Here fit_transform helps to fit and transform the X_s
#Hence type(X_poly) is numpy.array while type(X_s) is pandas.DataFrame 
#X_poly = poly.fit_transform(X_s)
#Similarly capture the coefficients and intercepts of this polynomial feature model


# # 4. Model Parameter Tuning

# * r^2 is not a reliable metric as it always increases with addition of more attributes even if the attributes have no influence on the predicted variable. Instead we use adjusted r^2 which removes the statistical chance that improves r^2 
# 
# (adjusted r^2 = r^2 - fluke)
# * Scikit does not provide a facility for adjusted r^2... so we use statsmodel, a library that gives results similar to what you obtain in R language
# * This library expects the X and Y to be given in one single dataframe

# In[ ]:


data_train_test = pd.concat([X_train, y_train], axis =1)
data_train_test.head()


# In[ ]:


import statsmodels.formula.api as smf
ols1 = smf.ols(formula = 'mpg ~ cyl+disp+hp+wt+acc+yr+car_type+origin_america+origin_europe+origin_asia', data = data_train_test).fit()
ols1.params


# In[ ]:


print(ols1.summary())


# In[ ]:


#Lets check Sum of Squared Errors (SSE) by predicting value of y for test cases and subtracting from the actual y for the test cases
mse  = np.mean((regression_model.predict(X_test)-y_test)**2)

# root of mean_sq_error is standard deviation i.e. avg variance between predicted and actual
import math
rmse = math.sqrt(mse)
print('Root Mean Squared Error: {}'.format(rmse))


# **So there is an avg. mpg difference of 0.37 from real mpg**

# In[ ]:


# Is OLS a good model ? Lets check the residuals for some of these predictor.

fig = plt.figure(figsize=(10,8))
sns.residplot(x= X_test['hp'], y= y_test['mpg'], color='green', lowess=True )


fig = plt.figure(figsize=(10,8))
sns.residplot(x= X_test['acc'], y= y_test['mpg'], color='green', lowess=True )


# In[ ]:


# predict mileage (mpg) for a set of attributes not in the training or test set
y_pred = regression_model.predict(X_test)

# Since this is regression, plot the predicted y value vs actual y values for the test data
# A good model's prediction will be close to actual leading to high R and R2 values
#plt.rcParams['figure.dpi'] = 500
plt.scatter(y_test['mpg'], y_pred)


# # 5. Inference

# **Both Ridge & Lasso regularization performs very well on this data, though Ridge gives a better score. The above scatter plot depicts the correlation between the actual and predicted mpg values.**
# 
# ***This kernel is a work in progress.***
