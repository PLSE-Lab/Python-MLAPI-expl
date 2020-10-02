#!/usr/bin/env python
# coding: utf-8

# I am a newbie to coding, data science and machine learning. 
# I wanted to test out my skills, improve my knowledge about regression model 
# I tested multiple linear regression model to determine the price of houses in King County, USA based on different variables.
# 
# BTW Special thanks to Burhan Y. Kiyakoglu for his kernel on this dataset. It was base of my work
# 
# Feel free to comment on my work. I can improve my skills with feedback
# 
# 1. [Pearson Correlation Matrix](#1) 
# 1. [Split the Data](#2)
# 1. [Check Train Data Scatter Plot](#3)
# 1. [Create Multiple Linear Regression Model](#4)
# 1. [Model Evaluation](#5)
# 1. [Polynomial Regression](#6)

# In[ ]:


# import all libraries used in the code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Open the data
df = pd.read_csv("../input/kc_house_data.csv")

#show summary data and detail information about data
# df.describe()
# df.info()
df.head()
df.columns


# # <a id="1"></a> Pearson Correlation Matrix
#  I created the matrix to check which variables are mostly correlated to price. 

# In[ ]:


# Pearson Correlation Matrix to determine which variables to choose
features = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
            'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
            'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']

mask = np.zeros_like(df[features].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=23)

sns.heatmap(df[features].corr(),linewidths=0.25, vmax=1.0, square=True, cmap="BuPu",
            linecolor='w', annot=True, mask=mask, cbar_kws={"shrink": .75})


# # <a id="2"></a> Split the Data

# In[ ]:



train_data, test_data = train_test_split(df, train_size=0.8)


# # <a id="3"></a> Check Train Data Scatter Plot
# I found top 3 correlator as living spaces, grade and square footage of house apart from basement. 
# Created new subset with those 3 variables
# I wanted to check the scatter plot visually of each variable 

# In[ ]:


# check train data scatter plot
plt.scatter(train_data.sqft_living, train_data.price, color='darkblue', alpha=.1)
plt.xlabel('Living Space (sqft)')
plt.ylabel('Price (USD)')


# In[ ]:


# check train data scatter plot
plt.scatter(train_data.grade, train_data.price, color='darkblue', alpha=.1)
plt.xlabel('Grade')
plt.ylabel('Price (USD)')


# In[ ]:


# check train data scatter plot
plt.scatter(train_data.sqft_above, train_data.price, color='darkblue', alpha=.1)
plt.xlabel('Space except Basement (sqft)')
plt.ylabel('Price (USD)')


# # <a id="4"></a> Create Multiple Linear Regression Model
# Create the model with top 3 correlating variable

# In[ ]:


# Create the model
regr = linear_model.LinearRegression()
x = np.asanyarray(train_data[['sqft_living', 'grade', 'sqft_above']])
y = np.asanyarray(train_data[['price']])
regr.fit(x, y)


# In[ ]:


# The coefficients
print ('Coefficients: ', regr.coef_)


# In[ ]:


# Prediction and model evaluation
y_hat = regr.predict(test_data[['sqft_living', 'grade', 'sqft_above']])
x = np.asanyarray(test_data[['sqft_living', 'grade', 'sqft_above']])
y = np.asanyarray(test_data[['price']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))


# # <a id="5"></a> Model Evaluation
# R2 score is %56. This means that I need to try different models to reach higher accuracy to estimate the price. 

# In[ ]:


# Explained variance score: 1 is perfect prediction
print('Variance score: %.3f' % regr.score(x, y))


# # <a id="6"></a> Polynomial Regression
# As I mentioned above multiple linear regression model doesnt provide enough accuracy to estimate the price
# I added more features to try polynomial regression. 
# With this model I achieved better accuracy than linear model

# In[ ]:


# create an array for top correlating features to be used in Polynomial Regression
features_poly = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'bedrooms', 'yr_built', 'waterfront', 'condition']

# create the polynomial regression model
regr_poly = PolynomialFeatures()
train_data_poly = regr_poly.fit_transform(train_data[features_poly])
test_data_poly = regr_poly.fit_transform(test_data[features_poly])
poly = linear_model.LinearRegression().fit(train_data_poly, train_data['price'])

# test the model with testing data
predp = poly.predict(test_data_poly)

# Explained variance score: 1 is perfect prediction
print('Variance score of Polynomial Regression: %.3f' % poly.score(test_data_poly, test_data['price']))

