#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This program attempts to visualize media planning spend data by media channel against sales data
# After visualizing variable pairs, we try to fit a linear regression model for the dataset
# After fitting the LR model, we predict sales based on past media spend patterns
# we estimate the accuracy of the prediction
# we have to use a few external python libraries to complete the prediction
# for visualisation, both pandas and seaborn were used
# This dataset has been constructed from scratch as in the real world, it is difficult to assemble channel level sales data

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# This kernel covers starter concepts of Linear Regression Machine Learning
# Inspiration for this kernel is from Pierian Data


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Import Ad spend data by media channel and observe the data

# In[ ]:


mediamix = pd.read_csv("../input/mediamix_sales.csv")
mediamix.tail()


# In[ ]:


mediamix.describe()


# **Visualising data

# In[ ]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='tv_cricket',y='sales',data=mediamix)
# trying to visualise Sales when Cricket ads were shown on TV channels


# In[ ]:


sns.jointplot(x='tv_RON',y='sales',data=mediamix)


# Sales is not explained from any one dependent variable eg: TV ads on Cricket or TV ads on RON (Run of network)

# In[ ]:


sns.jointplot(x='tv_sponsorships',y='sales',data=mediamix)


# In[ ]:


sns.jointplot(x='tv_sponsorships',y='sales',kind='hex', data=mediamix)
# change visualisation to observe relationships between dependent and independent variables


# seaborne offers a simple visualisation summary across variable pairs. this is very useful in selecting indepenent variables for further analysis

# In[ ]:


sns.pairplot(mediamix)
#explore facetgrid in seaborn documenteantation for customising pairplot visualisations


# Correlation Matrix
# A pairplot is usually difficult to analyse. however, it still helps to establish correlation among independent variables to remove or group variables before finalising prediction algorithm.
# 

# In[ ]:


#correlation matrix
corr_media = mediamix.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_media, vmax=.8, square=True);


# Establishing data ranges

# For those interested in custom visualisations for specific variables, refer https://seaborn.pydata.org/tutorial/axis_grids.html

# Create a Linear Model Plot using Seaborn using the most correlated variable for Sales eg: radio

# In[ ]:


sns.lmplot(x='radio', y='sales', data=mediamix)


# please note that the line of best fit does not explain the sales outcomes. The prediction error is very likely to high. Commonsense says Univariate regression does not work for forecasting sales. Lets us fit a model with multiple variables. However, fitting all variables at once creates input data issues. I can fix the data issues and continue...but in this basic kernel, I am going to model after limited variables

# In[ ]:


y=mediamix['sales']
X=mediamix[['tv_RON', 'tv_sponsorships', 'tv_cricket','radio', 'NPP','Magazines','OOH', 'Social', 'Display_Rest', 
            'Search', 'Programmatic']]
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


mm_model = LinearRegression()


# In[ ]:


mm_model.fit(X_train,y_train)


# In[ ]:


# Each variable in the dataset weights the predictions diferently
# these weights are referred to as coefficients
print('Coefficients: \n', mm_model.coef_)


# note the relative weight different for the independent variables.
# 
# Let us attempt our first prediction

# In[ ]:


sales_forecast = mm_model.predict(X_test)


# In[ ]:


plt.scatter(y_test,sales_forecast)
plt.xlabel('Y test')
plt.ylabel('Predicted Y')


# So Is this is a good prediction model ? Lets check

# In[ ]:


from sklearn import metrics
print ('MAE :', metrics.mean_absolute_error(y_test, sales_forecast))
print ('MSE :', metrics.mean_squared_error(y_test, sales_forecast))
print ('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, sales_forecast)))


# Are the predictions normally distributed ?

# In[ ]:


sns.distplot(sales_forecast, bins=50)


# Both Skew and Kurtosis are seen in the above distribution of predicted values.
# This is a basic kernel and hope to add more meaningful analysis in the kernels to come.

# In[ ]:




