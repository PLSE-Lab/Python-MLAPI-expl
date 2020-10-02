#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## Problem Statement
# 
# Use Multiple Linear Regression to **predict the consumption of petrol** given relevant variables are the petrol tax, the per capita, income, the number of miles of paved highway, and the proportion of the population with driver's licenses.
# 
# ## Dataset
# 
# There are 48 rows of data.  The data include:
# 
#       I,  the index;
#       A1, the petrol tax;
#       A2, the per capita income;
#       A3, the number of miles of paved highway;
#       A4, the proportion of drivers;
#       B,  the consumption of petrol.
# 
# ### Reference 
# 
#     Helmut Spaeth,
#     Mathematical Algorithms for Linear Regression,
#     Academic Press, 1991,
#     ISBN 0-12-656460-4.
# 
#     S Weisberg,
#     Applied Linear Regression,
#     New York, 1980, pages 32-33.
# 
# ## Exploratory Data Analysis
# 
# *Read the dataset given in file named **'petrol.csv'**. Check the statistical details of the dataset.*
# 
# **Hint:** You can use **df.describe()**

# In[ ]:


import os
os.getcwd()


# In[ ]:


import numpy as np
import pandas as pd
df=pd.read_csv('../input/petrol-consumption/petrol_consumption.csv')
df.head()


# In[ ]:


df1=df
df.shape


# # Cap outliers 
# 
# Find the outliers and cap them. (Use (Q1 - 1.5 * IQR) as the minimum cap and (Q3 + 1.5 * IQR) as the max cap. The decision criteria is you should consider the datapoints which only falls within this range. The data points which fall outside this range are outliers and the entire row needs to be removed

# * Petrol_tax

# In[ ]:


q1=df1['Petrol_tax'].quantile(0.25)
q3=df1['Petrol_tax'].quantile(0.75)
iqr=q3-q1
print(iqr)
ll=q1-1.5*iqr
ul=q3+1.5*iqr
df1=df1[~((df['Petrol_tax']<ll) | (df1['Petrol_tax']>ul))]
df1.shape


# * Average_income

# In[ ]:


q1=df1['Average_income'].quantile(0.25)
q3=df1['Average_income'].quantile(0.75)
iqr=q3-q1
print(iqr)
ll=q1-1.5*iqr
ul=q3+1.5*iqr
df1=df1[~((df1['Average_income']<ll) | (df1['Average_income']>ul))]
df1.shape


# * Paved_Highways

# In[ ]:


q1=df1['Paved_Highways'].quantile(0.25)
q3=df1['Paved_Highways'].quantile(0.75)
iqr=q3-q1
print(iqr)
ll=q1-1.5*iqr
ul=q3+1.5*iqr
df1=df1[~((df1['Paved_Highways']<ll) | (df1['Paved_Highways']>ul))]
df1.shape


# * Population_Driver_licence(%)

# In[ ]:


q1=df1['Population_Driver_licence(%)'].quantile(0.25)
q3=df1['Population_Driver_licence(%)'].quantile(0.75)
iqr=q3-q1
print(iqr)
ll=q1-1.5*iqr
ul=q3+1.5*iqr
df1=df1[~((df1['Population_Driver_licence(%)']<ll) | (df1['Population_Driver_licence(%)']>ul))]
df1.shape


# * Petrol_Consumption

# In[ ]:


q1=df1['Petrol_Consumption'].quantile(0.25)
q3=df1['Petrol_Consumption'].quantile(0.75)
iqr=q3-q1
print(iqr)
ll=q1-1.5*iqr
ul=q3+1.5*iqr
df1=df1[~((df1['Petrol_Consumption']<ll) | (df1['Petrol_Consumption']>ul))]
df1.shape


# # Transform the dataset 
# Divide the data into feature(X) and target(Y) sets.

# In[ ]:


df.corr()


# In[ ]:


df1.corr()


# In[ ]:


Y=df1.Petrol_Consumption
X=df1.drop(['Petrol_Consumption'],axis=1)


# # Split data into train, test sets 
# Divide the data into training and test sets with 80-20 split using scikit-learn. Print the shapes of training and test feature sets.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


Y=df1.Petrol_Consumption
X=df1.Petrol_tax
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state =0)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# # Build Model 
# Estimate the coefficients for each input feature. Construct and display a dataframe with coefficients and X.columns as columns

# In[ ]:


import statsmodels.formula.api as smf
import statsmodels.api as sm
model=sm.OLS(Y_train,X_train).fit()
model.summary()


# # R-Square 

# # Evaluate the model 
# Calculate the accuracy score for the above model.

# In[ ]:


def mean_abs_per_err(pred,actual):
    per=(abs(actual-pred))/actual
    value=per*100
    return value


# ### Training Accuracy

# In[ ]:


pred_y=model.predict(X_train)
from sklearn import metrics
MSE=metrics.mean_squared_error(pred_y,Y_train)
print('Mean Squared Error:',MSE)
RMSE=np.sqrt(MSE)
print('Root Mean Squared Error:',RMSE)
MAPE=(mean_abs_per_err(pred_y,Y_train).sum())/Y_train.size
print("Mean Absolute Percentage Error:", MAPE)


# ### Testing Accuracy

# In[ ]:


pred_y=model.predict(X_test)
from sklearn import metrics
MSE=metrics.mean_squared_error(pred_y,Y_test)
print('Mean Squared Error:',MSE)
RMSE=np.sqrt(MSE)
print('Root Mean Squared Error:',RMSE)
MAPE=(mean_abs_per_err(pred_y,Y_test).sum())/Y_test.size
print("Mean Absolute Percentage Error:", MAPE)


# # Repeat the same Multi linear regression modelling by adding Population_Driver_licence(%), Income and Highway features
# Find R2 
# 

# In[ ]:


Y=df1.Petrol_Consumption
X=df1.drop(['Petrol_Consumption'],axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state =0)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


model=sm.OLS(Y_train,X_train).fit()
model.summary()


# ### Training Accuracy

# In[ ]:


pred_y=model.predict(X_train)
from sklearn import metrics
MSE=metrics.mean_squared_error(pred_y,Y_train)
print('Mean Squared Error:',MSE)
RMSE=np.sqrt(MSE)
print('Root Mean Squared Error:',RMSE)
MAPE=(mean_abs_per_err(pred_y,Y_train).sum())/Y_train.size
print("Mean Absolute Percentage Error:", MAPE)


# ### Testing Accuracy

# In[ ]:


pred_y=model.predict(X_test)
from sklearn import metrics
MSE=metrics.mean_squared_error(pred_y,Y_test)
print('Mean Squared Error:',MSE)
RMSE=np.sqrt(MSE)
print('Root Mean Squared Error:',RMSE)
MAPE=(mean_abs_per_err(pred_y,Y_test).sum())/Y_test.size
print("Mean Absolute Percentage Error:", MAPE)


# #  Print the coefficients of the multilinear regression model

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[ ]:


regressor.coef_


# * When there was only one independent variable Petrol_tax in our regression the $R^2$ value was 95%
# * Adjusted $R^2$ value was 95.3%
# * When there were three independent variable Petrol_tax,Average_income,Paved_Highways and Population_Driver_licence(%) in our regression the $R^2$ value was 99%
# * Adjusted $R^2$ value was 98.8%
# * On addition of other varibles the $R^2$ value increased by a considerable amount of 4%. However the Adjusted $R^2$ value did not increase as much as $R^2$. This is because Adjusted $R^2$ only increases if the variable contributes in the prediction on dependent variable
