#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the libraries 
import pandas as pd
import numpy as np # data manupulation
from sklearn import metrics #metrics mse rme
import matplotlib.pyplot as plt
import seaborn as sns #visulization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing the Boston Housing dataset
from sklearn.datasets import load_boston #
boston = load_boston()
# Initializing the dataframe
data = pd.DataFrame(boston.data)
# See head of the dataset
data.head()


# In[ ]:


#Adding the feature names to the dataframe
data.columns = boston.feature_names
data.head()


# In[ ]:


#Adding target variable to dataframe
data['PRICE'] = boston.target 
# Median value of owner-occupied homes in $1000s


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.dtypes


# In[ ]:


# Identifying the unique number of values in the dataset
data.nunique()


# In[ ]:


# Check for missing values
data.isnull().sum()


# In[ ]:


# See rows with missing values
data[data.isnull().any(axis=1)]


# In[ ]:


# Viewing the data statistics
data.describe()


# In[ ]:


# Finding out the correlation between the features
corr = data.corr()
corr.shape


# In[ ]:


# Plotting the heatmap of correlation between features
plt.figure(figsize=(10,10))
sns.heatmap(corr, cbar=False, square= True, fmt='.1f', annot=True, annot_kws={'size':10}, cmap='Greens')


# # Linear Regression

# In[ ]:


# Import library for Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


# Spliting target variable and independent variables
X = data.drop(['PRICE'], axis = 1)
y = data['PRICE']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)


# In[ ]:


# Create a Linear regressor
lm = LinearRegression()

# Train the model using the training sets 
lm.fit(X_train, y_train)


# In[ ]:


# Value of y intercept
lm.intercept_


# In[ ]:


#Converting the coefficient values to a dataframe
coeffcients = pd.DataFrame([X_train.columns,lm.coef_]).T
coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})
coeffcients


# In[ ]:


# Model prediction on train data
y_pred= lm.predict(X_train)


# In[ ]:


# Model Evaluation on train data
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# In[ ]:


# Model prediction on test data
y_pred_test= lm.predict(X_test)


# In[ ]:


# Model Evaluation on test data
print('R^2:',metrics.r2_score(y_test, y_pred_test))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_pred_test))*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_pred_test))
print('MSE:',metrics.mean_squared_error(y_test, y_pred_test))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))


# In[ ]:


# Visualizing the differences between actual prices(train data) and predicted values
plt.scatter(y_train, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[ ]:


# Visualizing the differences between actual prices(test data) and predicted values
plt.scatter(y_test, y_pred_test)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[ ]:


# Checking residuals train data
plt.scatter(y_pred,y_train-y_pred)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()


# In[ ]:


# Checking residuals test data
plt.scatter(y_pred_test,y_test-y_pred_test)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()


# In[ ]:


# Checking Normality of errors for train data
sns.distplot(y_train-y_pred)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


# Checking Normality of errors for test data
sns.distplot(y_test-y_pred_test)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

