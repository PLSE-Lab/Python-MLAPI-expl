#!/usr/bin/env python
# coding: utf-8

# # **Multiple Linear Regression on Pyramid scheme data**

# In[ ]:


#import the required libaries for reading the csv file and for plotting the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Read the data using pandas
data = pd.read_csv('/kaggle/input/pyramid-scheme-profit-or-loss/pyramid_scheme.csv')


# In[ ]:


# Check for the top 5 rows of data
data.head()


# In[ ]:


# This provides the basic information the data
data.info()


# In[ ]:


# this explains the basic stat behind the data
data.describe()


# In[ ]:


# Remove the column unnamed 
data = data.iloc[:,1:]
data.head()


# In[ ]:


# paiplot the data to check the linearity
plt.figure(figsize=(12,6))
sns.pairplot(data,kind='scatter')
plt.show()


# In[ ]:


# Correlation matrix
plt.figure(figsize=(12,6))
cor = data.corr()
sns.heatmap(cor,annot=True)


# In[ ]:


# oulier detection
sns.boxplot(x=data['cost_price'])


# In[ ]:


sns.boxplot(x=data['profit_markup'])


# In[ ]:


sns.boxplot(x=data['depth_of_tree'])


# In[ ]:


sns.boxplot(data['sales_commission'])


# In[ ]:


sns.boxplot(data['profit'])


# In[ ]:


# Hence there are no ouliers cook the data
X = data[['cost_price','profit_markup','depth_of_tree','sales_commission']]
X.head(2)


# In[ ]:


y = data['profit']
y.head(2)


# In[ ]:


# Import Linear Regresion model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[ ]:


# Split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)


# In[ ]:


# Fit the model with the train dataset
lr.fit(X_train,y_train)


# In[ ]:


# predict the model with test data
y_pred = lr.predict(X_test)


# In[ ]:


# Coeff and intercept
print('coef:',lr.coef_)
print('Intercept:',lr.intercept_)


# In[ ]:


# Model evaluation
from sklearn.metrics import r2_score,mean_squared_error
mse = mean_squared_error(y_test,y_pred)
rsq = r2_score(y_test,y_pred)


# In[ ]:


# R square value provides how accurate the model is for the data
print('mean sq error:',mse)
print('r square:',rsq)


# In[ ]:


# visualising the actual and predicted
plt.figure(figsize=(12,6))
c = [i for i in range(1,len(y_test)+1,1)]
plt.plot(c,y_test,color='b',linestyle='-')
plt.plot(c,y_pred,color='r',linestyle='-')
plt.xlabel('index')
plt.ylabel('Profit')
plt.title('Actual vs Predicted')
plt.show()


# In[ ]:


# Plot the error value
plt.figure(figsize=(12,6))
c = [i for i in range(1,len(y_test)+1,1)]
plt.plot(c,(y_test-y_pred),color='b',linestyle='-')
#plt.plot(c,y_pred,color='r',linestyle='-')
plt.xlabel('index')
plt.ylabel('Profit')
plt.title('Actual vs Predicted')
plt.show()


# In[ ]:


# import stat model
import statsmodels.api as sm


# In[ ]:


# Add constant to the train data
X_train_new = X_train
X_train_new = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit()


# In[ ]:


lm.params


# In[ ]:


# This helps to identify the column which are really significant
print(lm.summary())


# In[ ]:


# Finally the predicted and the actual plot
plt.figure(figsize=(12,6))
plt.plot(y_test,y_pred,color='green',linestyle='-',linewidth=1.5)
plt.show()


#  ***` If you like this approach please give this kernel an UPVOTE to show your appreciation `***
