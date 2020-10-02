#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Importing USA Housing.csv
data = pd.read_csv('../input/USA_Housing.csv')


# ### EDA 

# In[ ]:


data.head()


# In[ ]:


# Checking for Null Values
data.info()


# In[ ]:


# Getting the summary of Data
data.describe()


# ### Data Preparation

# 1. There are no null values, so there is no need of deleting or replacing the data.
# 2. There is no necessity of having Address column/feature, so i am dropping it.

# In[ ]:


# Dropping Address Column
data.drop(['Address'],axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


# Let's plot a pair plot of all variables in our dataframe
sns.pairplot(data)


# In[ ]:


# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population'], y_vars='Price',height=7, aspect=0.7, kind='scatter')


# In[ ]:


sns.heatmap(data.corr(),annot=True)


# In[ ]:


data.corr().Price.sort_values(ascending=False)


# In[ ]:


sns.distplot(data.Price)


# #### Creating a Base Model

# In[ ]:


from sklearn import preprocessing
pre_process = preprocessing.StandardScaler()


# In[ ]:


# Putting feature variable to X
X = data[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]

# Putting response variable to y
y = data['Price']


# In[ ]:


X = pd.DataFrame(pre_process.fit_transform(X))


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=2)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# Importing RFE and LinearRegression
from sklearn.linear_model import LinearRegression


# In[ ]:


# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()


# In[ ]:


# fit the model to the training data
lm.fit(X_train, y_train)


# In[ ]:


# print the intercept
print(lm.intercept_)


# In[ ]:


# Let's see the coefficient
coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])
coeff_df


# From the above result we may infer that coefficient of Columns like 'Avg. Area House Age','Avg. Area Number of Rooms' and 'Avg. Area Number of Bedrooms' are influencing more as compared to other, hence we need to do scaling.

# In[ ]:


# Making predictions using the model
y_pred = lm.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[ ]:


from math import sqrt

rms = sqrt(mse)
rms


# From the above result we may infer that, mse is huge which shouldn't be, hence we need to improve our model.

# In[ ]:


# Actual and Predicted
c = [i for i in range(1,1501,1)] # generating index 
fig = plt.figure(figsize=(12,8))
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=15)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Housing Price', fontsize=16)                       # Y-label


# ### Also checking through Statistical Method

# In[ ]:


import statsmodels.api as sm
X_train_sm = X_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
X_train_sm = sm.add_constant(X_train_sm)
# create a fitted model in one line
lm_1 = sm.OLS(y_train,X_train_sm).fit()

# print the coefficients
lm_1.params


# In[ ]:


print(lm_1.summary())


# #### Dropping 'Avg. Area Number of Bedrooms' Column

# In[ ]:


X.head()


# In[ ]:


X.drop([3],axis=1, inplace=True)


# In[ ]:


X.head()


# In[ ]:


#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=2)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# Importing RFE and LinearRegression
from sklearn.linear_model import LinearRegression


# In[ ]:


# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()


# In[ ]:


# fit the model to the training data
lm.fit(X_train, y_train)


# In[ ]:


# print the intercept
print(lm.intercept_)


# In[ ]:


# Let's see the coefficient
coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])
coeff_df


# In[ ]:


# Making predictions using the model
y_pred = lm.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[ ]:


from math import sqrt

rms = sqrt(mse)
rms


# In[ ]:


# Actual and Predicted
c = [i for i in range(1,1501,1)] # generating index 
fig = plt.figure(figsize=(12,8))
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=15)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Housing Price', fontsize=16)                       # Y-label


# In[ ]:




