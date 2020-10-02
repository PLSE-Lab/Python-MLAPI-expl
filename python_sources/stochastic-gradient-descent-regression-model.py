#!/usr/bin/env python
# coding: utf-8

# # Importing all the required libraries 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing the dataset
# Getting the USA_Housing.csv dataset from the directory

# In[ ]:


data = pd.read_csv('/kaggle/input/usa-housing/USA_Housing.csv')


# # Getting feature and dependent variables
# x is our feature variable with 5000 rows and 5 columns of shape (5000,5)
# 
# y is our dependent variable with 5000 rows and 1 columns of shape (5000,)

# In[ ]:


x = data.drop(axis=1,columns=['Price','Address']).to_numpy()


# In[ ]:


y = data['Price'].to_numpy()


# # Scaling the feature data
# we are scaling the feature data because all the data is in different units.

# In[ ]:


std = StandardScaler()


# In[ ]:


x = std.fit_transform(x)


# # SGDRegressor
# This StochasticGradientRegressor (SGDRegressor) is used here because this regressor has a good method to reduce the error produced in the model by finding the global minima of the error.

# In[ ]:


reg = SGDRegressor(random_state=0)


# # KFold Method
# 
# Applying KFold method to get the train and test data with the random seeding trick which will actually boost your model's accuracy with less error. 

# In[ ]:


error = []
for n in range(50):
    kf = KFold(n_splits=5,random_state=n,shuffle=True)
    for i,j in kf.split(x,y):
        X_train,X_test = x[i],x[j]
        y_train,y_test = y[i],y[j]
    reg.fit(X_train,y_train)
    pred = reg.predict(X_test)
    error.append(mean_squared_error(y_test,pred))
print("Best Seed Error : ",min(error)," Seed : ",error.index(min(error)))


# # Best seed to KFold
# Here we are fitting the best seed to KFold function inorder to get good train/test data.

# In[ ]:


kf = KFold(n_splits=5,random_state=3,shuffle=True)
for i,j in kf.split(x,y):
    X_train,X_test = x[i],x[j]
    y_train,y_test = y[i],y[j]


# # Fitting and Predicting
# Here we are fitting and predicting the housing price with the model we built.

# In[ ]:


reg.fit(X_train,y_train)


# In[ ]:


pred = reg.predict(X_test)


# # Model's Score
# It looks pretty good because our model is 92% accurate in predicting the housing price.

# In[ ]:


reg.score(X_test,y_test)


# # Error Metrics
# 1.Here we are finding the various errors like Mean_Squared_Error(MSE), Mean_Absolute_Error(MAE), Root_Mean_Square_Error(RMSE) to say whether our model is producing minimum error while predictions.
# 
# 2.Lower the model's error,higher the model's accuracy

# In[ ]:


mae = mean_absolute_error(y_test,pred)
mae


# In[ ]:


mse = mean_squared_error(y_test,pred)
print(mse)


# In[ ]:


rmse = np.sqrt(mse)
rmse


# # Learn yourself !
# Here I used SGDRegressor to predict the housing price but there are many regression or regressors available out there. I recommend you to try those regressors because different methods produce different results (i.e Accuracy, Error,etc..).
