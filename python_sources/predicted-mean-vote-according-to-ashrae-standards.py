#!/usr/bin/env python
# coding: utf-8

# # Predicted Mean Vote according to ASHRAE standards

# ## What is PMV or Predicted Mean Vote?
# Predicted Mean Vote (PMV) is an index that aims to predict the mean value of votes of a group of occupants on a thermal sensation scale. Factors for thermal sensation are:
# 
# * operating temperature
# * relative humidity
# * airspeed 
# * metabolic rate 
# * clothing insulation 
# 
# PMV equal to zero is representing thermal neutrality. PMV in range of 0 to 3 is for warm sensation and pmv 0 to -3 is for cold sensation.
# 
# ## What is ASHRAE standard?
# ASHRAE Standard is the recognized standards for ventilation system design and acceptable indoor air quality (IAQ). 

# ## Aim of this notebook
# For standardizing the dataset the following two conditions arise:
# 
# 1. Standardize the whole dataset before splitting into a test and train.
# 2. Standardize after splitting into a test and train. Standardized both of them separately.
# 
# In the following notebook both the condition are checked.
# 
# * You can also have a track on [this](https://www.kaggle.com/questions-and-answers/159183) discussion for more details.

# ### Importing lib

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


# ### Importing dataset
# * Top --> Operating temp
# * Va --> Air speed
# * RH --> Relative Humidity
# * clo_tot --> clothing level
# * met --> metabolic rate
# * PMV --> Predicted Mean Vote
# 

# In[ ]:


data=pd.read_excel('../input/pmv-using-ashrae-standard/HVAC.xlsx',sheet_name=1)
data=data[['Top','Va','RH','met','clo_tot','PMV']]
data.head()


# ### Separating the feature and label

# In[ ]:


train_data=data[['Top','Va','RH','met','clo_tot']]
train_label=data['PMV']


# # First split then standardize

# Splitting in ratio 80:20 and using random seed also.

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(train_data,train_label,test_size=0.2,random_state=200)


# Standardizing test and train separately after splitting.

# In[ ]:


scaler = StandardScaler()
scaler.fit(x_train) #fit Compute the mean and std to be used for later scaling.
x_train=(scaler.transform(x_train)) #transform-Perform standardization by centering and scaling


# In[ ]:


scaler = StandardScaler()
scaler.fit(x_test)
x_test=(scaler.transform(x_test))


# Stochastic gradient descent and find MSE on test dataset.

# In[ ]:


sgd_reg = SGDRegressor(loss='squared_loss',alpha=0.05, fit_intercept=True, max_iter=200, random_state=123, learning_rate='constant',
                       eta0=0.01)
sgd_reg.fit(x_train, y_train)

y_pred2 = sgd_reg.predict(x_test)
mse = mean_squared_error(y_test,y_pred2)
print((mse))


# # Standardize whole dataset then split

# Standardize whole dataset.

# In[ ]:


scaler = StandardScaler()
scaler.fit(train_data) #fit Compute the mean and std to be used for later scaling.
train_data=(scaler.transform(train_data)) #transform-Perform standardization by centering and scaling


# Splitting

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(train_data,train_label,test_size=0.1,random_state=200)


# Stochastic gradient descent and find MSE on test dataset.

# In[ ]:


sgd_reg = SGDRegressor(loss='squared_loss',alpha=0.005, fit_intercept=True, max_iter=200, random_state=123, learning_rate='constant',
                       eta0=0.01)
sgd_reg.fit(x_train, y_train)

y_pred2 = sgd_reg.predict(x_test)
mse = mean_squared_error(y_test,y_pred2)
print((mse))


# # Conclusion
# 
# ## Mean Squared Error (MSE) for:
# ### First split then standardize = 0.0736
# ### Standardize whole dataset then split= 0.0724
# 
# We can see the MSE is less when dataset is first standardized the split.
# 
# *Note: This solution is valid only for the case when whole dataset is provided at once, not like in the real world scenario.*
# 
# ### Upvote the work if you like it.
