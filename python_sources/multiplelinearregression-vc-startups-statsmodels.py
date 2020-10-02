#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm


# # Import the dataset

# In[ ]:


dataset = pd.read_csv('/kaggle/input/vc-startups/VC_Startups.csv')
dataset


# # Declare the independent and the dependent variables

# In[ ]:


# Independent variables - R&D Spend, Administration, Marketing Spend and State
X = dataset.iloc[:,:-1].values
X


# In[ ]:


# Dependent variable - Profit
y = dataset.iloc[:,4].values
y


# # Dummy Variables and Encoders

# In[ ]:


# Encoding 'State' categorical variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

st = ColumnTransformer([("State",OneHotEncoder(), [3])], remainder='passthrough')
X = st.fit_transform(X)
X


# In[ ]:


# Avoiding the Dummy Variable Trap

X = X[:,1:]
X


# # Getting ready for Regression

# In[ ]:


X.shape


# In[ ]:


# For StatsModels regression, we need the additional column as bias
# y = mx + c implies y = mx + c*1, so we will add an array of ones to X

X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X.shape


# # Regression with all the variables

# In[ ]:


X_opt = np.array(X[:,[0,1,2,3,4,5]], dtype=float)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# ## Interpreting the above results
# pval is the probability. The lower the pval the more the significance of the independent variables.
# x1 and x2 are the Dummy variables for State.
# x3 is R&D Spend, x4 is Admin Spend and x5 is Marketing Spend.
# Since x2 has the highest pval, we will remove it and run the Regression with the rest.

# In[ ]:


X_opt = np.array(X[:,[0,1,3,4,5]], dtype=float)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# ## Interpreting the above results
# x1 which is a Dummy variable for State has highest pval, we will remove it and run the Regression with the rest

# In[ ]:


X_opt = np.array(X[:,[0,3,4,5]], dtype=float)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# ## Interpreting the above results
# x2 which is of Administration has the highest pval, we will remove it and run the Regression with the rest

# In[ ]:


X_opt = np.array(X[:,[0,3,5]], dtype=float)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# ## Interpreting the above results

# In[ ]:


X_opt = np.array(X[:,[0,3]], dtype=float)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# ## Interpreting the above results
# This looks ideal when we consider the pval alone but the Adj. R-squared value decreased for the 1st time which means the change in predictors is not improving our model. Hence we will revert to the previous step.

# In[ ]:


X_opt = np.array(X[:,[0,3,5]], dtype=float)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# # Make predictions based on the created regression

# In[ ]:


# Create new data
new_data = pd.DataFrame({'const':1, 'R&D Spend':[50000, 200000, 400000], 'Marketing Spend':[300000, 150000, 75000]})
new_data = np.array(new_data)
new_data


# In[ ]:


predictions = regressor_OLS.predict(new_data)
predictions

