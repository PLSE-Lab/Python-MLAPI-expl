#!/usr/bin/env python
# coding: utf-8

# # Electric Motor Temperature

# ## Dataset Provded by: kaggle.com

# ### Author: Jeffrey Cabrera

# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Columns
# #### ambient: Ambient temperature as measured by a thermal sensor located closely to the stator.
# #### coolant: Coolant temperature. The motor is water cooled. Measurement is taken at outflow.
# #### u_d: Voltage d-component
# #### u_q: Voltage q-component
# #### motor_speed: Motor speed
# #### torque: Torque induced by current.
# #### i_d: Current d-component
# ####  i_q: Current q-component
# #### pm: Permanent Magnet surface temperature representing the rotor temperature. This was measured with an infrared thermography unit.
# #### stator_yoke: Stator yoke temperature measured with a thermal sensor.
# #### stator_tooth: Stator tooth temperature measured with a thermal sensor.
# ####  stator_winding: Stator winding temperature measured with a thermal sensor.
# #### profile_id: Each measurement session has a unique ID. Make sure not to try to estimate from one session onto the other as they are strongly independent.

# ##### Purpose of this notebook is to predict a surface temperature of a permanent Magnet Surface

# In[ ]:


# import the data set
motor_data = pd.read_csv("../input/pmsm_temperature_data.csv")
motor_data.head()


# In[ ]:


# Display the shape
motor_data.shape


# In[ ]:


# Display the data types
motor_data.dtypes


# In[ ]:


# Display any missing values from the data
motor_data.isnull().sum()


# In[ ]:


# Display the statisitcsal desctiption
motor_data.describe().T


# In[ ]:


# Display any correlation
motor_corr = motor_data.corr()
motor_corr


# In[ ]:


# Display the correlation data with non-numerical values
for col in list(motor_data.columns):
    motor_corr[col] = pd.cut(motor_corr[col],
                            (-1, -0.5, -.1, 0.1, 0.5, 1),
                            labels=["NegStrong", "NegMedium", "Weak",
                                   "PosMedium", "PosStrong"])
motor_corr


# In[ ]:


# Save the response variable
Y = motor_data['pm']


# In[ ]:


# Save the predictor variables
X = motor_data.drop('pm', axis=1)


# In[ ]:


# Standardize the predictor variables
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)


# In[ ]:


# Feature Selection with backward elimiation use using statsmodel linear regression
# add the Y-intercept
X_std0 = sm.add_constant(X_std)


# In[ ]:


# Run OLS on the data
Y_model0 = sm.OLS(Y, X_std0)
Y_model0 = Y_model0.fit()
Y_model0 = Y_model0.summary()
Y_model0


# In[ ]:


# Use Linear Regression with cross_val_score
lr = linear_model.LinearRegression()
scores = cross_val_score(lr, X_std, Y, cv=10)
print("Mean R-Squared Score", sum(scores)/len(scores))


# In[ ]:


# Use train_tet_split on the data
X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y,
                                                   random_state=33,
                                                   test_size=0.33)


# In[ ]:


# Train the data
lr = linear_model.LinearRegression()
model = lr.fit(X_train, Y_train)


# In[ ]:


# Predict on the test data
predictions = lr.predict(X_test)
predictions[:5]


# In[ ]:


# Display the score
print("R-Sqaured Score:", model.score(X_test, Y_test))


# In[ ]:


# Plot the scores
plt.scatter(predictions, Y_test)
plt.xlabel("Predicted values")
plt.ylabel("Actual Values");


# # Any feeedback would be greatly appreciated.
