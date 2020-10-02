#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Check the relationship and the impact of minimum temperatue on the maximum temperature.

# In[ ]:


# Import all the needed packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Import the complete dataset.
df_data = pd.read_csv("../input/Istanbul Weather Data.csv")


# In[ ]:


# Check the dataset.
df_data.head()


# In[ ]:


# Check the correlation between the variables.
df_data.corr()


# In[ ]:


# Check the stats of the variables.
df_data.describe()


# In[ ]:


# Plot a scatter chart to visualise the relationship between the Min & Max Temp.
df_data.plot(kind='scatter',x='MinTemp',y='MaxTemp', figsize=(10,6))
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.title('MinTemp Vs MaxTemp')

plt.show()


# In[ ]:


# Create a train & Test DataSet
msk = np.random.rand(len(df_data)) < 0.8
train_set = df_data[msk]
test_set = df_data[~msk]

#Check the shape of the datasets
print(train_set.shape)
print(test_set.shape)


# In[ ]:


# Creating arrays to train the model.
train_x = np.asanyarray(train_set[["MinTemp"]])
train_y = np.asanyarray(train_set[["MaxTemp"]]).flatten()
print(train_x)
print(train_y)


# In[ ]:


# import the class LinearRegression from pachage sklearn.
from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression() # Creating an instance of the class LinearRegression.
model.fit(train_x,train_y) #Fitting the model to the data. 
r_sq = model.score(train_x,train_y) # Check the R2 score for model evaluation.
print(r_sq)


# In[ ]:


# Check the values of the parameters or coefficients.
intercept = model.intercept_
slope = model.coef_
print('Intercept : ', intercept)
print('Slope : ', slope)


# In[ ]:


# Plot a scatter chart with line of best fit.
train_set.plot(kind='scatter',x='MinTemp',y='MaxTemp',figsize=(10,6),color='blue')
plt.plot(train_x, slope*train_x + intercept, '-r')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.title('MinTemp Vs MaxTemp')

plt.show()


# In[ ]:


# Feeding the training data set value in the model to predict the output.
y_pred = model.predict(train_x)
print(y_pred)


# In[ ]:


# Creating arrays to test the model with the test data set.
test_x = np.asanyarray(test_set[["MinTemp"]])
test_y = np.asanyarray(test_set[["MaxTemp"]])


# In[ ]:


# Predict the values from the test data.
y_pred_test = model.predict(test_x)
print(y_pred_test)


# In[ ]:


mse = np.mean((test_y - y_pred_test)**2)          # Check the mean squared error.
mae = np.mean(np.absolute(test_y - y_pred_test))  # Check the mean absolute error.
print('Mean squared error : ', mse)
print('Mean absolute error : ', mae)


# In[ ]:


r_square = model.score(test_y, y_pred_test)
print(r_square)


# In[ ]:




