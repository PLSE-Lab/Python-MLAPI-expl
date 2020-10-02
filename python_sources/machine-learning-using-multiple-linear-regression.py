#!/usr/bin/env python
# coding: utf-8

# # Machine Learning: Using Multiple Linear Regression to predict 2014 cars CO2 Emission

# # Introduction
# 
# In this Notebook I'll show you how to implement Multiple Linear Regression to create accurate predictions. 
# 
# The dataset presented on this notebooks brings data of 2014 cars such as engine size, fuel consumption and CO2 emission. 
# 
# At the end of this analysis we'll be able to create a predictive model to predict the CO2 emission of 2014 cars based on the data of the car models and compare it with the actual CO2 emission values.
# 
# To measure and understand the CO2 emission is extremely important to maintain a suitable environment for all living beings on earth.

# ### Importing necessary libraries
# 
# _This libraries are essential for working, analyzing and visualizing data as such as creating machine learning models._

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
print('Libraries imported successfully')


# ### Importing the dataset

# In[ ]:


df = pd.read_csv('../input/FuelConsumptionCo2.csv')
print('Dataset imported successfully')


# # Exploratory Data Analysis

# ### Familiarizing with data
# 
# Here we'll check the basics to understand the data we're dealing with.
# We'll take a look on descriptive statistics, data types and search for missing data.

# In[ ]:


df.shape


# _So we have a 1067 rows and 13 columns dataset._

# In[ ]:


df.head()


# _We can see that some informations are not really necessary to create a predictive model, but we will analyze that further on the notebook._

# In[ ]:


df.info()


# _We can check that there's no missing data and all the attributes have the correct data type, so we can continue on creating the predictive model._

# # Choosing the attributes for the model
# ## Checking the correlation between attributes
# This step is necessary to pick the best attributes for creating the predictive model.
# 
# In here, we'll analyze the correlation between the attributes and visualize them after choosing.
# 
# **Observation:** The correlation we're trying to analyze is between attributes x CO2EMISSIONS

# In[ ]:


df.corr()


# _We can see that most of them have a good correlation with CO2EMISSIONS, so to avoid overfitting we'll choose three to create the model. After evaluation, if necessary, we'll add more attributes to the model._
# 
# _The attributes I chose are: ENGINESIZE, CYLINDERS and FUELCONSUMPTIONCOMB._

# ## Creating and analyzing the dataframe with the chosen attributes

# In[ ]:


cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print('Dataframe CDF created successfully')


# ### Let's take a look at the df created:

# In[ ]:


cdf.head()


# In[ ]:


cdf.corr()


# In[ ]:


sns.regplot(cdf.ENGINESIZE, cdf.CO2EMISSIONS)
sns.regplot(cdf.CYLINDERS, cdf.CO2EMISSIONS)
sns.regplot(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS)
plt.ylabel('CO2 Emission')
plt.legend(('Engine Size', 'Cylinders', 'Fuel Consupmtion'))
plt.show()


# **We can see at the scatter plot that is possible to create a linear model to fit the actual values in the chose dataset attributes.**

# # Creating the dataframes to train and test the model

# In[ ]:


# creating x,y
x = cdf.iloc[:,0:3]
y = cdf.CO2EMISSIONS
y = y.to_frame()


# In[ ]:


# taking a look at x,y


# In[ ]:


x.head()


# In[ ]:


y.head()


# ## Creating and checking Train/Test Dataframes

# ### Creating the dataframes

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)


# ### Checking the dataframes

# In[ ]:


print ('x_train quantity: ', len(x_train))
print ('y_train quantity: ', len(y_train))
print ('x_test quantity: ', len(x_test))
print ('y_test quantity: ', len(y_test))


# # Creating the predictive model

# ### Creating Linear Regression Object

# In[ ]:


lm = LinearRegression()


# ### Creating the model

# In[ ]:


lm.fit(x_train, y_train)


# **Considering yhat = b0 + b1*x1 + b2*x2 + b3*x3,**

# In[ ]:


# b0 = lm.intercept_ 
# b1 = lm.coef_


# In[ ]:


lm.intercept_


# In[ ]:


lm.coef_


# **We can now create the model:**
# 
# **yhat = 65.57 + 11.165*ENGINESIZE + 7.31*CYLINDERS + 9.58*FUELCONSUMPTION_COMB**

# ### Predicting CO2 Emission (y_test) using the data from x_test:

# In[ ]:


yhat = lm.predict(x_test)


# # Evaluating the model

# ## Checking the R-square

# In[ ]:


lm.score(x_test, y_test)


# **The closer this value is to 1, the better becomes the model. This score looks like a good value for our model so there's no need to create a new one.**

# ## Comparing actual values with predicted values

# ### One-to-one comparison

# In[ ]:


yhatdf = pd.DataFrame(yhat)
yhatdf.columns = ['PredictedValues']
yhatdf = yhatdf.PredictedValues.astype(int).to_frame()


# In[ ]:


actualval = y_test
actualval.columns = ['ActualValues']


# In[ ]:


lastdf = pd.concat([yhatdf.reset_index(drop=True), actualval.reset_index(drop=True)], axis = 1, sort = False)


# In[ ]:


lastdf


# 

# ### Creating a residual plot to check if it's well distributed

# In[ ]:


sns.residplot(y_test, yhat)
plt.title('Residual plot of YHAT x Y_TEST')
plt.show()


# **We can see that the residual plot shows us a well-distributed data, confirming the quality of the model.**

# **Finally, let's compare graphically yhat and y_test:**

# In[ ]:


sns.distplot(y_test, hist = False, label = 'Actual values')
sns.distplot(yhat, hist = False, label = 'Predicted values')
plt.title('Comparison of predicted values with actual values')
plt.show()


# **We can see throughout this graph that the model is accurate and can successfully predict the CO2 Emission with a low rate of error.**

# # Authors
# **Ruben Acevedo**,
# _Bachelor in Computer Engineering._
# 
# 
# 
# 
# 
# If you need to contact me, please send me an e-mail: rubenfsolorzano@hotmail.com.
# 
# ### Thanks!

# In[ ]:





# In[ ]:




