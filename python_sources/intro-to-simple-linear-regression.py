#!/usr/bin/env python
# coding: utf-8

# # <center>Gentle Introduction to Simple Linear Regression</center>
# 
# 
# ### About this Notebook
# This notebook presents the use scikit-learn to implement a simple linear regression. The dataset used is related to fuel consumption and Carbon dioxide emission of cars. which is split our data into training and test sets, create a model using training set, Evaluate your model using test set, and finally use model to predict unknown values

# Importing Needed packages

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Downloading Data
# 
# To download the data, we will use !wget to download it from IBM Object Storage

# In[ ]:


get_ipython().system('wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')


# ## Reading the Data

# In[ ]:


df = pd.read_csv("FuelConsumption.csv")
df.head()


# ## Data Exploration

# In[ ]:


df.describe()


# In[ ]:


vi = df
vi.hist()
plt.show()


# ### Selecting Specific Features

# In[ ]:


cdf = df[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(10)


# ### Some Visualizations

# In[ ]:


viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# ## Some Scatter Plots

# ## Fuel Consumption Versus Emmision

# In[ ]:


plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS, color ='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()


# ## Engine Size Versus Emmision

# In[ ]:


plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS, color ='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# ## Cylinders Versus Emmision

# In[ ]:


plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS, color ='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("CO2EMISSIONS")
plt.show()


# # The Regression Model
# ## Train data distribution

# In[ ]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
# Plotting the distribution of train and test data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# # Building the Simple Linear Regression Model

# In[ ]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# Printing the Parameters
print("Coefficient: ", regr.coef_)
print("Intercept: ", regr.intercept_)


# ## Plotting the Output

# In[ ]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# # Evaluation

# In[ ]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


# ## Predicting CO2 Emmision of Random Engine Sizes

# In[ ]:


print(regr.predict([[6.5]]))


# In[ ]:


print(regr.predict(np.array([4.0]).reshape(1, 1)))


# In[ ]:




