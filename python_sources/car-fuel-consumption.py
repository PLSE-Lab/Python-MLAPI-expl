#!/usr/bin/env python
# coding: utf-8

# This dataset is related to fuel consumption and Carbon dioxide emission of cars.
# Here I am creating a model using training set, evaluating  model using test set, and finally use model to predict unknown value.

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# In[ ]:


#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


#  The data contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale.

# In[ ]:


# Reading data
df = pd.read_csv("/kaggle/input/FuelConsumption.csv")

# take a look at the dataset
df.head()


# Lets first have a descriptive exploration on our data.

# In[ ]:


# summarize the data
df.describe()


# Lets select some features to explore more.

# In[ ]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# Lets plot each of these features:

# In[ ]:


viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


#  lets plot each of these features vs the Emission, to see how linear is their relation:

# In[ ]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


# In[ ]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[ ]:


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("cylinder")
plt.ylabel("co2emmision")
plt.show()


# Creating train and test dataset

# In[ ]:


msk = np.random.rand(len(df)) < 0.8
#print(msk)
train = cdf[msk]
test = cdf[~msk]


# Train data distribution

# In[ ]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# Modeling:
# Using sklearn package to model data.

# In[ ]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# Plot outputs

# In[ ]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# > **Evaluation**:
# I compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.
# 
# 

# In[ ]:


from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


# I use MSE(model evaluation metrics) here to calculate the accuracy of our model based on the test set.
