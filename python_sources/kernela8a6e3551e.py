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
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Reading the data in

# In[ ]:


df=pd.read_csv('../input/FuelConsumption.csv')
df.head()


# <h2 id="data_exploration">Data Exploration</h2>
# Lets first have a descriptive exploration on our data.

# In[ ]:


# summarize the data
df.describe()


# Lets select some features to explore more.

# In[ ]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# In[ ]:


cdf.hist()


# Now, lets plot each of these features vs the Emission, to see how linear is their relation:

# In[ ]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS)
plt.xlabel('Engine Size')
plt.ylabel('Co2 Emissions')


# In[ ]:


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS)
plt.xlabel('Cylinders')
plt.ylabel('Co2 Emissions')


# In[ ]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS)
plt.xlabel('Fuel Consumption')
plt.ylabel('Co2 Emissions')


# ## Creating train and test dataset
# Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set. This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. It is more realistic for real world problems.
# 
# This means that we know the outcome of each data point in this dataset, making it great to test with! And since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an out-of-sample testing.
# 
# Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing. We create a mask to select random rows using np.random.rand() function:

# In[ ]:


dataset=np.random.rand (len(df))<0.8
train=cdf[dataset]
test=cdf[~dataset]


# ### Simple Regression Model

# ### Train data distribution

# In[ ]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS)
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')


# ## Modeling

# Using sklearn package to model data.

# In[ ]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# ## Plot Outputs

# In[ ]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[ ]:





# In[ ]:




