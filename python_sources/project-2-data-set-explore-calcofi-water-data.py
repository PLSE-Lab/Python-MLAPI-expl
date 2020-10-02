#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import linear_model as lm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Read the data**

# In[ ]:


data = pd.read_csv('../input/bottle.csv')


# **Look at the head of the data**

# In[ ]:


data.head()


# In[ ]:


dataOfProject = data[["Depthm", "T_degC", "Salnty", "O2ml_L", "PO4q", "SiO3qu","NO3q","NH3q","NO2q"]]
dataOfProject.corr()


# **Create the data frame**

# In[ ]:


dataSet = pd.DataFrame()
dataSet["T_degC"] = data["T_degC"]
dataSet["Salnty"] = data["Salnty"]
print(dataSet)


# In[ ]:


dataSet.describe()


# **Eliminate N/A value**

# In[ ]:


dataSet.isnull().sum()


# In[ ]:


dataSet = dataSet.dropna()


# In[ ]:


dataSet.isnull().sum()


# **Find correlation**

# In[ ]:


dataSet.corr()


# **Draw a scatterplot with the linear model as a line**

# In[ ]:


plt.scatter(dataSet["Salnty"], dataSet["T_degC"], alpha=0.1)


# **Create data set for generating the model**

# In[ ]:


salnty = dataSet["Salnty"]
salnty = salnty.values.reshape(-1, 1)
print(salnty)


# In[ ]:


t_degC = dataSet["T_degC"]
t_degC = t_degC.values.reshape(-1, 1)
print(t_degC)


# **Create the model**

# In[ ]:


model = lm.LinearRegression()
model.fit(salnty, t_degC)
print(model.intercept_, model.coef_)


# **Predict the value with the model**

# In[ ]:


predictedValue = model.predict(salnty)
print(predictedValue)


# **Compare the predicted value by ploting the graph**

# In[ ]:


plt.scatter(salnty,t_degC)
plt.plot(salnty,predictedValue, color="blue")
plt.show()


# **Find sum of square error**

# In[ ]:


square_error = ((predictedValue - t_degC) ** 2)
print(square_error.sum())


# ** Part B**

# **Select 5 variables from your dataset. For each, draw a boxplot and analyze your observations.**

# In[ ]:


data.boxplot(column=["T_degC","Salnty","O2ml_L","STheta", "T_prec"])


# In[ ]:


data.boxplot(column=["O2Sat"])


# In[ ]:


data.corr()


# In[ ]:


plt.scatter(data["Depthm"], data["T_degC"], alpha=0.1)


# In[ ]:


plt.scatter(data["Depthm"], data["Salnty"], alpha=0.1)


# In[ ]:


plt.scatter(data["T_degC"], data["O2ml_L"], alpha=0.1)


# In[ ]:


plt.scatter(data["Depthm"], data["O2Sat"], alpha=0.1)

