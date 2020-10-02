#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# importing python libraries

# In[ ]:


from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt


# reading and storing data csv data in a variable
# The data has been uploaded from my PC.

# In[ ]:


df = pd.read_csv("../input/insurance/insurance.csv")


# In[ ]:


df


# In[ ]:


df.head()


# deciding upon dependent and independent variables .Dependent variable is one whose value is to be predicted based on independent data 

# In[ ]:


y= df.charges


# In[ ]:


x = df.iloc[:,0:1].values


# In[ ]:


y_predict = model.predict(x)


# Training the machine with the dependent and independent variable so that it can predict value for an unknown independent variable  

# In[ ]:


model = LinearRegression()
model.fit(x,y)


# predicting dependent variable or creating a best fit line

# In[ ]:


y_predict = model.predict(x)
y_predict


# y = mx + c
# the linear relation of dependent variable with independent variable
# here m is the slope or rate of change of y with respect to x
# c is the intercept

# In[ ]:


m = model.coef_
m


# In[ ]:


c = model.intercept_
c


# verifying that  predicted value of y works the same way as decided by mathematical relation

# In[ ]:


ypredict = m*x + c
ypredict


# taking independent variable out of the training set

# In[ ]:


x1 = 20
x2 = 30
w = model.predict([[x1],[x2]])
w


# plotting the dependent variable against independent one as scatter plot
# and prected one in line form which gives the best fit line through the scatter plot

# In[ ]:


plt.xlabel("Age")
plt.ylabel("Charges")
plt.scatter(x,y, color = "green")
plt.plot(x,y_predict, c= "red")
plt.scatter([x1,x2], w , color= "blue")


# In[ ]:




