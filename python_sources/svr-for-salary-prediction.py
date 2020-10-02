#!/usr/bin/env python
# coding: utf-8

# In this kernel we will be using Support vector regression to predict the salary of employees.In the data set we have the details of the desigination and the salary if the employee.This kernal is a work in process.Please do vote if you like my work.

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


# **Importing Python Modules**

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/polynomial-position-salary-data/Position_Salaries.csv')
df.head(10)


# From the dataset we can see that there ten positions in the company and we can see the corresponding salary for each level.Now we will implement SVR to predict the salary based on the level.

# **Creating the Matrix of features**

# In[ ]:


X=df.iloc[:,1:2].values  # For the features we are selecting all the rows of column Level represented by column position 1 or -1 in the data set.
y=df.iloc[:,2].values    # for the target we are selecting only the salary column which can be selected using -1 or 2 as the column location in the dataset
y


# **Feature Scaling:**
# 
# Feature scaling is applied to the Salary.Feature scaling helps us to get better result from our SVR algorithm.y is a vector and we nned to convert it into 2d array before doing a feature scaling.

# In[ ]:


y = y.reshape(len(y),1)


# In[ ]:


print(y)


# Now we have conveted y to a 2d Array

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# In[ ]:


print(X)


# In[ ]:


print(y)


# **Training the SVR on the whole dataset**

# In[ ]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)


# **Predicting New Result**

# In[ ]:


sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))


# **Vizualizing the SVR Results **

# In[ ]:


plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)),color='blue')
plt.title('Truth or Bluff (Support Vector Regressor)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# **Vizualising the SVR results (for higher resolution and smoother curve)**
