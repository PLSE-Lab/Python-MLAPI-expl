#!/usr/bin/env python
# coding: utf-8

# **Content of this Kernel:**
# 1. [Import and Data Manipulation:](#1)
#     1. [Scaling for SVR](#5)
# 1. [Support Vector Regression:](#2)
# 1. [Decision Tree Regression](#3)
# 1. [Random Forest Regression:](#4)
# 1. [R Square Evaulation:](#6)
# 1. [Adjusted R Square Evaulation:](#7)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="1"></a> <br>
# # 1. Import and Data Manipulation

# In[ ]:


df = pd.read_csv("../input/maaslar.csv", sep=",")
df.columns = ['Title', 'Education Level', 'Salary']


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


educationLevel = df.iloc[:,1:2]
salary = df.iloc[:,2:3]


# <a id="5"></a> <br>
# ## a. Scaling for Support Vector Regression

# In[ ]:


sc1 = StandardScaler()
scaledEducationLevel = sc1.fit_transform(educationLevel)
sc2 = StandardScaler()
scaledSalary = sc2.fit_transform(salary)


# <a id="2"></a> <br>
# # 2. Support Vector Regression

# In[ ]:


modelSVR = SVR(kernel='rbf')
modelSVR.fit(scaledEducationLevel, scaledSalary)
y_pred = modelSVR.predict(scaledEducationLevel)
y_pred


# In[ ]:


plt.scatter(scaledEducationLevel, scaledSalary, color = "red")
plt.plot(scaledEducationLevel, y_pred, color = "blue")      


# <a id="3"></a> <br>
# # 3. Decision Tree Regression

# In[ ]:


modelDT = DecisionTreeRegressor(random_state = 0)
modelDT.fit(educationLevel, salary)


# In[ ]:


plt.scatter(educationLevel, salary, color = "red")
plt.plot(educationLevel, modelDT.predict(educationLevel), color="blue")
plt.show()


# In[ ]:


A = educationLevel + .5
B = educationLevel - .4
plt.scatter(educationLevel, salary, color = "red")
plt.plot(educationLevel, modelDT.predict(educationLevel), color="blue")
plt.plot(educationLevel, modelDT.predict(A), color ="green") 
plt.plot(educationLevel, modelDT.predict(B), color = "orange") 
plt.show()


# In[ ]:


print(modelDT.predict(educationLevel))


# In[ ]:


print(modelDT.predict(np.array([[10],[11],[20],[50]])))


# <a id="4"></a> <br>
# # 4. Random Forest Regression

# In[ ]:


modelRF = RandomForestRegressor(n_estimators = 10, random_state = 0)
modelRF.fit(educationLevel, salary)


# In[ ]:


y_pred = model.predict(educationLevel)
plt.scatter(educationLevel, salary, color = "red")
plt.plot(educationLevel, y_pred, color ="blue")
plt.show()


# In[ ]:


plt.scatter(educationLevel, salary, color = "red")
plt.plot(educationLevel, y_pred, color ="blue")
plt.plot(educationLevel, modelRF.predict(A), color ="green") 
plt.plot(educationLevel, modelRF.predict(B), color = "orange")
plt.show()


# In[ ]:


print(modelRF.predict(np.array([[1],[5],[20],[50]])))


# <a id="6"></a> <br>
# # 5. R Squared Evaluation

# In[ ]:


print(r2_score(salary, modelRF.predict(educationLevel)))


# In[ ]:


print(r2_score(salary,modelDT.predict(educationLevel)))


# In[ ]:


print(r2_score(scaledSalary,modelSVR.predict(scaledEducationLevel)))

