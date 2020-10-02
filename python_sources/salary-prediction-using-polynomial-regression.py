#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plots

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Getting the data**

# In[ ]:


data = pd.read_csv('/kaggle/input/position-salaries/Position_Salaries.csv')
data.head()


# **Getting to know my data**

# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(data.Level,data.Salary)
plt.title("Level Vs Salary plot")


# # The graph is non-linear in nature. Thus instead of using simple linear regression algorithm, we need to use polynomial features.

# In[ ]:


X=data.iloc[:,1:2].values
y=data.iloc[:,-1].values


# **Importing necessary packages**

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# **Function to get the R2 score for my testing data **

# In[ ]:


def polynomialRegressionScore(X,y,k):

    poly = PolynomialFeatures(degree=k)
    X_poly = poly.fit_transform(X)
    lr = LinearRegression()
    lr.fit(X_poly,y)
  
    X_test_poly =poly.fit_transform(X)
    y_pred=lr.predict(X_test_poly)

    test_score = r2_score(y,y_pred)
    
  
    return test_score


# In[ ]:


#process to get the appropriate degree to get a good accuracy for the model
train=[]
test=[]
for i in range(1,10):
    r2test=polynomialRegressionScore(X,y,k=i)
    test.append(r2test)
x=np.arange(9)+1
plt.plot(x,test,label="Test")
plt.legend()
plt.xlabel("Degree for my model")
plt.ylabel("r2-Score")
plt.title("R2-Score");
plt.show()


# **Here, we see levels 2 and 3 will give a good accuracy to my model.Levels below 2 will make my model under-fitted and levels above 3 will make my model over-fitted.**

# In[ ]:


def polynomialRegression(X,y,k):

    poly = PolynomialFeatures(degree=k)
    X_poly = poly.fit_transform(X)
    lr = LinearRegression()
    lr.fit(X_poly,y)
  
    X_test_poly =poly.fit_transform(X)
    y_pred=lr.predict(X_test_poly)
    
    plt.plot(X,y_pred, label="Model",color='red')
    plt.scatter(X, y, label="data",color='green')
    plt.legend()
    plt.show()
    
    print("The accuracy of the model is " ,r2_score(y,y_pred)*100,"%")   


# # Over-fitted graph

# In[ ]:


plt.title("Curve when Level = 5. Over-fitted graph")
polynomialRegression(X,y,5)

plt.title("Curve when Level = 4.Over-fitted graph")
polynomialRegression(X,y,4)


# # Under-fitted graph

# In[ ]:


plt.title("Curve when Level = 1. Under-fitted graph")
polynomialRegression(X,y,1)


# # Appropriate curve

# In[ ]:


plt.title("Curve when Level = 3")
polynomialRegression(X,y,3)


# In[ ]:


plt.title("Curve when Level = 2")
polynomialRegression(X,y,2)


# **I personally prefer degree 3 for my model.**
