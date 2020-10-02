#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

anormality = data[data["class"] == "Abnormal"]

x = np.array(anormality.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(anormality.loc[:,'sacral_slope']).reshape(-1,1)

data.head()


# Reading data from target file and naming as data.
# Naming the data which are called "Abnormal" in anormality and, x and y stand for our features. 

# In[ ]:


data.info()


# If we take a look to our output, we can observe that we have 210 range and float64 datatyped data.

# In[ ]:


data.describe()

Describe function helps us to make our data meaningful. Without plotting, we can understand the data thanks to those significant numbers.
# In[ ]:


color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()


# red stars show "Abnormal" patients and greens show "Normal" ones.
# pd.plotting.scatter_matrix:
# 
# * green: normal and red: abnormal
# * c: color
# * figsize: figure size
# * diagonal: histohram of each features
# * alpha: opacity
# * s: size of marker
# * marker: marker type

# In[ ]:


sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()


# Countplot function design a bar chart for illustrating "class" which is included in seaborn library.
# value_counts, show the numeric value of "class" data.

# **REGRESSION**
# * Supervised learning
# * We will use linear regression and polynomial regression.
# 

# In[ ]:


plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()


# **Linear Regression**
# After visualizing our data in a plot, we can move on to make linear regression.

# Score: Score uses R^2 method that is ((y_pred - y_mean)^2 )/(y_actual - y_mean)^2

# In[ ]:


from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

linear_reg.fit(x,y)

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)

prediction = linear_reg.predict(x)

print('R^2 score: ',linear_reg.score(x, y))
# Plot regression line and scatter
plt.plot(x, prediction, color='black', linewidth=3)
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()


# **Polynomial Regression**
# * To make our regression polynomial we demand help from PolynomialFeatures in sklearn.preprocessing.
# * In this method, our most crucial parameter is degree (n). Polynomial functions are formulized as y =b0 + b1*x + b2*x^2 + .. +bn*x^n . When n increases, we will get more complicated function. To be more obvious, it has both advantages and disadvantages. It does not mean that more degree, more reliable.

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 3) #degree increases reliability

x_poly = polynomial_regression.fit_transform(x) 

linear_regression2 = LinearRegression()
linear_regression2.fit(x_poly,y)


prediction_poly = linear_regression2.predict(x_poly)
plt.scatter(x,y)
plt.plot(x,prediction_poly,color="green",label = "poly")
plt.legend()
plt.show()


# As you can see, our 3rd degree polynom is not appropriately fitted. That means we cannot find a correlation between x_poly and prediction_poly in polynomial regression.
