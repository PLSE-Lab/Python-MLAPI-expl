#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.


# In[ ]:


data2_weka = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
print(plt.style.available) # look at available plot styles
plt.style.use('ggplot')


# Exploratory Data Analysis
# 
# * In order to make something in data, as you know you need to explore data.
# * We always start with *head()* to see features that are *pelvic_incidence,	pelvic_tilt numeric,lumbar_lordosis_angle,	sacral_slope,	pelvic_radius* and 	*degree_spondylolisthesis* and target variable that is *class*
# * head(): default value of it shows first 5 rows(samples). If you want to see for example 20 rows just write head(20)
# 

# In[ ]:


data2_weka.head()


# In[ ]:


data2_weka.info()


# In[ ]:


data2_weka.describe()


# In[ ]:


color_list = ['red' if i=='Abnormal' else 'green' for i in data2_weka.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data2_weka.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()


# Create the New Data 
# 
# * We create the new data model that only include Abnormal in class column.

# In[ ]:


data1 = data2_weka[data2_weka['class'] =='Abnormal']
data1.head()


# We state the feature and target variable in new data model.
# 
# x : feature 
# y : target variable
# 
# > feature -- pelvic_incidence
# > target  -- lumbar_lordosis_angle

# In[ ]:


x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:,'lumbar_lordosis_angle']).reshape(-1,1)
#Scatter
plt.figure(figsize=[10,10])
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('lumbar_lordosis')
plt.show()


# Linear Regression Method
# 
# * **
# 
# y = ax + b
# fit() : fits the data1, train the data1
# predict() : predict the data1

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

y_head = lr.predict(x)
plt.plot(x,y_head, color="black")
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('lumbar_lordosis')
plt.show()


# Linear Regression Score
# 
# Score: Score uses R^2 method that is ((y_pred - y_mean)^2 )/(y_actual - y_mean)^2
# Score : predict and give accuracy. 
# 
# It should be very close to 1. 

# In[ ]:


from sklearn.metrics import r2_score
print("r_square_score", r2_score(y,y_head))


# Polynomial Regression
# 
# y = a*x + b* x^2 + c*x^3 + d* x^4+ e*x^5
# 
# coefficent = 5
# pr : polynomialfeatures

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 5)

x_polynomial = pr.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_polynomial,y)

y_head2 = lr2.predict(x_polynomial).reshape(-1,1)
plt.plot(x,y_head2,color = "yellow",linewidth =3,label = "poly_reg")
plt.legend()
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('lumbar_lordosis')
plt.show()


# In[ ]:


from sklearn.metrics import r2_score
print("r_square_score", r2_score(y,y_head2))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=50, random_state=42 )
rf.fit(x,y)

x_ = np.arange(min(x),max(x),0.05).reshape(-1,1)
y_head3 = rf.predict(x_)
plt.figure(figsize=[10,10])
plt.plot(x_,y_head3,color="green",label="randomforest")
plt.scatter(x,y,color="red")
plt.xlabel('pelvic_incidence')
plt.ylabel('lumbar_lordosis')
plt.show()


# Decision Tree Regressor
# 
# > * Decision Tree is a decision-making tool that uses a flowchart-like tree structure or is a model of decisions and all of their possible results, including outcomes, input costs and utility.
# > * Decision tree regression observes features of an object and trains a model in the structure of a tree to predict data in the future to produce meaningful continuous output.
# 
# https://media.geeksforgeeks.org/wp-content/uploads/decision-tree.jpg
# 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

y_head4 = tree_reg.predict(x)
plt.figure(figsize=[10,10])
plt.scatter(x,y,color="red")
plt.plot(x,y_head4,color="green",label="decisiontree")
plt.xlabel('pelvic_incidence')
plt.ylabel('lumbar_lordosis')
plt.show()


# In[ ]:


from sklearn.metrics import r2_score
print("r_square_score", r2_score(y,y_head4))

