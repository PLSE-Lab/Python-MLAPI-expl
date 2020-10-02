#!/usr/bin/env python
# coding: utf-8

# ### Problem : Analyze diamonds by their cut, color, clarity, price, and other attributes

# ### Content
# ### price price in US dollars (\$326--\$18,823)
#  
# ### carat weight of the diamond (0.2--5.01)
#  
# ### cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
#  
# ### color diamond colour, from J (worst) to D (best)
#  
# ### clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
#  
# ### x length in mm (0--10.74)
#  
# ### y width in mm (0--58.9)
#  
# ### z depth in mm (0--31.8)
#  
# ### depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
#  
# ### table width of top of diamond relative to widest point (43--95)

# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score  #Classification metrics
from sklearn.metrics import mean_squared_error, r2_score #Regression metrics

from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os


# ###  DATA HANDLING

# #### Getting sense of the data

# In[ ]:


diamond = pd.read_csv("../input/diamonds.csv")


# In[ ]:


diamond.info()


# In[ ]:


diamond.head()


# ## Dropping unneccesary column

# In[ ]:


diamond=diamond.drop('Unnamed: 0',axis=1)


# In[ ]:


diamond.head()


# In[ ]:





# ### Checking for null values

# In[ ]:


diamond.isnull().sum()


# ### Univariate Analysis

# In[ ]:


sns.distplot(diamond.carat , color = 'red')


# Carat is right skewed.
# # And we can see multiple peaks in the data. Which suggests there are different types of diamonds.

# In[ ]:





# In[ ]:


sns.distplot(diamond.depth , color = 'green')


# depth seems to follow normal distribution.

# In[ ]:





# In[ ]:


sns.distplot(diamond.table , color = 'yellow')


# Table seems to follow normal distribution.

# In[ ]:





# In[ ]:


sns.distplot(diamond.price , color = 'blue')


# Price is right skewed.

# In[ ]:





# In[ ]:


plt.figure(1,figsize=[10,7])
plt.subplot(221)
sns.distplot(diamond.x)

plt.subplot(222)
sns.distplot(diamond.y)

plt.subplot(223)
sns.distplot(diamond.z)

plt.show()


# In[ ]:





# In[ ]:


diamond.cut.unique()  # count 5


# In[ ]:


diamond.color.unique()   # count 7


# In[ ]:


diamond.clarity.unique()   # count 8


# In[ ]:





# ### Bivariate analysis

# In[ ]:


diamond.head()


# In[ ]:





# Carat vs Price

# In[ ]:


plt.scatter(diamond.carat , diamond.price)
plt.xlabel('Carat')
plt.ylabel('Price')
plt.show()


# With small increase in carat value the price increases by large value.

# In[ ]:





# In[ ]:


plt.scatter(diamond.depth , diamond.price)
plt.xlabel('Depth')
plt.ylabel('Price')
plt.show()


# It seems that depth and price are not correlated.

# In[ ]:





# In[ ]:


plt.scatter(diamond.table , diamond.price)
plt.xlabel('Table')
plt.ylabel('Price')
plt.show()


# It seems that table and price are not correlated.

# In[ ]:





# In[ ]:


plt.figure(1,figsize=[10,7])
plt.subplot(221)
plt.scatter(diamond.x , diamond.price)
plt.xlabel('x')
plt.ylabel('Price')

plt.subplot(222)
plt.scatter(diamond.y , diamond.price)
plt.xlabel('y')
plt.ylabel('Price')

plt.subplot(223)
plt.scatter(diamond.z , diamond.price)
plt.xlabel('z')
plt.ylabel('Price')
plt.show()


# Upward trend can be seen between x and price. For y and z no relation is visible.

# In[ ]:





# In[ ]:


sns.boxplot(diamond.cut , diamond.price)


# In[ ]:


diamond.groupby('cut').price.median()


# In[ ]:


diamond.groupby('cut').price.mean()


# Mean price for different types of cuts is in following order: 
# # 
# #     Premium > Fair > Very Good > Good > Ideal

# In[ ]:





# In[ ]:


sns.boxplot(diamond.color , diamond.price)
plt.show()
# color vs price


# In[ ]:





# In[ ]:


sns.boxplot(diamond.clarity , diamond.price)
plt.show()


# In[ ]:





# In[ ]:


plt.figure(figsize=[7,6])
sns.heatmap(diamond.corr() , annot = True)
plt.show()


# price is highly correlated with carat, x, y and z.
# # 
# # But there is also high correlation among carat, x, y and z.

# In[ ]:





# ### NULL value treatment

# In[ ]:


diamond.isnull().sum()


# No null values found.

# In[ ]:





# ### Linear Regression

# In[ ]:


X=diamond.drop('price',axis = 1)
y=diamond.price


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# ### One hot encoding

# In[ ]:


dummyXtrain = pd.get_dummies(X_train)
dummyXtest = pd.get_dummies(X_test)


# In[ ]:


dummyXtrain.head()


# In[ ]:





# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(dummyXtrain,y_train)


# In[ ]:


y_pred = lr.predict(dummyXtest)


# In[ ]:


mean_squared_error(y_test,y_pred)


# In[ ]:


lr.score(dummyXtest,y_test)


# ### Random Forest regressor

# In[ ]:


rfr = RandomForestRegressor()


# In[ ]:


rfr.fit(dummyXtrain,y_train)


# In[ ]:


y_pred = rfr.predict(dummyXtest)


# In[ ]:


rfr.score(dummyXtest,y_test)   #r2_score


# In[ ]:


mean_squared_error(y_test,y_pred)


# In[ ]:





# ### Classification

# In[ ]:


diamond.head()


# In[ ]:


X=diamond.drop('cut',axis=1)


# In[ ]:


y=diamond.cut


# In[ ]:





# In[ ]:


diamond.cut.unique()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=100)


# In[ ]:


dummyXtrain = pd.get_dummies(X_train)
dummyXtest = pd.get_dummies(X_test)


# In[ ]:





# ### KNN

# In[ ]:


knn = KNeighborsClassifier()


# In[ ]:


knn.fit(dummyXtrain,y_train)


# In[ ]:


y_pred = knn.predict(dummyXtest)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:





# ### Decision Tree Classifier

# In[ ]:


dtree = DecisionTreeClassifier()


# In[ ]:


dtree.fit(dummyXtrain,y_train)


# In[ ]:


y_pred = dtree.predict(dummyXtest)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:





# ### Random Forest Classifier

# In[ ]:


rfr = RandomForestClassifier()


# In[ ]:


rfr.fit(dummyXtrain,y_train)


# In[ ]:


y_pred = rfr.predict(dummyXtest)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:




