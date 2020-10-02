#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/column_2C_weka.csv")


# In[ ]:


data


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.describe()


# In[ ]:


data.columns


# In[ ]:


data.corr()


# In[ ]:


f,axis = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.4, fmt= '.2f', ax = axis)
plt.show()


# In[ ]:


sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()


# In[ ]:


sns.pairplot(data,hue="class",palette="Set2")
plt.show()


# **As we can see easily; Abnormal datas are located on more separated than Normal datas. **

# In[ ]:


#Linear Regression
data1=data[data['class'] == "Abnormal"] #We create a shorter dataset from data
data1


# In[ ]:


#Linear Regression
#Firstly, we should import our library for Linear regression

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1) 
#define an variable use by pelvic_incidance from data1
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
#define an variable use by sacral_slope from data1
linear_reg.fit(x,y) #We should fit these.
y_head = linear_reg.predict(x)
plt.figure(figsize=(15,5))
plt.scatter(x,y,color="green") #Let's see our figure
plt.plot(x,y_head,color="black")
plt.xlabel("pelvic_incidence")
plt.ylabel("sacrel_scope")
plt.show()


# In[ ]:


#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(x,y)
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_ = tree.predict(x_)
y_val = tree.predict(x)
#Visualize
plt.figure(figsize=(15,5))
plt.scatter(x,y,color="green")
plt.plot(x_,y_,color="red")
plt.xlabel("pelvic_incidence")
plt.ylabel("sacrel_scope")
plt.show()


# In[ ]:


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
random_reg = RandomForestRegressor(n_estimators=100,random_state=42)
random_reg.fit(x,y)
x_1 = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_predicted = random_reg.predict(x_1)
y_val1 = random_reg.predict(x)
#visualize
plt.figure(figsize=(15,5))
plt.scatter(x,y,color="green")
plt.xlabel("pelvic_incidence")
plt.ylabel("sacrel_scope")
plt.plot(x_1,y_predicted,color="blue")
plt.show()


# **Let's see all of our visualizations!**

# In[ ]:


plt.figure(figsize=(15,5))
plt.scatter(x,y,color="green")
plt.plot(x,y_head,color="green",label="Linear Regression")
plt.plot(x_,y_,color="red",label="Decision Tree Regression")
plt.plot(x_1,y_predicted,color="pink",label="Random Forest Regression")
plt.legend()
plt.xlabel("pelvic_incidence")
plt.ylabel("sacrel_scope")
plt.show()


# In[ ]:


#Let's see our R^2(R-square) for all of regressions
from sklearn.metrics import r2_score #This is the library of R-square with linear regression!
print ("For Linear Regression: ",r2_score(y,y_head))
print ("For Decision Tree Regression: ",r2_score(y,y_val) )
print ("For Random Forest Regression: ",r2_score(y,y_val1) )


# **As we can see; Decision Tree Regression gave us to truest predictions.**

# **KNN ALGORITHM**

# In[ ]:


#Read data again!
data = pd.read_csv("../input/column_2C_weka.csv")


# In[ ]:


data.info()


# In[ ]:


y1 = data.loc[:,'class']
x1 = data.loc[:,data.columns!='class']
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
x_train, x_test,y_train, y_test = train_test_split(x1,y1,test_size=0.3,random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)


# In[ ]:


#See our predictions
prediction


# In[ ]:


print("K={} nn score is: {}".format(3,knn.score(x_test,y_test)))

