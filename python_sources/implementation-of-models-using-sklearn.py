#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Reading data

# In[ ]:


def read_data(path,label_name):
    data=pd.read_csv(path)
    label=data[label_name]
    data=data.drop(label_name,axis=1)
    test_data=data[:100]
    test_label=label[:100]
    data=data[100:]
    label=label[100:]
    return data,label,test_data,test_label


# # LOGISTIC REGRESSION
# the Logistic model is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick.  Logistic Regression is best suited for binary classification problem.  
# ![Logistic Regression](https://miro.medium.com/max/2400/1*RqXFpiNGwdiKBWyLJc_E7g.png)

# In[ ]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
data,label,test_data,test_label=read_data('../input/oranges-vs-grapefruit/citrus.csv','name')
model.fit(data,label)
pred=model.predict(data)
print("Accuracy : "+str(accuracy_score(label,pred)))
plt.plot(data[['red','green','blue']],pred)
plt.scatter(data['red'],pred)
plt.scatter(data['green'],pred)
plt.scatter(data['blue'],pred)
plt.xlabel('Color Codes')
plt.ylabel('Fruit')
plt.legend(['Red','Green','Blue'])


# # LINEAR REGRESSION
# Regression analysis is a set of statistical processes for estimating the relationships between a dependent variable and one or more independent variables.
# ![Linear Regression](https://miro.medium.com/max/640/1*LEmBCYAttxS6uI6rEyPLMQ.png)

# In[ ]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
data,label,test_data,test_label=read_data('../input/weight-and-heightcsv/weight-height.csv','Weight')
data['Gender']=data['Gender'].replace({'Male':0,'Female':1})
test_data['Gender']=test_data['Gender'].replace({'Male':0,'Female':1})   #Changing string data to int type
model.fit(data,label)
pred=model.predict(test_data)
plt.plot(test_data['Height'],pred)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.scatter(test_data['Height'],pred)


# # DECISION TREE
#  decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label
# ![Decision Tree](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSf6ZCldSJYqWkCm49qEKvSj1a90Pj8mFCG0NSI_lmIZU_GY7eD&usqp=CAU) 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
data,label,test_data,test_label=read_data('../input/oranges-vs-grapefruit/citrus.csv','name')
model.fit(data,label)
pred=model.predict(data)
print("Accuracy : "+str(accuracy_score(label,pred)))
plt.plot(data[['red','green','blue']],pred)
plt.scatter(data['red'],pred)
plt.scatter(data['green'],pred)
plt.scatter(data['blue'],pred)
plt.xlabel('Color Codes')
plt.ylabel('Fruit')
plt.legend(['Red','Green','Blue'])


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
data,label,test_data,test_label=read_data('../input/weight-and-heightcsv/weight-height.csv','Weight')
data['Gender']=data['Gender'].replace({'Male':0,'Female':1})
test_data['Gender']=test_data['Gender'].replace({'Male':0,'Female':1})   #Changing string data to int type
model.fit(data,label)
pred=model.predict(test_data)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.plot(test_data['Height'][:10],pred[:10])
plt.scatter(test_data['Height'][:10],pred[:10])


# # RANDOM FOREST
# A random forest is a meta estimator that contains a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
# ![Random Forest](https://miro.medium.com/max/1170/1*58f1CZ8M4il0OZYg2oRN4w.png)

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(max_depth=2)
data,label,test_data,test_label=read_data('../input/oranges-vs-grapefruit/citrus.csv','name')
model.fit(data,label)
pred=model.predict(data)
print("Accuracy : "+str(accuracy_score(label,pred)))
plt.plot(data[['red','green','blue']],pred)
plt.scatter(data['red'],pred)
plt.scatter(data['green'],pred)
plt.scatter(data['blue'],pred)
plt.xlabel('Color Codes')
plt.ylabel('Fruit')
plt.legend(['Red','Green','Blue'])


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(max_depth=4,random_state=2)
data,label,test_data,test_label=read_data('../input/weight-and-heightcsv/weight-height.csv','Weight')
data['Gender']=data['Gender'].replace({'Male':0,'Female':1})
test_data['Gender']=test_data['Gender'].replace({'Male':0,'Female':1})   #Changing string data to int type
model.fit(data,label)
pred=model.predict(test_data)
plt.plot(test_data['Height'],pred)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.scatter(test_data['Height'],pred)


# In[ ]:




