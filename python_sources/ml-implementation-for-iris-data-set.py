#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt


# In[ ]:


# Loading data set 
iris = pd.read_csv("../input/Iris.csv")
iris.head()


# In[ ]:


iris.info() # To check if there is any inconsistency in the data


# In[ ]:


# Drop Id column because it is useless for us. 

iris.drop("Id", axis = 1, inplace = True)


# In[ ]:


iris.head()


# Exploratory Data Analysis (EDA)

# In[ ]:


sns.scatterplot(x="SepalLengthCm", y='SepalWidthCm', hue='Species', data = iris)


# In[ ]:


sns.scatterplot(x="PetalLengthCm", y='PetalWidthCm', hue='Species', data = iris)


# Here we can see than petal features are giving better grouping compared to Sepal features.

# In[ ]:


iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# In[ ]:


# Now let us analyze about how Length and Width Vary for different Species. Idea here is to use violin plot. 

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)


# In[ ]:


import sklearn
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm


# In[ ]:


iris.shape


# In[ ]:


plt.figure(figsize=(7,4)) 
sns.heatmap(iris.corr(),annot=True) 
plt.show()


# >  Petal length and Petal Width are highly correlated. 
# >  But sepal Length and Sepal Width are Not Correlated at all. 

# In[ ]:


train, test = train_test_split(iris, test_size = 0.3)
print(train.shape)
print(test.shape)


# In[ ]:


train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y=train.Species
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] 
test_y =test.Species


# In[ ]:


train_X.head(2)


# In[ ]:


test_X.head(2)


# In[ ]:


train_y.head()


# In[ ]:


model = svm.SVC()
model.fit(train_X,train_y)  # Training the Algorithm 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




