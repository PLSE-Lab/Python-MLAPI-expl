#!/usr/bin/env python
# coding: utf-8

# This is a project on classifying the flowers based on their Petal and Sepal features.Data set has the measurnment for Pedal and Sepal length and width.Using different machine learning algorithms we will be classifying the flowers into corresponding species.Plant and nature is one of my area of interest.If you want to learn more about plants please do follow my facebook page on plants https://www.facebook.com/Plants-956543137797435/?ref=aymt_homepage_panel

# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image
#%matplotlib inline
import numpy as np
img=np.array(Image.open('../input/iris-flower/Iris.jpg'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


plt.style.use('seaborn')


# **Picture of Iris flower**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Importing Python modules needed for doing the project

# In[ ]:


import pandas as pd
import seaborn as sns


# # Importing the data 

# In[ ]:


iris=pd.read_csv('../input/iris/Iris.csv')


# # Displaying the first five values of dataset

# In[ ]:


iris.head()


# # Checking if there are missing values in the data

# In[ ]:


iris.info()


# The data set has 150 rows of data and there are no missing values.The column **id** will not be needed for our analysis and can be removed from the data set.

# In[ ]:


iris.drop('Id',axis=1,inplace=True)


# In[ ]:


iris.head()


# axis=1 drops the columns axis=0 drops rows.
# inplace=True replaces the old data frame with the new one.New data frame doesnt have the column for** Id**

# **After importing and cleaning the data,next step will be to explore the data.We will plot the figures with the features available in the data set.This ill gives us direction on which features are more important in the dataset to make the prediction of the species.**

# In[ ]:


sc=iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red',label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green',label='Versicolor',ax=sc)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='virginica', ax=sc)
sc.set_xlabel('Sepal Length in cm')
sc.set_ylabel('Sepal Width in cm')
sc.set_title('Sepal Length Vs Sepal Width')
sc=plt.gcf()
sc.set_size_inches(10,6)


# In[ ]:


scp = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=scp)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=scp)
scp.set_xlabel("Petal Length")
scp.set_ylabel("Petal Width")
scp.set_title(" Petal Length VS Width")
scp=plt.gcf()
scp.set_size_inches(10,6)
plt.show()


# The features of petal length and petal width show better clustering.This indicates that the Features of Petals would give better result while applying machine learning to the data

# In[ ]:


iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.boxplot(x='Species',y='SepalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.boxplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.boxplot(x='Species',y='PetalLengthCm',data=iris)


# **Getting the correlation between the features in the datset**

# In[ ]:


plt.figure(figsize=(7,4))
sns.heatmap(iris.corr(),annot=True,cmap='summer')


# From the heat map it can be seen that feature Sepal width has the least correlation with other features.

# # Starting with machine learning models

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split # for spliting the data into training and test set
from sklearn import metrics # for validating the accuracy of the model


# **Splitting the data into training and test data**

# In[ ]:


train,test=train_test_split(iris,test_size=0.15)
train.head()


# **Creating train data set for dependent and independent variable**

# In[ ]:


X_train=train[['PetalLengthCm','PetalWidthCm','SepalLengthCm','SepalWidthCm']]
y_train=train.Species


# **Creating test data set for dependent and independent variable**

# In[ ]:


X_test=test[['PetalLengthCm','PetalWidthCm','SepalLengthCm','SepalWidthCm']]
y_test=test.Species


# In[ ]:


y_train.shape


# # Using Support vector machine (SVM) Model 

# In[ ]:


model=svm.SVC()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('Accuracy of the SVM is:',metrics.accuracy_score(prediction,y_test))


# SVM Algorithm has a very high accuracy to predict species of Iris data

# #  Logistic regression

# In[ ]:


model=LogisticRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('Accuracy of the Logistic Regression is:',metrics.accuracy_score(prediction,y_test))


# # Decision Tree

# In[ ]:


model=DecisionTreeClassifier()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('Accuracy of decision tree is:',metrics.accuracy_score(prediction,y_test))


# # K-Nearest Neighbours

# In[ ]:


model=KNeighborsClassifier(n_neighbors=6)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('Accuracy of KNeighbors is:',metrics.accuracy_score(prediction,y_test))


# # Checking the increase in accuracy of K-Nearest Neighours when change in index

# In[ ]:


a_index=list(range(1,11))
a=pd.Series()
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    print('Accuracy of KNeighbors for n=',i,metrics.accuracy_score(prediction,y_test))   
    


# In[ ]:


a_index=list(range(1,16))
a=pd.Series()
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for i in list(range(1,16)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))
plt.plot(a_index, a)
plt.xticks(x)


# As we increase the value of n the accuracy of KNeighbours increases.We have see where the results dont imprve further and stop further iterations
