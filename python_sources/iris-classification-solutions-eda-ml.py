#!/usr/bin/env python
# coding: utf-8

# In this notebook we will try to analyse one of the most famous datasets using seaborn, matplotlip and plotly libraries, and we will then use different ML classification algorithmes to classify the type o species based on their petal and sepal width and lenght. 

# let's load the required libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import random
import pandas as pd
from pandas.tools import plotting
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import  accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelBinarizer

import xgboost as xgb
import lightgbm as  lgb
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


#  Let's begin by importing the Iris dataset, and showing the first 5 rows.

# In[ ]:


data=pd.read_csv("../input/Iris.csv")
table = ff.create_table(data.head())
py.iplot(table,filename='jupyter-table1')


# Let's do some Exploratpory data analysis on this data 

# In[ ]:


print("the dataset has {} rows and {} features".format(data.shape[0],data.shape[1]))


# In[ ]:


data.info()


# The data types are correct and we can see that there is no null values in the dataset 

# we will remove the ID column from the data

# In[ ]:


data.drop('Id',axis=1,inplace=True)


# Let's take a look at the summary statistics

# In[ ]:


data.describe()
#py.iplot(ff.create_table(data.describe()),filename='describe_table')


# In[ ]:


print(" we have {} types of species in this dataset".format(data.Species.nunique()))
print("the names of this species are:",data.Species.unique())


# In[ ]:


data.Species.value_counts()


# So we have equally distributed species all are of 50

# Let's visualize the data with a scatter plots based on Sepal and Petal width and lenght

# In[ ]:


fig = data[data.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
data[data.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# we can clearly see that sepal characteristics differentiate setosa but not versicolor and virginica 

# In[ ]:


fig = data[data.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
data[data.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
data[data.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# With Petal characteristics we can easily classify the species but there is a little thin line between the virginica and versicolor

# Let's check the distribution of each column

# In[ ]:


data.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(15,6)
plt.show()


# Let's take a look at the Probability Density

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=data)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=data)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=data)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=data)


# On each side of the gray line is a kernel density estimation to show the distribution shape of the data.
# the skinnier sections represent a lower probability.

# Andrews curves are a method for visualizing multidimensional data by mapping each observation onto a function.

# In[ ]:


plt.figure(figsize=(10,8))
plotting.andrews_curves(data,'Species')


# In the plot above, the each color used represents a class and we can easily note that the lines that represent samples from the same class have similar curves.
# 

# Let's check if we have any correlation between features 

# In[ ]:


plt.figure(figsize=(7,4)) 
sns.heatmap(data.corr(),annot=True,fmt="f",cmap="YlGnBu")
plt.show()


# we have a strong correlation between Petal Width and Petal length , and also between Petal characteristics and Sepal Length.

# Let's create a regression plot for both petal and sepal

# In[ ]:


sns.lmplot(x="SepalWidthCm", y="SepalLengthCm",hue="Species",data=data)


# In[ ]:


sns.lmplot(x="PetalWidthCm", y="PetalLengthCm",hue="Species",data=data)


# **Let's now build our classification algorithmes**

# we will split the dataset into training and testing data

# In[ ]:


X_train, X_test, y_train,y_test = train_test_split(data.iloc[:,:4],data.Species,stratify=data.Species,test_size = 0.3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


SVC=SVC()
SVC.fit(X_train,y_train)
SVC_prediction=SVC.predict(X_test)
print("the accuracy of the Support Vector Machine Classifier model is :",accuracy_score(y_test,SVC_prediction))
confusion_matrix(SVC_prediction,y_test)


# In[ ]:


LG=LogisticRegression()
LG.fit(X_train,y_train)
LG_prediction=LG.predict(X_test)
print("the accuracy of the Logistic Regression model is :",accuracy_score(LG_prediction,y_test))
confusion_matrix(LG_prediction,y_test)


# In[ ]:


DTC=DecisionTreeClassifier(max_leaf_nodes=3)
DTC.fit(X_train,y_train)
DTC_prediction=DTC.predict(X_test)
print('The accuracy of the Decision Tree Classifier model is',accuracy_score(DTC_prediction,y_test))
confusion_matrix(DTC_prediction,y_test)


# In[ ]:


ETC=ExtraTreesClassifier()
ETC.fit(X_train,y_train)
ETC_prediction=ETC.predict(X_test)
print('The accuracy of the Extra Trees Classifier model is',accuracy_score(ETC_prediction,y_test))
confusion_matrix(ETC_prediction,y_test)


# In[ ]:


KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train,y_train)
KNN_prediction=KNN.predict(X_test)
print('The accuracy of the KNeighborsClassifier model is',accuracy_score(KNN_prediction,y_test))
confusion_matrix(KNN_prediction,y_test)


# In[ ]:


GNB=GaussianNB()
GNB.fit(X_train,y_train)
GNB_prediction=GNB.predict(X_test)
print('The accuracy of the GaussionNB model is',accuracy_score(GNB_prediction,y_test))
confusion_matrix(GNB_prediction,y_test)


# In[ ]:


RFC=RandomForestClassifier()
RFC.fit(X_train,y_train)
RFC_prediction=RFC.predict(X_test)
print('The accuracy of the RandomForestClassifier model is',accuracy_score(RFC_prediction,y_test))
confusion_matrix(RFC_prediction,y_test)


# In[ ]:


XGB=XGBClassifier()
XGB.fit(X_train,y_train)
XGB_prediction=XGB.predict(X_test)
print('The accuracy of the XGBClassifier model is',accuracy_score(XGB_prediction,y_test))
confusion_matrix(XGB_prediction,y_test)


# In[ ]:


LGB=LGBMClassifier()
LGB.fit(X_train,y_train)
LGB_prediction=LGB.predict(X_test)
print('The accuracy of the LGBMClassifier model is',accuracy_score(LGB_prediction,y_test))
confusion_matrix(LGB_prediction,y_test)


# In[ ]:


CBC=CatBoostClassifier()
CBC.fit(X_train,y_train)
CBC_prediction=CBC.predict(X_test)
print('The accuracy of the CatBoostClassifier model is',accuracy_score(CBC_prediction,y_test))
confusion_matrix(CBC_prediction,y_test)


# Let's try to build a Deep Learning Model with Keras

# In[ ]:


from sklearn.preprocessing import StandardScaler, LabelBinarizer
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

X = StandardScaler().fit_transform(X)
y = LabelBinarizer().fit_transform(y)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[ ]:


model = Sequential()
model.add(Dense( 4, input_dim=4, activation = 'relu'))
model.add(Dense( units = 10, activation= 'relu'))
model.add(Dense( units = 3, activation= 'softmax'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[ ]:


model_training = model.fit(x_train, y_train, epochs = 150, validation_data = (x_test, y_test))

