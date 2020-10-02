#!/usr/bin/env python
# coding: utf-8

# # Classifying different Iris species based on their sepal and petal characteristics

# # Importing all the necessary modules

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier


# **Reading the data and renaming columns for quicker access**

# In[ ]:


df=pd.read_csv('../input/iris/Iris.csv')
df.head(150)


# In[ ]:


df.rename(columns={'SepalLengthCm':'SL','SepalWidthCm':'SW','PetalLengthCm':'PL','PetalWidthCm':'PW'},inplace=True)


# # Visualizing the data

# In[ ]:


sns.distplot(df['SL'],bins=50)


# In[ ]:


sns.distplot(df['PL'],bins=50,color='purple')


# In[ ]:


sns.distplot(df['SW'],bins=50,color='green')


# In[ ]:


sns.distplot(df['PW'],bins=50,color='red')


# Dropping the unwanted ID column

# In[ ]:


df.drop('Id',axis=1,inplace=True)


# In[ ]:


sns.boxplot(data=df.drop('Species',axis=1),palette='Set2')


# Checking for null values 

# In[ ]:


df.isna().sum()


# In[ ]:


df['Species'].value_counts()


# Converting the 3 different species from strings to numbers using Label encoding

# In[ ]:


lenc=LabelEncoder()
df['Species']=lenc.fit_transform(df['Species'])

df.head()


# In[ ]:


df['Species'].value_counts()


# # Splitting the dataset into training and testing data

# In[ ]:


X=df.iloc[:,:-1]
y=df['Species']


# In[ ]:


Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=12)


# # Scaling the data

# In[ ]:


scale=StandardScaler()
Xtrain=scale.fit_transform(Xtrain)
Xtest=scale.transform(Xtest)


# # Using 3 different models and checking the accuracy of each one

# Logistic Regression

# In[ ]:


logreg=LogisticRegression()
logreg.fit(Xtrain,ytrain)
yprd=logreg.predict(Xtest)
sc=accuracy_score(ytest,yprd)
print('Accuracy score: {}'.format(round(sc*100),2))


# Support Vector Machine

# In[ ]:


mod=svm.SVC()
mod.fit(Xtrain,ytrain)
yspred=mod.predict(Xtest)
sco=accuracy_score(ytest,yspred)
print('Accuracy score: {}'.format(round(sco*100),2))


# Gradient Boosting

# In[ ]:


gb=GradientBoostingClassifier(n_estimators=20,learning_rate=0.1,max_depth=2)
gb.fit(Xtrain,ytrain)
ypred=gb.predict(Xtest)
scc=accuracy_score(ytest,ypred)
print('Learning rate: 0.1')
print('Accuracy score: {}'.format(round(scc*100),2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(ytest,ypred)),2))


# # Predicting the species of Iris flower for a given set of input values using Gradient Boosting

# In[ ]:


X_new=[[1,0.5,0.3,0.2]]
ygb=gb.predict(X_new)
if ygb==[0]:
    print('Iris-setosa')
elif ygb==[1]:
    print('Iris-versicolor')
else:
    print('Iris-virginica')

