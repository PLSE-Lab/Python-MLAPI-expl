#!/usr/bin/env python
# coding: utf-8

# **Classification Algorithms**
# 
# I will explain at this kernel the below to classification algorithms with examples.
# 
# We are predict feature of "survived" at this kernel 
# 
# 1. Logistic Regression Classification
# 2. K-NN (K-Nearest Neighbour) Classification
# 3. Support Vector Machine Classification
# 4. Naive Bayes Classification
# 5. Decision Tree Classification
# 6. Random Forest Classification
# 7. Evaluation Classification Models

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


#  

# In[ ]:


pre_dataTest = pd.read_csv("../input/test.csv")
pre_dataTrain = pd.read_csv("../input/train.csv")
data_sonuc = pd.read_csv("../input/gender_submission.csv")


# In[ ]:


pre_dataTrain.head()


# In[ ]:


pre_dataTest.head()


# In[ ]:


pre_dataTrain.info()
print("_"*25)
pre_dataTest.info()


# In[ ]:


pre_dataTrain.describe()


# 

# In[ ]:


pre_dataTrain.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(pre_dataTrain.corr(), annot=True, linewidths=0.5, linecolor="red", fmt='.1f', ax=ax)
plt.show()


# **Which we can use features?**
# * As we can see the higest correlation with "survived" is feature of "Pclass". And there is negative correlation between 2 feature.
# * There is a positive correlation between "survived" and "Fare" with a correlation of 0.25.
# * "There isn't a significant correlation between other features." that we can interpret.

# **In that case we can visualization of data.**

# In[ ]:


g = sns.jointplot(pre_dataTrain.Survived, pre_dataTrain.Pclass, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()


# In[ ]:


g = sns.jointplot(pre_dataTrain.Survived, pre_dataTrain.Fare, color="green" , kind="kde", size=7)
plt.savefig('graph.png')
plt.show()


# **Model and predict**
# 
# We want classification whether passengers survived. Therefore we will use the following algorithms. 
# 
# * Logistic Regression Classification
# * KNN or K-Nearest Neighbors Classification
# * Support Vector Machines Classification
# * Naive Bayes Classification
# * Decision Tree Classification
# * Random Forest Classification
# * Evaluation Classification Models
# 

# In[ ]:


pre_dataTrain.dropna(inplace=True)
pre_dataTest.dropna(inplace=True)


# In[ ]:


# Data drop
dataTrain = pre_dataTrain.drop(["PassengerId","Name","Sex", "SibSp", "Parch","Ticket","Cabin", "Embarked"], axis=1)
dataTest = pre_dataTest.drop(["Name","Sex","SibSp", "Parch", "Ticket","Cabin", "Embarked"], axis=1)


# In[ ]:


dataTest.info()


# In[ ]:


dataTest.head()


# In[ ]:


dataTrain.info()


# In[ ]:


pre_x_train = dataTrain.drop("Survived", axis=1)
x_train = (pre_x_train-np.min(pre_x_train))/(np.max(pre_x_train)-np.min(pre_x_train)).values
y_train = dataTrain["Survived"]
pre_x_test = dataTest.drop("PassengerId", axis=1)
x_test = (pre_x_test-np.min(pre_x_test))/(np.max(pre_x_test)-np.min(pre_x_test)).values
x_train.shape, y_train.shape, x_test.shape


# In[ ]:


#Logistic Regression Classisification
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_head = lr.predict(x_test)
result_lr = round(lr.score(x_train, y_train)*100,2)
result_lr
print("Result Survived Predict to Logistic Regression Class: ", lr.score(x_train, y_train))


# In[ ]:


# KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_head = knn.predict(x_test)
result_knn = round(knn.score(x_train, y_train)*100,2)
result_knn
print("Result Survived Predict to KNN Class.: ", knn.score(x_train, y_train))


# In[ ]:


# Support Vector Machine Classification
from sklearn.svm import SVC
svm = SVC(random_state=3)
svm.fit(x_train, y_train)
y_head = svm.predict(x_test)
result_svm = round(svm.score(x_train, y_train)*100,2)
result_svm
print("Result Survived Predict to SVM Class.: ", svm.score(x_train, y_train))


# In[ ]:


#Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
y_head = nb.predict(x_test)
result_nb = round(nb.score(x_train, y_train)*100,2)
result_nb
print("Result Survived Predict to Naive Bayes Class.: ", nb.score(x_train, y_train))


# In[ ]:


#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=5)
dt.fit(x_train, y_train)
y_head = dt.predict(x_test)
result_dt = round(dt.score(x_train, y_train)*100,2)
result_dt
print("Result Survived Predict to Decision Tree Class.: ", dt.score(x_train, y_train))


# In[ ]:


# Random Forest Classification and Evaluation Classification Models
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit (x_train, y_train)
y_head = rf.predict(x_test)
result_rf = round(rf.score(x_train, y_train)*100,2)
result_rf
print("Result Survived Predict to Random Forest Class.: ", rf.score(x_train, y_train))


# In[ ]:


# Evaluation Classification Models
models = pd.DataFrame({
    'Model' : ['Logistic Regression', 'KNN', 'SVM',
               'Naive Bayes', 'Decison Tree', 'Random Forest'],
    'Score' : [result_lr,result_knn,result_svm,
              result_nb, result_dt, result_rf]
})

models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": dataTest["PassengerId"],
        "Survived": y_head
    })
submission.to_csv('submission.csv', index=False)


# **CONCLUSION**
# 
# We can choose rank our evaluation of all the models the best result. While both Random Forest Class and Decision Tree Class result the same, we choose Random Forest Class due to its strcuture.According to this Random Forest Class. model is the best result.
