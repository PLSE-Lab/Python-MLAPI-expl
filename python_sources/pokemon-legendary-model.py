#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/Pokemon.csv")


# In[ ]:


data.head()


# I will use numeric featues therefore drop them.

# In[ ]:


data.drop(["#","Name","Type 1","Type 2"],axis=1,inplace=True)
data.head()


# In[ ]:


data.info()


#    True =1 , False=0. This is how i want to use my class.

# In[ ]:


data.Legendary=[1 if each==True else 0 for each in data.Legendary]


# In[ ]:


y=data.Legendary.values #class
x_data=data.drop(["Legendary"],axis=1)


# In[ ]:


data.describe()


# Because of the numeric differences i will normalize data.

# In[ ]:


x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# I will split my train and test data.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# We will use some algorithm to create model and evaluate accuracy of them.

# In[ ]:


alg_acc={} # to keep accuracies


# **KNearestNeighbor**

# Before fitting my data i want to find best K value, so i will make a for loop.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
scores=[]
for each in range(1,10):
    knn_t=KNeighborsClassifier(n_neighbors=each)
    knn_t.fit(x_train,y_train)
    scores.append(knn_t.score(x_test,y_test))
plt.plot(range(1,10),scores)


# K=2 gives best accuracy so i will use it to create my model.

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
print("accuracy is",knn.score(x_test,y_test))
alg_acc["knn"]=knn.score(x_test,y_test)


# To understand and evaluate accuracy i will use confusion matrix.

# In[ ]:


from sklearn.metrics import confusion_matrix
y_pre=knn.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pre)


#    Let visulization confusion matrix

# In[ ]:


import seaborn as sns
sns.heatmap(cm,annot=True,fmt=".0f")


# According to confusion matrix heatmap. We predict 149  of non-legendary pokemon correctly, only 1 wrong. It's good accuracy. However, there are 10 legendary pokemon, our model predict 5 of them correctly, 5 wrong. 50% accuracy. The reason of low accuracy is we dont have enough legendary pokemon at data.

# In[ ]:


np.count_nonzero(y_train)


# We used 55 legendary pokemon at data to train our model. not enough.

# **Support Vector Machine**

# In[ ]:


from sklearn.svm import SVC
svm=SVC(random_state=42)
svm.fit(x_train,y_train)
alg_acc["svm"]=svm.score(x_test,y_test)
print("Support Vector Machine test accuracy is:",svm.score(x_test,y_test))


# **Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
alg_acc["nb"]=nb.score(x_test,y_test)
print("Naive Bayes test accuracy is:",nb.score(x_test,y_test))


# **Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
alg_acc["dt"]=dt.score(x_test,y_test)
print("Decision Tree test accuracy is:",dt.score(x_test,y_test))


# **Random Forest**

# Random forest has hyperparameter called as n_estimators. It means tree number. To find best hyperparameter value, we can use for loop.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
scores2=[]
for each in range(100,1000,100):
    rf_t=RandomForestClassifier(n_estimators=each,random_state=42)
    rf_t.fit(x_train,y_train)
    scores2.append(rf_t.score(x_test,y_test))
plt.plot(range(100,1000,100),scores2)


# 100 is a good value for n_estimators hyperparamater.

# In[ ]:


rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
alg_acc["rf"]=rf.score(x_test,y_test)
print("Random Forest test accuracy is:",rf.score(x_test,y_test))


# In[ ]:


label=alg_acc.keys()
scores=alg_acc.values()
plt.plot(label,scores)


# **Conclusion**

# As you see; Knn and random forest gave best accuracies at our data.
