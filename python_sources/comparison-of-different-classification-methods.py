#!/usr/bin/env python
# coding: utf-8

# **EXPLANATION**
# 
# In this kernel, I compare 6 different classification methods in machine learning.
# 
# **CONTENTS**
# 
# **1. Data Cleaning and Regulation**
#    
#    I am looking data in general and change number from text for example, Yes-->1 and No-->0. In addtion to this I drop columns which is consist of texts.
# 
# **2. Normalization**
# 
# **3. Train Test Splite**
#     
# I separate data to 2 different part which is train and test. They are dataset.
# 
# **4. Classification Methods **
# 
# Logistic Regression, Decision Tree, Random Forest, Navie Byes, SVM, KNN
# 
# **5. Visualization**
# 
# **6. Conclusion**

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


# **1. Data Cleaning and Regulation**

# In[ ]:


data = pd.read_csv("../input/bank-full.csv",sep=",")


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


# we delete columns whose data type is text
data1= data.drop(["job","education","contact","month","poutcome"],axis=1)


# In[ ]:


data1["default"] = [0 if each== "no" else 1 for each in data1.default]


# In[ ]:


data1["loan"] = [0 if each== "no" else 1 for each in data1.loan]


# In[ ]:


data1["y"] = [0 if each== "no" else 1 for each in data1.y]


# In[ ]:


data1["marital"] = [1 if each == "married" else 0 if each == "single" else 0.5 for each in data1.marital]
data1["housing"] = [1 if each == "yes" else 0  for each in data1.housing]


# In[ ]:


data1.head()


# **2. Normalization**

# In[ ]:


# Defining to y and x_data values for train data
y = data1.y.values
x_data = data1.drop(["y"],axis=1)


# In[ ]:


# normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))


# **3. Train Test Splite**

# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size=0.15,random_state=42)


# **4. Classifications Methods**
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. Naive Byes
# 5. SVM Model
# 6. KNN Model
# 

# In[ ]:


# we define 2 list that one of them save results of models other list save name of model
labelList = []
resultList = []


# In[ ]:


# Logictic Regression with sklearn
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("test accuracy {}".format(lr.score(x_test,y_test)))

# adding result and label to lists
labelList.append("Log_Rec")
resultList.append(lr.score(x_test,y_test))


# In[ ]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("decison tree score : ",dt.score(x_test,y_test))

# adding result and label to lists
labelList.append("Dec_Tree")
resultList.append(dt.score(x_test,y_test))


# In[ ]:


# Random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,random_state = 1)
rf.fit(x_train, y_train)
print("Random forest algor. result: ",rf.score(x_test,y_test))

# adding result and label to lists
labelList.append("Rand_For")
resultList.append(rf.score(x_test,y_test))


# In[ ]:


# Naive Byes 
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))

# adding result and label to lists
labelList.append("Naive_Byes")
resultList.append(nb.score(x_test,y_test))


# In[ ]:


# SVM model
from sklearn.svm import SVC
svm = SVC(random_state=3)
svm.fit(x_train,y_train)
print("print accuracy of svm algo: ",svm.score(x_test,y_test))

# adding result and label to lists
labelList.append("SVM")
resultList.append(svm.score(x_test,y_test))


# In[ ]:


# KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) #n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)

# score
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))


# In[ ]:


# Finding optimum k value between 1 and 15
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each) # create a new knn model
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,15),score_list) # x axis is in interval of 1 and 15
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# In[ ]:


# finding max value in a list and it's index.
a = max(score_list) # finding max value in list
b = score_list.index(a)+1 # index of max value.

print("k = ",b," and maximum value is ", a)

# adding result and label to lists
labelList.append("KNN")
resultList.append(a)


# **5. Visualization**

# In[ ]:


plt.plot(labelList,resultList)
plt.show()


# Above chart give information to us about results of classification algorithms but it is not sorted and clear chart. We can improve this graph to read it easily.

# In[ ]:


# First of all we combine 2 lists (labelList and resultList) by using zip method
zipped = zip(labelList, resultList)
zipped = list(zipped)


# In[ ]:


df = pd.DataFrame(zipped, columns=['label','result'])


# In[ ]:


df


# In[ ]:


new_index = (df['result'].sort_values(ascending=False)).index.values 
sorted_data = df.reindex(new_index)


# In[ ]:


plt.plot(sorted_data.loc[:,"label"],sorted_data.loc[:,"result"])
plt.show()


# In[ ]:


sorted_data


# **6. Conclusion**
# 
# According to out results **Random Forest** has the biggest result value and others respectively KNN and Logistic Regression.

# 
