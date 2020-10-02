#!/usr/bin/env python
# coding: utf-8

#  **INTRODUCTION**
#  
#  We'll learn, practise and compare 6 classification models in this project. So, you'll see in this kernel:
# 
# * EDA (Exploratory Data Analysis)
# * What is Confusion Matrix?
# * Test-Train Datas Split
# * Logistic Regression Classification
# * KNN Classification
# * Support Vector Machine (SVM) Classification
# * Naive Bayes Classification
# * Desicion Tree Classification
# * Random Forest Classification
# * Compare all of these Classification Models
# * Conclusion

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#For confusion matrixes
from sklearn.metrics import confusion_matrix


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read our data from dataset.
data = pd.read_csv("../input/voice.csv")


# In[ ]:


#Let's looking at top 5 datas.
data.head()


# In[ ]:


#Let's looking at last 10 datas.
data.tail(10)


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


# Firstly, we must check our data. If we have NaN values, we should drop them.
data.info()
#As we can see easily, we have no NaN values.


# **Our 'label' feature has 2 valuable: male and female. These are string but we need integers for classification. Therefore, we must convert them from object to integer.**

# In[ ]:


data.label = [1 if each == "female" else 0 for each in data.label]
#We assign 1 to female, 0 to male.


# **Let's check it!**

# In[ ]:


data.info()


# **Confusion Matrix**
# 
# Before start the classifications, we should know one thing: Confusion Matrix!
# For example; we have 100 data point (dogs and cats) and we make a prediction. Our prediction score is 0.8 so we predict well %80. Confusion matrix gives us; 
# * How many true values we predict true  (TP = True Positive)
# * How many true values we predict false  (FP = False Positive)
# * How many false values we predict false  (TN = True Negative)
# * How many false values we predict true (FN = False Negative
# 

# **As you can see; our label features converted integer!**

# In[ ]:


#We should have x and y values for test-train datas.
y = data.label.values
x_data = data.drop(["label"],axis=1)


# In[ ]:


#Normalization
x = (x_data - np.min(x_data)) / (np.max(x_data)).values


# **After assign x and y value; we should train and test datas split.**

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
#test_size=0.2 means %20 test datas, %80 train datas
method_names = []
method_scores = []
#These are for barplot in conclusion


# **And now time to classification our data!**
# 
# **We start with:**
# 
# **LOGISTIC REGRESSION CLASSIFICATION**

# In[ ]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train) #Fitting
print("Logistic Regression Classification Test Accuracy {}".format(log_reg.score(x_test,y_test)))
method_names.append("Logistic Reg.")
method_scores.append(log_reg.score(x_test,y_test))

#Confusion Matrix
y_pred = log_reg.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# **KNN (K-Nearest Neighbour) CLASSIFICATION**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
print("Score for Number of Neighbors = 3: {}".format(knn.score(x_test,y_test)))
method_names.append("KNN")
method_scores.append(knn.score(x_test,y_test))

#Confusion Matrix
y_pred = knn.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# **n_neighbors is an optional parameter. I wrote it 3 but you can write anything. Let's learn the best value of n_neighbors parameter.**

# In[ ]:


score_list=[]
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("score")


# **As we can see; the best value of n_neighbor is 2. Let's find score when n_neighbors=2**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
print("Score for Number of Neighbors = 2: {}".format(knn.score(x_test,y_test)))

#Confusion Matrix
y_pred = knn.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# **SUPPORT VECTOR MACHINE (SVM)**

# In[ ]:


from sklearn.svm import SVC
svm = SVC(random_state=42)
svm.fit(x_train,y_train)
print("SVM Classification Score is: {}".format(svm.score(x_test,y_test)))
method_names.append("SVM")
method_scores.append(svm.score(x_test,y_test))

#Confusion Matrix
y_pred = svm.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# **NAIVE BAYES CLASSIFICATION**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(x_test,y_test)
print("Naive Bayes Classification Score: {}".format(naive_bayes.score(x_test,y_test)))
method_names.append("Naive Bayes")
method_scores.append(naive_bayes.score(x_test,y_test))

#Confusion Matrix
y_pred = naive_bayes.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# **DECISION TREE CLASSIFICATION**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train,y_train)
print("Decision Tree Classification Score: ",dec_tree.score(x_test,y_test))
method_names.append("Decision Tree")
method_scores.append(dec_tree.score(x_test,y_test))

#Confusion Matrix
y_pred = dec_tree.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# **RANDOM FOREST CLASSIFICATION**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rand_forest.fit(x_train,y_train)
print("Random Forest Classification Score: ",rand_forest.score(x_test,y_test))
method_names.append("Random Forest")
method_scores.append(rand_forest.score(x_test,y_test))

#Confusion Matrix
y_pred = rand_forest.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# **CONCLUSION**
# 
# We completed seven different classification on this data and we see; Random Forest Classification is the best way to make classification on this dataset. Of course not everytime but for this practice Random Forest gave us the best classifications!
# 
# Let's see differences between our methods scores!

# In[ ]:


plt.figure(figsize=(15,10))
plt.ylim([0.85,1])
plt.bar(method_names,method_scores,width=0.5)
plt.xlabel('Method Name')
plt.ylabel('Method Score')


# In[ ]:




