#!/usr/bin/env python
# coding: utf-8

# # Supervised Learning Practice
# 
# This notebook is a practice about some supervised learning algorithms.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Data Preparation

# There are two files in this data set. I will use "column_2C_weka.csv" first.
# 
# (Size of the dataset is small. I will find a better dataset to improve my practice later.)

# In[ ]:


data_2C = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data_2C.info()

#data_3C = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")
#data_3C.info()


# "class" column will be our classifier.
# There are two different results in it: "Abnormal" and "Normal"
# I will change the into numeric values: "Abnormal" = 1 and "Normal" = 0

# In[ ]:


print(data_2C.columns)
print(data_2C["class"].unique())
data_2C["class"] = [1 if (i=="Abnormal" or i==1) else 0 if (i=="Normal" or i==0) else None for i in data_2C["class"]]


# "x" will be input parameters, "y" will be result values.
# So "class" column should be in y, the rest should be x.

# In[ ]:


x = data_2C.drop(["class"], axis=1)
y = data_2C["class"].values


# Parameter values in the x should be normalised.
# (differ between 0 and 1)
# 

# In[ ]:


x = (x - np.min(x))/(np.max(x) - np.min(x))
print(x.head())


# Splitting the data into train and test sets

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)


# # Comparing Different Algorithms
# 
# I will create an empty dictionary and add the results of each algoritm to make a comparison.

# In[ ]:


result_dictionary = {}


# # **KNN Classification**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

n = 11 # neighbor number

knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(x_train,y_train)

knn_score = knn.score(x_test,y_test)

result_dictionary["KNearestNeighbor"] = knn_score

print("KNN score for n={} is => {:.1f}%".format(n, 100*knn_score))


# Finding the best neigbor number by trying for different values and comparing the fit scores.

# In[ ]:


neighbor_number_list = list(range(1,30))
score_list = []

for n in neighbor_number_list:
    knn_new = KNeighborsClassifier(n_neighbors=n)
    knn_new.fit(x_train, y_train)
    score_list.append(knn_new.score(x_test,y_test))
    
highest_score = max(score_list)
highest_score_index = score_list.index(highest_score)

best_neighbor_number = neighbor_number_list[highest_score_index]
print("Best value for neighbor number n is => {}".format(best_neighbor_number))


# Drawing a plot for "n" vs "score"

# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,6))

plt.plot(neighbor_number_list, score_list, color="black", lw=1.2, label="Score Graph")

plt.axhline(y=highest_score, color="red", ls="--", lw=1, label="Highest Score ({:.2f})".format(highest_score))
plt.axvline(x=best_neighbor_number, color="blue", ls="--", lw=1, label="Best n ({})".format(best_neighbor_number))

plt.xlabel("Neighbor Number")
plt.ylabel("Fit Score")

plt.legend()
plt.grid(True)
plt.show()


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)

lr_score = lr.score(x_test,y_test)

result_dictionary["LogisticRegression"] = lr_score

print("Logistic Regression Test Score: {:.1f} %".format(100*lr_score))


# # Support Vector Machine

# In[ ]:


from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(x_train,y_train)

svm_score = svm.score(x_test,y_test)

result_dictionary["SupportVectorMachine"] = svm_score

print("Support Vector Machine Test Score: {:.1f} %".format(100*svm_score))


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

nb_score = nb.score(x_test,y_test)

result_dictionary["NaiveBayes"] = nb_score

print("Naive Bayes Test Score: {:.1f} %".format(100*nb_score))


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

dt_score = dt.score(x_test,y_test)

result_dictionary["DecisionTree"] = dt_score

print("Decision Tree Test Score: {:.1f} %".format(100*dt_score))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=5, random_state=4)
rf.fit(x_train, y_train)

rf_score = rf.score(x_test,y_test)

result_dictionary["RandomForest"] = rf_score

print("Decision Tree Test Score: {:.1f} %".format(100*rf_score))


# Finding the best estimator number by trying for different values and comparing the fit scores.

# In[ ]:


estimator_number_list = list(range(1,21))
score_list = []

for n in estimator_number_list:
    rf_new = RandomForestClassifier(n_estimators=n, random_state=4)
    rf_new.fit(x_train, y_train)
    score_list.append(rf_new.score(x_test,y_test))
    
highest_score = max(score_list)
highest_score_index = score_list.index(highest_score)

best_estimator_number = estimator_number_list[highest_score_index]
print("Best value for estimator number n is => {}".format(best_estimator_number))


# Drawing a plot for "n" vs "score"

# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,6))

plt.plot(estimator_number_list, score_list, color="black", lw=1.2, label="Score Graph")

plt.axhline(y=highest_score, color="red", ls="--", lw=1, label="Highest Score ({:.2f})".format(highest_score))
plt.axvline(x=best_estimator_number, color="blue", ls="--", lw=1, label="Best n ({})".format(best_estimator_number))

plt.xlabel("Estimator Number")
plt.ylabel("Fit Score")

plt.legend()
plt.grid(True)
plt.show()


# # Results and Comparison

# In[ ]:


keys,values = zip(*result_dictionary.items())

best_score = max(values)
best_model = ""

for k,v in result_dictionary.items():
    print("Fit Score of {}: {:.3f}".format(k,v))
    if v==best_score:
        best_model = k
        
print("#"*45)
print("Best Model: {}".format(best_model))
print("Fit Score of Best Model: {:.5f}".format(best_score))
print("#"*45)


# # Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

y_predict = rf.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true, y_predict)

f, ax = plt.subplots(figsize=(7,7))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_predict")
plt.ylabel("y_true")

plt.title("Confusion Matrix")
plt.show()


# In[ ]:




