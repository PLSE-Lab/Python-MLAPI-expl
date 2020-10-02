# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Importing Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#For confusion matrixes
from sklearn.metrics import confusion_matrix

#Importing the data
data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

#Checking for NAN Values
data.info()

#Preprocessing
bins = (2, 6.5, 8)
labels = ['bad', 'good']
data['quality'] = pd.cut(x = data['quality'], bins = bins, labels = labels)
data['quality'].value_counts()

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
data['quality'] = labelencoder_y.fit_transform(data['quality'])

data.head()

# x and y values
x = data.iloc[:, 0:-2].values
y = data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

method_names = []
method_scores = []

# Fitting Logistic Regression

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train) #Fitting
print("Logistic Regression Classification Test Accuracy {}".format(log_reg.score(x_test,y_test)))
method_names.append("Logistic Reg.")
method_scores.append(log_reg.score(x_test,y_test))

#Confusion Matrix for Logistic Regression
y_pred = log_reg.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test, y_pred))


# Fitting KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
print("Score for Number of Neighbors = 3: {}".format(knn.score(x_test,y_test)))
method_names.append("KNN")
method_scores.append(knn.score(x_test,y_test))

#Confusion Matrix for KNeighborsClassifier
y_pred = knn.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test, y_pred))

# Fitting SVM
from sklearn.svm import SVC
svm = SVC(random_state=42)
svm.fit(x_train,y_train)
print("SVM Classification Score is: {}".format(svm.score(x_test,y_test)))
method_names.append("SVM")
method_scores.append(svm.score(x_test,y_test))

#Confusion Matrix for SVM
y_pred = svm.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test, y_pred))


# Fitting Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(x_test,y_test)
print("Naive Bayes Classification Score: {}".format(naive_bayes.score(x_test,y_test)))
method_names.append("Naive Bayes")
method_scores.append(naive_bayes.score(x_test,y_test))

#Confusion Matrix for Naive Bayes Classification
y_pred = naive_bayes.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test, y_pred))

# Fitting Decision Tree
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train,y_train)
print("Decision Tree Classification Score: ",dec_tree.score(x_test,y_test))
method_names.append("Decision Tree")
method_scores.append(dec_tree.score(x_test,y_test))

#Confusion Matrix for Decision Tree
y_pred = dec_tree.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test, y_pred))


# Fitting Random Forest
from sklearn.ensemble import RandomForestClassifier
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rand_forest.fit(x_train,y_train)
print("Random Forest Classification Score: ",rand_forest.score(x_test,y_test))
method_names.append("Random Forest")
method_scores.append(rand_forest.score(x_test,y_test))

#Confusion Matrix for Random Forest
y_pred = rand_forest.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test, y_pred))


# visualization
plt.figure(figsize=(15,10))
plt.ylim([0.5,1])
plt.bar(method_names,method_scores,width=0.5)
plt.xlabel('Method Name')
plt.ylabel('Method Score')
