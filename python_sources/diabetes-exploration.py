#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


data = pd.read_csv(os.path.join('../input','diabetes.csv'))


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data['Pregnancies'].hist()


# In[ ]:


max(data['Pregnancies'])


# In[ ]:


data[data['Pregnancies']==17]


# In[ ]:


data['Outcome'].hist()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


print(data.columns)


# In[ ]:


print("dimension of data: {}".format(data.shape))


# In[ ]:


print(data.groupby('Outcome').size())


# In[ ]:


import seaborn as sns


# In[ ]:


sns.countplot(data['Outcome'],label="Count")


# In[ ]:


data.info()


# # K-Nearest Neighbors

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data.loc[:,data.columns!='Outcome'],
                                                   data['Outcome'],
                                                   stratify=data['Outcome'],
                                                   random_state=66)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.hist()


# In[ ]:


y_test.hist()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []
# try n_neighbours from 1 to 10
neighbours_settings = range(1,11)

for n_neighbours in neighbours_settings:
    #build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbours)
    knn.fit(X_train,y_train)
    #record training set accuracy
    training_accuracy.append(knn.score(X_train,y_train))
    #record test set accuracy
    test_accuracy.append(knn.score(X_test,y_test))


# In[ ]:


plt.plot(neighbours_settings,training_accuracy,label="Training accuracy")
plt.plot(neighbours_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train, y_train)


# In[ ]:


print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


# In[ ]:


logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set accuracy: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(logreg001.score(X_test, y_test)))


# In[ ]:


logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set accuracy: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(logreg100.score(X_test, y_test)))


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


# In[ ]:


tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))


# In[ ]:


rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf1.score(X_test, y_test)))


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


gnb = GaussianNB()


# In[ ]:


gnb.fit(X_train,y_train)


# In[ ]:


print("Accuracy on training set: {:.3f}".format(gnb.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gnb.score(X_test, y_test)))


# In[ ]:




