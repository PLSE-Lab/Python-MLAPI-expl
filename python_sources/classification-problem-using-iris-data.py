#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
iris=pd.read_csv("../input/Iris.csv")


# In[ ]:


iris.head()


# **To count different types of species **

# In[ ]:


iris['Species'].value_counts()


# In[ ]:


iris.describe()


# In[ ]:


#split the data set for train and test
X = iris.drop('Species',axis=1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# **Data Visualization using seaborn and matplotlib**

# In[ ]:


import seaborn as sns
ax=sns.violinplot(iris['Species'],iris['SepalLengthCm'])


# In[ ]:


ax=sns.violinplot(iris['Species'],iris['SepalWidthCm'])


# In[ ]:


ax=sns.violinplot(iris['Species'],iris['PetalLengthCm'])


# In[ ]:


ax=sns.violinplot(iris['Species'],iris['PetalWidthCm'])


# In[ ]:


ax=sns.FacetGrid(iris, hue="Species", size=5) 
ax.map(plt.scatter, "SepalLengthCm", "SepalWidthCm") 
ax.add_legend()


# **Classification of Iris dataset using different classification Algorithm** 

# In[ ]:


#Logistic Regression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
#print(X_test[0:5])
#print(y_pred[0:5])
print(confusion_matrix(y_pred,y_test))
print('accuracy is',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[ ]:


#Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred =dtc.predict(X_test)
print('accuracy is',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[ ]:


#naive bayes 
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
y_pred=nb.predict(X_test)
print('accuracy ',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[ ]:


#Support Vector Machine
from sklearn.svm import LinearSVC
sv=LinearSVC()
sv.fit(X_train,y_train)
y_pred=sv.predict(X_test)
print('accuracy ',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[ ]:


# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print('accuracy ',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))

Cross validation to select a model 
# In[ ]:


print("logistc regression ")
print(cross_val_score(logreg,X_train,y_train,scoring='accuracy',cv=10).mean() *100)


# In[ ]:


print("decision tree classification ")
print(cross_val_score(dtc,X_train,y_train,scoring='accuracy',cv=10).mean() *100)


# In[ ]:


print("naive bayes ")
print(cross_val_score(nb,X_train,y_train,scoring='accuracy',cv=10).mean() *100)


# In[ ]:


print("support vector machine  ")
print(cross_val_score(sv,X_train,y_train,scoring='accuracy',cv=10).mean() *100)


# In[ ]:


print("k nearest neighbours ")
print(cross_val_score(knn,X_train,y_train,scoring='accuracy',cv=10).mean() *100)


# ****Conclusion****
# From the above cross validation test and accuracy score we can use the "naive bayes model " for classification of flower.
