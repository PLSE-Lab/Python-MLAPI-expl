#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import homogeneity_score
import numpy as np
import pandas as pd


# In[ ]:


iris = load_iris()


# In[ ]:


print("""


EXERCISE 01
Write a program to classify IRIS flowers by using the K-Means algorithm. Compare
the output when apply or not apply scaling and PCA techniques.


""")


# In[ ]:


km = KMeans(n_clusters=3,random_state=321)
km.fit(iris.data)
ypred = km.labels_
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


# In[ ]:


print(classification_report(iris.target, ypred, target_names=target_names))


# In[ ]:


print (confusion_matrix(iris.target, ypred))


# In[ ]:


scaler = MaxAbsScaler()
Xtrain = scaler.fit_transform(iris.data)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(0.95)
Xtf = pca.fit_transform(Xtrain)
Xtf[0:10]


# In[ ]:


km = KMeans(n_clusters=3,random_state=321)
km.fit(Xtf)
ypred = km.labels_
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


# In[ ]:


print(classification_report(iris.target, ypred, target_names=target_names))


# In[ ]:


print (confusion_matrix(iris.target, ypred))


# In[ ]:


# EXERCISE 02
# Write a program to classify IRIS flowers by using the Hierarchical algorithm.
# Compare the output when apply or not apply scaling and PCA techniques.
from sklearn.cluster import AgglomerativeClustering


# In[ ]:


ward = AgglomerativeClustering(n_clusters=3).fit(iris.data)
y_pred = ward.labels_
y_pred


# In[ ]:


print(classification_report(iris.target, ypred, target_names=target_names))


# In[ ]:


print (confusion_matrix(iris.target, ypred))


# In[ ]:


# PCA and scaler
scaler = MaxAbsScaler()
Xtrain = scaler.fit_transform(iris.data)
pca = PCA(0.95)
Xtf = pca.fit_transform(Xtrain)
Xtf[0:10]


# In[ ]:


ward = AgglomerativeClustering(n_clusters=3,linkage='average').fit(Xtf)
y_pred = ward.labels_
y_pred


# In[ ]:


print(classification_report(iris.target, ypred, target_names=target_names))


# In[ ]:


print (confusion_matrix(iris.target, ypred))


# In[ ]:


print("Compute structured hierarchical clustering...")
from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(iris.data, n_neighbors=10, include_self=False)
connectivity


# In[ ]:


# PCA and scaler
scaler = MaxAbsScaler()
Xtrain = scaler.fit_transform(iris.data)
pca = PCA(0.95)
Xtf = pca.fit_transform(Xtrain)
ward = AgglomerativeClustering(n_clusters=3, connectivity=connectivity,
                               linkage='single').fit(Xtf)
ward.labels_


# In[ ]:


print(classification_report(iris.target, ypred, target_names=target_names))


# In[ ]:


print (confusion_matrix(iris.target, ypred))


# In[ ]:


# EXERCISE 03
# Write a program to classify and predict fishes in fish.csv. Compare the output
# among supervised and unsupervised algorithms and apply or not apply scaling and
# PCA techniques.


# In[ ]:


dt_fish = pd.read_csv('../input/dataset/fish.csv',names=['C1','C2','C3','C4','C5','C6','C7','C8'])
dt_fish.head()


# In[ ]:


#Classifier
X = dt_fish.iloc[:,1:-1]
y = dt_fish.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
clf = tree.DecisionTreeClassifier()
y_pred = cross_val_score(clf, X, y, cv=10)
print('Decision tree= ',y_pred.mean())
clf = svm.SVC(gamma='scale')
y_pred = cross_val_score(clf, X, y, cv=10)
print('SVM= ',y_pred.mean())
clf = GaussianNB()
y_pred = cross_val_score(clf, X, y, cv=10)
print('Navibayes= ',y_pred.mean())
knn = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print('KNN= ',scores.mean())
clf = LogisticRegression(solver='newton-cg', multi_class='multinomial')
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print('Logistic= ',scores.mean())


# In[ ]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.3,random_state=31)
clf = tree.DecisionTreeClassifier()
clf.fit(Xtrain,ytrain)
clf.score(Xtest,ytest)


# In[ ]:


from sklearn.cluster import KMeans
km = KMeans(4,random_state=19)
km.fit(X)
ypred = km.labels_
ypred


# In[ ]:


print(classification_report(y, ypred))


# In[ ]:


print(confusion_matrix(y, ypred))


# In[ ]:


# PCA and scaler
scaler = StandardScaler()
Xtrain = scaler.fit_transform(X)
pca = PCA(0.95)
Xtf = pca.fit_transform(Xtrain)
Xtf[0:10]


# In[ ]:


km = KMeans(4,random_state=42)
km.fit(Xtf)
ypred = km.labels_
ypred


# In[ ]:


print(classification_report(y, ypred))


# In[ ]:


print(confusion_matrix(y, ypred))


# In[ ]:


ward = AgglomerativeClustering(n_clusters=4).fit(X)
y_pred = ward.labels_
y_pred


# In[ ]:


print(classification_report(y, ypred))


# In[ ]:


print(confusion_matrix(y, ypred))


# In[ ]:


ward = AgglomerativeClustering(n_clusters=4).fit(Xtf)
y_pred = ward.labels_
y_pred


# In[ ]:


print(classification_report(y, ypred))


# In[ ]:


print(confusion_matrix(y, ypred))


# In[ ]:




