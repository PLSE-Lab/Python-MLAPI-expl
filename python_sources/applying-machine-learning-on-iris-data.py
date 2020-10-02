#!/usr/bin/env python
# coding: utf-8

# ##Application of Machine learning algorithms to find the trends in Iris data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
iris = pd.read_csv("../input/Iris.csv")
iris.head()


# In[ ]:


iris.shape


# In[ ]:


iris['Species'].unique()


# In[ ]:


iris['Species'].value_counts()


# ##Visualize the data
# Find correlation between variables

# In[ ]:


sns.heatmap(iris.corr(), annot=True);


# In[ ]:


sns.lmplot('SepalLengthCm', 'SepalWidthCm', 
           data=iris, 
           hue="Species")
plt.title('SepalLength vs SepalWidth')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidthCm');


# In[ ]:


sns.lmplot('PetalLengthCm', 'PetalWidthCm', 
           data=iris, 
           hue="Species")
plt.title('PetalLength vs PetalWidth')
plt.xlabel('PetalLength')
plt.ylabel('PetalWidth');


# In[ ]:


ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray");


# The above visualizations show that sepal length and width are weakly correlated while petal length and width show strong correlation.

# ##Machine Learning
# Separate the features and labels

# In[ ]:


petal = iris[['PetalLengthCm', 'PetalWidthCm']]
sepal = iris[['SepalLengthCm', 'SepalLengthCm']]
cat = {'Iris-setosa': 0,'Iris-versicolor': 1,'Iris-virginica': 2}
y = iris['Species'].map(cat)


# Split the dataset into train and test

# In[ ]:


from sklearn.cross_validation import train_test_split

X_train_S, X_test_S, y_train_S, y_test_S = train_test_split(sepal,y,test_size=0.2,random_state=42)

X_train_P, X_test_P, y_train_P, y_test_P = train_test_split(petal,y,test_size=0.2,random_state=42);


# Feature Scaling
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train_S)
X_train_S_std = scaler.transform(X_train_S)
X_test_S_std = scaler.transform(X_test_S)

scaler.fit(X_train_P)
X_train_P_std = scaler.transform(X_train_P)
X_test_P_std = scaler.transform(X_test_P)


# Use classification algorithm to predict labels for the test data

# ##Support Vector Machine

# In[ ]:


from sklearn import svm
clf = svm.SVC(kernel = 'rbf', C = 10)
clf.fit(X_train_S_std, y_train_S)

print('SVM Training Accuracy Sepal data= {}'.format(clf.score(X_train_S_std, y_train_S)))
print('SVM Testing  Accuracy Sepal data= {}'.format(clf.score(X_test_S_std, y_test_S)))

clf.fit(X_train_P_std, y_train_P)

print('SVM Training Accuracy Petal data = {}'.format(clf.score(X_train_P_std, y_train_P)))
print('SVM Testing  Accuracy Petal data = {}'.format(clf.score(X_test_P_std, y_test_P)))


# ##Decision Tree

# In[ ]:


from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split=2)
clf.fit(X_train_S_std, y_train_S)

print('Decision Tree Training Accuracy Sepal data= {}'.format(clf.score(X_train_S_std, y_train_S)))
print('Decision Tree Testing  Accuracy Sepal data= {}'.format(clf.score(X_test_S_std, y_test_S)))

clf.fit(X_train_P_std, y_train_P)

print('Decision Tree Training Accuracy Petal data = {}'.format(clf.score(X_train_P_std, y_train_P)))
print('Decision Tree Testing  Accuracy Petal data = {}'.format(clf.score(X_test_P_std, y_test_P)))


# ##K-Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_S_std, y_train_S)

print('KNN Training Accuracy Sepal data= {}'.format(knn.score(X_train_S_std, y_train_S)))
print('KNN Testing  Accuracy Sepal data= {}'.format(knn.score(X_test_S_std, y_test_S)))

knn.fit(X_train_P_std, y_train_P)

print('KNN Training Accuracy Petal data = {}'.format(knn.score(X_train_P_std, y_train_P)))
print('KNN Testing  Accuracy Petal data = {}'.format(knn.score(X_test_P_std, y_test_P)))


# ##XGBoost

# In[ ]:


import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train_S_std, y_train_S)

print('XGB Training Accuracy Sepal data= {}'.format(xgb_clf.score(X_train_S_std, y_train_S)))
print('XGB Testing  Accuracy Sepal data= {}'.format(xgb_clf.score(X_test_S_std, y_test_S)))

xgb_clf.fit(X_train_P_std, y_train_P)

print('XGB Training Accuracy Petal data = {}'.format(xgb_clf.score(X_train_P_std, y_train_P)))
print('XGB Testing  Accuracy Petal data = {}'.format(xgb_clf.score(X_test_P_std, y_test_P)))


# In[ ]:





# In[ ]:




