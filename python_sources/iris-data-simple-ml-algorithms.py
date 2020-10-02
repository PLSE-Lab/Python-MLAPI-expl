#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')


# In[ ]:


iris = df.copy()
iris.head()


# In[ ]:


iris.info()
iris.species.value_counts()


# In[ ]:


iris.describe().T


# ## Visualization

# In[ ]:


plt.figure(figsize=(8,6));
sns.pairplot(iris,kind='reg',hue ='species',palette="husl" );


# In[ ]:


plt.figure(figsize=(8,6));
sns.scatterplot(x=iris.sepal_length,y=iris.sepal_width,hue=iris.species).set_title("Sepal length and Sepal width distribution of three flowers");


# In[ ]:


plt.figure(figsize=(8,6));
cmap = sns.cubehelix_palette(dark=.5, light=.9, as_cmap=True)
ax = sns.scatterplot(x="petal_length", y="petal_width",hue="species",size="species",sizes=(20,200),legend="full",data=iris);


# ## Creating ML Classify Models

# In[ ]:


#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
iris['species'] = lb_make.fit_transform(iris['species'])
iris.sample(3)


# In[ ]:


# # PCA ===> if data consist of too many parameters/variables(columns) then we need to use PCA; in this data it is not necessary
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2,whiten = True) #whitten = normalize
# pca.fit(iris)
# x_pca = pca.transform(iris)


# In[ ]:


# Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


y = iris.species
X = iris.drop('species',axis = 1)


# In[ ]:


#Train and Test split,cross_val,k-fold
from sklearn.model_selection import KFold,train_test_split,cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# ### KNN Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)


# In[ ]:


y_pred = knn.predict(X_test)


# In[ ]:


# Summary of the predictions made by the KNN
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:


# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# ### 2) Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)


# In[ ]:


y_pred = nb.predict(X_test)


# In[ ]:


# Summary of the predictions made by the NB///Accuracy Score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('accuracy is',accuracy_score(y_pred,y_test))


# ### 3) Support Vector Machine

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc = SVC()
svc.fit(X_train,y_train)


# In[ ]:


y_pred = svc.predict(X_test)


# In[ ]:


# Summary of the predictions made by the SVC///Accuracy Score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('accuracy is',accuracy_score(y_pred,y_test))


# ### 4) Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)


# In[ ]:


y_pred = rfc.predict(X_test)


# In[ ]:


# Summary of the predictions made by the Random Forest///Accuracy Score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('accuracy is',accuracy_score(y_pred,y_test))


# ### 5) Logistic Regression

# In[ ]:


df = iris[50:]
y = df.species
X = df.drop('species',axis = 1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[ ]:


y_pred = lr.predict(X_test)


# In[ ]:


# Summary of the predictions made by the Logistic Reg//Accuracy Score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('accuracy is',accuracy_score(y_pred,y_test))


# ### Creating validated model.

# In[ ]:


from sklearn.model_selection import GridSearchCV #Grid search CV method import


# In[ ]:


grid = {"C":np.logspace(-3,3,7),"penalty":["l1",'l2']}


# In[ ]:


lr_cv = GridSearchCV(lr,grid,cv =10)
lr_cv.fit(X_train,y_train)


# In[ ]:


lr_cv.best_params_


# In[ ]:


lr_cv_model = LogisticRegression(C=1.0, penalty='l2')
lr_cv_model.fit(X_train,y_train)


# In[ ]:


y_pred = lr_cv_model.predict(X_test)


# In[ ]:


# Summary of the predictions made by the Logistic Reg Validated model//Accuracy Score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('accuracy is',accuracy_score(y_pred,y_test))


# ### I have tried to use classification models for predicting 3 different species with ML models. Becasue of the Logistic Regression model need to 2 choices I made a prediction model about versicolor and virginica.
# 
# ### If you like it please upvote
# ### References :
# **** https://www.kaggle.com/sherli/iris-complete-eda-classification-and-clustering#Plot-decision-boundary-for-Logistic-Regression-classifier
# **** https://www.kaggle.com/gauravahujaravenclaw/iris-dataset-analysis-and-machine-learning

# In[ ]:




