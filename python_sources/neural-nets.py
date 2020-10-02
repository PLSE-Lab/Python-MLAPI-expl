#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


dataset=pd.read_csv("/kaggle/input/logistic-regression/Social_Network_Ads.csv")
dataset.head()


# In[ ]:


dataset.isnull().sum()
# no Null values
dataset.shape
dataset.Gender.value_counts()


# In[ ]:


dataset.Purchased.value_counts().plot.bar()
# A slightly imbalanced dataset


# In[ ]:


import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X=dataset.iloc[:,[3,4]].values
wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,10),wcss)
    


# In[ ]:


kmeans=KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(X)
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,color="orange")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,color="black")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,color="blue")


# In[ ]:


X=dataset.iloc[:,[1,2,3]]
Y=dataset.iloc[:,[4]]
X.head()
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
X=ct.fit_transform(X)
X=pd.DataFrame(X)
X.head()


# 

# In[ ]:


#classification using logistic regression
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)
xtrain.reset_index()
xtest.reset_index()
ytest.reset_index()
ytrain.reset_index()
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scale.fit_transform(xtrain)
scale.fit_transform(ytest)
scale.fit_transform(ytrain)
scale.fit_transform(xtest)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=300)
classifier.fit(xtrain,ytrain)
ypred=classifier.predict(xtest)
print(ypred)


# In[ ]:


from sklearn.metrics import r2_score
print(r2_score(ypred,ytest)*100)


# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=xtrain,y=ytrain,cv=10)
print("accuracies:{:.2f}".format(accuracies.mean()*100))


# In[ ]:


#Testing with neural networks
from keras.layers import Dense
from keras.models import Sequential
classifier=Sequential()
classifier.add(Dense(2,input_dim=4,kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(2,kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(1,kernel_initializer="uniform",activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
classifier.fit(xtrain,ytrain,batch_size=10,epochs=200)
ypred=classifier.predict(xtest)


# In[ ]:


ypred=(ypred>0.5)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,ypred))
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

