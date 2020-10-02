#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


ds=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")


# In[ ]:


ds.head()


# In[ ]:


#checking the number of rows in the dataset
len(ds)


# In[ ]:


#checking the null values in each column
ds.isna().sum()


# In[ ]:


#giving the inputs and outputs
x=ds.iloc[:,2:-1].values
y=ds.iloc[:,1:2].values
y


# In[ ]:


#data preprocessing 
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
y=l.fit_transform(y)
y


# In[ ]:


#splitting the data into training and split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)


# In[ ]:


#algorithm approach
from sklearn.tree import DecisionTreeClassifier
d=DecisionTreeClassifier()
d.fit(xtrain,ytrain)


# In[ ]:


ypred=d.predict(xtest)
ypred


# In[ ]:


#confusion matrix for accuracy
from sklearn.metrics import confusion_matrix
a=confusion_matrix(ytest,ypred)
a


# In[ ]:


accuracy=(a[0][0]+a[1][1])/(len(xtest))
accuracy=accuracy*100
accuracy


# In[ ]:


#algorithm approach random forest
from sklearn.ensemble import RandomForestClassifier
r=RandomForestClassifier()
r.fit(xtrain,ytrain)


# In[ ]:


rpred=r.predict(xtest)
rpred


# In[ ]:


#confusion matrix for accuracy
from sklearn.metrics import confusion_matrix
a1=confusion_matrix(ytest,rpred)
a1


# In[ ]:


accuracy1=(a1[0][0]+a1[1][1])/(len(xtest))
accuracy1=accuracy1*100
accuracy1


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


# In[ ]:


model = Sequential()
model.add(Dense(45, input_shape=(30,)))
model.add(Activation('relu'))                            
model.add(Dense(45))
model.add(Activation('relu'))
model.add(Dense(45))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[ ]:


model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])


# In[ ]:


model.fit(xtrain,ytrain,batch_size=10,epochs=100)


# In[ ]:


model.save('pra.yml')


# In[ ]:


apred=model.predict(xtest)
apred


# In[ ]:


apre=[]
for i in apred:
    if(i<0.5):
        apre.append(0)
    else:
        apre.append(1)


# In[ ]:


#confusion matrix for accuracy
from sklearn.metrics import confusion_matrix
a2=confusion_matrix(ytest,apre)
a2


# In[ ]:


accuracy2=(a2[0][0]+a2[1][1])/(len(xtest))
accuracy2=accuracy2*100
accuracy2


# In[ ]:


#using naive bayes
from sklearn.naive_bayes import GaussianNB
cla3=GaussianNB()
cla3.fit(xtrain,ytrain)


# In[ ]:


gpred=cla3.predict(xtest)
gpred


# In[ ]:


#accuracy for gausianNB
a3=confusion_matrix(ytest,gpred)
a3


# In[ ]:


accuracy3=(a3[0][0]+a3[1][1])/(len(xtest))
accuracy3=accuracy3*100
accuracy3


# In[ ]:


#k nearest neighbours approach
from sklearn.neighbors import KNeighborsClassifier
cla4=KNeighborsClassifier(n_neighbors=7)
cla4.fit(xtrain,ytrain)


# In[ ]:


kpred=cla4.predict(xtest)
kpred


# In[ ]:


#accuracy for knn
a4=confusion_matrix(ytest,kpred)
a4


# In[ ]:


accuracy4=(a4[0][0]+a4[1][1])/(len(xtest))
accuracy4=accuracy4*100
accuracy4


# In[ ]:


#implementing svm
from sklearn.svm import SVC
cla5=SVC()
cla5.fit(xtrain,ytrain)


# In[ ]:


spred=cla5.predict(xtest)
spred


# In[ ]:


#accuracy for svm
a5=confusion_matrix(ytest,spred)
a5


# In[ ]:


accuracy5=(a5[0][0]+a5[1][1])/(len(xtest))
accuracy5=accuracy5*100
accuracy5


# In[ ]:


#implementing logistic regression
from sklearn.linear_model import LogisticRegression
cla6=LogisticRegression()
cla6.fit(xtrain,ytrain)


# In[ ]:


lpred=cla6.predict(xtest)
lpred


# In[ ]:


#accuarcy for logistic
a6=confusion_matrix(ytest,lpred)
a6


# In[ ]:


accuracy6=(a6[0][0]+a6[1][1])/(len(xtest))
accuracy6=accuracy6*100
accuracy6


# In[ ]:




