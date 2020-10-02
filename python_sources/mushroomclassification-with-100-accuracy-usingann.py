#!/usr/bin/env python
# coding: utf-8

# Importing the necessary libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Let's look at the data. 

# In[ ]:


train=pd.DataFrame(pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv'))
train.columns


# Renaming the columns 

# In[ ]:


train.columns=['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
       'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
       'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
       'stalk_surface_below_ring', 'stalk_color_above_ring',
       'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number',
       'ring_type', 'spore_print_color', 'population', 'habitat']


# Checking for missing values

# In[ ]:


train.head(5)
train.isna().sum()
#train['stalk-root'].unique()


# In[ ]:


train.dtypes


# Let's map Categorical variables

# In[ ]:


mapping=[{'e':1,'p':0},
         {'b':0,'c':1,'x':2,'f':3, 'k':4,'s':5},
         {'f':0,'g':1,'y':2,'s':3},
         {'n':0,'b':1,'c':2,'g':3,'r':4,'p':5,'u':6,'e':7,'w':8,'y':9},
         {'t':1,'f':0},
         {'a':1,'l':2,'c':3,'y':4,'f':5,'m':6,'n':0,'p':7,'s':8},
         {'a':0,'d':1, 'f':2, 'n':3},
         {'c':0,'w':1,'d':2},
         {'b':0,'n':1},
         {'k':0,'n':1,'b':2,'h':3,'g':4,'r':5,'o':6,'p':7,'u':8,'e':9,'w':10,'y':11},
         { 'e':0,'t':1},{'b':0,'c':1,'u':2,'e':3,'z':4,'r':5,'?':6},
         {'f':0,'y':1,'k':2,'s':3},
         {'f':0,'y':1,'k':2,'s':3},
         {'n':0,'b':1,'c':2,'g':3,'o':4,'p':5,'e':5,'w':6,'y':7},
         {'n':0,'b':1,'c':2,'g':3,'o':4,'p':5,'e':6,'w':7,'y':8},
         {'p':0,'u':1},
         {'n':0,'o':1,'w':2,'y':3},
         {'n':0,'o':1,'t':2},
         {'c':4,'e':1,'f':2,'l':3,'n':0,'p':5,'s':6,'z':7},
         {'k':0,'n':1,'b':2,'h':3,'r':4,'o':5,'u':6,'w':7,'y':8},
         {'a':0,'c':1,'n':2,'s':3,'v':4,'y':5},
         {'g':0,'l':1,'m':2,'p':3,'u':4,'w':5,'d':6}]
len(mapping),len(train.columns)


# In[ ]:


for i in range(len(train.columns)):
    train[train.columns[i]]=train[train.columns[i]].map(mapping[i]).astype(int)


# Data types have changed from object to int

# In[ ]:


train.shape


# Separating depend varible from predictors and splitting the dataset

# In[ ]:


x=train.iloc[:,1:]
y=train.iloc[:,0]
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2)


# Predictions using Logistic regression

# In[ ]:


lr=LogisticRegression()
lr.fit(x_tr,y_tr)
accuracy_score(y_ts,lr.predict(x_ts)),confusion_matrix(y_ts,lr.predict(x_ts))


# Predictions using KNN

# In[ ]:


kn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
kn.fit(x_tr,y_tr)
accuracy_score(y_ts,kn.predict(x_ts)),confusion_matrix(y_ts,kn.predict(x_ts))


# Linear SVM

# In[ ]:


svm=SVC(kernel='linear',random_state=0)
svm.fit(x_tr,y_tr)
ysvc_pred=svm.predict(x_ts)
accuracy_score(y_ts,ysvc_pred),confusion_matrix(y_ts,ysvc_pred)


# Kernel SVM

# In[ ]:


kersvm=SVC(kernel='rbf',random_state=0)
kersvm.fit(x_tr,y_tr)
yksvm_pred=kersvm.predict(x_ts)
accuracy_score(y_ts,yksvm_pred),confusion_matrix(y_ts,yksvm_pred)


# Naive Bayes

# In[ ]:


gnb=GaussianNB()
gnb.fit(x_tr,y_tr)
ygnb_pred=gnb.predict(x_ts)
accuracy_score(y_ts,ygnb_pred),confusion_matrix(y_ts,ygnb_pred)


# Decision Trees

# In[ ]:


dct=DecisionTreeClassifier(random_state=0)
dct.fit(x_tr,y_tr)
ydct_pred=dct.predict(x_ts)
accuracy_score(y_ts,ydct_pred),confusion_matrix(y_ts,ydct_pred)


# Random Forest

# In[ ]:


rf=RandomForestClassifier(random_state=0,n_estimators=100)
rf.fit(x_tr,y_tr)
yrf_pred=rf.predict(x_ts)
accuracy_score(y_ts,yrf_pred),confusion_matrix(y_ts,yrf_pred)


# XGBoost

# In[ ]:


xgb=XGBClassifier()
xgb.fit(x_tr,y_tr)
y_xgb=xgb.predict(x_ts)
accuracy_score(y_ts,y_xgb),confusion_matrix(y_ts,y_xgb)


# In[ ]:


x_tr.shape


# For the task submission , Using ANN

# In[ ]:


classifier=Sequential()
classifier.add(Dense(64,activation='relu',input_dim=22))
#classifier.add(Dense(output_dim=1,init='uniform',activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_tr,y_tr,batch_size=10,epochs=100)


# In[ ]:


y_pred=classifier.predict(x_ts)
y_pred=y_pred>0.5


# In[ ]:


confusion_matrix(y_ts,y_pred)


# In[ ]:





# The unmaaping can be done in the same using dictonaries . This shows how simple encoding helps model predict give accurate results. If you found this helpful do upvote.  Thanks in advance !. 
