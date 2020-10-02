#!/usr/bin/env python
# coding: utf-8

# **Objective: KNN - Predict whether a person will be diagnosed with Diabetes or not.****

# Libraries:
# train_test_split --> For splitting the data. Part of it for Training the model and other for Testing how good it is.
# Preprocessing --> To do the scaling using standardScaler as there should not be any bias
# Neighbors --> We will be using KNeighborsClassifier
# Metrics --> For Testing the model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


diadf=pd.read_csv('../input/diabetes.csv')
#print(len(diadf))
diadf.head()


# In[ ]:


diadf.isna().sum()


# Columns like 'Glucose','bloodpressure' cannot be accepted as zeroes because it will affect the outcomes. We can replace such values with the mean of the respective columns.

# In[ ]:


#Replace zeroes
zero_not_accepted=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
for column in zero_not_accepted:
    diadf[column]=diadf[column].replace(0,np.NaN)
    mean=int(diadf[column].mean(skipna=True))
    diadf[column]=diadf[column].replace(np.NaN,mean)


# Split the Dataset.

# In[ ]:


X=diadf.iloc[:,0:8]
y=diadf.iloc[:,8]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


# **Feature scaling**
# We scale only the data which is going in (i.e.,)Independent data.
# We dont need to scale Target variable

# In[ ]:


sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# Define the model using KNeighborsClassifier and fit the train data in the model.

# In[ ]:


import math
math.sqrt(len(y_test))


# Since we got even number as 12 we will reduce 1 and use k=11

# In[ ]:


#Define the model: Init K-NN
classifier=KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')


# In[ ]:


#fit the model
classifier.fit(X_train,y_train)


# In[ ]:


#predict the test set results
y_pred=classifier.predict(X_test)
y_pred


# Evaluate the model, use Confusion Matrix

# In[ ]:


#evaluate the model
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


print(f1_score(y_test,y_pred))


# In[ ]:


print(accuracy_score(y_test,y_pred))

