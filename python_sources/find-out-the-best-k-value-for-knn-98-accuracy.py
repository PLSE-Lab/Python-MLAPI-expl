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


# # Read in data. Drop the useless column (Serial No.). Rename the column.

# In[ ]:


data=pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
data.drop('Serial No.',axis=1,inplace=True)
data=data.rename(columns={'Chance of Admit ':'Chance of Admit','LOR ':'LOR'})
data.head()


# # Data Preprocessing

# ## Set the threshold of admit to 0.8

# In[ ]:


# The threshold will influence the final accuracy. The lower the threshold, the higher the accuracy.
data['Chance of Admit']=[1 if i>=0.8 else 0 for i in data['Chance of Admit']]
data['Chance of Admit'].head()


# ## Checking categorical data

# In[ ]:


data.info()


# ## Checking missing values

# In[ ]:


data.isnull().any()


# # Building machine learning model

# ## Import packages

# In[ ]:


#Visualisation
import matplotlib.pyplot as plt

#Machine learning model and evaluation methods

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[ ]:


data.columns


# ## Setting independent variables and target variable. Split the data to training set and test set. Test set size is 10%.

# In[ ]:


X=data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA',
       'Research']]
y=data['Chance of Admit']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1,random_state=42)


# # Standardisation

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


X_train


# In[ ]:


X_test


# ## Use iteration to find out the best k value for KNN model.
# 

# In[ ]:


error_rate=[]
for i in range(1,50):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    predictions=knn.predict(X_test)
    error_rate.append(np.mean(predictions!=y_test))

num=np.array(error_rate).argmin()+1

print ('The lowest error rate is ',np.min(error_rate),'when k=',num)


# ## Visualise the iteration

# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)


# # Use the best k value to build KNN model

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=num)
knn.fit(X_train, y_train)
predictions=knn.predict(X_test)


# ## Evaluation

# In[ ]:


confusion_matrix(y_test,predictions)


# In[ ]:




