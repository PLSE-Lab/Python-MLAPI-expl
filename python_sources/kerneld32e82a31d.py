#!/usr/bin/env python
# coding: utf-8

# In[38]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import sys
from pandas import Series
import pandas as pd
import numpy as np
import traceback
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[39]:


data = pd.read_csv("../input/VitalSigns.csv")
data.head()


# In[40]:


print('Shape of the data set: ' + str(data.shape))


# In[41]:


Temp = pd.DataFrame(data.isnull().sum())
Temp.columns = ['Environmental_Temperature']
print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Environmental_Temperature'] > 0])) )


# In[42]:


Temp = pd.DataFrame(data.isnull().sum())
Temp.columns = ['PRbpm']
print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['PRbpm'] > 0])) )


# In[43]:


Temp = pd.DataFrame(data.isnull().sum())
Temp.columns = ['Airflow']
print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Airflow'] > 0])) )


# In[44]:


plt.title('Environmental Temperature')
plt.xlabel('Time(s)')
plt.ylabel('Temperature')
aX=data['Environmental_Temperature']
Time=data['Second']
plt.plot(aX,  color='red')
plt.legend() 


# In[45]:


plt.title('Pulse Rate')
plt.xlabel('Time(s)')
plt.ylabel('Pulse Rate')
aX=data['PRbpm']
Time=data['Second']
plt.plot(aX,  color='red')


# In[46]:



plt.title('Air Flow')
plt.xlabel('Time(s)')
plt.ylabel('Air flows')
aX=data['Airflow']
Time=data['Second']
plt.plot(aX,  color='red')


# In[48]:


X = data.drop('class',axis=1).values
y = data['class'].values


# In[51]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)


# In[52]:


neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)


# In[53]:


plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[54]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=1)


# In[55]:


knn.fit(X_train,y_train)


# In[56]:


knn.score(X_test,y_test)


# In[57]:


#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)


# In[58]:


confusion_matrix(y_test,y_pred)


# In[59]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[60]:


print(classification_report(y_test,y_pred))


# In[37]:


data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
sns.pairplot(data);

