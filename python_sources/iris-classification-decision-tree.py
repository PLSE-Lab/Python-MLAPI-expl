#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df=pd.read_csv('../input/iris/Iris.csv')


# In[ ]:


df.head()


# In[ ]:


#checking if missing values are there
df.isnull().all()


# In[ ]:


#independent and dependent variable
X=df.iloc[:,:-1]
X


# In[ ]:


Y=df.iloc[:,5]
Y


# In[ ]:


#Encoded dependent variable
from sklearn.preprocessing import LabelEncoder
Label_Y=LabelEncoder()
Y=Label_Y.fit_transform(Y)


# In[ ]:


Y


# In[ ]:


##splitting train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[ ]:


#applying algorithm
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)


# In[ ]:


#predicting
Y_pred=classifier.predict(X_test)


# In[ ]:


Y_pred


# In[ ]:


#checking Accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_test,Y_pred)
print(cm)


# In[ ]:


accuracy=accuracy_score(Y_test,Y_pred)


# In[ ]:


accuracy


# In[ ]:




