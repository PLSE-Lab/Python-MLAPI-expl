#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
file=pd.read_csv('../input/winequality-red.csv')
file.head()


# In[ ]:


#input variables
x_test=file.iloc[:,0:10]
x_test


# In[ ]:


# from info i know about what data it is & data type
x_test.info()


# In[ ]:


#basic information about dataset
x_test.describe()


# In[ ]:


#1.Skewness, in statistics, is the degree of distortion from the symmetrical bell curve, 
# or normal distribution, in a set of data. Skewness can be negative, positive
from scipy.stats import skew
x_test.skew()


# In[ ]:


# kurtosis is used for know about outliers
from scipy.stats import kurtosis
x_test.kurtosis()


# In[ ]:


#1. An image histogram is a type of histogram that acts as a graphical
#representation of the tonal distribution in a digital image 
import matplotlib.pyplot as pl
x_test.hist()


# In[ ]:


#boxplot for know about outliers
box=x_test.plot(kind='box')


# In[ ]:


# iam doing the normalization for same scale
x=(x_test-x_test.min()/x_test.max()-x_test.min())
x.head()


# In[ ]:


#iam change the 7 above good wine quality below 7 are bad quality in string.
bins=(2,6.5,8)
group_names=['bad','good']
file['quality']=pd.cut(file['quality'],bins=bins,labels=group_names)


# In[ ]:


# y is output variable
y=file['quality']
y.head()


# In[ ]:


#dividing train , test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state = 0)


# In[ ]:


#apply decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


#predict the x_test
y_pred = classifier.predict(X_test)


# In[ ]:


#finding the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


#finding accuracy
from sklearn.metrics import accuracy_score 
Accuracy_Score = accuracy_score(y_test, y_pred)


# In[ ]:


Accuracy_Score


# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[ ]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


from sklearn.metrics import accuracy_score 
Accuracy_Score = accuracy_score(y_test, y_pred)


# In[ ]:


Accuracy_Score


# In[ ]:


#randomforest give the more accuracy.


# In[ ]:





# In[ ]:





# In[ ]:




