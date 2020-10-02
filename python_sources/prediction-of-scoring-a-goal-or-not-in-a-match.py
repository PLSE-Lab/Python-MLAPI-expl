#!/usr/bin/env python
# coding: utf-8

# Import Necessary Files

# In[ ]:


from sklearn import metrics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree


# Load Dataset.

# In[ ]:


train = pd.read_csv('../input/Goal_Pred.csv')


# Print Shape of the Training Dataset.

# In[ ]:


print( train.shape )


# Let us see the top 5 data in the Training Data

# In[ ]:


print( train.head() )


# In[ ]:


print( train.info() )


# In[ ]:


y = train['Target']
print( y )


# In[ ]:


x = train.drop(['Name', 'Target' ], axis=1)
print( x )


# In[ ]:


X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(x, y, test_size = 0.33, random_state = 15)


# In[ ]:


print(X_train.shape)


# In[ ]:


print(X_test.shape)


# In[ ]:


print(Y_train.shape)


# In[ ]:


print(Y_test.shape)


# In[ ]:


lm = linear_model.SGDClassifier()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error of SGDC Classifier: ",mse,"\n")


# In[ ]:


print (metrics.classification_report(Y_test, Y_pred))


# In[ ]:


accuracy = accuracy_score(Y_pred, Y_test)
print("Accuracy of SGDC Classifier: ",accuracy,"\n")


# In[ ]:


clf_test = GaussianNB() 
clf_test = clf_test.fit(X_train, Y_train)
pred = clf_test.predict(X_test)
print (metrics.classification_report(Y_test,pred))


# In[ ]:


accuracy = accuracy_score(pred, Y_test)
print("Accuracy of Naive Bayes Classifier: ",accuracy,"\n")


# In[ ]:


clf_test2 = tree.DecisionTreeClassifier(min_samples_split = 11) 
clf_test2 = clf_test2.fit(X_train, Y_train)
pred2 = clf_test2.predict(X_test)
print (metrics.classification_report(Y_test, pred2))


# In[ ]:


accuracy2 = accuracy_score(pred2, Y_test)
print("Accuracy of Decision Tree Classifier: ",accuracy2,"\n")


# In[ ]:


clf_test3 = svm.SVC(kernel = 'rbf', C = 1.0) 
clf_test3 = clf_test3.fit(X_train, Y_train)
pred3 = clf_test3.predict(X_test)
print (metrics.classification_report(Y_test, pred3))


# In[ ]:


accuracy3 = accuracy_score(pred3, Y_test)
print("Accuracy of Support Vector MAchine Classifier: ",accuracy3,"\n")

