#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


dataset = pd.read_csv('../input/diabetes.csv')
dataset.info()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.head()


# In[ ]:


X = dataset.iloc[:,:-1]
y = dataset.iloc[:,[-1]]


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2,random_state = 0)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 6, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


y=pd.DataFrame(y_pred)
y.to_csv('out.csv',index=False,header=False)


# In[ ]:




