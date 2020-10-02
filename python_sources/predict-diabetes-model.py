#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


dataset = pd.read_csv('../input/diabetes.csv')


# In[ ]:


dataset.head()


# In[ ]:


y = dataset.iloc[:,-1]
x = dataset.iloc[:,:7]


# In[ ]:


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 6, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# In[ ]:


y_predict = classifier.predict(x_test)


# In[ ]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
cm


# In[ ]:


y = pd.DataFrame(y_predict)
y.to_csv('out.csv',index=False,header=False)


# In[ ]:




