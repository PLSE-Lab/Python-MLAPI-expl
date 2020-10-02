#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('../input/heart.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop('target',axis=1),df['target'],test_size=0.3,random_state=101)


# ***Prediction using Decision Tree***

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[ ]:


pred_tree = dtree.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report , confusion_matrix
print(classification_report(y_test,pred_tree))
print("\n")
print(confusion_matrix(y_test,pred_tree))


# **Prediction using Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=400)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


pred = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report , confusion_matrix
print(classification_report(y_test,pred))
print("\n")
print(confusion_matrix(y_test,pred))


# **END**
