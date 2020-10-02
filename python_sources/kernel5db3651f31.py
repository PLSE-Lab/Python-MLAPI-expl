#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv("../input/heart-disease-uci/heart.csv")


# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


df.head(3)


# In[ ]:


df.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), df['target'], test_size = .2, random_state=25) 


# In[ ]:


X_train.shape


# In[ ]:


y_test.shape


# In[ ]:


X_train.head(3)


# In[ ]:


y_train.shape


# In[ ]:


y_train.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier 
classifier=RandomForestClassifier(n_estimators=20,random_state=30,max_features=13,max_depth=5)
classifier.fit(X_train,y_train)


# In[ ]:


y_predict = classifier.predict(X_test)
y_pred_quant = classifier.predict_proba(X_test)[:, 1]


# In[ ]:


y_predict


# In[ ]:


y_pred_quant


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predict)
cm


# In[ ]:


print('Accuracy:{}'.format((21+28)/61))

