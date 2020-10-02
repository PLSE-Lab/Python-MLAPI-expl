#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/kyphosis.csv')


# In[ ]:


df.head()


# In[ ]:


#Age is in months
#Number is number of vertabrae involved in operation
#Start is the number of the first (top most) vertabrae operated on


# In[ ]:


df.info()


# In[ ]:


sns.pairplot(df,hue='Kyphosis')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop('Kyphosis',axis=1)


# In[ ]:


y = df['Kyphosis']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtree = DecisionTreeClassifier()


# In[ ]:


dtree.fit(X_train,y_train)


# In[ ]:


predictions = dtree.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))


# In[ ]:


df['Kyphosis'].value_counts()


# In[ ]:




