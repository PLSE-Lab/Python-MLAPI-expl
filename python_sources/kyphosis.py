#!/usr/bin/env python
# coding: utf-8

# In[8]:




import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# In[3]:


file = '../input/kyphosis.csv'
df = pd.read_csv(file)


# In[4]:


df.head()


# In[6]:


df.info()


# In[9]:


sns.pairplot(df,hue = 'Kyphosis',size = 3,markers=["o", "D"])


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X = df.drop(['Kyphosis'],axis = 1)


# In[12]:


y = df['Kyphosis']


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


dtree = DecisionTreeClassifier()


# In[16]:


dtree.fit(X_train,y_train)


# In[17]:


predictions = dtree.predict(X_test)


# In[18]:


from sklearn.metrics import classification_report,confusion_matrix


# In[19]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[20]:


from sklearn.ensemble import RandomForestClassifier


# In[21]:


rfc = RandomForestClassifier(n_estimators = 100)


# In[22]:


rfc.fit(X_train,y_train)


# In[23]:


rfc_pred = rfc.predict(X_test)


# In[24]:


print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))


# In[ ]:




