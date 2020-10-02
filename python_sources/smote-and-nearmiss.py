#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv("../input/creditcardfraud/creditcard.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


# In[ ]:


df = df.drop(['Time'], axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


amt = np.array(df['Amount'])
amt = amt.reshape(-1,1)
amt.shape
amt[:10]


# In[ ]:


sc = StandardScaler()
df['Amount'] = sc.fit_transform(amt)


# In[ ]:


df.head()


# In[ ]:


x = df.drop('Class', axis=1)
y = df['Class']
x.shape, y.shape


# In[ ]:


lr = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[ ]:


lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:


smote = SMOTE()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
x_train, y_train = smote.fit_sample(x_train, y_train)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[ ]:


np.bincount(y_train), np.bincount(y_test)


# In[ ]:


lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:


nm = NearMiss()
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)
x_train, y_train = nm.fit_sample(x_train, y_train)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[ ]:


np.bincount(y_train), np.bincount(y_test)


# In[ ]:


lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:




