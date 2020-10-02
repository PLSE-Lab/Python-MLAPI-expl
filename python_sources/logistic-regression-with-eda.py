#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ds = pd.read_csv('../input/advertising.csv')


# In[ ]:


ds.info()


# In[ ]:


ds.head(5)


# In[ ]:


sns.pairplot(ds)


# In[ ]:


ds.head(2)


# In[ ]:


ds['Age'].plot.hist(bins=40)


# In[ ]:


sns.jointplot(data=ds, x='Age', y='Area Income')


# In[ ]:


sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ds, kind='kde', color='red')


# In[ ]:


sns.pairplot(ds, hue='Clicked on Ad')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


ds.head(2)


# In[ ]:


y = ds['Clicked on Ad']
X = ds[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))
print('\n')
print(confusion_matrix(y_test, y_pred))


# In[ ]:


# Predicting Model based on only two variables(Age, Daily Internet Usage)


# In[ ]:


y = ds['Clicked on Ad']
X = ds[['Daily Internet Usage', 'Age']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))
print('\n')
print(confusion_matrix(y_test, y_pred))


# In[ ]:




