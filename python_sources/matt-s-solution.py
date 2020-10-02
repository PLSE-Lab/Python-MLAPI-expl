#!/usr/bin/env python
# coding: utf-8

# In[139]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# In[140]:


# Load data
df = pd.read_csv('../input/Iris.csv')
df


# In[141]:


# One-hot encode classes
lb = LabelBinarizer()
lb.fit(df['Species'])
lb.classes_


# In[142]:


# Build dataset
X = df.iloc[:,1:-1].as_matrix()
y = lb.transform(df['Species'])
print(X.shape, y.shape)


# In[143]:


# Analyze dataset
plt.figure(2, figsize=(8, 6))
plt.clf()

cols = list(df.columns)
attr0 = 0
attr1 = 1

plt.scatter(X[:,attr0], X[:,attr1], c=y, edgecolor='k')
plt.xlabel(cols[attr0+1])
plt.ylabel(cols[attr1+1])
plt.show()


# In[144]:


# Divide dataset into train & test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
print(X_train.shape, X_test.shape)


# In[145]:


# Build classifier and fit on training data
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)


# In[146]:


# Analyze feature importance
imp = rf.feature_importances_

plt.figure()
plt.title('Feature importances')
plt.bar(range(X.shape[1]), imp, color='b', align='center')
plt.xlim([-1, X.shape[1]])
plt.xticks(range(X.shape[1]))
plt.show()


# In[147]:


# Test model performance
y_hats = rf.predict(X_test)
print(classification_report(y_test, y_hats, target_names=lb.classes_))

