#!/usr/bin/env python
# coding: utf-8

# IMPORT

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# Read

# In[ ]:



df = pd.read_csv('../input/heart.csv')
df = df.fillna(0)


# X and y

# In[ ]:


df.columns


# In[ ]:


y = df['target'].values
df = df.fillna(0)
X = df.drop(columns=['target'], axis=1).values


# Model

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)
model = GradientBoostingClassifier(random_state=39, n_estimators=50)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = np.mean(pred == y_test)
print('accuracy: ', accuracy*100, '%')


# Model with RandomForest

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)
model = RandomForestClassifier(random_state=39, n_estimators=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = np.mean(pred == y_test)
print('accuracy: ', accuracy*100, '%')


# With more estimators Random Forest has the same accuracy

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
model = RandomForestClassifier(random_state=33, n_estimators=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = np.mean(pred == y_test)
print('accuracy: ', accuracy*100, '%')


# Because of small file it depends on random state.
