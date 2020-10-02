#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()


# In[ ]:


df.describe()


# In[ ]:


X = df.drop('target', axis=1)
X.head()


# In[ ]:


y = df["target"]
y.head()


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(random_state=42)
# clf = LogisticRegression(random_state=42)


# In[ ]:


from sklearn.pipeline import Pipeline
pipeline = Pipeline([
        ('clf', clf)
])


# In[ ]:


pipeline.fit(X_train, y_train)


# In[ ]:


pipeline.score(X_test, y_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = pipeline.predict(X_test)

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.model_selection import GridSearchCV

hyperparameters = { 
                    'clf__n_estimators': np.arange(1, 80, 2),
                    'clf__max_depth': np.arange(1, 8, 1)
                  }
grd = GridSearchCV(pipeline, hyperparameters,  cv=5)
grd.fit(X_train, y_train)
grd.best_params_


# In[ ]:


pipeline = pipeline.set_params(**grd.best_params_)
pipeline.fit(X_train, y_train)


# In[ ]:


y_pred = pipeline.predict(X_test)

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred)) 

