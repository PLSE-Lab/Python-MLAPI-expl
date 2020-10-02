#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


df.head()


# In[ ]:


col = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


# In[ ]:


label = df['Outcome'].values
features = df[list(col)].values


# In[ ]:


label


# In[ ]:


features


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.30)


# In[ ]:


clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(x_train, y_train)


# In[ ]:


acc = clf.score(x_train, y_train)
print(acc*100)


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


pred = clf.predict(x_train)
print(classification_report(y_train, pred))


# In[ ]:


print(confusion_matrix(y_train, pred))


# In[ ]:


arr = confusion_matrix(y_train, pred)


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


plt.plot(arr[0],'green',label='Accuracy')
plt.plot(arr[1],'red',label='Loss')
plt.title('Tconfusion matrix')
plt.xlabel('Epoch')
plt.figure()

