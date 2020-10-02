#!/usr/bin/env python
# coding: utf-8

# Kyphosis is a spinal disorder. This project attempts to predict the success rate of corrective spinal surgery. The dataset contains the columns Kyphosis(indicating if the surgery was successful), Age (indicating the age of the patient), Number (the number of vertebrae involved in the operation) and Start (the topmost vertebrae that was operated on).

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/kyphosis.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


#handling categorical data
kyphosis = pd.get_dummies(df['Kyphosis'], drop_first=True)
df = pd.concat([df, kyphosis], axis=1).drop(['Kyphosis'], axis=1)
df.head()


# In[ ]:


sns.pairplot(df, hue='present')


# In[ ]:


#splitting train and test data
from sklearn.model_selection import train_test_split
X = df.drop('present', axis=1)
y = df['present']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


#using descision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
prediction = dtree.predict(X_test)
#checking performance of the model
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))


# In[ ]:


#using random forests
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
predictionRFC = rfc.predict(X_test)
print(classification_report(y_test, predictionRFC))
print(confusion_matrix(y_test, predictionRFC))

