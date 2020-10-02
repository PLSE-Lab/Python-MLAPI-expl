#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Import necessary modules

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# # Loading the dataset

# In[ ]:


df = pd.read_csv('../input/Iris.csv')
df.head(10)


# # Inspecting the data

# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.describe(include='all')


# In[ ]:


df.drop('Id', inplace=True, axis=1)


# In[ ]:


df.head()


# In[ ]:


df.isnull().any()


# In[ ]:


df['Species'].value_counts()


# # Visualizing the data

# In[ ]:


sns.pairplot(df, hue='Species', palette='Set2');


# In[ ]:


sns.distplot(df['SepalLengthCm'], color='red');


# In[ ]:


sns.boxplot(df['SepalLengthCm'])


# # Preparing the data

# Since algorithm only accepts numerical values first we have to encode the 'Species' column using LabelEncoder from scikit-learn 

# ### Label Encoding

# In[ ]:


label = LabelEncoder()
df['Species'] = label.fit_transform(df['Species'])


# ### Normalizing features

# In[ ]:


scaler = StandardScaler()
scaled_df = scaler.fit_transform(df.drop('Species', axis=1))
X = scaled_df
y = df['Species'].as_matrix()


# In[ ]:


df.head(12)


# ### Splitting into training and testing sets

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# # Modelling

# ### Logistic Regression

# In[ ]:


clf_lr = LogisticRegression(C=10)
clf_lr.fit(X_train, y_train)
y_pred = clf_lr.predict(X_test)
print(accuracy_score(y_test, y_pred))


# ### KNN

# In[ ]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))


# ### Linear Support Vector Machine (SVM)

# In[ ]:


linear_svm = LinearSVC()
linear_svm.fit(X_train, y_train)
y_pred = linear_svm.predict(X_test)
print(accuracy_score(y_test, y_pred))


# ### SVM (with 'rbf' kernel)

# In[ ]:


svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(accuracy_score(y_test, y_pred))


# In[ ]:




