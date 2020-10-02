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


# ## Importing the libraries

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Importing the dataset

# In[ ]:


df = pd.read_csv('/kaggle/input/iris/Iris.csv')


# **EDA**

# In[ ]:


sns_plot = sns.pairplot(df, hue='Species')


# In[ ]:


df.head()


# In[ ]:


df.shape
X = df.iloc[:, 1:5]
y = df[['Species']]


# In[ ]:


X.head()


# In[ ]:


y.head()


# ## Splitting into train and test sets

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## KNeighborsClassifier model

# In[ ]:


classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)


# **Predicted Value**

# In[ ]:


pred = classifier.predict(X_test)


# ## Accuracy

# In[ ]:


accuracy = accuracy_score(y_test, pred)
print(accuracy)


# ## Let's find best k_value for KNeighborsClassifier().

# In[ ]:


error = []
k_value = []


# In[ ]:


for k in range(40):
    k_value.append(k+1)
    # Using KNN
    knn = KNeighborsClassifier(n_neighbors=k+1)
    print(knn.fit(X_train, y_train))
    pred = knn.predict(X_test)

    error_ = 1 - accuracy_score(y_test, pred)
    error.append(error_)

    # %%
    # confusion_matrix_ = confusion_matrix(y_test, pred)
    # print(confusion_matrix_)
    # classification_report_ = classification_report(y_test, pred)
    # print(classification_report_)


# ## Plot Error and k_value

# In[ ]:


# %%
plt.plot(k_value, error)
plt.xlabel('K value')
plt.ylabel('Error')
plt.show()


# In[ ]:


def knn_fun(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    print(knn.fit(X_train, y_train))
    pred = knn.predict(X_test)

    error_ = 1 - accuracy_score(y_test, pred)
    error.append(error_)

    confusion_matrix_ = confusion_matrix(y_test, pred)
    print(confusion_matrix_)
    classification_report_ = classification_report(y_test, pred)
    print(classification_report_)


# In[ ]:


error_np = np.array(error)
k_value_np = np.array(k_value)

error_min_index = error_np.argmin().item() # numpy int to python int
k_value_ = k_value_np[error_min_index]

#%%
# for minimum error
print('knn for k={}'.format(k_value_, knn_fun(k_value_)))


# In[ ]:


print('Best K_value is {}.'.format(k_value_))

